"""One-document Stages 1-7 orchestration.

This module only sequences existing stage implementations. It does not duplicate
detection, grouping, callout, component, enrichment or export logic.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import uuid

from . import callouts, components, detection, enrichment, exports, grouping, indexer, progress, delivery, page_markdown
from .provenance import code_fingerprint
from .qwen import ModelConfig, ModelError, QwenClient, capture_diagnostics
from . import async_inference, performance
from .graph import page_results

VERSION = '2'
STATE_VERSION = 1
RUN_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')


def _view_budget(manifest, minimum, retries, enabled, extra=0):
    """Compatibility helper: manual budgets never scale with views or retries."""
    if type(minimum) is not int or minimum <= 0:
        raise ValueError('Request budget must be a finite positive integer')
    return minimum


def _component_config(args):
    return components.Config(image_max=args.image_max,context_fraction=.02,details_only=True,
        paragraph_descriptions=False,observed_profiles=False,
        profile_inventory_max_requests=getattr(args,'profile_inventory_max_requests',6),
        profile_verification_max_requests=getattr(args,'profile_verification_max_requests',6),
        paragraph_max_requests=getattr(args,'paragraph_max_requests',6))


def _page_worker(task):
    """One source page per process; no PDF handles or DB connections are shared."""
    stage, positional, keywords, diagnostics = task[:4]
    page = task[4] if len(task)>4 else None
    if progress.cancelled():
        return {'result':None,'stats':asdict(Stats()),'error':'Cancelled before page processing'}
    operations={2:'Detecting Assembly Details',4:'Detecting and extracting Callouts',5:'Extracting source-supported component fields'}
    progress.start(stage, page, operations[stage])
    stats = Stats()
    model = keywords['model']
    keywords = dict(keywords, client=_client(model, stats, directory=diagnostics, stage=stage))
    try:
        result = {2: detection, 4: callouts, 5: components}[stage].run(*positional, **keywords)
        # A detection worker intentionally selects one page of the full PDF.
        # Its document-level 'incomplete/limited' flag is not a page failure.
        ok = not result.get('errors') and not result.get('model_failures') and (stage == 2 or not result.get('incomplete'))
        progress.done(stage, page, ok=ok, failed=bool(result.get('errors') or result.get('model_failures')),
                      callouts=result.get('occurrences') if stage==4 else None,
                      empty=stage==2 and result.get('page_decisions')=={'NO_DETAIL':1} and ok,
                      reused=bool(result.get('pages_reused') or result.get('views_reused')),
                      reason='Detection review flags retained' if stage==2 and result.get('page_decisions',{}).get('REVIEW') else '')
        return {'result': result, 'stats': asdict(stats), 'error': None}
    except BaseException as exc:
        progress.done(stage, page, ok=False, reason=_safe_error(exc))
        if not isinstance(exc, Exception): raise
        return {'result': None, 'stats': asdict(stats), 'error': _safe_error(exc)}


def _worker_result(packed, stats):
    for key, value in packed['stats'].items():
        setattr(stats, key, getattr(stats, key) + value)
    if packed['error']:
        raise RuntimeError(packed['error'])
    return packed['result']


def _parallel_detection(source, output, config, model, index_path, selected_count, args, stats):
    # Initialize schema before workers open independent SQLite connections.
    with sqlite3.connect(index_path) as db:
        detection.migrate(db)
    selected = [p-1 for p in args.pages] if getattr(args,'pages',None) else list(range(selected_count))
    tasks = [(2, (source, output), dict(config=config, model=model, index_path=index_path,
              pages=[page + 1], force=args.force), output/'diagnostics', page+1)
             for page in selected]
    progress.plan(2, [p+1 for p in selected])
    result = dict(run_id='parallel-'+uuid.uuid4().hex, pages_completed=0, pages_reused=0,
                  candidates=0, review=0, errors=0, model_failures=0, model_requests=0,
                  limited=False, page_decisions={}, manifests=[], page_failures=[])
    with ProcessPoolExecutor(max_workers=args.page_workers, mp_context=get_context('spawn'), **async_inference.worker_options(progress.worker_options())) as pool:
        # map yields original page order even when workers finish out of order.
        completed = (((task[-1]-1),future.result()) for task,future in
                     performance.bounded_results(pool,_page_worker,tasks,args.page_workers,async_inference.cancel)) if async_inference.enabled() else zip(selected,pool.map(_page_worker,tasks))
        for page, packed in completed:
            try:
                item = _worker_result(packed, stats)
                for key in ('pages_completed','pages_reused','candidates','review','errors','model_failures','model_requests'):
                    result[key] += item.get(key, 0)
                result['manifests'].extend(item.get('manifests', []))
                for decision, count in item.get('page_decisions', {}).items():
                    result['page_decisions'][decision] = result['page_decisions'].get(decision, 0) + count
                if item.get('stop_reason'):
                    result['limited'] = True
            except Exception as exc:
                result['errors'] += 1
                result['page_failures'].append({'page':page+1,'error':_safe_error(exc)})
            if not getattr(args, 'live_progress', False):
                print(f'stage 2 page {page+1}: outcome saved', flush=True)
    return result


def _page_jobs(stage, expected, args, directories, index_path, stage3_pages=None):
    ordered = sorted(expected.items())
    progress.plan(stage, [index+1 for index in sorted(expected)])
    if args.page_workers == 1:
        for index, upstream in ordered:
            yield index, upstream, None
        return
    # Avoid concurrent schema migration; all subsequent transactions are short.
    with sqlite3.connect(index_path) as db:
        {4:callouts,5:components}[stage].migrate(db)
    tasks = []
    for index, upstream in ordered:
        minimum = args.stage4_max_requests if stage == 4 else args.stage5_max_requests
        model = _model(args, max_requests=_view_budget(upstream['manifest'], minimum, args.vision_retries, args.all_views, extra=2 * int(stage == 5)),
                       timeout_seconds=args.vision_timeout, max_tokens=8192 if stage==4 else 4096,
                       retries=args.vision_retries)
        output = directories[f'stage_{stage}']
        keywords = dict(index_path=index_path, model=model, force=args.force)
        if args.all_views:
            keywords['complete_views'] = True
        if stage == 4:
            positional = (upstream['stage2_manifest'], output)
            keywords.update(stage3=upstream['manifest'], config=callouts.Config(image_max=args.image_max, details_only=True))
        else:
            positional = (stage3_pages[str(index)]['manifest'], upstream['manifest'], output)
            keywords['config'] = _component_config(args)
        tasks.append((stage, positional, keywords, output/'diagnostics', index+1))
    with ProcessPoolExecutor(max_workers=args.page_workers, mp_context=get_context('spawn'), **async_inference.worker_options(progress.worker_options())) as pool:
        if async_inference.enabled():
            for task,packed in page_results(pool,_page_worker,tasks,args.page_workers,async_inference.cancel):
                index=task[-1]-1
                yield index,expected[index],packed
        else:
            for (index, upstream), packed in zip(ordered, pool.map(_page_worker, tasks)):
                yield index, upstream, packed


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_digest(path):
    return grouping.file_digest(path)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _safe_error(exc):
    message = str(exc).replace('\n', ' ').strip()
    return (type(exc).__name__ + (': ' + message if message else ''))[:400]


def _one_based(value):
    return value + 1 if isinstance(value, int) else value


positive = indexer.positive


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def timeout(value):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 600:
        raise argparse.ArgumentTypeError('must be finite and between 0 and 600 seconds')
    return number


@dataclass
class Stats:
    requests: int = 0
    wall_seconds: float = 0.0
    transport_failures: int = 0
    timeouts: int = 0
    budget_exhaustions: int = 0


class TimedClient:
    """Transparent request count/wall-time recorder; never stores payloads."""

    def __init__(self, client, stats, *, directory=None, stage=None):
        self._client = client
        self.stats = stats
        self.config = client.config
        self.directory = directory
        self.stage = stage

    @property
    def requests(self):
        return self._client.requests

    @property
    def response_models(self):
        return self._client.response_models

    def request_json(self, *args, **kwargs):
        before = self._client.requests
        started = time.monotonic()
        request_error = None
        prior_diagnostics = len(getattr(self._client, 'request_diagnostics', []))
        progress.request_start()
        try:
            scope = capture_diagnostics(self._client, self.directory, stage=self.stage,
                subject_id='pipeline', validate=False) if self.directory and not getattr(
                    self._client, '_diagnostic_scope', None) else nullcontext()
            with scope:
                return self._client.request_json(*args, **kwargs)
        except BaseException as exc:
            request_error = _safe_error(exc)
            message = str(exc).lower()
            if isinstance(exc,ModelError) and 'budget exhausted' in message:
                self.stats.budget_exhaustions += 1
            if (isinstance(exc, ModelError) and 'budget exhausted' not in message and
                    'qwen request failed' in message):
                self.stats.transport_failures += 1
                if 'timeout' in message:
                    self.stats.timeouts += 1
            raise
        finally:
            self.stats.requests += self._client.requests - before
            self.stats.wall_seconds += time.monotonic() - started
            events = getattr(self._client, 'request_diagnostics', [])[prior_diagnostics:]
            progress.request_end(time.monotonic()-started, error=request_error, timeouts=sum(bool(e.get('timeout_classification')) for e in events))


def _model(args, *, max_requests, timeout_seconds, max_tokens, retries):
    return ModelConfig(
        args.qwen_endpoint or os.getenv('AS_OCR_QWEN_ENDPOINT'),
        args.qwen_model or os.getenv('AS_OCR_QWEN_MODEL'),
        args.qwen_identity or os.getenv('AS_OCR_QWEN_IDENTITY'),
        timeout_seconds, max_tokens, retries, max_requests,
        os.getenv('AS_OCR_QWEN_API_KEY'),
    )


def _public_model(model):
    result = model.public()
    return result


def _client(model, stats, *, directory=None, stage=None):
    return TimedClient(QwenClient(model), stats, directory=directory, stage=stage)


def _stage_public(name, stage):
    keys = {
        'stage_1': ('documents', 'completed', 'reused', 'errors', 'limited', 'incomplete'),
        'stage_2': ('pages_completed', 'pages_reused', 'candidates', 'review', 'errors',
                    'model_failures', 'model_requests', 'limited', 'page_decisions',
                    'incomplete', 'stop_reason'),
        'stage_3': ('pages_completed', 'pages_reused', 'errors', 'views', 'details',
                    'confirmed_multiview_groups', 'model_requests', 'model_failures',
                    'pages_skipped'),
        'stage_4': ('views_completed', 'views_reused', 'errors', 'model_requests',
                    'occurrences', 'markers', 'incomplete'),
        'stage_5': ('views_completed', 'views_reused', 'errors', 'model_requests',
                    'candidates', 'observations', 'incomplete'),
        'stage_6': ('lookups_completed', 'lookups_reused', 'errors', 'model_requests',
                    'execution_status', 'source_coverage_status', 'native_hits',
                    'supported_assignments'),
        'stage_7': ('export_status', 'execution_status', 'source_coverage_status',
                    'acceptance_status', 'complete', 'validation_errors', 'counts', 'reused'),
    }.get(name, ())
    return {key: stage[key] for key in keys if key in stage}


def _stage_result(summary, stage, *, started, stats, status='COMPLETE', reused=False,
                  summary_value=None, error=None, outputs=None):
    finished = _now()
    item = {
        'status': status,
        'started_at': started,
        'finished_at': finished,
        'duration_seconds': round(time.monotonic() - summary['_clock'][stage], 3),
        'reused_from_previous_invocation': reused,
        'model': {
            'requests': stats.requests if stats else 0,
            'request_wall_seconds': round(stats.wall_seconds, 3) if stats else 0,
            'transport_failures': stats.transport_failures if stats else 0,
            'client_observed_timeouts': stats.timeouts if stats else 0,
            'request_budget_exhaustions':stats.budget_exhaustions if stats else 0,
        },
        'public': _stage_public(stage, summary_value or {}),
        'outputs': outputs or {},
    }
    if error:
        item['error'] = error
    summary['stages'][stage] = item
    return item


def _iter_manifests(paths, stage):
    result = []
    for path in paths:
        path = Path(path)
        try:
            saved = _load_json(path)
        except (OSError, ValueError):
            continue
        if saved.get('stage') == stage:
            result.append((path, saved))
    return result


def _source_paths(source, output_root, run_id):
    root = output_root.resolve()
    folder = root / delivery.identifier(source.name, indexer.fingerprint(source)) / '_internal' / 'runs' / run_id
    if folder.resolve() != folder.absolute():
        raise ValueError('Output path resolves through a symbolic link outside the intended root')
    if folder.resolve() == root or root not in folder.resolve().parents:
        raise ValueError('Run directory must be inside the selected output root')
    return folder


def _load_state(run_dir):
    path = Path(run_dir) / 'run.json'
    if not path.is_file():
        return None
    try:
        state = _load_json(path)
    except (OSError, ValueError) as exc:
        raise ValueError('Existing run state is unreadable; use a new run ID') from exc
    if not isinstance(state, dict):
        raise ValueError('Existing run state is invalid; use a new run ID')
    if state.get('state_version') != STATE_VERSION:
        raise ValueError('Existing run state has an unsupported version; use a new run ID')
    return state


def _validate_run_target(run_dir, source, *, resume, force, source_sha=None, check_hash=True):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        if resume:
            raise ValueError('Resume requested but the run directory does not exist')
        return None
    if not run_dir.is_dir():
        raise ValueError('Run target exists and is not a directory')
    entries = list(run_dir.iterdir())
    if not entries:
        if resume:
            raise ValueError('Resume requested but the run directory is empty')
        return None
    state = _load_state(run_dir)
    if state is None:
        raise ValueError('Run target contains unrelated data without a valid run.json; choose a new run ID')
    recorded = Path(state.get('source', {}).get('path', '')).resolve()
    if recorded != source.resolve():
        raise ValueError('Run target belongs to a different source; choose a new run ID')
    if not (resume or force):
        raise ValueError('Run target already exists; use --resume to continue it or choose a new run ID')
    if check_hash and source_sha and state.get('source', {}).get('sha256') not in {None, source_sha}:
        raise ValueError('Source content changed for this run; choose a new run ID')
    return state


def _index_page_count(index_path, document_id):
    with sqlite3.connect(index_path) as db:
        row = db.execute('SELECT page_count FROM documents WHERE id=?', (document_id,)).fetchone()
    if row is None:
        raise ValueError('Selected source is absent from the Stage 1 index')
    return int(row[0])


def _stage2_budget(args, page_count):
    """Finite ceiling: full page + bounded crops + coverage, including retries.

    A manual override remains exact. Unknown dry-run page counts are reported as
    pending instead of inventing a fixed total. No request is forced by this cap.
    """
    if args.stage2_max_requests is not None:
        return args.stage2_max_requests
    if not page_count:
        return None
    return page_count * (2 + args.stage2_max_followups) * (1 + args.stage2_retries)


def _stage2_key(pages, settings):
    return _digest({'logic_sha256':code_fingerprint(),'settings': settings, 'pages': {
        str(index): value['sha256'] if value else None for index, value in pages.items()
    }})


def _stage_key(upstream, settings):
    return _digest({'logic_sha256':code_fingerprint(),'upstream': upstream, 'settings': settings})


def _validate_stage3(prior, key, expected):
    if not prior or prior.get('key') != key or prior.get('summary', {}).get('errors'):
        return False
    records = prior.get('pages', {})
    for index, upstream in expected.items():
        if upstream is None:
            continue
        record = records.get(str(index))
        if not record:
            return False
        if upstream['decision'] == 'NO_DETAIL':
            if not record.get('skipped'):
                return False
            continue
        path = Path(record.get('manifest', ''))
        if not path.is_file() or _file_digest(path) != record.get('sha256'):
            return False
        try:
            saved = _load_json(path)
        except (OSError, ValueError):
            return False
        if saved.get('stage') != 3 or saved.get('upstream', {}).get('manifest_sha256') != upstream['sha256']:
            return False
    return True


def _validate_stage4(prior, key, expected):
    if not prior or prior.get('key') != key or prior.get('summary', {}).get('errors'):
        return False
    records = prior.get('pages', {})
    for index, upstream in expected.items():
        if upstream is None:
            continue
        record = records.get(str(index))
        if not record or not record.get('manifest'):
            return False
        path = Path(record['manifest'])
        if not path.is_file() or _file_digest(path) != record.get('sha256'):
            return False
        try:
            saved = _load_json(path)
        except (OSError, ValueError):
            return False
        if (saved.get('stage') != 4 or
                saved.get('upstream', {}).get('stage3', {}).get('sha256') != upstream['sha256']):
            return False
    return True


def _validate_stage5(prior, key, expected):
    if not prior or prior.get('key') != key or prior.get('summary', {}).get('errors'):
        return False
    records = prior.get('pages', {})
    for index, upstream in expected.items():
        if upstream is None:
            continue
        record = records.get(str(index))
        if not record or not record.get('manifest'):
            return False
        path = Path(record['manifest'])
        if not path.is_file() or _file_digest(path) != record.get('sha256'):
            return False
        try:
            saved = _load_json(path)
        except (OSError, ValueError):
            return False
        up = saved.get('upstream', {})
        if (saved.get('stage') != 5 or up.get('stage3', {}).get('sha256') != upstream['stage3'] or
                up.get('stage4', {}).get('sha256') != upstream['stage4']):
            return False
    return True


def _validate_stage6(prior, key, expected):
    if not prior or prior.get('key') != key:
        return False
    summary_value = prior.get('summary', {})
    if summary_value.get('execution_status') != 'COMPLETE':
        return False
    path = Path(prior.get('manifest', ''))
    if not path.is_file() or _file_digest(path) != prior.get('manifest_sha256'):
        return False
    try:
        saved = _load_json(path)
    except (OSError, ValueError):
        return False
    if saved.get('stage') != 6 or saved.get('execution_status') != 'COMPLETE':
        return False
    sources = {str(Path(item['stage5_path']).resolve()): item['stage5_sha256']
               for item in saved.get('sources', [])}
    if len(sources) != len(expected):
        return False
    for item in expected.values():
        if item is None:
            return False
        if sources.get(str(Path(item['manifest']).resolve())) != item['sha256']:
            return False
    return True


def _skip_records(page_records, stage):
    result = []
    for index, record in sorted(page_records.items()):
        if record.get('skipped'):
            result.append({
                'stage_skipped': stage,
                'page_index_zero_based': index,
                'page_index_one_based': _one_based(index),
                'page_id': record.get('page_id'),
                'stage_2_manifest': record.get('stage2_manifest'),
                'decision': record.get('decision', 'NO_DETAIL'),
                'reason': record.get('reason', []),
                'downstream_skipped': True,
            })
    return result


def _write_skips(stage_dir, records):
    _write_json(Path(stage_dir) / 'skipped-pages.json', records)


def _manifest_page_map(paths, stage):
    result = {}
    for path, saved in _iter_manifests(paths, stage):
        index = saved.get('page_index')
        if isinstance(index, int):
            result[index] = {'manifest': str(Path(path).resolve()), 'sha256': _file_digest(path),
                             'saved': saved}
    return result


def _run_pipeline(args, source, run_dir, state):
    summary = {
        'schema_version': 1,
        'run_id': args.run_id,
        'pipeline_version': VERSION,
        'source': {'path': str(source)},
        'started_at': _now(),
        'finished_at': None,
        'status': 'RUNNING',
        'execution_status': 'RUNNING',
        'completeness_status': 'UNKNOWN',
        'review_status': 'UNKNOWN',
        'export_status': 'NOT_RUN',
        'successful_empty': False,
        'effective_limits': {},
        'counts': {},
        'pages': {},
        'stages': {},
        'enrichment': {},
        'outputs': {},
        'failures': [],
        '_started_monotonic': time.monotonic(),
        '_clock': {f'stage_{number}': time.monotonic() for number in range(1, 8)},
    }
    stage_dirs = {f'stage_{number}': run_dir / f'stage_{number}' for number in range(1, 8)}
    for directory in stage_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    index_path = stage_dirs['stage_1'] / 'index.sqlite3'
    source_hash = indexer.fingerprint(source)
    summary['source']['sha256'] = source_hash
    summary['outputs'] = {
        'run_dir': str(run_dir),
        'public_root': str(args.output_root.resolve() / delivery.identifier(source.name, source_hash)),
        **{name: str(path) for name, path in stage_dirs.items()},
        'run_summary': str(run_dir / 'run-summary.json'),
        'run_state': str(run_dir / 'run.json'),
    }

    state.update({
        'state_version': STATE_VERSION,
        'pipeline_version': VERSION,
        'run_id': args.run_id,
        'source': summary['source'],
        'created_at': state.get('created_at') or _now(),
        'last_started_at': _now(),
        'run_dir': str(run_dir),
        'stages': state.get('stages', {}),
    })
    selection = {'pages':args.pages,'max_pages':args.max_pages}
    if state.get('selection') is not None and state['selection'] != selection:
        raise ValueError('Resume selection differs from original pages; use a fresh isolated run')
    state['selection']=selection

    stage1_started = _now()
    summary['_clock']['stage_1'] = time.monotonic()
    stage1_stats = Stats()
    try:
        stage1 = indexer.run(source, stage_dirs['stage_1'], max_documents=None,
                             max_pages=args.max_pages, preview_max=args.preview_max,
                             force=args.force, **({'pages':args.pages} if args.pages else {}))
    except Exception as exc:
        summary['failures'].append({'stage': 'stage_1', 'code': 'EXECUTION_FAILED',
                                    'reason': _safe_error(exc)})
        _stage_result(summary, 'stage_1', started=stage1_started, stats=stage1_stats,
                      status='FAILED', error=_safe_error(exc),
                      outputs={'index': str(index_path)})
        return _finish(summary, state, run_dir, fatal=True)

    try:
        page_count = _index_page_count(index_path, source_hash)
        selected_pages = args.pages or list(range(1,(min(page_count,args.max_pages) if args.max_pages else page_count)+1))
        if max(selected_pages)>page_count:
            raise ValueError('Selected original page exceeds source page count')
        selected_count = len(selected_pages)
        budget = _stage2_budget(args, selected_count)
        summary['effective_limits'] = _effective_limits(args, selected_count, budget)
    except Exception as exc:
        summary['failures'].append({'stage': 'stage_1', 'code': 'INDEX_UNAVAILABLE',
                                    'reason': _safe_error(exc)})
        _stage_result(summary, 'stage_1', started=stage1_started, stats=stage1_stats,
                      status='FAILED', summary_value=stage1, error=_safe_error(exc),
                      outputs={'index': str(index_path)})
        state['stages']['stage_1'] = {'summary': stage1, 'completed_at': _now()}
        return _finish(summary, state, run_dir, fatal=True)
    summary['pages']['page_count'] = page_count
    summary['pages']['selected'] = selected_count
    summary['pages']['limited'] = selected_count < page_count
    summary['pages']['original_pages'] = selected_pages
    summary['scope'] = 'SELECTED_PAGES' if summary['pages']['limited'] else 'FULL_CATALOG'
    def transcribe_page_text():
        markdown_model = _model(args,max_requests=args.markdown_max_requests,
            timeout_seconds=args.vision_timeout,max_tokens=8192,retries=args.vision_retries)
        markdown_stats=Stats()
        try:
            if getattr(args,'markdown_engine','legacy')=='compact':
                summary['page_markdown']=page_markdown.run_compact(source,index_path,summary['outputs']['public_root'],
                    run_dir/('page_ocr-'+uuid.uuid4().hex),selected_pages,markdown_model,
                    page_markdown.CompactConfig(max_requests=args.markdown_max_requests,
                        timeout_seconds=args.vision_timeout,
                        prefer_native=getattr(args,'markdown_prefer_native',False)),workers=args.page_workers)
                markdown_stats.requests=summary['page_markdown']['requests']
                markdown_stats.wall_seconds=sum(q.get('duration_seconds',0) for p in summary['page_markdown']['pages'] for q in p.get('requests',[]))
            else:
                summary['page_markdown']=page_markdown.run(source,index_path,summary['outputs']['public_root'],
                    run_dir/('page_ocr-'+uuid.uuid4().hex),selected_pages,
                    lambda:_client(markdown_model,markdown_stats,directory=stage_dirs['stage_1']/'diagnostics',stage=1),
                    page_markdown.Config(image_max=args.image_max,max_crops=args.markdown_max_crops))
        except Exception as exc:
            summary['page_markdown']={'complete':False,'pages':[], 'error':_safe_error(exc)}
            summary['failures'].append({'stage':'stage_1','code':'PAGE_MARKDOWN_FAILED','reason':_safe_error(exc)})
        summary['page_markdown']['requests']=markdown_stats.requests
        summary['page_markdown']['finished_at']=_now()
        if 'stage_1' in summary['stages']:
            summary['stages']['stage_1']['model']={
                'requests':markdown_stats.requests,'request_wall_seconds':round(markdown_stats.wall_seconds,3),
                'transport_failures':markdown_stats.transport_failures,'client_observed_timeouts':markdown_stats.timeouts,
                'request_budget_exhaustions':markdown_stats.budget_exhaustions}
            state['page_markdown']={k:v for k,v in summary['page_markdown'].items() if k!='pages'}
        return markdown_stats

    if args.markdown_after_extraction:
        summary['_deferred_markdown']=transcribe_page_text
        summary['page_markdown']={'complete':False,'pages':[],'status':'PENDING_AFTER_EXTRACTION'}
    else:
        stage1_stats=transcribe_page_text()

    _stage_result(summary, 'stage_1', started=stage1_started, stats=stage1_stats,
                  summary_value=stage1,
                  outputs={'index': str(index_path), 'previews': str(stage_dirs['stage_1'] / 'previews')})
    state['stages']['stage_1'] = {'summary': stage1, 'completed_at': _now()}
    _write_json(run_dir / 'run.json', state)
    _write_summary(summary, run_dir)
    print(f'stage 1: {stage1.get("completed", 0)} pages indexed; {stage1.get("reused", 0)} reused')
    print(f'stage 2: request budget {budget} for {selected_count} selected pages')

    stage2_started = _now()
    summary['_clock']['stage_2'] = time.monotonic()
    stage2_stats = Stats()
    stage2_config = detection.Config(args.stage2_analysis_max, args.stage2_crop_max,
                                     args.stage2_overview_max, args.stage2_context_margin,
                                     args.stage2_max_followups, args.stage2_followup_max)
    stage2_model = _model(args, max_requests=budget, timeout_seconds=args.stage2_timeout,
                          max_tokens=args.stage2_max_tokens, retries=args.stage2_retries)
    async_inference.set_stage2_budget(budget)
    try:
        if args.page_workers > 1 or args.live_progress:
            stage2 = _parallel_detection(source, stage_dirs['stage_2'], stage2_config, stage2_model,
                                         index_path, selected_count, args, stage2_stats)
        else:
            stage2 = detection.run(source, stage_dirs['stage_2'], stage2_config, stage2_model,
                                   index_path, max_pages=args.max_pages, force=args.force,
                                   **({'pages':args.pages} if args.pages else {}),
                                   client=_client(stage2_model, stage2_stats,
                                                  directory=stage_dirs['stage_2']/'diagnostics', stage=2))
    except Exception as exc:
        summary['failures'].append({'stage': 'stage_2', 'code': 'EXECUTION_FAILED',
                                    'reason': _safe_error(exc)})
        _stage_result(summary, 'stage_2', started=stage2_started, stats=stage2_stats,
                      status='FAILED', error=_safe_error(exc))
        return _finish(summary, state, run_dir, fatal=True)

    stage2_pages = _manifest_page_map(stage2.get('manifests', []), 2)
    for index, record in stage2_pages.items():
        record['decision'] = record['saved'].get('page_decision')
        record['reason'] = record['saved'].get('skip_reason') or []
        record['page_id'] = record['saved'].get('page_id')
    _stage_result(summary, 'stage_2', started=stage2_started, stats=stage2_stats,
                  summary_value=stage2,
                  outputs={'page_count': len(stage2_pages),
                           'manifests': stage2.get('manifests', [])})
    state['stages']['stage_2'] = {
        'summary': stage2,
        'completed_at': _now(),
        'pages': {str(index): {'manifest': value['manifest'], 'sha256': value['sha256'],
                              'decision': value.get('decision'), 'reason': value.get('reason'),
                              'page_id': value.get('page_id')}
                  for index, value in stage2_pages.items()},
    }
    _write_json(run_dir / 'run.json', state)
    _write_summary(summary, run_dir)
    decisions = {}
    for record in stage2_pages.values():
        decisions[record.get('decision')] = decisions.get(record.get('decision'), 0) + 1
    print(f'stage 2: {stage2.get("pages_completed", 0)} completed, '
          f'{stage2.get("pages_reused", 0)} reused; decisions {decisions}')
    if args.stop_after_stage2:
        summary.update(status='PAUSED_STAGE_2', execution_status='PAUSED',
                       completeness_status='INCOMPLETE', export_status='NOT_RUN',
                       duration_seconds=round(time.monotonic()-summary['_started_monotonic'],3))
        state.update(status='PAUSED_STAGE_2',execution_status='PAUSED')
        _write_json(run_dir/'run.json',state)
        _write_summary(summary,run_dir)
        print('Paused at the saved Stage 2 checkpoint; Stage 3 and later stages have not started.')
        return summary

    expected_upstream = {}
    for index in (p-1 for p in selected_pages):
        record = stage2_pages.get(index)
        if record is None or record.get('decision') not in {
                'NO_DETAIL', 'CANDIDATES', 'REVIEW'}:
            expected_upstream[index] = None
        else:
            expected_upstream[index] = {'sha256': record['sha256'],
                                        'manifest': record['manifest'],
                                        'decision': record.get('decision'),
                                        'page_id': record.get('page_id'),
                                        'reason': record.get('reason', [])}

    # Stage 3: page-local grouping, deterministic and offline by default.
    stage3_started = _now()
    summary['_clock']['stage_3'] = time.monotonic()
    stage3_stats = Stats()
    stage3_settings = {
        'version': grouping.VERSION,
        'prompt_sha256': grouping.digest(grouping.GROUP_PROMPT),
        'configuration': asdict(grouping.Config(mode=args.grouping_mode)),
        'model':_public_model(_model(args,max_requests=args.stage3_max_requests,timeout_seconds=args.vision_timeout,max_tokens=4096,retries=args.vision_retries)) if args.grouping_mode=='hybrid' else None,
        'page_count': selected_count,
    }
    stage3_key = _stage2_key(expected_upstream, stage3_settings)
    prior3 = None if args.force else state.get('stages', {}).get('stage_3')
    stage3_pages = {}
    if _validate_stage3(prior3, stage3_key, expected_upstream):
        stage3 = prior3.get('summary', {})
        stage3_pages = prior3.get('pages', {})
        _stage_result(summary, 'stage_3', started=stage3_started, stats=stage3_stats,
                      reused=True, summary_value=stage3,
                      outputs={'page_count': len(stage3_pages),
                               'manifests': stage3.get('manifests', [])})
        print(f'stage 3: reused {stage3.get("pages_completed", 0)} completed and '
              f'{stage3.get("pages_skipped", 0)} skipped pages')
    else:
        stage3 = {'run_id': uuid.uuid4().hex, 'pages_completed': 0, 'pages_reused': 0,
                  'errors': 0, 'views': 0, 'details': 0,
                  'confirmed_multiview_groups': 0, 'model_requests': 0,
                  'model_failures': 0, 'manifests': [], 'pages_skipped': 0, 'skipped': []}
        progress.plan(3, selected_pages)
        for index, upstream in sorted(expected_upstream.items()):
            if upstream is None:
                progress.done(3,index+1,ok=False,reason='Upstream detection failed or unavailable; not an empty page')
                continue
            if upstream['decision'] == 'NO_DETAIL':
                stage3['pages_skipped'] += 1
                stage3_pages[str(index)] = {
                    'manifest': None, 'skipped': True, 'decision': 'NO_DETAIL',
                    'reason': upstream.get('reason', []),
                    'page_id': upstream.get('page_id'), 'stage2_manifest': upstream['manifest'],
                }
                progress.done(3,index+1,empty=True,reason='Grouping not required after confirmed NO_DETAIL')
                continue
            progress.start(3, index+1, 'Cropping and grouping Detail views')
            result = None
            try:
                result = grouping.run([upstream['manifest']], stage_dirs['stage_3'],
                                      index_path=index_path, force=args.force,
                                      **(dict(config=grouping.Config(mode='hybrid'),
                                      model=_model(args,max_requests=args.stage3_max_requests,timeout_seconds=args.vision_timeout,max_tokens=4096,retries=args.vision_retries),
                                      client=_client(_model(args,max_requests=args.stage3_max_requests,timeout_seconds=args.vision_timeout,max_tokens=4096,retries=args.vision_retries),stage3_stats,directory=stage_dirs['stage_3']/'diagnostics',stage=3)) if args.grouping_mode=='hybrid' else {}))
                if result.get('errors'):
                    stage3['errors'] += result['errors']
                stage3['pages_completed'] += result.get('pages_completed', 0)
                stage3['pages_reused'] += result.get('pages_reused', 0)
                for key in ('views', 'details', 'confirmed_multiview_groups', 'model_failures'):
                    stage3[key] += result.get(key, 0)
                manifesto = (result.get('manifests') or [None])[-1]
                if manifesto:
                    stage3['manifests'].append(manifesto)
                    saved = _load_json(manifesto)
                    stage3_pages[str(index)] = {
                        'manifest': str(Path(manifesto).resolve()),
                        'sha256': _file_digest(manifesto), 'skipped': False,
                        'decision': 'GROUPED', 'page_id': saved.get('page_id'),
                        'stage2_manifest': upstream['manifest'],
                    }
                else:
                    stage3['errors'] += 1
                    stage3_pages[str(index)] = {'manifest': None, 'skipped': False,
                                                'error': 'No Stage 3 manifest returned'}
            except Exception as exc:
                stage3['errors'] += 1
                stage3_pages[str(index)] = {'manifest': None, 'skipped': False,
                                            'error': _safe_error(exc)}
                summary['failures'].append({'stage': 'stage_3', 'page': _one_based(index),
                                            'code': 'EXECUTION_FAILED', 'reason': _safe_error(exc)})
                print(f'stage 3 page {_one_based(index)}: {_safe_error(exc)}', file=sys.stderr)
            progress.done(3, index+1, ok=bool(stage3_pages[str(index)].get('manifest')), details=(result or {}).get('details',0))
            state['stages']['stage_3'] = {
                'key': stage3_key, 'summary': stage3, 'pages': stage3_pages,
                'completed_at': _now() if index == max(expected_upstream) else None,
            }
            _write_json(run_dir / 'run.json', state)
            _write_summary(summary, run_dir)
        stage3['model_requests'] = stage3_stats.requests
        _stage_result(summary, 'stage_3', started=stage3_started, stats=stage3_stats,
                      status='PARTIAL' if stage3['errors'] or stage3['model_failures'] else 'COMPLETE',
                      summary_value=stage3,
                      outputs={'page_count': len(stage3_pages),
                               'manifests': stage3.get('manifests', [])})
        state['stages']['stage_3'] = {'key': stage3_key, 'summary': stage3,
                                      'pages': stage3_pages, 'completed_at': _now()}
        _write_json(run_dir / 'run.json', state)
        print(f'stage 3: {stage3["views"]} views, {stage3["details"]} Details, '
              f'{stage3["pages_skipped"]} NO_DETAIL skips')
    _write_skips(stage_dirs['stage_3'], _skip_records(
        {int(k): v for k, v in stage3_pages.items()}, 'stage_3'))
    _write_summary(summary, run_dir)

    # Stage 4: callout occurrence extraction per page.
    stage4_started = _now()
    summary['_clock']['stage_4'] = time.monotonic()
    stage4_stats = Stats()
    stage4_settings = {
        'version': callouts.VERSION, 'prompt_version': callouts.PROMPT_VERSION,
        'schema_version': callouts.SCHEMA_VERSION,
        'prompt_sha256': grouping.digest(callouts.DETAIL_PROMPT),
        'schema_sha256': grouping.digest(callouts.Analysis.model_json_schema()),
        'configuration': asdict(callouts.Config(image_max=args.image_max, details_only=True)),
        'model': _public_model(_model(args, max_requests=args.stage4_max_requests,
                                      timeout_seconds=args.vision_timeout,
                                      max_tokens=8192, retries=args.vision_retries)),
    }
    stage4_expected = {
        int(index): {'sha256': record['sha256'], 'manifest': record['manifest'],
                     'stage2_manifest': record.get('stage2_manifest')}
        for index, record in stage3_pages.items()
        if record.get('manifest') and not record.get('skipped')
    }
    stage4_key = _stage_key(stage4_expected, stage4_settings)
    prior4 = None if args.force else state.get('stages', {}).get('stage_4')
    stage4_pages = {}
    if _validate_stage4(prior4, stage4_key, stage4_expected):
        stage4 = prior4.get('summary', {})
        stage4_pages = prior4.get('pages', {})
        reused_partial = bool(stage4.get('incomplete')) or any(
            record.get('state') in {'PARTIAL', 'FAILED'}
            for record in stage4_pages.values())
        _stage_result(summary, 'stage_4', started=stage4_started, stats=stage4_stats,
                      status='PARTIAL' if reused_partial else 'COMPLETE',
                      reused=True, summary_value=stage4,
                      outputs={'page_count': len(stage4_pages),
                               'manifests': stage4.get('manifests', [])})
        print(f'stage 4: reused {stage4.get("views_completed", 0)} completed views')
    else:
        stage4 = {'views_completed': 0, 'views_reused': 0, 'errors': 0,
                  'model_requests': 0, 'occurrences': 0, 'markers': 0,
                  'incomplete': False, 'manifests': []}
        for index, upstream, packed in _page_jobs(4, stage4_expected, args, stage_dirs, index_path):
            model = _model(args, max_requests=_view_budget(upstream['manifest'], args.stage4_max_requests, args.vision_retries, args.all_views),
                           timeout_seconds=args.vision_timeout, max_tokens=8192,
                           retries=args.vision_retries)
            try:
                if packed is not None:
                    result = _worker_result(packed, stage4_stats)
                else:
                    progress.start(4,index+1,'Detecting and extracting Callouts')
                    result = callouts.run(upstream['stage2_manifest'], stage_dirs['stage_4'],
                                          index_path=index_path, stage3=upstream['manifest'],
                                          config=callouts.Config(image_max=args.image_max, details_only=True),
                                          model=model, client=_client(model, stage4_stats,
                                              directory=stage_dirs['stage_4']/'diagnostics', stage=4),
                                          force=args.force, **({'complete_views': True} if args.all_views else {}))
                if packed is None:
                    progress.done(4,index+1,ok=not result.get('errors') and not result.get('incomplete'),failed=bool(result.get('errors')),callouts=result.get('occurrences') if 4==4 else None)
                stage4['views_completed'] += result.get('views_completed', 0)
                stage4['views_reused'] += result.get('views_reused', 0)
                stage4['errors'] += result.get('errors', 0)
                stage4['occurrences'] += result.get('occurrences', 0)
                stage4['markers'] += result.get('markers', 0)
                stage4['incomplete'] |= bool(result.get('incomplete'))
                manifesto = result.get('manifest')
                if manifesto and Path(manifesto).is_file():
                    stage4['manifests'].append(manifesto)
                    saved = _load_json(manifesto)
                    stage4_pages[str(index)] = {
                        'manifest': str(Path(manifesto).resolve()),
                        'sha256': _file_digest(manifesto), 'state': saved.get('state'),
                        'page_id': saved.get('upstream', {}).get('page_id'),
                    }
                else:
                    stage4['errors'] += 1
            except Exception as exc:
                stage4['errors'] += 1
                stage4_pages[str(index)] = {'manifest': None, 'error': _safe_error(exc)}
                summary['failures'].append({'stage': 'stage_4', 'page': _one_based(index),
                                            'code': 'EXECUTION_FAILED', 'reason': _safe_error(exc)})
                print(f'stage 4 page {_one_based(index)}: {_safe_error(exc)}', file=sys.stderr)
            stage4['model_requests'] = stage4_stats.requests
            state['stages']['stage_4'] = {'key': stage4_key, 'summary': stage4,
                                          'pages': stage4_pages, 'completed_at': None}
            _write_json(run_dir / 'run.json', state)
            _write_summary(summary, run_dir)
        stage4['model_requests'] = stage4_stats.requests
        stage4['incomplete'] = bool(stage4['errors']) or any(
            record.get('state') in {'PARTIAL', 'FAILED'} for record in stage4_pages.values())
        _stage_result(summary, 'stage_4', started=stage4_started, stats=stage4_stats,
                      status='PARTIAL' if stage4['incomplete'] else 'COMPLETE',
                      summary_value=stage4,
                      outputs={'page_count': len(stage4_pages),
                               'manifests': stage4.get('manifests', [])})
        state['stages']['stage_4'] = {'key': stage4_key, 'summary': stage4,
                                      'pages': stage4_pages, 'completed_at': _now()}
        _write_json(run_dir / 'run.json', state)
        print(f'stage 4: {stage4["occurrences"]} occurrences, '
              f'{stage4["markers"]} markers, {stage4["errors"]} failed views')
    _write_skips(stage_dirs['stage_4'], _skip_records(
        {int(k): v for k, v in stage3_pages.items()}, 'stage_4'))
    _write_summary(summary, run_dir)

    # Stage 5: component candidates and source-grounded properties per page.
    stage5_started = _now()
    summary['_clock']['stage_5'] = time.monotonic()
    stage5_stats = Stats()
    stage5_settings = {
        'paragraph_contract':None,
        'profile_contract':None,
        'profile_count_contract':'distinct_extracted_associated_components_v1',
        'version': components.VERSION, 'prompt_version': components.PROMPT_VERSION,
        'schema_version': components.SCHEMA_VERSION,
        'prompt_sha256': grouping.digest(components.DETAIL_PROMPT),
        'schema_sha256': grouping.digest(components.Analysis.model_json_schema()),
        'configuration': asdict(_component_config(args)),
        'model': _public_model(_model(args, max_requests=args.stage5_max_requests,
                                      timeout_seconds=args.vision_timeout,
                                      max_tokens=4096, retries=args.vision_retries)),
    }
    stage5_expected = {
        int(index): {'stage3': stage3_pages[str(index)]['sha256'],
                     'stage4': record['sha256'], 'manifest': record['manifest']}
        for index, record in stage4_pages.items()
        if record.get('manifest') and stage3_pages.get(index, {}).get('manifest')
    }
    stage5_key = _stage_key(stage5_expected, stage5_settings)
    prior5 = None if args.force else state.get('stages', {}).get('stage_5')
    stage5_pages = {}
    if _validate_stage5(prior5, stage5_key, stage5_expected):
        stage5 = prior5.get('summary', {})
        stage5_pages = prior5.get('pages', {})
        reused_partial = bool(stage5.get('incomplete')) or any(
            record.get('state') in {'PARTIAL', 'FAILED'}
            for record in stage5_pages.values())
        _stage_result(summary, 'stage_5', started=stage5_started, stats=stage5_stats,
                      status='PARTIAL' if reused_partial else 'COMPLETE',
                      reused=True, summary_value=stage5,
                      outputs={'page_count': len(stage5_pages),
                               'manifests': stage5.get('manifests', [])})
        print(f'stage 5: reused {stage5.get("candidates", 0)} candidates')
    else:
        stage5 = {'views_completed': 0, 'views_reused': 0, 'errors': 0,
                  'model_requests': 0, 'candidates': 0, 'observations': 0,
                  'incomplete': False, 'manifests': []}
        for index, upstream, packed in _page_jobs(5, stage5_expected, args, stage_dirs, index_path, stage3_pages):
            model = _model(args, max_requests=_view_budget(upstream['manifest'], args.stage5_max_requests, args.vision_retries, args.all_views, extra=2),
                           timeout_seconds=args.vision_timeout, max_tokens=4096,
                           retries=args.vision_retries)
            try:
                if packed is not None:
                    result = _worker_result(packed, stage5_stats)
                else:
                    progress.start(5,index+1,'Extracting source-supported component fields')
                    result = components.run(stage3_pages[str(index)]['manifest'],
                                            upstream['manifest'], stage_dirs['stage_5'],
                                            index_path=index_path, config=_component_config(args),
                                            model=model, client=_client(model, stage5_stats,
                                                directory=stage_dirs['stage_5']/'diagnostics', stage=5),
                                            force=args.force, **({'complete_views': True} if args.all_views else {}))
                if packed is None:
                    progress.done(5,index+1,ok=not result.get('errors') and not result.get('incomplete'),failed=bool(result.get('errors')),callouts=result.get('occurrences') if 5==4 else None)
                stage5['views_completed'] += result.get('views_completed', 0)
                stage5['views_reused'] += result.get('views_reused', 0)
                stage5['errors'] += result.get('errors', 0)
                stage5['candidates'] += result.get('candidates', 0)
                stage5['observations'] += result.get('observations', 0)
                stage5['incomplete'] |= bool(result.get('incomplete'))
                manifesto = result.get('manifest')
                if manifesto and Path(manifesto).is_file():
                    stage5['manifests'].append(manifesto)
                    saved = _load_json(manifesto)
                    stage5_pages[str(index)] = {
                        'manifest': str(Path(manifesto).resolve()),
                        'sha256': _file_digest(manifesto), 'state': saved.get('state'),
                        'page_id': saved.get('upstream', {}).get('page_id'),
                    }
                else:
                    stage5['errors'] += 1
            except Exception as exc:
                stage5['errors'] += 1
                stage5_pages[str(index)] = {'manifest': None, 'error': _safe_error(exc)}
                summary['failures'].append({'stage': 'stage_5', 'page': _one_based(index),
                                            'code': 'EXECUTION_FAILED', 'reason': _safe_error(exc)})
                print(f'stage 5 page {_one_based(index)}: {_safe_error(exc)}', file=sys.stderr)
            stage5['model_requests'] = stage5_stats.requests
            state['stages']['stage_5'] = {'key': stage5_key, 'summary': stage5,
                                          'pages': stage5_pages, 'completed_at': None}
            _write_json(run_dir / 'run.json', state)
            _write_summary(summary, run_dir)
        stage5['model_requests'] = stage5_stats.requests
        stage5['incomplete'] = bool(stage5['errors']) or any(
            record.get('state') in {'PARTIAL', 'FAILED'} for record in stage5_pages.values())
        _stage_result(summary, 'stage_5', started=stage5_started, stats=stage5_stats,
                      status='PARTIAL' if stage5['incomplete'] else 'COMPLETE',
                      summary_value=stage5,
                      outputs={'page_count': len(stage5_pages),
                               'manifests': stage5.get('manifests', [])})
        state['stages']['stage_5'] = {'key': stage5_key, 'summary': stage5,
                                      'pages': stage5_pages, 'completed_at': _now()}
        _write_json(run_dir / 'run.json', state)
        print(f'stage 5: {stage5["candidates"]} candidates, '
              f'{stage5["observations"]} observations, {stage5["errors"]} failed views')
    _write_skips(stage_dirs['stage_5'], _skip_records(
        {int(k): v for k, v in stage3_pages.items()}, 'stage_5'))
    _write_summary(summary, run_dir)

    # Stage 6: same-document native retrieval and optional bounded visual windows.
    stage6_started = _now()
    summary['_clock']['stage_6'] = time.monotonic()
    stage6_stats = Stats()
    stage5_inputs = [record['manifest'] for _,record in sorted(stage5_pages.items(),key=lambda item:int(item[0]))
                     if record.get('manifest') and Path(record['manifest']).is_file()]
    stage6_expected = {index: {'manifest': record['manifest'], 'sha256': record['sha256']}
                       for index, record in stage5_pages.items()
                       if record.get('manifest') and Path(record['manifest']).is_file()}
    plan_hash = _file_digest(args.lookup_plan) if args.lookup_plan else None
    stage6_settings = {
        'version': enrichment.VERSION, 'prompt_version': enrichment.PROMPT_VERSION,
        'schema_version': enrichment.SCHEMA_VERSION,
        'prompt_sha256': grouping.digest(enrichment.PROMPT),
        'schema_sha256': grouping.digest(enrichment.Analysis.model_json_schema()),
        'configuration': asdict(enrichment.Config(args.verify_hits,
                                                  args.max_hit_windows,
                                                  args.image_max)),
        'model': _public_model(_model(args, max_requests=args.stage6_max_requests,
                                      timeout_seconds=args.vision_timeout,
                                      max_tokens=4096, retries=args.vision_retries)),
        'lookup_plan_sha256': plan_hash,
    }
    stage6_key = _stage_key(stage6_expected, stage6_settings)
    prior6 = None if args.force else state.get('stages', {}).get('stage_6')
    stage6_manifest = None
    if not stage5_inputs:
        stage6 = {'execution_status': 'SKIPPED', 'source_coverage_status': 'UNAVAILABLE',
                  'failures': [], 'model_requests': 0, 'native_hits': 0,
                  'supported_assignments': 0}
        _stage_result(summary, 'stage_6', started=stage6_started, stats=stage6_stats,
                      status='SKIPPED', summary_value=stage6)
        state['stages']['stage_6'] = {'key': stage6_key, 'summary': stage6,
                                      'manifest': None, 'completed_at': _now()}
        _write_json(run_dir / 'run.json', state)
        print('stage 6: skipped (no eligible Stage 5 inputs)')
    elif _validate_stage6(prior6, stage6_key, stage6_expected):
        stage6 = prior6.get('summary', {})
        stage6_manifest = prior6.get('manifest')
        _stage_result(summary, 'stage_6', started=stage6_started, stats=stage6_stats,
                      status='PARTIAL' if stage6.get('execution_status') != 'COMPLETE' else 'COMPLETE',
                      reused=True, summary_value=stage6,
                      outputs={'manifest': stage6_manifest})
        print(f'stage 6: reused {stage6_manifest}')
    else:
        model = _model(args, max_requests=args.stage6_max_requests,
                       timeout_seconds=args.vision_timeout, max_tokens=4096,
                       retries=args.vision_retries)
        progress.plan(6, [None])
        progress.start(6, None, 'Searching source-supported cross-page references')
        try:
            stage6 = enrichment.run(
                stage5_inputs, stage_dirs['stage_6'], index_path=index_path,
                plan=args.lookup_plan,
                config=enrichment.Config(args.verify_hits, args.max_hit_windows,
                                         args.image_max),
                model=model, client=_client(model, stage6_stats,
                    directory=stage_dirs['stage_6']/'diagnostics', stage=6), force=args.force)
            stage6_manifest = stage6.get('manifest')
            progress.done(6, None, ok=stage6.get('execution_status')=='COMPLETE')
            _stage_result(summary, 'stage_6', started=stage6_started, stats=stage6_stats,
                          status='PARTIAL' if stage6.get('execution_status') != 'COMPLETE' else 'COMPLETE',
                          summary_value=stage6, outputs={'manifest': stage6_manifest})
            state['stages']['stage_6'] = {
                'key': stage6_key, 'summary': stage6, 'manifest': stage6_manifest,
                'manifest_sha256': (_file_digest(stage6_manifest)
                                    if stage6_manifest and Path(stage6_manifest).is_file()
                                    else None),
                'completed_at': _now()}
            _write_json(run_dir / 'run.json', state)
            print(f'stage 6: {stage6.get("execution_status")}; '
                  f'coverage {stage6.get("source_coverage_status")}; '
                  f'{stage6.get("native_hits", 0)} native hits')
        except Exception as exc:
            stage6 = {'execution_status': 'FAILED', 'source_coverage_status': 'UNAVAILABLE',
                      'failures': [{'code': 'EXECUTION_FAILED', 'reason': _safe_error(exc)}],
                      'model_requests': stage6_stats.requests, 'native_hits': 0,
                      'supported_assignments': 0}
            summary['failures'].append({'stage': 'stage_6', 'code': 'EXECUTION_FAILED',
                                        'reason': _safe_error(exc)})
            _stage_result(summary, 'stage_6', started=stage6_started, stats=stage6_stats,
                          status='FAILED', summary_value=stage6, error=_safe_error(exc))
            _write_summary(summary, run_dir)
            progress.done(6, None, ok=False, reason=_safe_error(exc))
            print(f'stage 6: failed ({_safe_error(exc)})', file=sys.stderr)
    _write_summary(summary, run_dir)

    # Stage 7: validation and linked export; complete artifacts are never replaced
    # by a partial publication (enforced by exports.run()).
    stage7_started = _now()
    summary['_clock']['stage_7'] = time.monotonic()
    stage7_stats = Stats()
    stage7_summary = None
    if stage6_manifest:
        try:
            stage7_summary = exports.run(stage6_manifest, stage_dirs['stage_7'],
                                         index_path=index_path, force=args.force)
            status = 'COMPLETE' if stage7_summary.get('complete') else (
                'FAILED' if stage7_summary.get('export_status') == 'FAILED' else 'PARTIAL')
            _stage_result(summary, 'stage_7', started=stage7_started, stats=stage7_stats,
                          status=status, summary_value=stage7_summary,
                          reused=bool(stage7_summary.get('reused')),
                          outputs=stage7_summary.get('artifacts', {}))
            print(f'stage 7: {stage7_summary.get("export_status")} export; '
                  f'validation errors {stage7_summary.get("validation_errors", 0)}')
        except Exception as exc:
            summary['failures'].append({'stage': 'stage_7', 'code': 'EXECUTION_FAILED',
                                        'reason': _safe_error(exc)})
            _stage_result(summary, 'stage_7', started=stage7_started, stats=stage7_stats,
                          status='FAILED', error=_safe_error(exc))
            print(f'stage 7: failed ({_safe_error(exc)})', file=sys.stderr)
    elif stage6.get('execution_status') == 'SKIPPED':
        _stage_result(summary, 'stage_7', started=stage7_started, stats=stage7_stats,
                      status='SKIPPED')
        state['stages']['stage_7'] = {'summary': {}, 'status': 'SKIPPED',
                                      'completed_at': _now()}
        _write_json(run_dir / 'run.json', state)
        print('stage 7: skipped (successful empty result; no eligible pages)')
    else:
        _stage_result(summary, 'stage_7', started=stage7_started, stats=stage7_stats,
                      status='NOT_RUN')
        print('stage 7: not run because Stage 6 did not publish a manifest', file=sys.stderr)

    if (stage7_summary is None and
            summary['stages'].get('stage_7', {}).get('status') == 'FAILED'):
        stage7_summary = {'export_status': 'FAILED', 'execution_status': 'FAILED',
                          'complete': False}
    state['stages']['stage_7'] = {
        'summary': stage7_summary or {},
        'status': summary['stages'].get('stage_7', {}).get('status'),
        'completed_at': _now()}
    _write_json(run_dir / 'run.json', state)
    return _finish(summary, state, run_dir, fatal=False, stage7=stage7_summary or {},
                   stage6=stage6)


def _effective_limits(args, selected_count, budget):
    stage2 = {
        'max_requests': budget,
        'request_budget_source': 'manual override' if args.stage2_max_requests is not None else 'selected pages * (2 + max followups) * (1 + retries)',
        'max_attempts_per_page': (2 + args.stage2_max_followups) * (1 + args.stage2_retries),
        'max_followups_per_page': args.stage2_max_followups,
        'retries': args.stage2_retries,
        'timeout_seconds': args.stage2_timeout,
        'max_tokens': args.stage2_max_tokens,
        'image_max': {'analysis': args.stage2_analysis_max,
                      'followup': args.stage2_followup_max,
                      'crop': args.stage2_crop_max,
                      'overview': args.stage2_overview_max},
    }
    plan = None
    if args.lookup_plan:
        try:
            saved = enrichment.LookupPlan.model_validate_json(Path(args.lookup_plan).read_text())
            plan = {'path': str(Path(args.lookup_plan).resolve()),
                    'index_pages': len(saved.index_pages), 'windows': len(saved.windows)}
        except (OSError, ValueError) as exc:
            raise ValueError('Invalid --lookup-plan: ' + _safe_error(exc)) from None
    return {
        'source': str(args.source),
        'async_inference':getattr(args,'async_inference',False),
        'recognition_scheduler':'langgraph' if getattr(args,'async_inference',False) and args.page_workers>1 else 'serial',
        'max_in_flight':getattr(args,'max_in_flight',2) if getattr(args,'async_inference',False) else None,
        'run_id': args.run_id,
        'page_limit': args.max_pages, 'all_views': args.all_views, 'page_workers': args.page_workers, 'fresh': args.fresh,
        'selected_pages': selected_count,
        'stage_1': {'preview_max': args.preview_max},
        'stage_2': stage2,
        'stage_3': {'mode': args.grouping_mode, 'max_requests_per_page':args.stage3_max_requests if args.grouping_mode=='hybrid' else 0, 'model_requests':0 if args.grouping_mode=='deterministic' else 'bounded'},
        'page_markdown':{'engine':getattr(args,'markdown_engine','legacy'),
                         'prefer_native':getattr(args,'markdown_prefer_native',False),
                         'max_requests_per_page':args.markdown_max_requests,'max_crops':args.markdown_max_crops,
                         'order':'after_extraction' if args.markdown_after_extraction else 'before_detection'},
        'stage_4': {'max_requests_per_page': args.stage4_max_requests,
                    'retries': args.vision_retries, 'timeout_seconds': args.vision_timeout,
                    'max_tokens': 8192, 'image_max': args.image_max},
        'stage_5': {'max_requests_per_page': args.stage5_max_requests,
                    'physical_profile_inventory_enabled':False,'profile_type_inference_enabled':False,
                    'paragraph_description_extraction_enabled':False,
                    'profile_count_source':'stage4_verified_component_leaders',
                    'profile_inventory_max_requests_per_page':getattr(args,'profile_inventory_max_requests',6),
                    'profile_verification_max_requests_per_page':getattr(args,'profile_verification_max_requests',6),
                    'paragraph_max_requests_per_page':getattr(args,'paragraph_max_requests',6),
                    'retries': args.vision_retries, 'timeout_seconds': args.vision_timeout,
                    'max_tokens': 4096, 'image_max': args.image_max},
        'stage_6': {'max_requests_total': args.stage6_max_requests,
                    'verify_native_hits': args.verify_hits,
                    'max_hit_windows': args.max_hit_windows,
                    'retries': args.vision_retries, 'timeout_seconds': args.vision_timeout,
                    'max_tokens': 4096, 'image_max': args.image_max,
                    'lookup_plan': plan,
                    'native_search_scope': 'all indexed pages of the selected PDF',
                    'visual_search_scope': ('saved complete Stage 5 page crops plus explicit '
                                            'lookup-plan windows; no automatic visual scan')},
        'stage_7': {'model_requests': 0},
        'model': {
            'endpoint': args.qwen_endpoint or os.getenv('AS_OCR_QWEN_ENDPOINT'),
            'alias': args.qwen_model or os.getenv('AS_OCR_QWEN_MODEL'),
            'identity': args.qwen_identity or os.getenv('AS_OCR_QWEN_IDENTITY'),
            'api_key_configured': bool(os.getenv('AS_OCR_QWEN_API_KEY')),
        },
    }


def _collect_counts(stage2_pages, stage3_pages, stage4_pages, stage5_pages,
                    stage6_manifest, stage7):
    counts = {
        'candidate_views': 0, 'details': 0, 'review_details': 0, 'alternative_group_proposals':0,
        'callout_occurrences': 0, 'unresolved_occurrence_targets': 0,
        'components': 0, 'accepted_properties': 0,
        'unresolved_component_memberships': 0,
        'enrichment_assignments': 0, 'enrichment_review_matches': 0,
        'exported_rows': 0,
    }
    for record in stage3_pages.values():
        if not record.get('manifest'):
            continue
        try:
            saved = _load_json(record['manifest'])
        except (OSError, ValueError):
            continue
        counts['candidate_views'] += len(saved.get('views', []))
        primary=[d for d in saved.get('details',[]) if d.get('membership_role')!='alternative']
        counts['details'] += len(primary)
        counts['alternative_group_proposals'] += len(saved.get('details',[]))-len(primary)
        counts['review_details'] += sum(detail.get('review_status') == 'REVIEW'
                                        for detail in primary)
    for record in stage4_pages.values():
        if not record.get('manifest'):
            continue
        try:
            saved = _load_json(record['manifest'])
        except (OSError, ValueError):
            continue
        occurrences = saved.get('occurrences', [])
        counts['callout_occurrences'] += len(occurrences)
        counts['unresolved_occurrence_targets'] += sum(
            occurrence.get('target_point') is None for occurrence in occurrences)
    for record in stage5_pages.values():
        if not record.get('manifest'):
            continue
        try:
            saved = _load_json(record['manifest'])
        except (OSError, ValueError):
            continue
        counts['components'] += len(saved.get('candidates', []))
        counts['accepted_properties'] += sum(
            bool(observation.get('accepted'))
            for observation in saved.get('observations', []))
        for candidate in saved.get('candidates', []):
            counts['unresolved_component_memberships'] += sum(
                membership.get('status') not in {'SUPPORTED_BY_STAGE4', 'CONFIRMED'}
                for membership in candidate.get('memberships', []))
    if stage6_manifest:
        try:
            saved = _load_json(stage6_manifest)
            counts['enrichment_assignments'] = len(saved.get('assignments', []))
            counts['enrichment_review_matches'] = sum(
                match.get('status') == 'REVIEW' for match in saved.get('matches', []))
            counts['accepted_properties'] += len(saved.get('assignments', []))
        except (OSError, ValueError):
            pass
    if stage7:
        exported = stage7.get('counts', {})
        counts['internal_evidence_rows'] = sum(exported.values())
        # Public CSV row count is assigned only after delivery succeeds.
    return counts


def _finish(summary, state, run_dir, *, fatal, stage7=None, stage6=None):
    transcribe=summary.pop('_deferred_markdown',None)
    if transcribe is not None and not fatal:
        transcribe()
    summary['finished_at'] = _now()
    stage1 = summary['stages'].get('stage_1', {})
    stage2 = summary['stages'].get('stage_2', {})
    stage3 = summary['stages'].get('stage_3', {})
    stage4 = summary['stages'].get('stage_4', {})
    stage5 = summary['stages'].get('stage_5', {})
    if stage6 is None:
        stage6 = state.get('stages', {}).get('stage_6', {}).get('summary', {})
    if stage7 is None:
        stage7 = state.get('stages', {}).get('stage_7', {}).get('summary')
    stage7 = stage7 or {}
    state3 = state.get('stages', {}).get('stage_3', {}).get('pages', {})
    state4 = state.get('stages', {}).get('stage_4', {}).get('pages', {})
    state5 = state.get('stages', {}).get('stage_5', {}).get('pages', {})
    state2 = state.get('stages', {}).get('stage_2', {}).get('pages', {})
    stage2_pages = {int(k): v for k, v in state2.items()}
    stage3_pages = {int(k): v for k, v in state3.items()}
    stage4_pages = {int(k): v for k, v in state4.items()}
    stage5_pages = {int(k): v for k, v in state5.items()}
    if fatal:
        stage2_pages, stage3_pages, stage4_pages, stage5_pages = {}, {}, {}, {}
        stage6, stage7 = {}, {}

    decisions = {}
    no_detail = []
    review_pages = []
    failed_pages = []
    for index in (p-1 for p in summary['pages'].get('original_pages',range(1,summary['pages'].get('selected',0)+1))):
        record = stage2_pages.get(index)
        if record is None:
            failed_pages.append(index)
            continue
        decision = record.get('decision')
        if decision not in {'NO_DETAIL', 'CANDIDATES', 'REVIEW', 'FAILED'}:
            decision = 'FAILED'
        decisions[decision] = decisions.get(decision, 0) + 1
        if decision == 'NO_DETAIL':
            no_detail.append({'page_index_zero_based': index,
                              'page_index_one_based': _one_based(index),
                              'page_id': record.get('page_id'),
                              'reason': record.get('reason', [])})
        elif decision == 'REVIEW':
            review_pages.append(index)
        elif decision == 'FAILED':
            failed_pages.append(index)

    summary['pages'].update({
        'stage_2_processed': len(stage2_pages),
        'decisions': decisions,
        'no_detail_count': len(no_detail),
        'no_detail_skipped': no_detail,
        'review_pages_one_based': [_one_based(index) for index in review_pages],
        'failed_pages_one_based': [_one_based(index) for index in failed_pages],
    })
    summary['counts'] = _collect_counts(stage2_pages, stage3_pages, stage4_pages,
                                        stage5_pages, stage6.get('manifest') if isinstance(stage6, dict) else None,
                                        stage7)
    coverage = {}
    if stage6.get('manifest'):
        try:
            enriched = _load_json(stage6['manifest'])
            source_coverage = enriched.get('coverage', {})
            coverage = {
                'source_coverage_status': enriched.get('source_coverage_status'),
                'execution_status': enriched.get('execution_status'),
                'indexed_pages_searched_one_based': [
                    _one_based(page) for page in source_coverage.get('indexed_pages_searched', [])],
                'unindexed_pages_one_based': [
                    _one_based(page) for page in (source_coverage.get('unindexed_pages') or [])],
                'visual_unsearched_pages_one_based': [
                    _one_based(page) for page in source_coverage.get('visual_unsearched_pages', [])],
                'unsearched_page_count': len(source_coverage.get('unindexed_pages') or []) + len(
                    source_coverage.get('visual_unsearched_pages', [])),
                'visual_unsearched_page_count': len(source_coverage.get('visual_unsearched_pages', [])),
            }
        except (OSError, ValueError):
            coverage = {}
    summary['enrichment'] = coverage

    execution_errors = sum(int(item.get('public', {}).get('errors', 0) or 0)
                           for name, item in summary['stages'].items()
                           if name in {'stage_1', 'stage_2', 'stage_3', 'stage_4',
                                       'stage_5', 'stage_6'})
    execution_errors += sum(int(item.get('public', {}).get('model_failures', 0) or 0)
                            for name, item in summary['stages'].items()
                            if name in {'stage_2', 'stage_3'})
    execution_errors += len(summary.get('failures', []))
    execution_errors += sum(int(s.get('model',{}).get(k,0)) for s in summary['stages'].values()
                            for k in ('transport_failures','request_budget_exhaustions'))
    partial_stage = any(item.get('status') in {'PARTIAL', 'FAILED', 'NOT_RUN'}
                        for name, item in summary['stages'].items()
                        if name != 'stage_7')
    stage7_execution_partial = stage7.get('execution_status') == 'PARTIAL'
    execution_partial = execution_errors or partial_stage or stage7_execution_partial
    coverage_partial = bool(summary['pages'].get('limited')) or bool(failed_pages)
    if stage1.get('public', {}).get('errors') or stage1.get('public', {}).get('incomplete'):
        coverage_partial = True
    if (stage2.get('public', {}).get('limited') or
            stage2.get('public', {}).get('errors') or
            stage2.get('public', {}).get('model_failures')):
        coverage_partial = True
    if stage4.get('status') == 'PARTIAL' or stage5.get('status') == 'PARTIAL':
        coverage_partial = True
    if coverage.get('source_coverage_status') not in {None, 'COMPLETE'}:
        coverage_partial = True
    if stage7 and stage7.get('source_coverage_status') not in {None, 'COMPLETE'}:
        coverage_partial = True
    if stage7 and stage7.get('export_status') == 'PARTIAL':
        coverage_partial = True
    if not summary.get('page_markdown',{}).get('complete',False):
        coverage_partial = True

    review = bool(review_pages) or int(summary['counts'].get('review_details', 0)) > 0
    review = review or int(summary['counts'].get('enrichment_review_matches', 0)) > 0
    review = review or (stage7.get('acceptance_status') == 'REVIEW')

    selected_empty = (bool(decisions) and set(decisions) == {'NO_DETAIL'} and
                      not failed_pages)
    stage1_incomplete = bool(stage1.get('public', {}).get('errors') or
                             stage1.get('public', {}).get('incomplete'))
    stage2_incomplete = bool(stage2.get('public', {}).get('limited') or
                             stage2.get('public', {}).get('errors') or
                             stage2.get('public', {}).get('model_failures'))
    successful_empty = (selected_empty and not summary['pages'].get('limited') and
                        not stage1_incomplete and not stage2_incomplete and
                        not partial_stage and summary.get('page_markdown',{}).get('complete',False))
    no_export = stage7.get('export_status') not in {'COMPLETE', 'PARTIAL'}
    fatal = fatal or bool(failed_pages and not stage2_pages) or (no_export and not selected_empty)
    if fatal or stage7.get('export_status') == 'FAILED':
        status = 'FAILED'
        execution = 'FAILED'
        completeness = 'FAILED'
    elif execution_partial or coverage_partial or stage7.get('export_status') == 'PARTIAL':
        status = 'PARTIAL'
        execution = 'PARTIAL' if execution_partial else 'COMPLETE'
        completeness = 'PARTIAL'
    elif review:
        status = 'REVIEW'
        execution = 'COMPLETE'
        completeness = 'COMPLETE'
    else:
        status = 'COMPLETE'
        execution = 'COMPLETE'
        completeness = 'COMPLETE'
    summary.update({
        'status': status,
        'execution_status': execution,
        'completeness_status': completeness,
        'review_status': 'REVIEW' if review else 'NONE',
        'export_status': stage7.get('export_status', 'NOT_RUN'),
        'successful_empty': bool(successful_empty),
        'duration_seconds': round(time.monotonic() - summary['_started_monotonic'], 3),
        'selected_page_completion': ('INCOMPLETE' if fatal or execution_partial or failed_pages or
            any(p['status']=='FAILED' for p in summary.get('page_markdown',{}).get('pages',[])) else
            'PARTIAL' if not summary.get('page_markdown',{}).get('complete',False) else 'COMPLETE'),
        'full_catalog_completion': 'INCOMPLETE' if summary['pages'].get('limited') or completeness!='COMPLETE' else 'COMPLETE',
        'extraction_accuracy_verified':False,
    })
    state.update({
        'status': summary['status'],
        'execution_status': summary['execution_status'],
        'finished_at': summary['finished_at'],
        'outputs': summary['outputs'],
        'summary': {key: summary[key] for key in (
            'status', 'execution_status', 'completeness_status', 'review_status',
            'export_status', 'successful_empty', 'finished_at')},
    })
    _write_json(run_dir / 'run.json', state)
    _write_summary(summary, run_dir)
    export_path = stage7.get('artifacts', {}).get('summary')
    if selected_empty and summary.get('outputs', {}).get('public_root'):
        try:
            summary['outputs']['delivery'] = delivery.publish_empty(
                summary['outputs']['public_root'], summary['source']['path'], summary['source']['sha256'],
                complete=successful_empty)
        except (OSError, ValueError) as exc:
            summary['failures'].append({'stage':'stage_7', 'code':'DELIVERY_FAILED', 'reason':_safe_error(exc)})
            summary.update(status='FAILED', export_status='FAILED')
            summary['stages'].setdefault('stage_7', {})['delivery_error'] = _safe_error(exc)
        _write_summary(summary, run_dir)
    if export_path and Path(export_path).is_file() and stage7.get('export_status') in {'COMPLETE','PARTIAL'}:
        try:
            summary['outputs']['delivery'] = delivery.publish(export_path,
                output=Path(summary['outputs']['public_root']).parent,
                catalog_manifest=summary.get('page_markdown',{}).get('manifest'))
            summary['counts']['exported_rows'] = summary['outputs']['delivery']['rows']
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary['failures'].append({'stage':'stage_7', 'code':'DELIVERY_FAILED', 'reason':_safe_error(exc)})
            summary.update(status='FAILED', export_status='FAILED')
            summary['stages'].setdefault('stage_7', {})['delivery_error'] = _safe_error(exc)
        _write_summary(summary, run_dir)
    state['status'] = summary['status']
    if summary['export_status']=='FAILED':
        summary.update(execution_status='FAILED',completeness_status='FAILED',
            selected_page_completion='INCOMPLETE',full_catalog_completion='INCOMPLETE')
        state['execution_status']='FAILED'
        _write_summary(summary,run_dir)
    state['outputs'] = summary['outputs']
    state['summary'].update({key:summary[key] for key in ('status','export_status','execution_status','completeness_status')})
    _write_json(run_dir/'run.json', state)
    if summary['outputs'].get('public_root'):
        delivery.focused_report(summary)
    return summary


def _write_summary(summary, run_dir):
    public = {key: value for key, value in summary.items() if not key.startswith('_')}
    _write_json(Path(run_dir) / 'run-summary.json', public)
    for number in range(1, 8):
        stage = summary.get('stages', {}).get(f'stage_{number}', {})
        stage = dict(stage,status=stage.get('status','NOT_RUN'))
        source = summary.get('source', {})
        stage = dict(stage, source=source.get('path', 'Not recorded') if isinstance(source, dict) else str(source))
        if number == 7 and summary.get('outputs', {}).get('delivery'):
            published = summary['outputs']['delivery']
            stage = dict(stage, delivery_status=published['status'],
                         exported_details=published.get('details'), merged_rows=published.get('rows'))
        outputs = stage.get('outputs', {})
        paths = outputs.get('manifests', [])
        if outputs.get('manifest'):
            paths = [outputs['manifest']]
        if number == 7 and outputs.get('summary'):
            paths = [outputs['summary']]
        stage['original_pages']=summary.get('pages',{}).get('original_pages',[])
        stage['page_failures']=summary.get('failures',[])
        if number==1:
            stage['page_markdown']=summary.get('page_markdown',{})
        if number==7:
            stage['field_trace']=str(Path(summary.get('outputs',{}).get('public_root',run_dir))/'_internal/rows.jsonl')
        processed=set();incomplete=set();failed=set()
        for path in paths:
            if not Path(path).is_file():continue
            saved=_load_json(path);page=saved.get('page_index',saved.get('upstream',{}).get('page_index'))
            if isinstance(page,int):
                processed.add(page+1)
                if saved.get('state')=='FAILED' or saved.get('page_decision')=='FAILED':failed.add(page+1)
                elif saved.get('state')=='PARTIAL' or saved.get('page_decision')=='REVIEW' or saved.get('complete') is False:incomplete.add(page+1)
            if number>=6:
                processed.update(s['page_index']+1 for s in saved.get('sources',[]) if isinstance(s.get('page_index'),int))
        skipped=set()
        skip_path=Path(run_dir)/f'stage_{number}'/'skipped-pages.json'
        if skip_path.is_file():
            skipped={s['page_index_one_based'] for s in _load_json(skip_path)}
        if number==1:
            for p in summary.get('page_markdown',{}).get('pages',[]):
                processed.add(p['source_page'])
                if p['status']=='FAILED':failed.add(p['source_page'])
                elif p['status']=='PARTIAL':incomplete.add(p['source_page'])
        if number>=3:
            skipped.update(p['page_index_one_based'] for p in summary.get('pages',{}).get('no_detail_skipped',[]))
        failed.update(f['page'] for f in summary.get('failures',[]) if f.get('stage')==f'stage_{number}' and isinstance(f.get('page'),int))
        selected=set(stage['original_pages'])
        if number==7:
            # Stage 7 is an aggregate manifest, not a per-page execution file.
            for path in paths:
                if not Path(path).is_file():continue
                exported=_load_json(path)
                for result_path in exported.get('artifacts',{}).get('detail_results',[]):
                    detail=_load_json(result_path)
                    if isinstance(detail.get('page_index'),int):
                        page=detail['page_index']+1;processed.add(page)
                        if detail.get('review_status')=='REVIEW' or exported.get('acceptance_status')=='REVIEW':incomplete.add(page)
        stage['page_inventory']=dict(processed=sorted(processed),skipped=sorted(skipped),failed=sorted(failed),
            incomplete=sorted(incomplete),not_run_or_no_saved_page_manifest=sorted(selected-processed-skipped-failed))
        delivery.review_stage(Path(summary.get('outputs', {}).get('public_root', run_dir)), number,
                              [p for p in paths if Path(p).is_file()], outcome=stage or {'status':'NOT_RUN'})


def _parser():
    parser = argparse.ArgumentParser(
        prog='as-ocr pipeline',
        description='Run Stages 1-7 for exactly one explicit PDF and one run ID.')
    parser.add_argument('pdf', type=Path, help='Explicit PDF path; sibling PDFs are never enumerated')
    parser.add_argument('--run-id', help='Run identifier; default is a UTC timestamp')
    parser.add_argument('--output-root', type=Path, default=Path('output'))
    parser.add_argument('--fresh', action='store_true', help='Require a new run directory and disable every extraction-stage cache')
    parser.add_argument('--live-progress', action='store_true', help='Print worker events and 30-second status summaries')
    parser.add_argument('--resume', action='store_true', help='Continue the exact existing run ID')
    parser.add_argument('--stop-after-stage2', action='store_true',
                        help='Save all detection outcomes and exit before any downstream stage')
    parser.add_argument('--force', action='store_true',
                        help='Reprocess compatible work in the same run; never deletes prior artifacts')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate configuration and print effective limits; no writes or requests')
    parser.add_argument('--page-workers', type=positive, default=1, help='Bounded page processes, 1 through 32 with async inference; GPU concurrency is separate')
    parser.add_argument('--async-inference',action='store_true',help='Reuse async HTTP connections with a shared cross-process request limit')
    parser.add_argument('--max-in-flight',type=int,choices=range(1,33),default=32,help='Global inference limit for this async workload, including page workers (1-32); tune from measured server capacity')
    parser.add_argument('--markdown-engine',choices=['legacy','compact'],default='legacy')
    parser.add_argument('--markdown-prefer-native',action='store_true',help='Compact OCR: try native prose only with an independent complete visual audit')
    parser.add_argument('--all-views', action='store_true', default=True,
                        help='Cover every Detail view (enabled by default; no Detail count cap)')
    parser.add_argument('--max-pages', type=positive,
                        help='Diagnostic limit only; the default processes every page')
    parser.add_argument('--pages',type=positive,nargs='+',help='Explicit nonconsecutive original 1-based PDF pages')
    parser.add_argument('--focused-test',action='store_true',help='Require --pages and a new output root; disable all caches')
    parser.add_argument('--grouping-mode',choices=['deterministic','hybrid'],default='deterministic')
    parser.add_argument('--stage3-max-requests','--grouping-max-requests',type=positive,default=2,help='Manual grouping HTTP-attempt allowance per page; deterministic mode makes no requests')
    parser.add_argument('--markdown-after-extraction',action='store_true',help='Run independent full-page OCR after Stages 2-7; preserve extraction checkpoints before long page transcription')
    parser.add_argument('--markdown-max-requests',type=positive,default=5,help='Full-page OCR attempts per selected page, including crops/retries')
    parser.add_argument('--markdown-max-crops',type=nonnegative,default=4)
    parser.add_argument('--preview-max', type=positive, default=1200)
    parser.add_argument('--qwen-endpoint', help='Override AS_OCR_QWEN_ENDPOINT')
    parser.add_argument('--qwen-model', help='Override AS_OCR_QWEN_MODEL')
    parser.add_argument('--qwen-identity', help='Override AS_OCR_QWEN_IDENTITY')
    parser.add_argument('--stage2-max-requests','--detection-max-requests', type=positive, default=None,
                        help='Total detection attempt ceiling; automatic per selected page unless explicitly overridden')
    parser.add_argument('--stage2-retries', type=nonnegative, default=1)
    parser.add_argument('--stage2-timeout', type=timeout, default=600)
    parser.add_argument('--stage2-max-tokens', type=positive, default=8192)
    parser.add_argument('--stage2-analysis-max', type=positive, default=1280)#1536
    parser.add_argument('--stage2-followup-max', type=positive, default=2048)
    parser.add_argument('--stage2-crop-max', type=positive, default=2400)
    parser.add_argument('--stage2-overview-max', type=positive, default=1600)
    parser.add_argument('--stage2-context-margin', type=float, default=.01)
    parser.add_argument('--stage2-max-followups', type=nonnegative, default=2)
    parser.add_argument('--stage4-max-requests','--callout-max-requests', type=positive, default=10,help='Manual callout and component-leader census HTTP-attempt allowance per page')
    parser.add_argument('--stage5-max-requests','--component-max-requests', type=positive, default=6,help='Manual component-extraction HTTP-attempt allowance per page')
    parser.add_argument('--profile-inventory-max-requests',type=positive,default=6,help='Retained manual setting; physical-profile/type inventory is disabled in the five-column pipeline')
    parser.add_argument('--profile-verification-max-requests',type=positive,default=6,help='Retained manual setting; physical-profile verification is disabled; leader verification uses Stage 4 budget')
    parser.add_argument('--paragraph-max-requests',type=positive,default=6,help='Retained manual setting; CSV paragraph-description extraction is disabled')
    parser.add_argument('--stage6-max-requests','--enrichment-max-requests', type=positive, default=10,help='Manual enrichment HTTP-attempt allowance for the stage run')
    parser.add_argument('--vision-retries', type=nonnegative, default=0)
    parser.add_argument('--vision-timeout', type=timeout, default=180)
    parser.add_argument('--image-max', type=positive, default=2048)
    parser.add_argument('--lookup-plan', type=Path,
                        help='Explicit Stage 6 lookup JSON; otherwise native search only')
    parser.add_argument('--verify-hits', action='store_true',
                        help='Render bounded native-hit windows during Stage 6')
    parser.add_argument('--max-hit-windows', type=nonnegative, default=8)
    return parser


def _validate_args(args):
    if args.pages and (args.max_pages or len(set(args.pages))!=len(args.pages)):
        raise ValueError('--pages must be unique and cannot be combined with --max-pages')
    if not 0<=args.markdown_max_crops<=16:
        raise ValueError('Invalid Markdown crop budget')
    for name in ('stage3_max_requests','stage4_max_requests','stage5_max_requests','stage6_max_requests',
                 'markdown_max_requests','profile_inventory_max_requests','profile_verification_max_requests','paragraph_max_requests'):
        value=getattr(args,name,6)
        if type(value) is not int or value<=0:
            raise ValueError(name+' must be a finite positive integer')
    if args.stage2_max_requests is not None and (type(args.stage2_max_requests) is not int or args.stage2_max_requests<=0):
        raise ValueError('stage2_max_requests must be a finite positive integer')
    if getattr(args,'markdown_engine','legacy')=='compact':
        page_markdown.CompactConfig(max_requests=args.markdown_max_requests).validate()
    if not args.pdf.is_file():
        raise ValueError('Selected source is not an existing file: ' + str(args.pdf))
    if args.pdf.suffix.lower() != '.pdf':
        raise ValueError('Pipeline accepts exactly one explicit PDF file')
    if args.output_root.exists() and not args.output_root.is_dir():
        raise ValueError('Output root exists and is not a directory')
    worker_limit = 32 if getattr(args, 'async_inference', False) else 4
    if not 1 <= args.page_workers <= worker_limit:
        bound = 'thirty-two' if worker_limit == 32 else 'four'
        raise ValueError(f'Page workers must remain between one and {bound}; more than four requires async inference')
    if args.stage2_retries > 3 or args.vision_retries > 3:
        raise ValueError('Retries are limited to zero through three')
    if not 0 <= args.stage2_max_followups <= 4:
        raise ValueError('Stage 2 follow-ups are limited to zero through four')
    if not 256 <= args.image_max <= 2048:
        raise ValueError('Image bound must be between 256 and 2048 pixels')
    if not 0 <= args.max_hit_windows <= 20:
        raise ValueError('Hit-window limit must be between zero and twenty')
    if not 0 <= args.stage2_context_margin <= .2:
        raise ValueError('Stage 2 context margin must be between zero and 0.2')
    if args.stage2_max_tokens > 32768:
        raise ValueError('Stage 2 max tokens must not exceed 32768')
    detection.Config(args.stage2_analysis_max, args.stage2_crop_max, args.stage2_overview_max,
                     args.stage2_context_margin, args.stage2_max_followups,
                     args.stage2_followup_max).validate()
    callouts.Config(image_max=args.image_max).validate()
    components.Config(image_max=args.image_max, context_fraction=.02).validate()
    enrichment.Config(args.verify_hits, args.max_hit_windows, args.image_max).validate()
    if args.lookup_plan and not args.lookup_plan.is_file():
        raise ValueError('Lookup plan does not exist: ' + str(args.lookup_plan))
    if args.lookup_plan:
        plan=enrichment.LookupPlan.model_validate_json(args.lookup_plan.read_text())
        if args.pages and not {p+1 for p in plan.index_pages+[w.page_index for w in plan.windows]} <= set(args.pages):
            raise ValueError('Lookup plan refers to pages outside the explicit original-page selection')
    QwenClient(_model(args, max_requests=args.stage2_max_requests or 1,
                      timeout_seconds=args.stage2_timeout,
                      max_tokens=args.stage2_max_tokens, retries=args.stage2_retries))
    model = _model(args, max_requests=args.stage4_max_requests,
                   timeout_seconds=args.vision_timeout, max_tokens=8192,
                   retries=args.vision_retries)
    QwenClient(model)
    if not model.endpoint or not model.model:
        raise ValueError('Configure AS_OCR_QWEN_ENDPOINT and AS_OCR_QWEN_MODEL (or pass overrides)')
    if args.run_id is not None and not RUN_ID.fullmatch(args.run_id):
        raise ValueError('Run ID must contain only letters, digits, dot, underscore or hyphen')


def _prepare(args):
    if args.focused_test:
        if not args.pages or args.resume or args.output_root.exists():
            raise ValueError('Focused tests require --pages, an absent output root, and no --resume')
        args.fresh=True
    if args.pages:
        args.pages=sorted(args.pages)
    if args.fresh:
        if args.resume:
            raise ValueError('--fresh cannot be combined with --resume')
        args.force = True
    _validate_args(args)
    source = args.pdf.resolve()
    args.source = source
    if args.run_id is None:
        base = 'run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        candidate = base
        while _source_paths(source, args.output_root, candidate).exists():
            candidate = base + '-' + uuid.uuid4().hex[:6]
        args.run_id = candidate
    run_dir = _source_paths(source, args.output_root, args.run_id)
    if args.resume and not run_dir.exists():
        legacy = args.output_root.resolve()/source.parent.name/source.stem/args.run_id
        if legacy.exists():
            run_dir = legacy
    if args.fresh and run_dir.exists():
        raise ValueError('Fresh extraction requires a new, absent run directory')
    state = _validate_run_target(run_dir, source, resume=args.resume, force=args.force,
                                 check_hash=not args.dry_run, source_sha=None)
    return source, run_dir, state


def run_pipeline(args):
    source, run_dir, state = _prepare(args)
    if state is None:
        state = {}
    if args.dry_run:
        selected = len(args.pages) if args.pages else args.max_pages
        budget = _stage2_budget(args, selected or 0)
        limits = _effective_limits(args, selected or 0, budget)
        if selected is None:
            limits['selected_pages'] = 'all'
        return {'dry_run': True, 'source': str(source), 'run_id': args.run_id,
                'run_dir': str(run_dir), 'original_pages':args.pages, 'effective_limits': limits}
    source_hash = indexer.fingerprint(source)
    if state:
        state = _validate_run_target(run_dir, source, resume=True, force=True,
                                     source_sha=source_hash, check_hash=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'source: {source}')
    print(f'run: {args.run_id} -> {run_dir}')
    print('limits: pages=' + (','.join(map(str,args.pages)) if args.pages else 'all' if args.max_pages is None else str(args.max_pages)) +
          f'; stage 1 preview={args.preview_max}')
    print(f'  stage 2: requests={args.stage2_max_requests if args.stage2_max_requests is not None else "auto after indexing"}'
          f'; retries={args.stage2_retries}; timeout={args.stage2_timeout:g}s'
          f'; tokens={args.stage2_max_tokens}; follow-ups={args.stage2_max_followups}'
          f'; images={args.stage2_analysis_max}/{args.stage2_followup_max}/'
          f'{args.stage2_crop_max}/{args.stage2_overview_max}')
    print(f'  page workers: {args.page_workers}; GPU concurrency configured separately')
    print(f'  stage 3: {args.grouping_mode}; requests/page={args.stage3_max_requests if args.grouping_mode=="hybrid" else 0}')
    print(f'  stage 4: {args.stage4_max_requests}/page; retries={args.vision_retries}'
          f'; timeout={args.vision_timeout:g}s; tokens=8192; image={args.image_max}')
    print(f'  stage 5: {args.stage5_max_requests}/page; retries={args.vision_retries}'
          f'; timeout={args.vision_timeout:g}s; tokens=4096; image={args.image_max}')
    print(f'  stage 6: {args.stage6_max_requests}/run; retries={args.vision_retries}'
          f'; timeout={args.vision_timeout:g}s; tokens=4096; image={args.image_max}'
          f'; verify-hits={args.verify_hits}; hit-windows={args.max_hit_windows}')
    print('  stage 7: validation/export; 0 requests')
    with (async_inference.Workload(getattr(args,'max_in_flight',2),run_dir/'performance')
          if getattr(args,'async_inference',False) else nullcontext()):
        return _execute_pipeline(args,source,run_dir,state)


def _execute_pipeline(args,source,run_dir,state):
    if args.live_progress:
        with indexer.fitz.open(source) as document:
            total_pages = len(document)
        log_path=run_dir/'progress.log'
        if log_path.exists():
            log_path=run_dir/('progress-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.log')
        with progress.Reporter(source, total_pages, args.page_workers, log_path):
            return _run_pipeline(args, source, run_dir, state)
    return _run_pipeline(args, source, run_dir, state)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(args)
    except KeyboardInterrupt:
        print('Interrupted; completed stage artifacts and checkpoints were preserved.', file=sys.stderr)
        return 130
    except Exception as exc:
        print('Error: ' + _safe_error(exc), file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(f'status: {result["status"]} '
          f'(execution {result["execution_status"]}, completeness {result["completeness_status"]}, '
          f'export {result["export_status"]})')
    counts = result.get('counts') or {}
    if counts:
        print('counts: ' + ', '.join(
            f'{key}={value}' for key, value in sorted(counts.items())))
    if result.get('enrichment'):
        print('enrichment unsearched pages: '
              f'{result["enrichment"].get("unsearched_page_count", "unknown")}')
    print(f'summary: {result["outputs"]["run_summary"]}')
    export_summary = (result.get('stages', {}).get('stage_7', {})
                      .get('outputs', {}).get('summary'))
    if export_summary:
        print(f'export: {export_summary}')
    return 0 if result.get('status') in {'COMPLETE', 'REVIEW', 'PAUSED_STAGE_2'} else 1

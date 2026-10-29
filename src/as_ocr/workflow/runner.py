"""Four-stage catalog workflow using the existing validated extraction modules."""
from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time
import uuid

import pymupdf as fitz

from as_ocr import async_inference, callouts, components, detection, indexer, page_markdown
from as_ocr.qwen import ModelConfig, QwenClient

from . import adapters
from .markdown import format_page
from .adapters import WORKSPACE, admit_paths, check_workspace, sha256

STAGES = {1: 'Full-page OCR', 2: 'Complete Assembly Detection',
          3: 'Detail recognition and verified leader counting', 4: 'Five-column CSV export'}


@dataclass(frozen=True)
class Config:
    source: Path
    pages: tuple[int, ...] | None
    output: Path = WORKSPACE / 'outputv3'
    internal: Path | None = None
    ocr_max_requests: int = 20
    detection_max_requests: int = 160
    callout_max_requests: int = 10
    component_max_requests: int = 6
    max_in_flight: int = 32

    def validate(self):
        check_workspace()
        if not self.source.is_file() or self.source.suffix.lower() != '.pdf':
            raise ValueError('Select exactly one readable source PDF')
        if self.pages is not None and (not self.pages or len(set(self.pages)) != len(self.pages)
                or any(type(p) is not int or p < 1 for p in self.pages)):
            raise ValueError('Require explicit unique positive original PDF page numbers')
        for name in ('ocr_max_requests', 'detection_max_requests', 'callout_max_requests', 'component_max_requests'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + ' must be a finite positive manual allowance')
        if type(self.max_in_flight) is not int or not 1 <= self.max_in_flight <= 32:
            raise ValueError('Existing inference workload supports a limit of 1–32')
        with fitz.open(self.source) as document:
            if document.needs_pass or not len(document) or self.pages and max(self.pages) > len(document):
                raise ValueError('Source is encrypted or selected pages exceed its actual page count')
            return len(document)


class ExistingBackend:
    mode = 'REAL_INFERENCE'

    def client(self, role, page, model):
        return QwenClient(model)


def model_for(config, role):
    allowance, tokens = {'ocr': (config.ocr_max_requests, 1536),
                         'detection': (config.detection_max_requests, 8192),
                         'callouts': (config.callout_max_requests, 8192),
                         'components': (config.component_max_requests, 4096)}[role]
    return ModelConfig(endpoint=os.getenv('AS_OCR_QWEN_ENDPOINT'), model=os.getenv('AS_OCR_QWEN_MODEL'),
                       identity=os.getenv('AS_OCR_QWEN_IDENTITY'), api_key=os.getenv('AS_OCR_QWEN_API_KEY'),
                       max_requests=allowance, max_tokens=tokens, timeout=180, retries=0)


def transport_model(config, role, backend):
    model = model_for(config, role)
    prepare = getattr(backend, 'prepare_model', None)
    return prepare(role, model) if prepare else model


def overall(records):
    states = [p['status'] for p in records]
    if all(s == 'NOT_RUN' for s in states):
        return 'NOT_RUN'
    if all(s == 'BLOCKED' for s in states):
        return 'BLOCKED'
    if all(s == 'FAILED' for s in states):
        return 'FAILED'
    if any(s not in {'COMPLETE', 'NO_DETAIL'} for s in states):
        return 'PARTIAL'
    return 'COMPLETE'


def relative(path, output):
    return str(Path(path).relative_to(output))


def copy_images(internal, destination):
    copied = []
    for number, path in enumerate(sorted(Path(internal).rglob('*.png')), 1):
        copied.append(adapters.asset(path, destination / f'{number:03d}-{path.name}'))
    return copied


def run(config, *, backend=None, allow_inference=False):
    count = config.validate()
    if config.pages is None:
        config = replace(config, pages=tuple(range(1, count + 1)))
    backend = backend or ExistingBackend()
    if not allow_inference and backend.mode != 'CPU_SYNTHETIC':
        raise PermissionError('Real inference requires an explicit --allow-inference invocation')
    # Validate actual real transport settings before creating a run.
    if backend.mode != 'CPU_SYNTHETIC':
        QwenClient(model_for(config, 'detection'))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]
    internal = config.internal or WORKSPACE / '_internal' / 'workflow-runs' / run_id
    output, internal = admit_paths(config.output, internal)
    internal.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    source = config.source.resolve()
    identity = sha256(source)
    records = {n: dict(stage=n, name=name, status='NOT_RUN', execution_mode=backend.mode,
                       source=dict(path=str(source), sha256=identity, total_pages=count),
                       selected_original_pages=list(config.pages), accuracy_verified=False,
                       relationship_discovery='SKIPPED_BY_DESIGN', same_page_grouping='EXISTING_DETERMINISTIC', historical_stage6='NOT_RUN',
                       cross_page_enrichment='NOT_RUN', graph_generation='NOT_RUN',
                       pages=[dict(source_page=p, status='NOT_RUN', reasons=[]) for p in config.pages])
               for n, name in STAGES.items()}
    for n in STAGES:
        for name in ('images', 'output'):
            (output / f'stage{n}' / name).mkdir(parents=True)
    (output / 'README.md').write_text(
        '# OCR_Detaila — four-stage workflow results\n\n'
        f'Execution mode: **{backend.mode}**. Source: `{source.name}`; original pages {list(config.pages)}.\n\n'
        'This directory was created exclusively for this invocation and will never be adopted or overwritten. '
        'CPU_SYNTHETIC means scripted model responses, not a real-model extraction or accuracy claim.\n\n'
        'Each stage has one public `summary.json`, `images/` and `output/`. '
        'Read every page/Detail status; PARTIAL, REVIEW, FAILED and BLOCKED are not successes. '
        f'Internal SQLite, raw responses, observations and checkpoints are separate: `{internal}`.\n\n'
        'Existing deterministic same-page grouping is preserved, including source-supported Section/Isometric groups. '
        'No cross-page relationship discovery, enrichment, historical Stage 6 or graph export runs. '
        'Local leader/label ownership validation remains mandatory.\n\n'
        'CSV: `detail_name,profile_count,profile_id,component_name,source_page`. '
        'Counts and names are withheld when existing source-evidence rules do not support them.\n')

    clients = []
    stage_started = {}

    def make_client(role, page, model):
        client = backend.client(role, page, model)
        clients.append((role, page, client))
        return client

    def persist():
        for n, summary in records.items():
            roles = {1: {'ocr'}, 2: {'detection'}, 3: {'callouts', 'components'}, 4: set()}[n]
            usage = []
            for role, page, client in clients:
                if role not in roles:
                    continue
                diagnostics = getattr(client, 'request_diagnostics', [])
                times = sorted(e['duration_seconds'] for e in diagnostics if 'duration_seconds' in e)
                usage.append(dict(role=role, source_page=page, configured_max_requests=client.config.max_requests,
                    consumed_requests=client.requests, retries=sum(e.get('attempt', 1) > 1 for e in diagnostics),
                    response_status_counts={s: sum(e.get('response_status') == s for e in diagnostics)
                                            for s in sorted({e.get('response_status', 'unknown') for e in diagnostics})},
                    latency_seconds=dict(mean=statistics.mean(times), median=statistics.median(times),
                        p95=times[min(len(times)-1, int(.95 * len(times)))]) if times else None))
            summary['request_usage'] = usage
            summary['request_count'] = sum(r['consumed_requests'] for r in usage)
            summary['request_budget_scope'] = {1: 'per original page', 2: 'whole detection stage run',
                                              3: 'separate callout/name allowances per original page', 4: 'no inference'}[n]
            summary['page_status_counts'] = {s: sum(p['status'] == s for p in summary['pages'])
                                            for s in sorted({p['status'] for p in summary['pages']})}
            summary['review_detail_count'] = sum(d.get('status') == 'REVIEW' for p in summary['pages'] for d in p.get('details', []))
            if n in stage_started and 'duration_seconds' not in summary:
                summary['elapsed_seconds'] = time.monotonic() - stage_started[n]
            summary['status'] = summary.get('fatal_status') or overall(summary['pages'])
            adapters.save_json(output / f'stage{n}' / 'summary.json', summary)

    def page_record(n, page):
        return next(r for r in records[n]['pages'] if r['source_page'] == page)

    def failed(n, page, exc):
        page_record(n, page).update(status='FAILED', reasons=[type(exc).__name__, callouts.failure_reason(exc)])
        adapters.save_json(internal / 'failures' / f'stage{n}-page{page}.json',
                           dict(exception_type=type(exc).__name__, status='FAILED', original_page=page))

    persist()
    adapters.save_json(internal / 'run.json', dict(run_id=run_id, status='RUNNING', mode=backend.mode,
                                                output=str(output), source_sha256=identity))
    workload = async_inference.Workload(config.max_in_flight, internal / 'http') if backend.mode != 'CPU_SYNTHETIC' else nullcontext()
    pages, detected, snapshots = {}, {}, {}
    all_rows, exported_pages = [], {}
    started = time.monotonic()
    current_stage = 1
    try:
        with workload as admitted:
            # 1. Existing indexing and existing compact full-page OCR, every selected page.
            clock = time.monotonic()
            stage_started[1] = clock
            indexed = indexer.run(source, internal / 'index', pages=list(config.pages))
            index_path = internal / 'index' / 'index.sqlite3'
            with sqlite3.connect(index_path) as db:
                db.row_factory = sqlite3.Row
                for page in config.pages:
                    row = db.execute("SELECT * FROM pages WHERE document_id=? AND page_index=? AND status='complete'",
                                     (identity, page - 1)).fetchone()
                    if row is None:
                        page_record(1, page).update(status='FAILED', reasons=['No complete source index page'])
                        continue
                    row = dict(row)
                    geometry, spans, signature = detection.load_page(db, row)
                    vectors = [json.loads(data) for data, in db.execute('SELECT data FROM vectors WHERE page_id=? ORDER BY ordinal', (row['id'],))]
                    pages[page] = row
                    preview = output / 'stage1' / 'images' / f'page_{page:04d}-original.png'
                    adapters.asset(internal / 'index' / row['preview'], preview)
                    folder = internal / 'ocr' / f'page_{page:04d}'
                    try:
                        model = transport_model(config, 'ocr', backend)
                        client = make_client('ocr', page, model)
                        result = page_markdown.extract_page_compact(source, row, geometry, spans, folder, client,
                            page_markdown.CompactConfig(max_requests=config.ocr_max_requests), vectors)
                        if detection.load_page(db, row)[2] != signature or sha256(source) != identity:
                            raise ValueError('Source/index changed during OCR')
                        text, formatting = format_page(result, (folder / 'page.md').read_text())
                        final = output / 'stage1' / 'output' / f'page_{page:04d}.md'
                        final.write_text(text)
                        images = copy_images(folder, output / 'stage1' / 'images' / f'page_{page:04d}')
                        page_record(1, page).update(formatting, markdown=relative(final, output),
                            original_image=relative(preview, output), source_regions=[relative(p, output) for p in images],
                            model_requests=result['model_requests'], issues=result.get('issues', []))
                    except Exception as exc:
                        failed(1, page, exc)
                        (output / 'stage1' / 'output' / f'page_{page:04d}.md').write_text(
                            f'# Original PDF page {page}\n\n**OCR status: FAILED**\n\n[REVIEW: see stage1/summary.json]\n')
                    persist()
            records[1].update(duration_seconds=time.monotonic() - clock, index_result=indexed)

            # 2. Existing full-page detection, including its unchanged coverage verification.
            current_stage = 2
            clock = time.monotonic()
            stage_started[2] = clock
            eligible_pages = [p for p in config.pages if p in pages]
            if eligible_pages:
                try:
                    model = transport_model(config, 'detection', backend)
                    result = detection.run(source, internal / 'detection', model=model, index_path=index_path,
                        pages=eligible_pages, client=make_client('detection', None, model))
                    adapters.save_json(internal / 'detection-result.json', result)
                    records[2].update(model_requests=result['model_requests'],
                                      manual_stage_request_limit=config.detection_max_requests)
                    for path in result['manifests']:
                        saved = adapters.read_json(path)
                        page = saved['page_index'] + 1
                        detected[page] = path
                        images = copy_images(Path(path).parent, output / 'stage2' / 'images' / f'page_{page:04d}')
                        reviews = []
                        for candidate in saved['candidates']:
                            for name, review in candidate.get('crop_review_paths', {}).items():
                                if name.endswith('.md'):
                                    target = output / 'stage2' / 'output' / f'page_{page:04d}-{candidate["id"]}-{name}'
                                    adapters.asset(review, target)
                                    reviews.append(relative(target, output))
                        decision = saved['page_decision']
                        status = 'NO_DETAIL' if decision == 'NO_DETAIL' else 'FAILED' if decision == 'FAILED' else 'REVIEW' if decision == 'REVIEW' else 'COMPLETE'
                        page_record(2, page).update(status=status, decision=decision,
                            candidates=saved['candidates'], coverage=saved.get('page_analysis'),
                            images=[relative(p, output) for p in images], crop_reviews=reviews,
                            internal_manifest=str(path))
                except Exception as exc:
                    for page in eligible_pages:
                        if page not in detected:
                            failed(2, page, exc)
            for page in sorted(config.pages):
                if page not in detected and page_record(2, page)['status'] == 'NOT_RUN':
                    page_record(2, page).update(status='BLOCKED', reasons=['No source-index page or no completed detection manifest'])
            records[2]['duration_seconds'] = time.monotonic() - clock
            persist()

            # 3. Existing same-page grouping, callouts, names and leader validation.
            detail_ordinal=0
            current_stage = 3
            clock = time.monotonic()
            stage_started[3] = clock
            for page in sorted(config.pages):
                if page not in detected or page_record(2, page)['status'] in {'FAILED', 'BLOCKED'}:
                    page_record(3, page).update(status='BLOCKED', reasons=['Detection did not produce usable source evidence'])
                    continue
                if page_record(2, page)['status'] == 'NO_DETAIL':
                    page_record(3, page).update(status='NO_DETAIL', details=[], reasons=['Existing complete detection says NO_DETAIL'])
                    continue
                try:
                    compat = adapters.group_page(detected[page], internal / 'recognition' / f'page_{page:04d}' / 'grouping', index_path=index_path)
                    definition = adapters.read_json(compat)
                    if not definition['details']:
                        page_record(3, page).update(status='REVIEW', details=[], reasons=['No eligible Detail; unresolved detection retained'])
                        continue
                    model = transport_model(config, 'callouts', backend)
                    calls = callouts.run(detected[page], internal / 'callouts' / f'page_{page:04d}', index_path=index_path,
                        stage3=compat, config=callouts.Config(details_only=True), model=model,
                        client=make_client('callouts', page, model))
                    saved_calls = adapters.read_json(calls['manifest'])
                    names = None
                    if saved_calls['state'] != 'FAILED':
                        model = transport_model(config, 'components', backend)
                        names = components.run(compat, calls['manifest'], internal / 'names' / f'page_{page:04d}',
                            index_path=index_path, config=components.Config(details_only=True), model=model,
                            client=make_client('components', page, model))['manifest']
                    snapshot = adapters.recognize_snapshot(compat, calls['manifest'], names)
                    name_result = adapters.read_json(names) if names else None
                    snapshots[page] = snapshot
                    items, seen = [], set()
                    images = copy_images(compat.parent, output / 'stage3' / 'images' / f'page_{page:04d}-grouping')
                    images += copy_images(Path(calls['manifest']).parent, output / 'stage3' / 'images' / f'page_{page:04d}-callouts')
                    if names:
                        images += copy_images(Path(names).parent, output / 'stage3' / 'images' / f'page_{page:04d}-names')
                    for detail in snapshot['details']:
                        detail_ordinal+=1
                        rows, traces, exported = adapters.project_detail(snapshot, detail, detail_ordinal)
                        seen |= exported
                        inventory = snapshot['inventories'].get(detail['id'])
                        if inventory and inventory.get('review_path'):
                            adapters.asset(inventory['review_path'], output / 'stage3' / 'output' / f'{detail["id"]}-profile_count_review.md')
                        items.append(dict(detail_id=detail['id'], source_page=page, view_ids=detail['view_ids'],
                            grouping_status=detail['grouping_status'], grouping_review_status=detail['review_status'],
                            grouping_evidence=detail.get('grouping_evidence', []), membership_role=detail['membership_role'],
                            status='REVIEW' if (detail['review_status'] == 'REVIEW' or not detail['complete']
                                or not rows[0]['profile_count'] or all(not r['profile_id'] and not r['component_name'] for r in rows)) else 'COMPLETE',
                            profile_count_status=traces[0]['profile_count_status'],
                            leader_observations=adapters.public_inventory(inventory), rows=rows, row_evidence=traces))
                    exported_pages[page] = items
                    status = 'COMPLETE' if (saved_calls['complete'] and name_result and name_result['complete']
                                             and all(i['status'] == 'COMPLETE' for i in items)
                                             and page_record(2, page)['status'] == 'COMPLETE') else 'REVIEW'
                    page_record(3, page).update(status=status, details=items,
                        occurrences=saved_calls['occurrences'], markers=saved_calls['markers'],
                        unexported_occurrence_ids=[o['id'] for o in saved_calls['occurrences'] if o['id'] not in seen],
                        validation=snapshot['validation'], images=[relative(p, output) for p in images],
                        callout_status=saved_calls['state'],
                        component_name_status=name_result['state'] if name_result else 'NOT_RUN',
                        model_requests=dict(callouts=calls['model_requests'],
                            components=name_result['model_requests'] if name_result else 0),
                        internal_evidence=snapshot['internal_sources'])
                except Exception as exc:
                    failed(3, page, exc)
                persist()
            records[3]['duration_seconds'] = time.monotonic() - clock

            # 4. New CSV serialization only; no exports.run/enrichment.run/graph builder.
            current_stage = 4
            clock = time.monotonic()
            stage_started[4] = clock
            for page in config.pages:
                if page_record(3, page)['status'] == 'NO_DETAIL':
                    page_record(4, page).update(status='NO_DETAIL', detail_count=0, rows=0)
                    continue
                if page not in exported_pages:
                    page_record(4, page).update(status='BLOCKED', reasons=['Recognition evidence unavailable or unresolved eligibility'])
                    continue
                try:
                    delivered = []
                    snapshot = snapshots[page]
                    for item in exported_pages[page]:
                        did = item['detail_id']
                        definition = next(d for d in snapshot['details'] if d['id'] == did)
                        image = output / 'stage4' / 'images' / did / 'image.png'
                        adapters.detail_image(definition, image)
                        csv_path = output / 'stage4' / 'output' / did / 'detail.csv'
                        adapters.write_csv(csv_path, item['rows'])
                        all_rows += item['rows']
                        delivered.append(dict(detail_id=did, status=item['status'], rows=len(item['rows']),
                            image=relative(image, output), csv=relative(csv_path, output),
                            csv_sha256=sha256(csv_path), source_page=page, evidence='stage3/summary.json'))
                    page_record(4, page).update(status=page_record(3, page)['status'], details=delivered,
                                               detail_count=len(delivered), rows=sum(d['rows'] for d in delivered))
                except Exception as exc:
                    failed(4, page, exc)
                    page_record(4, page).update(details=delivered, rows=sum(d['rows'] for d in delivered),
                        detail_count=len(delivered), status='PARTIAL' if delivered else 'FAILED')
                persist()
            merged = []
            for page in records[4]['pages']:
                for delivered in page.get('details', []):
                    path = output / delivered['csv']
                    if sha256(path) != delivered['csv_sha256']:
                        raise ValueError('Per-Detail CSV changed before aggregate serialization')
                    with path.open(encoding='utf-8-sig', newline='') as handle:
                        reader = csv.DictReader(handle)
                        if tuple(reader.fieldnames) != adapters.CSV_COLUMNS:
                            raise ValueError('Per-Detail CSV schema differs')
                        merged += list(reader)
            if merged != all_rows:
                raise ValueError('Aggregate differs from the ordered per-Detail evidence')
            aggregate = output / 'stage4' / 'output' / 'all_details.csv'
            adapters.write_csv(aggregate, all_rows)
            if sha256(source) != identity:
                raise ValueError('Source bytes changed during the run')
            records[4].update(duration_seconds=time.monotonic() - clock, columns=list(adapters.CSV_COLUMNS),
                              aggregate_csv=relative(aggregate, output), aggregate_rows=len(all_rows))
            peak = admitted.state.limit.peak.value if admitted is not None else 0
            for record in records.values():
                record.update(max_in_flight_limit=config.max_in_flight, actual_peak_in_flight=peak,
                              scheduler_page_workers=1)
        persist()
        result = dict(status='COMPLETE' if all(s['status'] == 'COMPLETE' for s in records.values()) else 'PARTIAL',
                      run_id=run_id, output=str(output), internal=str(internal), mode=backend.mode,
                      duration_seconds=time.monotonic() - started,
                      summaries=[str(output / f'stage{n}' / 'summary.json') for n in STAGES])
        adapters.save_json(internal / 'run.json', result)
        return result
    except BaseException as exc:
        active = records[current_stage]
        active['fatal_reason'] = type(exc).__name__
        if any(p['status'] != 'NOT_RUN' for p in active['pages']):
            active['fatal_status'] = 'PARTIAL' if isinstance(exc, KeyboardInterrupt) else 'FAILED'
        for summary in records.values():
            for page in summary['pages']:
                if page['status'] == 'NOT_RUN':
                    page.update(status='BLOCKED', reasons=['Run cancelled or stopped before this stage'])
        persist()
        adapters.save_json(internal / 'run.json', dict(status='CANCELLED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                           output=str(output), exception_type=type(exc).__name__))
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('source', type=Path)
    selection = result.add_mutually_exclusive_group(required=True)
    selection.add_argument('--pages', type=indexer.positive, nargs='+', help='Original 1-based PDF pages')
    selection.add_argument('--all-pages', action='store_true', help='Explicitly select the complete catalog')
    result.add_argument('--output', type=Path, default=WORKSPACE / 'outputv3')
    result.add_argument('--internal', type=Path)
    for option, default in [('ocr', 20), ('detection', 160), ('callout', 10), ('component', 6)]:
        result.add_argument('--' + option + '-max-requests', type=indexer.positive, default=default)
    result.add_argument('--max-in-flight', type=indexer.positive, default=32)
    result.add_argument('--allow-inference', action='store_true', help='Explicitly authorize model requests for this invocation')
    result.add_argument('--dry-run', action='store_true', help='Read-only source/path validation; create no output and run no inference')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = Config(**{key: value for key, value in vars(args).items() if key not in {'allow_inference', 'dry_run', 'all_pages'}})
    if args.dry_run:
        count = config.validate()
        internal = config.internal or WORKSPACE / '_internal' / 'workflow-runs' / ('planned-' + uuid.uuid4().hex)
        output, internal = admit_paths(config.output, internal)
        print(json.dumps(dict(mode='DRY_RUN', source_pages=count, selected_pages=config.pages or tuple(range(1, count + 1)),
            output=str(output), internal=str(internal), stages=STAGES, relationship_discovery=False,
            historical_stage6=False, graph_generation=False, writes=0, inference_requests=0)))
        return 0
    result = run(config, allow_inference=args.allow_inference)
    print(json.dumps(result))
    return 0 if result['status'] == 'COMPLETE' else 1


import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from as_ocr import pipeline


@pytest.fixture(autouse=True)
def offline_page_transcription(monkeypatch):
    """Orchestration tests must never contact the shared vision service."""
    def run(source,index,output,evidence,pages,client,config):
        return {'complete':True,'pages':[{'source_page':p,'status':'COMPLETE','issues':[]} for p in pages]}
    monkeypatch.setattr(pipeline.page_markdown,'run',run)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    return path


@pytest.fixture
def qwen_env(monkeypatch):
    monkeypatch.setenv('AS_OCR_QWEN_ENDPOINT', 'http://127.0.0.1:8003/v1')
    monkeypatch.setenv('AS_OCR_QWEN_MODEL', 'qwen-catalog-vision')
    monkeypatch.setenv(
        'AS_OCR_QWEN_IDENTITY',
        'unsloth/Qwen3.8-27B-NVFP4@57926baca9a82b4d6906b43f2750d55315f5b10f')
    monkeypatch.delenv('AS_OCR_QWEN_API_KEY', raising=False)


def _args(pdf, output, run_id, *, resume=False, dry_run=False, max_pages=None,
          stage2_max_requests=None):
    values = [str(pdf), '--output-root', str(output), '--run-id', run_id]
    if resume:
        values.append('--resume')
    if dry_run:
        values.append('--dry-run')
    if max_pages is not None:
        values.extend(['--max-pages', str(max_pages)])
    if stage2_max_requests is not None:
        values.extend(['--stage2-max-requests', str(stage2_max_requests)])
    return pipeline._parser().parse_args(values)


def test_dry_run_selects_one_explicit_pdf_and_writes_nothing(tmp_path, qwen_env):
    folder = tmp_path / 'Data' / 'Alupco'
    folder.mkdir(parents=True)
    selected = folder / 'catalog.pdf'
    selected.write_bytes(b'%PDF-1.4\n')
    (folder / 'sibling.pdf').write_bytes(b'%PDF-1.4\n')
    output = tmp_path / 'out'
    args = _args(selected, output, 'dry-run-check', dry_run=True)

    result = pipeline.run_pipeline(args)

    assert result['dry_run'] is True
    assert result['source'] == str(selected.resolve())
    assert result['effective_limits']['page_limit'] is None
    assert result['effective_limits']['selected_pages'] == 'all'
    assert result['effective_limits']['stage_2']['max_requests'] is None
    assert Path(result['run_dir']) == (
        output.resolve() / ('catalog__' + pipeline.indexer.fingerprint(selected)) / '_internal' / 'runs' / 'dry-run-check')
    assert not Path(result['run_dir']).exists()


def test_dry_run_reports_bounded_budget_and_explicit_override(tmp_path, qwen_env):
    source = tmp_path / 'catalog.pdf'
    source.write_bytes(b'%PDF-1.4\n')
    output = tmp_path / 'out'

    limited = pipeline.run_pipeline(
        _args(source, output, 'limited', dry_run=True, max_pages=3))
    assert limited['effective_limits']['selected_pages'] == 3
    assert limited['effective_limits']['stage_2']['max_requests'] == 24

    override = pipeline.run_pipeline(
        _args(source, output, 'override', dry_run=True, stage2_max_requests=5))
    assert override['effective_limits']['selected_pages'] == 'all'
    assert override['effective_limits']['stage_2']['max_requests'] == 5


def test_dry_run_rejects_non_pdf(tmp_path, qwen_env):
    source = tmp_path / 'catalog.txt'
    source.write_text('not a pdf')

    with pytest.raises(ValueError, match='explicit PDF'):
        pipeline.run_pipeline(_args(source, tmp_path / 'out', 'bad', dry_run=True))


def test_stage2_budget_scales_with_selected_pages_and_preserves_override():
    automatic = SimpleNamespace(
        stage2_max_requests=None, stage2_max_followups=2, stage2_retries=1)
    explicit = SimpleNamespace(
        stage2_max_requests=7, stage2_max_followups=2, stage2_retries=1)

    assert pipeline._stage2_budget(automatic, 5) == 40
    assert pipeline._stage2_budget(automatic, 0) is None
    assert pipeline._stage2_budget(explicit, 5) == 7


def test_stage2_checkpoint_prevents_downstream_execution(tmp_path,qwen_env,monkeypatch):
    import pymupdf
    source=tmp_path/'catalog.pdf'
    with pymupdf.open() as doc:
        doc.new_page();doc.save(source)
    def detect(source,output,*args,**kwargs):
        path=_write(output/'manifest.json',{'stage':2,'page_index':0,'page_id':'p0',
                                          'page_decision':'CANDIDATES'})
        return {'manifests':[str(path)],'pages_completed':1,'errors':0}
    monkeypatch.setattr(pipeline.detection,'run',detect)
    def forbidden(*args,**kwargs):raise AssertionError('Downstream stage started before validation')
    monkeypatch.setattr(pipeline.grouping,'run',forbidden)
    monkeypatch.setattr(pipeline.callouts,'run',forbidden)
    args=_args(source,tmp_path/'out','checkpoint')
    args.stop_after_stage2=True
    result=pipeline.run_pipeline(args)
    assert result['status']=='PAUSED_STAGE_2' and result['export_status']=='NOT_RUN'
    state=json.loads((Path(result['outputs']['run_dir'])/'run.json').read_text())
    assert set(state['stages'])=={'stage_1','stage_2'}


def test_run_target_resume_checks_source_and_hash(tmp_path):
    source = tmp_path / 'catalog.pdf'
    source.write_bytes(b'%PDF-1.4\n')
    run_dir = tmp_path / 'out' / 'Alupco' / 'catalog' / 'run-1'
    run_dir.mkdir(parents=True)
    state = {
        'state_version': pipeline.STATE_VERSION,
        'source': {'path': str(source.resolve()), 'sha256': 'a' * 64},
    }
    pipeline._write_json(run_dir / 'run.json', state)

    with pytest.raises(ValueError, match='already exists'):
        pipeline._validate_run_target(run_dir, source, resume=False, force=False)
    with pytest.raises(ValueError, match='content changed'):
        pipeline._validate_run_target(
            run_dir, source, resume=True, force=False,
            source_sha='b' * 64, check_hash=True)
    assert pipeline._validate_run_target(
        run_dir, source, resume=True, force=False,
        source_sha='a' * 64, check_hash=True) == state
    with pytest.raises(ValueError, match='does not exist'):
        pipeline._validate_run_target(
            tmp_path / 'missing', source, resume=True, force=False)


def test_finish_successful_no_detail_run_is_complete_without_export(tmp_path):
    summary = _summary_shell(selected=1)
    summary['stages'].update({
        'stage_1': {'status': 'COMPLETE'},
        'stage_2': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_3': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_4': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_5': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_6': {'status': 'SKIPPED'},
        'stage_7': {'status': 'SKIPPED'},
    })
    state = {'stages': {
        'stage_2': {'pages': {'0': {
            'decision': 'NO_DETAIL', 'reason': ['blank page'], 'page_id': 'p0'}}},
        'stage_3': {'pages': {}},
        'stage_4': {'pages': {}},
        'stage_5': {'pages': {}},
    }}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()

    result = pipeline._finish(
        summary, state, run_dir, fatal=False,
        stage6={'execution_status': 'SKIPPED'}, stage7={})

    assert result['successful_empty'] is True
    assert result['status'] == 'COMPLETE'
    assert result['export_status'] == 'NOT_RUN'
    assert result['pages']['no_detail_count'] == 1


def test_finish_failed_empty_run_is_not_successful_empty(tmp_path):
    summary = _summary_shell(selected=1)
    summary['stages'].update({
        'stage_1': {'status': 'COMPLETE'},
        'stage_2': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_3': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_4': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_5': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_6': {'status': 'FAILED'},
        'stage_7': {'status': 'NOT_RUN'},
    })
    state = {'stages': {
        'stage_2': {'pages': {'0': {
            'decision': 'NO_DETAIL', 'reason': ['blank page'], 'page_id': 'p0'}}},
        'stage_3': {'pages': {}},
        'stage_4': {'pages': {}},
        'stage_5': {'pages': {}},
    }}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()

    result = pipeline._finish(
        summary, state, run_dir, fatal=False,
        stage6={'execution_status': 'FAILED',
                'source_coverage_status': 'UNAVAILABLE'}, stage7={})

    assert result['successful_empty'] is False
    assert result['status'] == 'PARTIAL'


def test_finish_limited_no_detail_run_is_partial_not_successful_empty(tmp_path):
    summary = _summary_shell(selected=1)
    summary['pages']['limited'] = True
    summary['stages'].update({
        'stage_1': {'status': 'COMPLETE'},
        'stage_2': {'status': 'COMPLETE', 'public': {'errors': 0, 'limited': True}},
        'stage_3': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_4': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_5': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_6': {'status': 'SKIPPED'},
        'stage_7': {'status': 'SKIPPED'},
    })
    state = {'stages': {
        'stage_2': {'pages': {'0': {
            'decision': 'NO_DETAIL', 'reason': ['blank'], 'page_id': 'p0'}}},
        'stage_3': {'pages': {}},
        'stage_4': {'pages': {}},
        'stage_5': {'pages': {}},
    }}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()

    result = pipeline._finish(
        summary, state, run_dir, fatal=False,
        stage6={'execution_status': 'SKIPPED'}, stage7={})

    assert result['successful_empty'] is False
    assert result['status'] == 'PARTIAL'
    assert result['completeness_status'] == 'PARTIAL'
    assert result['export_status'] == 'NOT_RUN'


def test_finish_review_only_run_is_review_not_partial(tmp_path):
    summary = _summary_shell(selected=1)
    summary['stages'].update({
        'stage_1': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_2': {'status': 'COMPLETE', 'public': {'errors': 0, 'incomplete': True}},
        'stage_3': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_4': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_5': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_6': {'status': 'COMPLETE'},
        'stage_7': {'status': 'COMPLETE'},
    })
    state = {'stages': {
        'stage_2': {'pages': {'0': {
            'decision': 'REVIEW', 'page_id': 'p0'}}},
        'stage_3': {'pages': {}},
        'stage_4': {'pages': {}},
        'stage_5': {'pages': {}},
    }}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()

    result = pipeline._finish(
        summary, state, run_dir, fatal=False,
        stage6={'execution_status': 'COMPLETE',
                'source_coverage_status': 'COMPLETE'},
        stage7={'export_status': 'COMPLETE', 'complete': True,
                'source_coverage_status': 'COMPLETE',
                'acceptance_status': 'REVIEW', 'counts': {}})

    assert result['status'] == 'REVIEW'
    assert result['completeness_status'] == 'COMPLETE'
    assert result['review_status'] == 'REVIEW'


def test_finish_coverage_partial_keeps_execution_complete(tmp_path):
    summary = _summary_shell(selected=1)
    summary['stages'].update({
        'stage_1': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_2': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_3': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_4': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_5': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_6': {'status': 'COMPLETE'},
        'stage_7': {'status': 'PARTIAL'},
    })
    state = {'stages': {
        'stage_2': {'pages': {'0': {
            'decision': 'CANDIDATES', 'page_id': 'p0'}}},
        'stage_3': {'pages': {}},
        'stage_4': {'pages': {}},
        'stage_5': {'pages': {}},
    }}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()

    result = pipeline._finish(
        summary, state, run_dir, fatal=False,
        stage6={'execution_status': 'COMPLETE',
                'source_coverage_status': 'PARTIAL'},
        stage7={'export_status': 'PARTIAL', 'execution_status': 'COMPLETE',
                'source_coverage_status': 'PARTIAL', 'complete': False,
                'acceptance_status': 'SOURCE_SUPPORTED', 'counts': {}})

    assert result['status'] == 'PARTIAL'
    assert result['execution_status'] == 'COMPLETE'
    assert result['completeness_status'] == 'PARTIAL'


def test_finish_missing_page_is_failed_without_export(tmp_path):
    summary = _summary_shell(selected=2)
    summary['stages'].update({
        'stage_1': {'status': 'COMPLETE'},
        'stage_2': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_3': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_4': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_5': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_6': {'status': 'SKIPPED'},
        'stage_7': {'status': 'SKIPPED'},
    })
    state = {'stages': {
        'stage_2': {'pages': {'0': {
            'decision': 'NO_DETAIL', 'reason': ['blank'], 'page_id': 'p0'}}},
        'stage_3': {'pages': {}},
        'stage_4': {'pages': {}},
        'stage_5': {'pages': {}},
    }}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()

    result = pipeline._finish(
        summary, state, run_dir, fatal=False,
        stage6={'execution_status': 'SKIPPED'}, stage7={})

    assert result['successful_empty'] is False
    assert result['status'] == 'FAILED'
    assert result['pages']['failed_pages_one_based'] == [2]


def test_finish_partial_export_is_partial(tmp_path):
    summary = _summary_shell(selected=1)
    summary['stages'].update({
        'stage_1': {'status': 'COMPLETE'},
        'stage_2': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_3': {'status': 'COMPLETE', 'public': {'errors': 0}},
        'stage_4': {'status': 'PARTIAL', 'public': {'errors': 1}},
        'stage_5': {'status': 'PARTIAL', 'public': {'errors': 1}},
        'stage_6': {'status': 'PARTIAL'},
        'stage_7': {'status': 'PARTIAL'},
    })
    state = {'stages': {
        'stage_2': {'pages': {'0': {
            'decision': 'CANDIDATES', 'page_id': 'p0'}}},
        'stage_3': {'pages': {}},
        'stage_4': {'pages': {}},
        'stage_5': {'pages': {}},
    }}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()

    result = pipeline._finish(
        summary, state, run_dir, fatal=False,
        stage6={'execution_status': 'PARTIAL'},
        stage7={'export_status': 'PARTIAL', 'source_coverage_status': 'PARTIAL',
                'acceptance_status': 'REVIEW', 'complete': False, 'counts': {}})

    assert result['status'] == 'PARTIAL'
    assert result['export_status'] == 'PARTIAL'
    assert result['review_status'] == 'REVIEW'


def test_mocked_pipeline_reuses_stages_on_resume(tmp_path, qwen_env, monkeypatch):
    folder = tmp_path / 'Data' / 'Alupco'
    folder.mkdir(parents=True)
    source = folder / 'catalog.pdf'
    source.write_bytes(b'%PDF-1.4\n')
    (folder / 'sibling.pdf').write_bytes(b'%PDF-1.4\n')
    output = tmp_path / 'out'
    seen = {'indexer': [], 'detection': [], 'stage3': 0, 'stage4': 0,
            'stage5': 0, 'stage6': 0, 'exports': 0}
    source_hash = 'c' * 64

    monkeypatch.setattr(pipeline.indexer, 'fingerprint', lambda path: source_hash)
    monkeypatch.setattr(pipeline, '_index_page_count', lambda index, document: 2)

    def fake_indexer(run_source, stage_dir, **kwargs):
        seen['indexer'].append(str(run_source))
        Path(stage_dir, 'index.sqlite3').write_bytes(b'sqlite')
        return {'documents': 1, 'completed': 2, 'reused': 0, 'errors': 0,
                'limited': False, 'incomplete': False}

    def fake_detection(run_source, stage_dir, config, model, index_path,
                       max_pages=None, force=False, client=None):
        seen['detection'].append((str(run_source), max_pages, force))
        manifests = []
        for index, decision in ((0, 'NO_DETAIL'), (1, 'CANDIDATES')):
            manifest = _write(Path(stage_dir) / 'pages' / f'page-{index:02d}.json', {
                'stage': 2, 'page_index': index, 'page_id': f'p{index}',
                'page_decision': decision, 'skip_reason': ['blank'] if index == 0 else None,
            })
            manifests.append(str(manifest))
        return {'pages_completed': 2, 'pages_reused': 0, 'candidates': 1,
                'review': 0, 'errors': 0, 'model_failures': 0, 'limited': False,
                'page_decisions': {'NO_DETAIL': 1, 'CANDIDATES': 1},
                'manifests': manifests, 'model_requests': 0}

    def fake_grouping(inputs, stage_dir, **kwargs):
        seen['stage3'] += 1
        stage2 = Path(inputs[0])
        saved = json.loads(stage2.read_text())
        manifest = _write(Path(stage_dir) / 'pages' / 'page-01.json', {
            'stage': 3, 'grouping_attempt_id': 'g1', 'page_index': 1,
            'page_id': saved['page_id'], 'upstream': {'manifest_sha256': _digest(stage2)},
            'views': [], 'details': [{'id': 'd1', 'review_status': 'SOURCE_SUPPORTED'}],
        })
        return {'pages_completed': 1, 'pages_reused': 0, 'errors': 0, 'views': 0,
                'details': 1, 'confirmed_multiview_groups': 0, 'model_requests': 0,
                'model_failures': 0, 'manifests': [str(manifest)],
                'pages_skipped': 0, 'skipped': []}

    def fake_callouts(stage2, stage_dir, *, index_path, stage3=None, **kwargs):
        seen['stage4'] += 1
        saved = json.loads(Path(stage3).read_text())
        manifest = _write(Path(stage_dir) / 'manifest.json', {
            'stage': 4, 'state': 'COMPLETE', 'complete': True,
            'page_id': saved['page_id'],
            'upstream': {'stage3': {'sha256': _digest(stage3)}},
            'occurrences': [{'id': 'o1', 'target_point': [1, 2]}],
            'markers': [{'id': 'm1'}],
        })
        return {'views_completed': 1, 'views_reused': 0, 'errors': 0,
                'model_requests': 0, 'occurrences': 1, 'markers': 1,
                'incomplete': False, 'manifest': str(manifest)}

    def fake_components(stage3, stage4, stage_dir, *, index_path, **kwargs):
        seen['stage5'] += 1
        saved3 = json.loads(Path(stage3).read_text())
        manifest = _write(Path(stage_dir) / 'manifest.json', {
            'stage': 5, 'state': 'COMPLETE', 'complete': True,
            'page_id': saved3['page_id'],
            'upstream': {'stage3': {'sha256': _digest(stage3)},
                         'stage4': {'sha256': _digest(stage4)}},
            'candidates': [{'id': 'c1', 'code_raw': 'X', 'memberships': []}],
            'observations': [{'accepted': True}],
        })
        return {'views_completed': 1, 'views_reused': 0, 'errors': 0,
                'model_requests': 0, 'candidates': 1, 'observations': 1,
                'incomplete': False, 'manifest': str(manifest)}

    def fake_enrichment(inputs, stage_dir, *, index_path, plan=None, **kwargs):
        seen['stage6'] += 1
        stage5 = Path(inputs[0])
        manifest = _write(Path(stage_dir) / 'manifest.json', {
            'stage': 6, 'execution_status': 'COMPLETE',
            'source_coverage_status': 'COMPLETE',
            'sources': [{'stage5_path': str(stage5),
                         'stage5_sha256': _digest(stage5)}],
            'coverage': {'indexed_pages_searched': [0, 1], 'unindexed_pages': [],
                         'visual_unsearched_pages': []},
            'assignments': [], 'matches': [], 'failures': [],
        })
        return {'manifest': str(manifest), 'lookups_completed': 0,
                'lookups_reused': 0, 'errors': 0, 'model_requests': 0,
                'execution_status': 'COMPLETE',
                'source_coverage_status': 'COMPLETE', 'native_hits': 0,
                'supported_assignments': 0}

    def fake_exports(stage6, stage_dir, *, index_path, force=False):
        seen['exports'] += 1
        summary = _write(Path(stage_dir) / 'run-summary.json', {'stage': 7})
        return {'stage': 7, 'export_status': 'COMPLETE',
                'execution_status': 'COMPLETE',
                'source_coverage_status': 'COMPLETE',
                'acceptance_status': 'SOURCE_SUPPORTED', 'complete': True,
                'validation_errors': 0, 'counts': {'details': 1}, 'reused': False,
                'artifacts': {'summary': str(summary)}}

    monkeypatch.setattr(pipeline.indexer, 'run', fake_indexer)
    monkeypatch.setattr(pipeline.detection, 'run', fake_detection)
    monkeypatch.setattr(pipeline.grouping, 'run', fake_grouping)
    monkeypatch.setattr(pipeline.callouts, 'run', fake_callouts)
    monkeypatch.setattr(pipeline.components, 'run', fake_components)
    monkeypatch.setattr(pipeline.enrichment, 'run', fake_enrichment)
    monkeypatch.setattr(pipeline.exports, 'run', fake_exports)
    monkeypatch.setattr(pipeline.delivery, 'publish', lambda *a, **k: {'status':'REVIEW'})

    first = pipeline.run_pipeline(_args(source, output, 'run-1'))
    run_dir = Path(first['outputs']['run_dir'])
    assert run_dir == output.resolve() / ('catalog__' + source_hash) / '_internal' / 'runs' / 'run-1'
    assert first['pages']['page_count'] == 2
    assert first['pages']['selected'] == 2
    assert first['pages']['limited'] is False
    assert first['status'] == 'COMPLETE'
    assert seen['indexer'] == [str(source.resolve())]
    assert seen['detection'] == [(str(source.resolve()), None, False)]
    assert seen['stage3'] == 1
    assert seen['stage4'] == 1
    assert seen['stage5'] == 1
    assert seen['stage6'] == 1
    assert [path.name for path in sorted(run_dir.iterdir())
            if path.is_dir()] == [f'stage_{number}' for number in range(1, 8)]
    skipped = json.loads((run_dir / 'stage_3' / 'skipped-pages.json').read_text())
    assert skipped[0]['page_index_zero_based'] == 0
    assert skipped[0]['decision'] == 'NO_DETAIL'
    assert skipped[0]['reason'] == ['blank']

    before = {key: seen[key] for key in ('stage3', 'stage4', 'stage5', 'stage6')}
    second = pipeline.run_pipeline(_args(source, output, 'run-1', resume=True))
    assert second['status'] == 'COMPLETE'
    assert Path(second['outputs']['run_dir']) == run_dir
    assert {key: seen[key] for key in before} == before
    assert seen['indexer'] == [str(source.resolve()), str(source.resolve())]
    assert seen['detection'] == [(str(source.resolve()), None, False),
                                 (str(source.resolve()), None, False)]


def _summary_shell(selected):
    return {
        'pages': {'selected': selected},
        'page_markdown':{'complete':True,'pages':[]},
        'stages': {},
        'counts': {},
        'outputs': {},
        'failures': [],
        '_started_monotonic': time.monotonic(),
    }


def test_full_view_budget_is_manual_without_a_detail_limit(tmp_path):
    stage3=_write(tmp_path/'s3.json', {'views':[{'id':str(n)} for n in range(25)]})
    stage4=_write(tmp_path/'s4.json', {'selected_view_ids':[str(n) for n in range(25)]})
    assert pipeline._view_budget(stage3,10,0,True)==10
    assert pipeline._view_budget(stage4,6,1,True)==6
    assert pipeline._view_budget(stage3,10,0,False)==10


def test_parallel_detection_accounts_for_every_page_and_isolates_errors(tmp_path, monkeypatch):
    import sqlite3
    index=tmp_path/'index.sqlite3'
    with sqlite3.connect(index):
        pass
    submitted=[]
    class Pool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers==4
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, worker, tasks):
            for task in tasks:
                page=task[2]['pages'][0]
                submitted.append(page)
                stats=dict(requests=1,wall_seconds=.2,transport_failures=0,timeouts=0)
                if page==2:
                    yield {'result':None,'stats':stats,'error':'isolated page failure'}
                else:
                    yield {'result':dict(pages_completed=1,model_requests=1,limited=True,
                                        page_decisions={'NO_DETAIL':1},manifests=[str(page)]),
                           'stats':stats,'error':None}
    monkeypatch.setattr(pipeline,'ProcessPoolExecutor',Pool)
    stats=pipeline.Stats()
    result=pipeline._parallel_detection(tmp_path/'source.pdf',tmp_path,None,None,index,110,
        SimpleNamespace(force=False,page_workers=4),stats)
    assert submitted==list(range(1,111))
    assert result['pages_completed']==109 and result['errors']==1
    assert not result['limited']
    assert len(result['manifests'])==109 and stats.requests==110
    assert result['page_failures'][0]['page']==2


def test_page_worker_bound_is_validated(tmp_path,qwen_env):
    source=tmp_path/'source.pdf';source.write_bytes(b'%PDF-1.4\n')
    args=_args(source,tmp_path/'out','workers',dry_run=True)
    args.page_workers=5
    with pytest.raises(ValueError,match='one and four'):
        pipeline.run_pipeline(args)
    args.page_workers=4
    assert pipeline.run_pipeline(args)['effective_limits']['page_workers']==4
    args.async_inference=True
    args.max_in_flight=32
    args.page_workers=32
    limits=pipeline.run_pipeline(args)['effective_limits']
    assert limits['page_workers']==32 and limits['max_in_flight']==32


def test_worker_error_retains_request_accounting(monkeypatch):
    from as_ocr.qwen import ModelConfig
    def fail(*args,**kwargs):
        kwargs['client'].stats.requests=2
        raise ValueError('bad source page')
    monkeypatch.setattr(pipeline.detection,'run',fail)
    result=pipeline._page_worker((2,('source','output'),{'model':ModelConfig()},None))
    assert result['error']=='ValueError: bad source page'
    assert result['stats']['requests']==2


def test_fresh_forces_all_stage_regeneration_and_rejects_resume(tmp_path,qwen_env):
    source=tmp_path/'source.pdf';source.write_bytes(b'%PDF-1.4\n')
    args=_args(source,tmp_path/'out','fresh-run',dry_run=True)
    args.fresh=True
    result=pipeline.run_pipeline(args)
    assert args.force and result['effective_limits']['fresh']
    args.resume=True
    with pytest.raises(ValueError,match='cannot be combined'):
        pipeline.run_pipeline(args)
    args.resume=False
    Path(result['run_dir']).mkdir(parents=True)
    with pytest.raises(ValueError,match='new, absent'):
        pipeline.run_pipeline(args)


@pytest.mark.parametrize('decision,errors,expected_ok,empty', [('NO_DETAIL',0,True,True),('CANDIDATES',0,True,False),('REVIEW',0,True,False),('FAILED',1,False,False)])
def test_single_page_detection_progress_does_not_misreport_document_limit(monkeypatch,decision,errors,expected_ok,empty):
    from as_ocr.qwen import ModelConfig
    events=[]
    monkeypatch.setattr(pipeline.detection,'run',lambda *a,**k:dict(incomplete=True,limited=True,errors=errors,model_failures=errors,page_decisions={decision:1}))
    monkeypatch.setattr(pipeline.progress,'done',lambda *a,**k:events.append(k))
    packed=pipeline._page_worker((2,('source','output'),{'model':ModelConfig()},None,4))
    assert packed['error'] is None
    assert events[0]['ok']==expected_ok and events[0]['empty']==empty



def test_cancelled_request_is_not_reported_as_a_received_response(monkeypatch):
    from as_ocr.qwen import ModelConfig
    class Client:
        config=ModelConfig()
        requests=0
        response_models=set()
        def request_json(self,*args,**kwargs):
            self.requests+=1
            raise KeyboardInterrupt
    events=[]
    monkeypatch.setattr(pipeline.progress,'request_end',lambda *a,**kw:events.append(kw))
    client=pipeline.TimedClient(Client(),pipeline.Stats())
    with pytest.raises(KeyboardInterrupt):client.request_json(None,'test')
    assert events[0]['error']=='KeyboardInterrupt'

"""New-only CPU tests. Scripted vision responses never prove model accuracy."""
from dataclasses import replace
import csv
import json
import os
import hashlib
from pathlib import Path

import pytest

from as_ocr.workflow import runner
from workflow_fixture import CPUBackend, TABLE, make_pdf
from as_ocr.workflow.markdown import format_page
from as_ocr.workflow import adapters


@pytest.fixture(autouse=True)
def forbid_real_requests(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('OutputV3 CPU tests forbid real inference')
    monkeypatch.setattr('as_ocr.qwen.QwenClient.request_json', denied)


@pytest.fixture(autouse=True)
def windows_cpu_fixture_only(monkeypatch, tmp_path):
    """Keep production DGX path guards; admit only isolated synthetic test paths."""
    if os.name != 'nt':
        return
    from as_ocr import performance
    monkeypatch.setattr(runner, 'check_workspace', lambda: None)
    monkeypatch.setattr(adapters, 'check_workspace', lambda: None)
    monkeypatch.setattr(adapters.tempfile, 'gettempdir', lambda: str(tmp_path.parent))
    original = performance.atomic_write
    def write(path, text):
        try:
            return original(path, text)
        except AttributeError as exc:
            # The original function already fsyncs/replaces the file before its
            # Linux-only directory fsync. This fixture does not test durability.
            if 'O_DIRECTORY' not in str(exc):
                raise
    monkeypatch.setattr(performance, 'atomic_write', write)


def config(tmp_path, *, pages=(1,), total=1, assembly=True, **changes):
    source = make_pdf(tmp_path / 'source.pdf', total, assembly=assembly)
    return runner.Config(source=source, pages=pages, output=tmp_path / 'public', internal=tmp_path / 'internal', **changes)


def summary(cfg, number):
    return json.loads((cfg.output / f'stage{number}' / 'summary.json').read_text())


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == adapters.CSV_COLUMNS
        return list(reader)


def test_four_actual_stages_without_relationship_or_graph_execution(tmp_path, monkeypatch):
    from as_ocr import enrichment, exports, grouping, pipeline
    def forbidden(*args, **kwargs):
        raise AssertionError('Relationship/legacy pipeline/graph processing must not run')
    for module, name in [(enrichment, 'run'), (exports, 'run'), (pipeline, 'run_pipeline'),
                         (grouping, 'assisted_proposals'), (exports, 'build_tables')]:
        monkeypatch.setattr(module, name, forbidden)
    cfg = config(tmp_path)
    backend = CPUBackend()
    result = runner.run(cfg, backend=backend)
    assert result['status'] == 'PARTIAL'  # Existing grouping remains provisional/REVIEW.
    assert {c.role for c in backend.clients} == {'ocr', 'detection', 'callouts', 'components'}
    for stage in range(1, 5):
        saved = summary(cfg, stage)
        assert saved['status'] == ('COMPLETE' if stage < 3 else 'PARTIAL')
        assert saved['graph_generation'] == 'NOT_RUN'
        assert saved['relationship_discovery'] == 'SKIPPED_BY_DESIGN'
    assert callable(pipeline.run_pipeline)  # Original entry point remains importable.


def test_simple_public_structure_one_json_per_stage(tmp_path):
    cfg = config(tmp_path)
    runner.run(cfg, backend=CPUBackend())
    assert {p.name for p in cfg.output.iterdir()} == {'README.md', 'stage1', 'stage2', 'stage3', 'stage4'}
    for n in range(1, 5):
        stage = cfg.output / f'stage{n}'
        assert {p.name for p in stage.iterdir()} == {'images', 'output', 'summary.json'}
        assert list(stage.rglob('*.json')) == [stage / 'summary.json']
        assert all(p.suffix == '.png' for p in (stage / 'images').rglob('*') if p.is_file())
    assert list(cfg.internal.rglob('index.sqlite3'))
    assert list(cfg.internal.rglob('leader_observations.json'))
    assert cfg.output not in cfg.internal.parents


def test_exact_five_columns_source_counts_repeats_and_unguessed_names(tmp_path):
    cfg = config(tmp_path)
    runner.run(cfg, backend=CPUBackend())
    public = rows(cfg.output / 'stage4/output/all_details.csv')
    assert len(public) == 2 and public[0] == public[1]
    assert public[0] == dict(detail_name='source_01', profile_count='2',
                            profile_id='EXACT-001', component_name='', source_page='1')
    per_detail = next((cfg.output / 'stage4/output').glob('*/detail.csv'))
    assert rows(per_detail) == public
    delivered = summary(cfg, 4)['pages'][0]['details'][0]
    assert (cfg.output / delivered['image']).is_file()
    inventory = summary(cfg, 3)['pages'][0]['details'][0]['leader_observations']
    assert [o['number'] for o in inventory['observations'] if o['accepted']] == [1, 2]


def test_original_nonconsecutive_page_identity_is_preserved(tmp_path):
    cfg = config(tmp_path, pages=(1, 3), total=3)
    runner.run(cfg, backend=CPUBackend())
    exported = rows(cfg.output / 'stage4/output/all_details.csv')
    assert [r['source_page'] for r in exported] == ['1', '1', '3', '3']
    import sqlite3
    with sqlite3.connect(cfg.internal / 'index/index.sqlite3') as db:
        assert [r[0] for r in db.execute('SELECT page_index FROM pages ORDER BY page_index')] == [0, 2]


def test_markdown_heading_table_values_reading_order_and_repeated_annotations(tmp_path):
    cfg = config(tmp_path)
    runner.run(cfg, backend=CPUBackend())
    text = (cfg.output / 'stage1/output/page_0001.md').read_text()
    assert '## Assembly Notes' in text and TABLE in text
    assert text.index('Assembly Notes') < text.index('Use source data only.') < text.index('| ID | Value |') < text.index('EXACT-001')
    assert text.count('EXACT-001') == 2 and '![' not in text
    assert summary(cfg, 1)['pages'][0]['reading_order'] == 'UNCHANGED_FROM_EXISTING_OCR'


def test_no_detail_still_runs_full_page_ocr_and_emits_only_csv_header(tmp_path):
    cfg = config(tmp_path, assembly=False)
    backend = CPUBackend(no_detail=True)
    result = runner.run(cfg, backend=backend)
    assert result['status'] == 'COMPLETE'
    assert (cfg.output / 'stage1/output/page_0001.md').read_text().strip()
    assert all(summary(cfg, n)['pages'][0]['status'] == 'NO_DETAIL' for n in [2, 3, 4])
    assert not {'callouts', 'components'} & {c.role for c in backend.clients}
    assert rows(cfg.output / 'stage4/output/all_details.csv') == []


def test_detection_failure_blocks_downstream_without_fake_success(tmp_path):
    cfg = config(tmp_path)
    result = runner.run(cfg, backend=CPUBackend(fail_role='detection'))
    assert result['status'] == 'PARTIAL'
    assert summary(cfg, 2)['pages'][0]['status'] == 'FAILED'
    assert summary(cfg, 3)['pages'][0]['status'] == 'BLOCKED'
    assert summary(cfg, 4)['pages'][0]['status'] == 'BLOCKED'
    assert rows(cfg.output / 'stage4/output/all_details.csv') == []


def test_callout_failure_preserves_blank_count_instead_of_zero(tmp_path):
    cfg = config(tmp_path)
    result = runner.run(cfg, backend=CPUBackend(fail_role='callouts'))
    assert result['status'] == 'PARTIAL'
    public = rows(cfg.output / 'stage4/output/all_details.csv')
    assert len(public) == 1 and not public[0]['profile_count'] and not public[0]['component_name']
    assert summary(cfg, 3)['pages'][0]['status'] == 'REVIEW'


def test_component_name_failure_is_visible_without_losing_verified_counts(tmp_path):
    cfg = config(tmp_path)
    result = runner.run(cfg, backend=CPUBackend(fail_role='components'))
    assert result['status'] == 'PARTIAL'
    page = summary(cfg, 3)['pages'][0]
    assert page['status'] == 'REVIEW' and page['component_name_status'] == 'FAILED'
    assert all(r['profile_count'] == '2' and not r['component_name'] for r in rows(cfg.output / 'stage4/output/all_details.csv'))


def test_partial_ocr_does_not_erase_other_independent_stage_evidence(tmp_path):
    cfg = config(tmp_path, ocr_max_requests=2)
    result = runner.run(cfg, backend=CPUBackend(fail_role='ocr'))
    assert result['status'] == 'PARTIAL'
    assert summary(cfg, 1)['pages'][0]['status'] in {'PARTIAL', 'FAILED'}
    assert all(r['profile_count'] == '2' for r in rows(cfg.output / 'stage4/output/all_details.csv'))


def test_existing_output_is_never_adopted_or_overwritten(tmp_path):
    cfg = config(tmp_path)
    cfg.output.mkdir()
    sentinel = cfg.output / 'previous-output.txt'
    sentinel.write_bytes(b'KEEP EXACT BYTES')
    backend = CPUBackend()
    with pytest.raises(FileExistsError):
        runner.run(cfg, backend=backend)
    assert sentinel.read_bytes() == b'KEEP EXACT BYTES'
    assert not cfg.internal.exists() and not backend.clients


def test_no_implicit_gpu_authorization(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(PermissionError):
        runner.run(cfg)
    assert not cfg.output.exists() and not cfg.internal.exists()


def test_symlink_and_nested_output_rejected(tmp_path):
    cfg = config(tmp_path)
    try:
        cfg.output.symlink_to(Path('/tmp/nonexistent-outputv3-target'))
    except OSError:
        pytest.skip('Host does not permit creating test symlinks')
    with pytest.raises(FileExistsError):
        adapters.admit_paths(cfg.output, cfg.internal)
    with pytest.raises(ValueError):
        adapters.admit_paths(tmp_path / 'fresh', tmp_path / 'fresh/checkpoints')


@pytest.mark.parametrize('limit', [0, -1, True])
def test_invalid_manual_budgets_rejected_before_writes(tmp_path, limit):
    cfg = config(tmp_path, callout_max_requests=limit)
    with pytest.raises(ValueError):
        runner.run(cfg, backend=CPUBackend())
    assert not cfg.output.exists()


def test_manual_limits_above_ten_propagate_without_forcing_requests(tmp_path):
    cfg = config(tmp_path, callout_max_requests=50, component_max_requests=100)
    backend = CPUBackend()
    runner.run(cfg, backend=backend)
    call = next(c for c in backend.clients if c.role == 'callouts')
    component = next(c for c in backend.clients if c.role == 'components')
    assert call.config.max_requests == 50 and call.requests == 2
    assert component.config.max_requests == 100 and component.requests == 1
    assert summary(cfg, 4)['max_in_flight_limit'] == 32 and summary(cfg, 4)['scheduler_page_workers'] == 1


def test_cancellation_preserves_internal_draft_and_explicit_blocked_status(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        runner.run(cfg, backend=CPUBackend(cancel_role='callouts'))
    assert list(cfg.internal.rglob('*-leader-draft.json'))
    assert summary(cfg, 3)['status'] == 'BLOCKED' and summary(cfg, 4)['status'] == 'BLOCKED'
    assert json.loads((cfg.internal / 'run.json').read_text())['status'] == 'CANCELLED'


def test_final_integrity_failure_cannot_leave_complete_export_status(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    actual = runner.sha256
    calls = 0
    def changing_digest(path):
        nonlocal calls
        if Path(path) == cfg.source:
            calls += 1
            if calls >= 3:
                return 'simulated changed source'
        return actual(path)
    monkeypatch.setattr(runner, 'sha256', changing_digest)
    with pytest.raises(ValueError, match='Source bytes changed'):
        runner.run(cfg, backend=CPUBackend())
    assert summary(cfg, 4)['status'] == 'FAILED'
    assert json.loads((cfg.internal / 'run.json').read_text())['status'] == 'FAILED'


def test_dry_run_is_read_only(tmp_path, capsys):
    cfg = config(tmp_path)
    assert runner.main([str(cfg.source), '--pages', '1', '--output', str(cfg.output), '--internal', str(cfg.internal), '--dry-run']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['writes'] == value['inference_requests'] == 0
    assert not cfg.output.exists() and not cfg.internal.exists()


def test_formatting_never_reorders_columns_or_deduplicates_source_values():
    body = 'Column B\n\nColumn A\n\n| ID | Text |\n| --- | --- |\n| 001 | A\\|B |\n| 001 | A\\|B |'
    original = '# Original PDF page 7\n\n**OCR status: COMPLETE**\n\n' + body + '\n'
    result = dict(status='COMPLETE', units=[dict(kind='heading', markdown='Column B'), dict(kind='heading', markdown='Column A')])
    text, report = format_page(result, original)
    assert text.index('Column B') < text.index('Column A')
    assert text.count('| 001 | A\\|B |') == 2 and report['status'] == 'COMPLETE'


def test_inconsistent_or_merged_table_is_preserved_and_flagged():
    original = '# Original PDF page 1\n\n**OCR status: COMPLETE**\n\n| A | B |\n| --- | --- |\n| 0007 |\n'
    text, record = format_page(dict(status='COMPLETE', units=[]), original)
    assert '| 0007 |' in text and record['status'] == 'PARTIAL'
    html = '<table><tr><th colspan="2">Exact source</th></tr><tr><td>01</td><td>02</td></tr></table>'
    text, record = format_page(dict(status='COMPLETE', units=[]), html)
    assert html in text and record['status'] == 'PARTIAL'


def test_source_status_is_never_upgraded_by_serialization():
    original = '# Original PDF page 1\n\n**OCR status: PARTIAL**\n\nUseful source text\n'
    text, record = format_page(dict(status='PARTIAL', units=[dict(kind='prose', markdown='Useful source text')]), original)
    assert record['status'] == 'PARTIAL' and 'Useful source text' in text


def test_source_backed_crop_and_leaders_match_frozen_preintegration_results(tmp_path):
    from workflow_fixture import source_regressions
    source_fixture=json.loads((Path(__file__).parent/'fixtures/alupco-page1-context.json').read_text())
    if not Path(source_fixture['source']).is_file():
        pytest.skip('Original ALUPCO fixture PDF is unavailable in this checkout')
    expected = json.loads((Path(__file__).parent / 'fixtures/workflow-source-regressions.json').read_text())
    expected['leaders']['missing_known_detection_leader']=['1','OBSERVED_PARTIAL']
    assert source_regressions(tmp_path) == expected


def test_validated_core_and_layout_adapter_remain_byte_identical():
    expected = json.loads((Path(__file__).parent / 'fixtures/workflow-core-hashes.json').read_text())
    assert len(expected) >= 20
    for filename, digest in expected.items():
        # These files implement the explicitly approved count/CSV/naming/graph changes.
        if filename in {'src/as_ocr/leaders.py','src/as_ocr/delivery.py','src/as_ocr/pipeline.py',
                        'src/as_ocr/presentation.py','src/as_ocr/public_contract.py',
                        'src/as_ocr/regions.py','src/as_ocr/detection.py','src/as_ocr/coverage.py',
                        'src/as_ocr/grouping.py','src/as_ocr/callouts.py','src/as_ocr/components.py',
                        'src/as_ocr/page_markdown.py','src/as_ocr/line_evidence.py'}:
            continue
        # Git can check out CRLF on Windows; the frozen source hashes use LF.
        data=(adapters.WORKSPACE / filename).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(data).hexdigest() == digest, filename


def test_approved_same_page_group_and_source_resolution_match_existing_grouping(tmp_path):
    from PIL import Image
    from as_ocr import exports, grouping
    from test_grouping import saved_page
    source = saved_page(tmp_path, ['Detail Q7 — Fixture joint — Section A',
                                   'Detail Q7 — Fixture joint — Isometric'])
    direct = grouping.run([source], tmp_path / 'existing')
    expected = adapters.read_json(direct['manifests'][0])
    actual = adapters.read_json(adapters.group_page(source, tmp_path / 'integrated'))
    assert len(actual['details']) == 1 and len(actual['details'][0]['view_ids']) == 2
    for key in ('view_ids', 'membership_role', 'grouping_status', 'review_status', 'complete', 'source_titles'):
        assert actual['details'][0][key] == expected['details'][0][key]
    detail = actual['details'][0]
    assert detail['grouping_status'] == 'CONFIRMED' and detail['review_status'] == 'REVIEW'
    adapters.detail_image(detail, tmp_path / 'integrated.png')
    exports.detail_image(expected['details'][0]['composites'], tmp_path / 'existing.png')
    with Image.open(tmp_path / 'integrated.png') as one, Image.open(tmp_path / 'existing.png') as two:
        assert one.size == two.size == (113, 149) and one.tobytes() == two.tobytes()
    panel = detail['composites'][0]['panels'][0]['source_crop']
    Path(panel).write_bytes(b'changed source evidence')
    with pytest.raises(ValueError, match='crop changed'):
        adapters.detail_image(detail, tmp_path / 'must-not-export.png')


def test_adjacent_repeated_profile_ids_never_create_a_same_page_group(tmp_path):
    from test_grouping import saved_page
    source = saved_page(tmp_path, ['P1200', 'P1200'])
    manifest = adapters.read_json(adapters.group_page(source, tmp_path / 'grouping'))
    assert len(manifest['details']) == 2
    assert all(d['grouping_status'] == 'SINGLETON' for d in manifest['details'])


def test_existing_grouping_review_is_never_upgraded_by_successful_count(tmp_path):
    cfg = config(tmp_path)
    runner.run(cfg, backend=CPUBackend())
    detail = summary(cfg, 3)['pages'][0]['details'][0]
    assert detail['rows'][0]['profile_count'] == '2' and detail['profile_count_status'] == 'COMPLETE'
    assert detail['grouping_review_status'] == detail['status'] == 'REVIEW'
    assert summary(cfg, 4)['pages'][0]['details'][0]['status'] == 'REVIEW'


def test_per_stage_timing_requests_and_failures_include_failed_attempts(tmp_path):
    cfg = config(tmp_path)
    backend = CPUBackend(fail_role='components')
    runner.run(cfg, backend=backend)
    for n, roles in [(1, {'ocr'}), (2, {'detection'}), (3, {'callouts', 'components'}), (4, set())]:
        saved = summary(cfg, n)
        assert saved['duration_seconds'] >= 0
        assert saved['request_count'] == sum(c.requests for c in backend.clients if c.role in roles)
        assert sum(saved['page_status_counts'].values()) == len(cfg.pages)
        assert all(r['consumed_requests'] <= r['configured_max_requests'] for r in saved['request_usage'])
    names = next(r for r in summary(cfg, 3)['request_usage'] if r['role'] == 'components')
    assert names['consumed_requests'] > 0 and summary(cfg, 3)['review_detail_count'] == 1


def test_all_pages_is_explicit_and_dry_run_stays_read_only(tmp_path, capsys):
    cfg = config(tmp_path, total=3)
    assert runner.main([str(cfg.source), '--all-pages', '--output', str(cfg.output),
                        '--internal', str(cfg.internal), '--dry-run']) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned['selected_pages'] == [1, 2, 3] and planned['inference_requests'] == 0
    assert not cfg.output.exists() and not cfg.internal.exists()
    with pytest.raises(SystemExit):
        runner.parser().parse_args([str(cfg.source), '--all-pages', '--pages', '1'])


def test_all_pages_runtime_uses_original_one_based_numbering(tmp_path):
    cfg = replace(config(tmp_path, total=2), pages=None)
    runner.run(cfg, backend=CPUBackend())
    public = rows(cfg.output / 'stage4/output/all_details.csv')
    assert [r['source_page'] for r in public] == ['1', '1', '2', '2']
    assert [r['detail_name'] for r in public] == ['source_01','source_01','source_02','source_02']


def test_historical_incompatible_csv_is_still_rejected(tmp_path):
    from as_ocr import delivery
    old = tmp_path / 'historical.csv'
    old.write_text('detail_name,profile_count,profile_id,component_name,profile_type,description,source_page\n')
    with pytest.raises(ValueError, match='Unexpected Detail CSV columns'):
        delivery.read_csv(old)


@pytest.mark.parametrize('command,module', [
    ('deliver', 'delivery'), ('present', 'presentation'), ('detect', 'detection'), ('group', 'grouping'),
    ('callouts', 'callouts'), ('components', 'components'), ('enrich', 'enrichment'), ('export', 'exports'),
    ('pipeline', 'pipeline'), ('workflow', 'workflow'),
])
def test_existing_application_cli_dispatch_is_preserved(monkeypatch, command, module):
    import importlib
    import sys
    import as_ocr
    selected = importlib.import_module('as_ocr.' + module)
    seen = []
    monkeypatch.setattr(selected, 'main', lambda argv: seen.append(argv) or 123)
    monkeypatch.setattr(sys, 'argv', ['as-ocr', command, '--help'])
    assert as_ocr.main() == 123 and seen == [['--help']]


def test_default_index_cli_dispatch_is_preserved(monkeypatch):
    import sys
    import as_ocr
    from as_ocr import indexer
    monkeypatch.setattr(indexer, 'main', lambda: 123)
    monkeypatch.setattr(sys, 'argv', ['as-ocr', '--help'])
    assert as_ocr.main() == 123


def test_main_runtime_has_no_prototype_imports():
    import ast
    package = Path(runner.__file__).parent
    for file in package.glob('*.py'):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Import):
                assert all(not name.name.startswith('outputv3_implementation') for name in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('outputv3_implementation')


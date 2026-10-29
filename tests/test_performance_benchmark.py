"""Benchmark safety and source-oracle checks; never perform model inference."""
import json
from types import SimpleNamespace
import urllib.error
import pytest
from as_ocr import benchmark,performance,pipeline,page_markdown


def args(tmp_path,**changes):
    source=tmp_path/'synthetic.pdf';source.write_bytes(b'not opened in dry-run')
    values=dict(pdf=source,pages=[1,5,7],output=tmp_path/'new',concurrency=2,workers=4,
        synthetic=False,transport='async',workload='markdown',prefer_native=False,
        expectations=None,metrics_interval=5,dry_run=True)
    values.update(changes);return SimpleNamespace(**values)


def test_dry_run_no_http_no_pdf_open_no_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark,'_get',lambda *a:pytest.fail('Network must not run'))
    result=benchmark.run(args(tmp_path))
    assert result['dry_run'] and result['pages']==[1,5,7] and result['workers']==4
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('pages',[list(range(1,111)),[0],[1,1],[-1]])
def test_full_catalog_or_invalid_page_selection_rejected(tmp_path,pages):
    with pytest.raises(ValueError):benchmark.run(args(tmp_path,pages=pages))
    assert not (tmp_path/'new').exists()


def test_no_overwrite_of_existing_benchmark(tmp_path):
    output=tmp_path/'preserved';output.mkdir();(output/'marker').write_text('historical')
    with pytest.raises(ValueError):benchmark.run(args(tmp_path,output=output))
    assert (output/'marker').read_text()=='historical'


def test_source_oracle_preserves_repeats_and_exact_table_cells(tmp_path):
    pages=tmp_path/'pages';pages.mkdir()
    text='# Original PDF page 7\n\n**OCR status: PARTIAL**\n\n- 32496\n- 6\n- 6\n\n| Code | Width |\n|---|---|\n| 001 | |\n| 001 | |\n\n[REVIEW: 6 is unresolved]\n'
    (pages/'page_0007.md').write_text(text)
    oracle=tmp_path/'oracle.json'
    oracle.write_text(json.dumps(dict(source_sha256='source',pages={'7':dict(contains=['32496'],occurrences={'6':2},
        tables=[dict(headers=['Code','Width'],rows=[['001',''],['001','']])])})))
    result=benchmark.compare_content(tmp_path,oracle,'source',[7])
    assert result['status']=='PASS' and not result['human_accuracy_verified']
    with pytest.raises(ValueError):benchmark.compare_content(tmp_path,oracle,'other',[7])
    with pytest.raises(ValueError):benchmark.compare_content(tmp_path,oracle,'source',[1])
    (pages/'page_0007.md').write_text(text.replace('| 001 | |','| 002 | |',1))
    assert benchmark.compare_content(tmp_path,oracle,'source',[7])['status']=='FAIL'


def test_missing_oracle_requires_review_not_accuracy_claim(tmp_path):
    assert benchmark.compare_content(tmp_path,None,'source',[7])['status']=='REVIEW'


def test_unavailable_server_metrics_are_explicit(tmp_path,monkeypatch):
    from as_ocr.qwen import ModelConfig
    def unavailable(*args):raise urllib.error.URLError('offline')
    monkeypatch.setattr(benchmark,'_get',unavailable)
    monkeypatch.setattr(benchmark.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='fixture GPU'))
    result=benchmark.resource_snapshot(ModelConfig(endpoint='http://127.0.0.1:1/v1'))
    assert not result['metrics_available'] and result['server_queue_time_seconds'] is None


def test_async_flags_are_opt_in_and_bounds_are_explicit():
    parser=pipeline._parser()
    args=parser.parse_args(['source.pdf'])
    assert not args.async_inference and args.markdown_engine=='legacy' and args.max_in_flight==2
    with pytest.raises(SystemExit):parser.parse_args(['source.pdf','--max-in-flight','33'])
    manual=parser.parse_args(['source.pdf','--markdown-engine','compact','--markdown-max-requests','100'])
    page_markdown.CompactConfig(max_requests=manual.markdown_max_requests).validate()


def test_missing_tokens_or_inflight_are_not_fabricated(tmp_path):
    performance.atomic_write(tmp_path/'requests-1.jsonl',json.dumps(dict(network_started=True,response_status='received',local_validation_status='NOT_RUN'))+'\n')
    result=performance.summarize(tmp_path,1)
    assert result['completion_tokens_per_wall_second'] is None and result['observed_peak_in_flight'] is None
    assert result['validation_not_run']==1 and result['usage_reported_requests']==0


@pytest.mark.parametrize('transport',['async','sync'])
def test_synthetic_benchmark_records_nested_diagnostics_and_real_overlap(tmp_path,transport):
    result=benchmark.run(args(tmp_path,synthetic=True,dry_run=False,transport=transport,
        concurrency=2,workers=2,pages=None))
    assert result['requests']==12 and result['validation_passed']==12
    assert result['synthetic_server_observation']['requests']==12
    assert result['synthetic_server_observation']['peak']==2
    assert result['pages_per_minute'] is None and result['quality']['status']=='NOT_MEASURED'
    assert (tmp_path/'new/_internal/benchmark-state.json').is_file()

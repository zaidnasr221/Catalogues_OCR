"""Focused contract regressions. Only synthetic sources and mocked OCR are used."""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pymupdf as fitz
import pytest

from as_ocr import delivery, indexer, page_markdown as md, pipeline, presentation, profile_observations as profiles, traceability
from as_ocr.qwen import ModelError
from test_profile_observations import analyze, classification
from test_presentation import fixture, save, table, update_callout


MAPPING=dict(source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0],pixel_size=[100,100],source_context_box=[0,0,100,100])


def block(text='Exact source',box=(.1,.1,.5,.2),**kw):
    return dict(text=text,box=list(box),kind='paragraph',heading_level=None,cells=[],native_span_ids=[],uncertainty=[],**kw)


def response(blocks=(),**kw):
    return dict(coverage='complete',blocks=list(blocks),crops=[],uncertainty=[],**kw)


class OCR:
    def __init__(self,*responses):self.responses=iter(responses);self.requests=0
    def request_json(self,*args,**kwargs):
        self.requests+=1
        value=next(self.responses)
        if isinstance(value,Exception):raise value
        return json.dumps(value)


def extract(tmp_path,monkeypatch,client,spans=()):
    monkeypatch.setattr(md.detection,'render',lambda *a:(Image.new('RGB',(100,100),'white'),MAPPING))
    return md.extract_page(Path('never-opened.pdf'),dict(id='p',document_id='doc',page_index=6),
        dict(source_rect=[0,0,100,100]),list(spans),tmp_path/'evidence',client,md.Config())


@pytest.mark.parametrize('record,status,value',[
    (None,'MISSING',''),
    (dict(observed_count=0,profiles=[],coverage='incomplete',potentially_incomplete=True),'INCOMPLETE_OR_FAILED',''),
    (dict(observed_count=0,profiles=[],coverage='complete',potentially_incomplete=False),'INCOMPLETE_OR_FAILED',''),
    (dict(observed_count=0,profiles=[],coverage='complete',potentially_incomplete=False,no_profile_reason='Only glass visible'),'VERIFIED_ZERO','0'),
    (dict(observed_count=1,profiles=[{}],coverage='incomplete',potentially_incomplete=True),'PARTIAL','1'),
])
def test_count_absence_failure_zero_and_partial_are_distinct(record,status,value):
    assert profiles.count_state(record)==(value,status)


@pytest.mark.parametrize('count',[-1,True,1.5,2])
def test_invalid_inventory_blocks_count_publication(count):
    with pytest.raises(ValueError):profiles.count_state(dict(observed_count=count,profiles=[]))


def test_explicit_name_type_does_not_need_target_and_cannot_create_ownership():
    context=dict(components=[dict(candidate_id='c1',profile_ids=[],component_names=['Frame'],targets=[])])
    result=analyze(types=[classification()],context=context)
    assert result['component_types'][0]['profile_type']=='Frame'
    assert result['profiles']==[]
    detail=dict(id='d',view_ids=['v'])
    candidate=dict(id='c1',observed_in_view_ids=['v'],memberships=[dict(view_id=None,target_point=None,status='UNRESOLVED')])
    fields=[dict(candidate_id='c1',field='component_name',value_raw='Frame',accepted=True,conditions=[],conflicts_with=[])]
    assert profiles.facts([candidate],fields,detail)[0]['targets']==[]
    assert not profiles.owner(candidate,detail)


def test_description_label_or_reference_table_cannot_reach_csv(tmp_path):
    assert 'description' not in presentation.project(fixture(tmp_path))[3][0]



def test_foreign_page_description_is_withheld(tmp_path):
    assert all('description' not in r for r in presentation.project(fixture(tmp_path))[3])



def test_missing_provider_does_not_block_filename_detail_name(tmp_path):
    rows,traces=presentation.project(fixture(tmp_path,provider=''))[3:5]
    assert rows[0]['detail_name']=='catalog_01' and traces[0]['row_key'][1]=='d1'



def test_no_secondary_type_is_published(tmp_path):
    assert all('profile_type' not in r for r in presentation.project(fixture(tmp_path))[3])



def test_occurrence_accounting_records_export_withheld_and_unresolved():
    occurrences=[dict(id='o'+str(n),document_id='doc',page_id='p',raw_code='001',association_method='leader' if n<3 else 'unresolved') for n in range(1,4)]
    records=traceability.audit(occurrences,[],[],[dict(row_key=['doc','d','o1'],csv_row_number=2,reasons=[])])
    assert [r['status'] for r in records]==['EXPORTED','WITHHELD','UNRESOLVED']
    assert all(r['reasons'] for r in records)


def test_markdown_combines_native_visual_and_distinct_repeat_locations(tmp_path,monkeypatch):
    native=[(7,dict(text='Exact source',bbox=[10,10,50,20]))]
    a=block();a['native_span_ids']=[7]
    b=block('Ø 12,5 mm',(.2,.5,.6,.6));c=block('Ø 12,5 mm',(.2,.7,.6,.8))
    result=extract(tmp_path,monkeypatch,OCR(response([a,a,b,c])),native)
    text=(tmp_path/'evidence/page.md').read_text()
    assert result['status']=='COMPLETE' and text.count('Exact source')==1 and text.count('Ø 12,5 mm')==2
    assert not result['csv_description_source'] and not result['accuracy_verified']


def test_markdown_rejects_corrupt_native_and_marks_unreadable(tmp_path,monkeypatch):
    b=block('guessed word');b['uncertainty']=['Unreadable glyphs']
    result=extract(tmp_path,monkeypatch,OCR(response([b]),ModelError('budget exhausted')),
        [(1,dict(text='bad\x81font',bbox=[10,10,50,20]))])
    text=(tmp_path/'evidence/page.md').read_text()
    assert result['status']=='FAILED' and 'REVIEW' in text and 'guessed word' not in text and 'bad\x81' not in text


def test_markdown_failed_model_keeps_native_with_partial_status(tmp_path,monkeypatch):
    result=extract(tmp_path,monkeypatch,OCR(ModelError('request timeout')),[(1,dict(text='عربي Δ',bbox=[10,10,50,20]))])
    assert result['status']=='PARTIAL' and 'عربي Δ' in (tmp_path/'evidence/page.md').read_text()
    assert any('timeout' in r for r in result['failures'])


def test_markdown_high_resolution_crop_is_retained(tmp_path,monkeypatch):
    first=response([]);first['crops']=[dict(box=[.1,.1,.5,.2],reason='Small outlined ID')]
    client=OCR(first,response([block('000123')]))
    result=extract(tmp_path,monkeypatch,client)
    assert client.requests==2 and (tmp_path/'evidence/crop-1.png').is_file()
    assert '000123' in (tmp_path/'evidence/page.md').read_text()
    assert result['requests'][1]['source_box']==[10,10,50,20]


def test_markdown_native_visual_disagreement_is_partial(tmp_path,monkeypatch):
    result=extract(tmp_path,monkeypatch,OCR(response([block('00124')])),[(1,dict(text='00123',bbox=[10,10,50,20]))])
    text=(tmp_path/'evidence/page.md').read_text()
    assert result['status']=='PARTIAL' and '00123' in text and '00124' not in text


def test_sparse_original_pages_are_indexed_without_intermediate_pages(tmp_path):
    source=tmp_path/'synthetic.pdf'
    with fitz.open() as doc:
        for n in range(8):doc.new_page().insert_text((40,40),f'Original page {n+1}')
        doc.save(source)
    output=tmp_path/'index'
    result=indexer.run(source,output,pages=[7,2])
    with sqlite3.connect(output/'index.sqlite3') as db:
        assert db.execute('SELECT page_index FROM pages ORDER BY page_index').fetchall()==[(1,),(6,)]
        assert db.execute('SELECT page_count FROM documents').fetchone()==(8,)
    assert result['completed']==2 and result['limited']


def test_focused_dry_run_requires_absent_root_and_keeps_original_selection(tmp_path,monkeypatch):
    monkeypatch.setenv('AS_OCR_QWEN_ENDPOINT','http://127.0.0.1:9999/v1');monkeypatch.setenv('AS_OCR_QWEN_MODEL','offline-fixture')
    source=tmp_path/'never-opened.pdf';source.write_bytes(b'PDF placeholder')
    args=pipeline._parser().parse_args([str(source),'--pages','7','2','--focused-test','--dry-run','--output-root',str(tmp_path/'fresh')])
    result=pipeline.run_pipeline(args)
    assert result['original_pages']==[2,7] and result['effective_limits']['fresh']
    assert result['effective_limits']['stage_3']['mode']=='deterministic'
    assert not (tmp_path/'fresh').exists()
    (tmp_path/'fresh').mkdir()
    with pytest.raises(ValueError,match='absent output root'):pipeline.run_pipeline(args)


@pytest.mark.parametrize('markdown_after_extraction',[False,True])
def test_selected_page_pipeline_no_details_still_transcribes_all_selected_pages(tmp_path,monkeypatch,markdown_after_extraction):
    monkeypatch.setenv('AS_OCR_QWEN_ENDPOINT','http://127.0.0.1:9999/v1');monkeypatch.setenv('AS_OCR_QWEN_MODEL','offline-fixture')
    source=tmp_path/'synthetic.pdf'
    with fitz.open() as doc:
        for n in range(8):doc.new_page().insert_text((40,40),f'Original page {n+1}')
        doc.save(source)
    class Client(OCR):
        response_models=['offline-fixture']
        config=None
        def request_json(self,image,prompt,context,**kw):
            self.requests+=1
            ctx=json.loads(context)
            spans=ctx['native_spans'];b=block(spans[0]['text'],(.01,.01,.9,.2));b['native_span_ids']=[spans[0]['id']]
            return json.dumps(response([b]))
    client=Client()
    monkeypatch.setattr(pipeline,'_client',lambda *a,**k:client)
    def detect(source,output,config,model,index_path,**kw):
        assert kw['pages']==[2,7] and kw['force']
        assert client.requests==(0 if markdown_after_extraction else 2)
        paths=[]
        with sqlite3.connect(index_path) as db:
            for page in kw['pages']:
                pid=db.execute('SELECT id FROM pages WHERE page_index=?',(page-1,)).fetchone()[0]
                path=output/f'page-{page}.json';save(path,dict(stage=2,page_index=page-1,page_id=pid,
                    page_decision='NO_DETAIL',skip_reason=['Synthetic blank assembly area'],candidates=[]))
                paths.append(str(path))
        return dict(manifests=paths,pages_completed=2,errors=0,limited=True)
    monkeypatch.setattr(pipeline.detection,'run',detect)
    args=pipeline._parser().parse_args([str(source),'--pages','2','7','--focused-test','--output-root',str(tmp_path/'focused')])
    args.markdown_after_extraction=markdown_after_extraction
    result=pipeline.run_pipeline(args)
    root=Path(result['outputs']['public_root'])
    assert {p.name for p in (root/'pages').glob('*.md')}=={'page_0002.md','page_0007.md'}
    assert 'Original page 7' in (root/'pages/page_0007.md').read_text()
    assert result['selected_page_completion']=='COMPLETE' and result['full_catalog_completion']=='INCOMPLETE'
    assert not result['extraction_accuracy_verified']
    assert delivery.read_csv(root/'all_details.csv')==[]
    assert len(list((root/'stages').glob('stage_*/review.md')))==7
    assert (root/'_internal/focused-report.json').is_file()


def test_implementation_fingerprint_changes_when_logic_changes(tmp_path,monkeypatch):
    from as_ocr import provenance
    path=tmp_path/'provenance.py';path.write_text('a=1\n')
    monkeypatch.setattr(provenance,'__file__',str(path))
    before=provenance.code_fingerprint();path.write_text('a=2\n')
    assert provenance.code_fingerprint()!=before


def test_page_ocr_coordinate_limits_are_on_the_wire_not_only_local():
    schema=md.Analysis.model_json_schema()
    for name in ('Block','Crop'):
        box=schema['$defs'][name]['properties']['box']
        assert all(item['minimum']==0 and item['maximum']==1 for item in box['prefixItems'])
    invalid=response([block(box=(96,50,430,850))])
    with pytest.raises(ValueError):md.Analysis.model_validate_json(json.dumps(invalid))


def test_duplicate_same_view_readings_do_not_duplicate_occurrence(tmp_path):
    from test_callouts import fixture,run,result,Client,observation,response
    path,index=fixture(tmp_path)
    saved=result(run(path,index,tmp_path,Client(response(observation('001'),observation('001')))))
    assert len(saved['occurrences'])==1
    assert len(saved['occurrences'][0]['readings'])==2


def test_missing_secondary_classification_preserves_verified_primary_type():
    draft=analyze(types=[classification()]);second=analyze()
    result=profiles.retain_observed(draft,second)
    assert result['component_types'][0]['profile_type']=='Frame'


def test_general_detail_paragraph_can_survive_without_component_identity():
    from as_ocr import paragraphs
    from test_paragraphs import setup,paragraph
    views,details,_,_,mapping=setup()
    payload=dict(coverage='complete',uncertainty=[],paragraphs=[paragraph(
        text='Install the assembly vertically.',scope='detail',candidate_ids=[],detail_ids=['d1'],
        applies_to_all=True,basis='heading',association_quote='Assembly A')])
    analysis=paragraphs.Analysis.model_validate_json(json.dumps(payload))
    records,fields=paragraphs.reconcile(analysis,dict(components=[]),views,details,[],mapping,[0,0,100,100],[])
    assert records[0]['accepted'] and not fields


def test_each_profile_panel_retains_its_own_source_box(monkeypatch):
    def render(source,page,geometry,box,maximum):
        return Image.new('RGB',(100,100),'white'),dict(MAPPING,source_context_box=list(box))
    monkeypatch.setattr(profiles.detection,'render',render)
    views=[dict(id='a',box=[0,0,40,40]),dict(id='b',box=[60,60,100,100])]
    _,layout=profiles.render(Path('not-opened.pdf'),0,{},dict(view_ids=['a','b']),views,2048)
    assert layout[0]['source_box']==[0,0,40,40]
    assert layout[1]['source_box']==[60,60,100,100]


def test_source_page_zero_is_rejected_before_publication(tmp_path):
    import csv
    summary=fixture(tmp_path);path=Path(json.loads(summary.read_text())['artifacts']['details'])
    with path.open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
    rows[0]['page_index']='-1';table(path,rows)
    with pytest.raises(ValueError,match='source page'):presentation.project(summary)


def test_markdown_preserves_model_column_reading_order(tmp_path,monkeypatch):
    left_top=block('A',(.1,.1,.3,.2));left_bottom=block('B',(.1,.6,.3,.7));right_top=block('C',(.6,.1,.8,.2))
    result=extract(tmp_path,monkeypatch,OCR(response([left_top,left_bottom,right_top])))
    assert result['status']=='COMPLETE'
    assert (tmp_path/'evidence/page.md').read_text().split()==['A','B','C']


def test_missing_page_index_still_writes_failed_markdown_for_selected_page(tmp_path):
    source=tmp_path/'identity.pdf';source.write_bytes(b'synthetic identity only')
    db=indexer.database(tmp_path/'index.sqlite3');db.close()
    result=md.run(source,tmp_path/'index.sqlite3',tmp_path/'public',tmp_path/'evidence',[7],OCR())
    assert result['pages'][0]['status']=='FAILED'
    assert 'FAILED' in (tmp_path/'public/pages/page_0007.md').read_text()


def test_ast_audit_does_not_claim_zero_callers_are_unused(tmp_path):
    import runpy
    tool=Path(__file__).resolve().parents[1]/'scripts/audit_code.py'
    (tmp_path/'src/demo').mkdir(parents=True)
    (tmp_path/'src/demo/main.py').write_text('def first():\n    return second()\ndef second():\n    return 1\ndef callback():\n    return 3\n')
    (tmp_path/'pyproject.toml').write_text('[project.scripts]\ndemo="demo.main:first"\n')
    result=runpy.run_path(str(tool))['audit'](tmp_path,{'demo.main.callback'})
    assert result['definitions']['demo.main.second']['callers']==['demo.main.first']
    assert result['definitions']['demo.main.callback']['observed']
    assert not result['confirmed_unused']


@pytest.mark.parametrize('category',['Gaskets','EPDM gasket','Glass/Infill','Screw','Bolts','Washer'])
def test_nonstructural_aliases_cannot_increment_profile_count(category):
    from test_profile_observations import profile
    record=analyze([profile(category=category)])
    assert record['observed_count']==0 and profiles.count_state(record)[0]==''


def test_overlapping_boxes_do_not_establish_physical_identity():
    from test_profile_observations import profile,mark
    one=profile();two=profile('two',[mark(box=(.105,.105,.305,.305))])
    two['identity_evidence']='A potentially separate inset extrusion; its material boundary needs review'
    record=analyze([one,two])
    assert len(record['profiles'])==2 and record['count_ambiguous']
    assert profiles.count_state(record)==('','AMBIGUOUS_IDENTITY')


def test_delivery_failure_updates_execution_and_selected_completion(tmp_path,monkeypatch):
    from test_pipeline import _summary_shell
    summary=_summary_shell(1)
    summary['source']=dict(path=str(tmp_path/'identity.pdf'),sha256='a'*64)
    summary['outputs']['public_root']=str(tmp_path/'public')
    summary['pages']['original_pages']=[1]
    summary['stages']={f'stage_{n}':{'status':'COMPLETE' if n<6 else 'SKIPPED'} for n in range(1,8)}
    state={'stages':{'stage_2':{'pages':{'0':dict(decision='NO_DETAIL',page_id='p',reason=['No assembly'])}}}}
    def fail(*a,**k):raise ValueError('Invalid publication inventory')
    monkeypatch.setattr(pipeline.delivery,'publish_empty',fail)
    result=pipeline._finish(summary,state,tmp_path/'run',fatal=False,stage6={'execution_status':'SKIPPED'},stage7={})
    assert result['export_status']==result['execution_status']=='FAILED'
    assert result['selected_page_completion']=='INCOMPLETE' and state['execution_status']=='FAILED'

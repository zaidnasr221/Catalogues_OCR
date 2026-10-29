"""Offline Stage 6/7 evidence, scope, graph and serialization contracts."""
from dataclasses import replace
import copy
import csv
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pymupdf as fitz
import pytest

from as_ocr import callouts,components,detection,enrichment as e,exports as x,grouping,indexer
from as_ocr.qwen import ModelError
from test_callouts import Client,MODEL,observation,response
from test_components import field,evidence,component,analysis


SCOPE=[dict(field=key,value_raw=value,evidence=[evidence(key.title()+': '+value,(.1,.7,.8,.75))],review_reasons=[])
       for key,value in [('manufacturer','SourceCo'),('system','System Q'),('edition','Edition 2020')]]
CODE='001.20 - A'


def source(tmp_path,texts=(CODE,CODE,'001.20-A'),*,index_pages=None):
    path=tmp_path/'source.pdf'
    with fitz.open() as doc:
        for text in texts:
            page=doc.new_page(width=300,height=240)
            if text:page.insert_text((30,40),text,fontsize=10)
        doc.save(path)
    indexer.run(path,tmp_path/'stage1',max_pages=index_pages)
    return path,tmp_path/'stage1/index.sqlite3'


def saved(tmp_path,src,index,*,page=0,known=False,fields=(),repeat=False,target=False):
    with sqlite3.connect(index) as db:
        db.row_factory=sqlite3.Row;row=db.execute('SELECT * FROM pages WHERE page_index=?',(page,)).fetchone()
        geo=json.loads(row['geometry']);row=dict(row)
    folder=tmp_path/f'detection-{page}';folder.mkdir()
    image,mapping=detection.render(src,page,geo,[0,0,300,240],1200);crop=folder/'view.png';image.save(crop)
    candidate=dict(id='c1',document_id=row['document_id'],page_id=row['id'],page_index=page,source_path=str(src),
        box=[0,0,300,240],structural_box=[0,0,300,240],kind='assembly_candidate',view_type='section',status='REVIEW',
        review_reasons=['Deferred source boundary review'],annotations=[],parent_id=None,child_ids=[],crop_path=str(crop),crop_geometry=mapping)
    s2=folder/'manifest.json';s2.write_text(json.dumps(dict(stage=2,detector_version='7',attempt_id=folder.name,document_id=row['document_id'],
        page_id=row['id'],page_index=page,source_path=str(src),candidates=[candidate],page_decision='CANDIDATES')))
    s3=grouping.run([s2],tmp_path/f'stage3-{page}',index_path=index)['manifests'][0]
    def labels(ctx):
        native=ctx['native_candidates']
        box=native[0]['text_box'] if native else [.1,.1,.3,.15]
        kwargs={}
        if target:
            kwargs=dict(owner_view_id=ctx['focus_view_id'],association_method='leader',leader_state='clear',
                        leader_points=[[box[2],(box[1]+box[3])/2],[.6,.5]],target_point=[.6,.5],association_evidence=['Visible continuous label leader'])
        obs=[observation(CODE,box,**kwargs)]
        if repeat:obs.append(observation(CODE,[.1,.4,.3,.45]))
        return response(*obs)
    s4=callouts.run(s2,tmp_path/f'stage4-{page}',index_path=index,stage3=s3,config=callouts.Config(context_fraction=0),model=MODEL,client=Client(labels))['manifest']
    def properties(ctx):
        part=component(ctx,fields)
        part['relation']['code_evidence']['box']=ctx['occurrences'][0]['text_box']
        part['relation']['bounds']=[0,0,.99,.99]
        return analysis(part,scope=copy.deepcopy(SCOPE) if known else [])
    s5=components.run(s3,s4,tmp_path/f'stage5-{page}',index_path=index,config=components.Config(context_fraction=0),
                      model=replace(MODEL,max_requests=6,max_tokens=4096),client=Client(properties))['manifest']
    return Path(s5)


def fixture(tmp_path,*,texts=(CODE,CODE,'001.20-A'),known=False,fields=(),repeat=False,target=False,index_pages=None):
    src,index=source(tmp_path,texts,index_pages=index_pages)
    s5=saved(tmp_path,src,index,known=known,fields=fields,repeat=repeat,target=target)
    return src,index,s5


def enrich(tmp_path,index,inputs,**kwargs):
    return e.run(inputs,tmp_path/'stage6',index_path=index,model=replace(MODEL,max_requests=1,max_tokens=4096),**kwargs)


def load(summary):return json.loads(Path(summary['manifest']).read_text())


def lookup_response(ctx,fields=(),*,scope=(),footnotes=(),kind='table_row'):
    part=component(ctx,fields,kind=kind,code=ctx['target_codes'][0],ids=[])
    body=json.loads(analysis(part,scope=scope));body['footnotes']=list(footnotes)
    return json.dumps(body)


def window_plan(src,index,page=1,box=(0,0,300,240)):
    with sqlite3.connect(index) as db:doc=db.execute('SELECT document_id FROM pages LIMIT 1').fetchone()[0]
    return e.LookupPlan(document_id=doc,selection_reason='Explicit synthetic reference page',index_pages=[],
                        windows=[e.Window(page_index=page,box=box,reason='Selected reference row')])


def exported(tmp_path,index,s6,**kwargs):return x.run(s6,tmp_path/'exports',index_path=index,**kwargs)


def rows(summary,name):
    with Path(summary['artifacts'][name]).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def test_exact_retrieval_normalization_collision_and_scope_unknown(tmp_path):
    _,index,s5=fixture(tmp_path)
    out=load(enrich(tmp_path,index,[s5]))
    assert [h['match'] for h in out['retrieval_hits']]==['EXACT_TEXT_CANDIDATE','EXACT_TEXT_CANDIDATE','NORMALIZED_COLLISION']
    assert all(h['status']=='UNVERIFIED' for h in out['retrieval_hits'])
    assert out['assignments']==[] and out['model_requests']==0


def test_numeric_prefix_suffix_are_not_exact_identifiers(tmp_path):
    _,index,s5=fixture(tmp_path,texts=(CODE,'1'+CODE+'9'))
    out=load(enrich(tmp_path,index,[s5]))
    assert len(out['retrieval_hits'])==1 and out['retrieval_hits'][0]['page_index']==0


def test_scanned_no_match_is_not_absence(tmp_path):
    _,index,s5=fixture(tmp_path,texts=(CODE,None))
    out=load(enrich(tmp_path,index,[s5]))
    assert out['coverage']['visual_unsearched_pages']==[1]
    assert out['source_coverage_status']=='PARTIAL'
    assert not out['coverage']['pages'][1]['absence_established']


def test_missing_scope_and_scope_collisions_block_transfers(tmp_path):
    src,index,s5=fixture(tmp_path,known=False)
    other=saved(tmp_path,src,index,page=1,known=True,fields=[field('linear_weight','2','kg/m',decimal_convention='integer')])
    out=load(enrich(tmp_path,index,[s5,other],candidate_ids=[json.loads(s5.read_text())['candidates'][0]['id']]))
    assert len(out['matches'])==1 and out['matches'][0]['status']=='REVIEW' and not out['assignments']
    target=json.loads(s5.read_text());identity=target['identities'][0]
    reference=json.loads(other.read_text())
    identity.update(scope_status='SOURCE_VERIFIED',scope={'manufacturer':'DifferentCo','system':'System Q','edition':'Edition 2020'},scope_evidence=copy.deepcopy(reference['identities'][0]['scope_evidence']))
    for item in identity['scope_evidence']:
        for ev in item['evidence']:ev.update(page_id=identity['page_id'],page_index=0,view_id=target['selected_view_ids'][0])
    identity['scope_evidence'][0]['value_raw']='DifferentCo';identity['scope_evidence'][0]['evidence'][0]['quote']='Manufacturer: DifferentCo'
    matches,assign=e.match_sources(target['candidates'],target['identities'],reference['candidates'],reference['identities'],reference['observations'])
    assert matches[0]['reason'].startswith('Scope collision') and not assign


def test_verified_cross_page_properties_preserve_null_membership_and_conflicts(tmp_path):
    src,index,s5=fixture(tmp_path,texts=(CODE,CODE),known=True,fields=[field('linear_weight','1','kg/m',decimal_convention='integer')])
    other=saved(tmp_path,src,index,page=1,known=True,fields=[field('linear_weight','2','kg/m',decimal_convention='integer')])
    cid=json.loads(s5.read_text())['candidates'][0]['id']
    summary=enrich(tmp_path,index,[s5,other],candidate_ids=[cid]);out=load(summary)
    assert len(out['assignments'])==1 and out['assignments'][0]['membership_effect']=='NONE'
    result=exported(tmp_path,index,summary['manifest'])
    assert result['export_status']=='COMPLETE',result
    props=[p for p in rows(result,'properties') if p['field']=='linear_weight']
    assert len(props)==2 and all(json.loads(p['conflict_ids']) for p in props)
    assert all(r['detail_id']=='' and r['target_point']=='' for r in rows(result,'detail_components'))
    assert json.loads(s5.read_text())['candidates'][0]['memberships'][0]['target_point'] is None


def test_cross_document_inputs_rejected(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    _,index,s5=fixture(a);_,_,foreign=fixture(b,texts=('different source',))
    with pytest.raises(ValueError,match='Cross-document'):
        enrich(tmp_path,index,[s5,foreign])


def test_selected_indexing_is_additive_and_at_most_ten(tmp_path):
    src,index,s5=fixture(tmp_path,index_pages=1)
    with sqlite3.connect(index) as db:
        db.row_factory=sqlite3.Row
        old=db.execute('SELECT id,geometry FROM pages').fetchall()
        doc=old[0]['id'].split(':')[0]
        added=e.index_selected(db,src,doc,[1],tmp_path/'selected')
        assert added==[1]
        assert db.execute('SELECT geometry FROM pages WHERE id=?',(old[0]['id'],)).fetchone()[0]==old[0]['geometry']
        assert e.index_selected(db,src,doc,[1],tmp_path/'selected')==[]
        with pytest.raises(ValueError,match='ten'):
            e.index_selected(db,src,doc,list(range(11)),tmp_path/'selected')
    out=load(enrich(tmp_path,index,[s5]))
    assert out['coverage']['indexed_pages_searched']==[0,1] and out['coverage']['unindexed_pages']==[2]


def test_explicit_visual_lookup_reference_properties_do_not_merge_unknown_scope(tmp_path):
    src,index,s5=fixture(tmp_path,texts=(CODE,None))
    plan=window_plan(src,index)
    client=Client(lambda ctx:lookup_response(ctx,[field('linear_weight','2500','g/m',decimal_convention='integer')]))
    summary=e.run([s5],tmp_path/'stage6',index_path=index,plan=plan,client=client,model=replace(MODEL,max_requests=1,max_tokens=4096))
    out=load(summary)
    weight=next(o for o in out['source_observations'] if o['field']=='linear_weight')
    assert weight['accepted'] and weight['normalized']=={'value':'2.5','unit':'kg/m'}
    assert out['matches'][0]['status']=='REVIEW' and out['assignments']==[]
    assert out['source_candidates'][0]['occurrence_ids']==[] and out['source_candidates'][0]['memberships']==[]
    assert out['reference_views'][0]['membership_effect']=='NONE'


def test_profile_caption_lookup_does_not_invent_callout_occurrence(tmp_path):
    src,index,s5=fixture(tmp_path,texts=(CODE,None))
    client=Client(lambda ctx:lookup_response(ctx,[field('description','Source caption')],kind='code_label'))
    out=load(e.run([s5],tmp_path/'stage6',index_path=index,plan=window_plan(src,index),client=client,model=replace(MODEL,max_requests=1)))
    assert out['source_candidates'][0]['occurrence_ids']==[]
    assert next(o for o in out['source_observations'] if o['field']=='description')['accepted']


def test_footnote_marker_scope_and_finish_are_conditional(tmp_path):
    src,index,s5=fixture(tmp_path,texts=(CODE,None))
    def reply(ctx):
        link=dict(component_local_id='part1',field_index=0,marker='*',marker_evidence=evidence('black anodized*'),
                  note_evidence=evidence('* '+CODE+' finish applies when specified',(.1,.6,.9,.7)),
                  scope_evidence=[evidence(CODE+' marked *')],explanation='This code and marker are explicitly named',uncertainty=[])
        return lookup_response(ctx,[field('surface_finish','black anodized*')],footnotes=[link])
    out=load(e.run([s5],tmp_path/'stage6',index_path=index,plan=window_plan(src,index),client=Client(reply),model=replace(MODEL,max_requests=1)))
    finish=next(o for o in out['source_observations'] if o['field']=='surface_finish')
    assert finish['accepted'] and finish['conditions']
    assert out['footnotes'][0]['supported'] and out['footnotes'][0]['observation_ids']==[finish['id']]
    result=exported(tmp_path,index,str(Path(tmp_path/'stage6/enrichments'/out['run_id']/'manifest.json')))
    assert any(r['kind']=='footnote_scope' for r in rows(result,'relationships'))


def test_general_note_and_adjacency_are_not_scope(tmp_path):
    src,index,s5=fixture(tmp_path,texts=(CODE,None))
    def reply(ctx):
        link=dict(component_local_id='part1',field_index=0,marker='*',marker_evidence=evidence('black anodized*'),
                  note_evidence=evidence('* all adjacent shapes aluminium'),scope_evidence=[evidence('Nearby drawings')],
                  explanation='Nearby note',uncertainty=[])
        return lookup_response(ctx,[field('surface_finish','black anodized*')],footnotes=[link])
    out=load(e.run([s5],tmp_path/'stage6',index_path=index,plan=window_plan(src,index),client=Client(reply),model=replace(MODEL,max_requests=1)))
    assert not out['footnotes'][0]['supported']
    assert not next(o for o in out['source_observations'] if o['field']=='surface_finish')['accepted']


def test_lookup_invalid_response_failure_and_cached_success_preserved(tmp_path):
    src,index,s5=fixture(tmp_path,texts=(CODE,None))
    plan=window_plan(src,index);model=replace(MODEL,max_requests=1,max_tokens=4096)
    good=lambda ctx:lookup_response(ctx,[field('description','Row description')])
    first=e.run([s5],tmp_path/'stage6',index_path=index,plan=plan,client=Client(good),model=model)
    old={str(p):p.read_bytes() for p in Path(first['manifest']).parent.rglob('*') if p.is_file()}
    bad=e.run([s5],tmp_path/'stage6',index_path=index,plan=plan,client=Client('{}'),model=model,force=True)
    assert bad['errors']==1 and load(bad)['execution_status']=='PARTIAL'
    resumed=e.run([s5],tmp_path/'stage6',index_path=index,plan=plan,client=Client(),model=replace(model,max_requests=1))
    assert resumed['lookups_reused']==1 and resumed['model_requests']==0
    assert all(Path(p).read_bytes()==raw for p,raw in old.items())


def test_incomplete_or_truncated_lookup_is_not_empty_success(tmp_path):
    src,index,s5=fixture(tmp_path,texts=(CODE,None))
    out=load(e.run([s5],tmp_path/'stage6',index_path=index,plan=window_plan(src,index),client=Client(ModelError('Model output incomplete or truncated')),model=replace(MODEL,max_requests=1)))
    assert out['failures'][0]['reason']=='model_output_truncated' and out['execution_status']=='PARTIAL'


def test_native_lookup_and_export_resumes_invalidate_inputs(tmp_path):
    _,index,s5=fixture(tmp_path,fields=[field()])
    s6=enrich(tmp_path,index,[s5]);first=exported(tmp_path,index,s6['manifest'])
    second=exported(tmp_path,index,s6['manifest'])
    assert second['reused'] and first['artifacts']==second['artifacts']
    changed=load(s6);changed['execution_status']='PARTIAL';Path(s6['manifest']).write_text(json.dumps(changed))
    third=exported(tmp_path,index,s6['manifest']);assert not third['reused'] and third['export_status']=='PARTIAL'


def test_repeated_occurrences_unicode_csv_json_and_no_context_membership(tmp_path):
    label='Dichtung, "Spezial"\nقطعة'
    _,index,s5=fixture(tmp_path,repeat=True,fields=[field('description',label)])
    s6=enrich(tmp_path,index,[s5]);result=exported(tmp_path,index,s6['manifest'])
    membership=rows(result,'detail_components')
    assert len(membership)==2 and len({r['occurrence_id'] for r in membership})==2
    assert all(r['detail_id']=='' and r['association_status']=='UNRESOLVED' for r in membership)
    prop=next(p for p in rows(result,'properties') if p['field']=='description')
    assert prop['value_raw']==label
    assert Path(result['artifacts']['properties']).read_bytes().startswith(b'\xef\xbb\xbf')
    assert json.loads(prop['evidence_ids'])
    for path in result['artifacts']['detail_results']:
        detail=json.loads(Path(path).read_text());assert detail['component_ids']==[] and detail['status']=='PROVISIONAL'
        assert detail['context_occurrence_ids']
    assert len(rows(result,'review'))==len({r['id'] for r in rows(result,'review')})


def test_supported_membership_still_marks_detail_provisional(tmp_path):
    _,index,s5=fixture(tmp_path,target=True)
    result=exported(tmp_path,index,enrich(tmp_path,index,[s5])['manifest'])
    assert any(row['detail_id'] for row in rows(result,'detail_components'))
    assert all(row['status']=='PROVISIONAL' for row in rows(result,'details'))


def altered(tmp_path,index,s5,mutate):
    raw=json.loads(s5.read_text());mutate(raw);s5.write_text(json.dumps(raw))
    return enrich(tmp_path,index,[s5])['manifest']


def test_missing_property_evidence_is_quarantined(tmp_path):
    _,index,s5=fixture(tmp_path,fields=[field()])
    def mutate(raw):next(o for o in raw['observations'] if o['field']=='description')['evidence']=[]
    result=exported(tmp_path,index,altered(tmp_path,index,s5,mutate))
    prop=next(p for p in rows(result,'properties') if p['field']=='description')
    assert prop['source_accepted']=='true' and prop['accepted']=='false' and result['validation_errors']>0


def test_invalid_coordinates_and_units_are_not_asserted(tmp_path):
    _,index,s5=fixture(tmp_path,fields=[field('linear_weight','2','kg/m',decimal_convention='integer')])
    def mutate(raw):
        weight=next(o for o in raw['observations'] if o['field']=='linear_weight')
        weight['evidence'][0]['box']=[-50,1,5,5];weight['normalized']={'value':'NaN','unit':'cm4'}
    result=exported(tmp_path,index,altered(tmp_path,index,s5,mutate))
    prop=next(p for p in rows(result,'properties') if p['field']=='linear_weight')
    assert prop['accepted']=='false' and prop['normalized']==''
    assert result['export_status']=='PARTIAL' and result['validation_errors']


def test_null_target_cannot_be_upgraded_in_exports(tmp_path):
    _,index,s5=fixture(tmp_path)
    def mutate(raw):
        member=raw['candidates'][0]['memberships'][0]
        member.update(target_point=[180,120],view_id=raw['candidates'][0]['observed_in_view_ids'][0],status='SUPPORTED_BY_STAGE4')
    result=exported(tmp_path,index,altered(tmp_path,index,s5,mutate))
    assert all(r['detail_id']=='' and r['target_point']=='' for r in rows(result,'detail_components'))
    assert any(r['code']=='NULL_MEMBERSHIP_UPGRADE' for r in rows(result,'review'))


def test_missing_upstream_reported_and_complete_export_preserved(tmp_path):
    _,index,s5=fixture(tmp_path,texts=(CODE,))
    s6=enrich(tmp_path,index,[s5]);complete=exported(tmp_path,index,s6['manifest'])
    assert complete['complete'],complete
    pointer=(tmp_path/'exports/latest-complete.json').read_bytes()
    historical={str(p):p.read_bytes() for p in Path(complete['artifacts']['summary']).parent.rglob('*') if p.is_file()}
    s5.rename(s5.with_name('preserved-source.json'))
    partial=exported(tmp_path,index,s6['manifest'])
    assert partial['export_status']=='PARTIAL' and partial['validation_errors']
    assert (tmp_path/'exports/latest-complete.json').read_bytes()==pointer
    assert all(Path(p).read_bytes()==raw for p,raw in historical.items())


def test_failed_csv_write_preserves_previous_complete_export(tmp_path,monkeypatch):
    _,index,s5=fixture(tmp_path,texts=(CODE,));s6=enrich(tmp_path,index,[s5]);complete=exported(tmp_path,index,s6['manifest'])
    pointer=(tmp_path/'exports/latest-complete.json').read_bytes()
    def fail(*args):raise ValueError('simulated write/round-trip failure')
    monkeypatch.setattr(x,'write_csv',fail)
    failed=exported(tmp_path,index,s6['manifest'],force=True)
    assert failed['export_status']=='FAILED' and not failed['complete']
    assert '/partial/' in failed['artifacts']['summary']
    assert (tmp_path/'exports/latest-complete.json').read_bytes()==pointer
    assert Path(complete['artifacts']['summary']).exists()


def test_missing_source_or_index_offline_partial_not_run(tmp_path):
    src,index,s5=fixture(tmp_path)
    src.rename(src.with_name('retained-source.pdf'))
    summary=enrich(tmp_path,index,[s5]);out=load(summary)
    assert out['execution_status']=='NOT_RUN' and out['model_requests']==0
    result=x.run(summary['manifest'],tmp_path/'exports',index_path=tmp_path/'missing.sqlite3')
    assert result['export_status']=='PARTIAL' and result['validation_errors']
    assert not (tmp_path/'missing.sqlite3').exists()


def test_csv_writer_structured_cells_quoting_and_leading_zeros(tmp_path):
    path=tmp_path/'cells.csv'
    expected=[{'code':'00 01/A','text':'"Gasket", Δ\nعربي','structured':{'condition':['A','B'], 'value':None}}]
    x.write_csv(path,['code','text','structured'],expected)
    with path.open(encoding='utf-8-sig',newline='') as f:row=next(csv.DictReader(f))
    assert row['code']==expected[0]['code'] and row['text']==expected[0]['text']
    assert json.loads(row['structured'])==expected[0]['structured']


def test_no_detail_not_exported_as_detail(tmp_path):
    _,index,s5=fixture(tmp_path);s6=enrich(tmp_path,index,[s5]);raw=load(s6)
    p4=Path(raw['sources'][0]['stage4_path']);s4=json.loads(p4.read_text());s4['upstream']['page_decision']='NO_DETAIL';p4.write_text(json.dumps(s4))
    raw['sources'][0]['stage4_sha256']=grouping.file_digest(p4);Path(s6['manifest']).write_text(json.dumps(raw))
    result=exported(tmp_path,index,s6['manifest'])
    assert rows(result,'details')==[]
    assert any(r['code']=='NO_DETAIL_VIOLATION' for r in rows(result,'review'))


def test_partial_enrichment_and_upstream_never_export_complete(tmp_path):
    _,index,s5=fixture(tmp_path)
    def mutate(raw):raw.update(state='PARTIAL',complete=False)
    result=exported(tmp_path,index,altered(tmp_path,index,s5,mutate))
    assert not result['complete'] and result['export_status']=='PARTIAL'
    assert result['acceptance_status']=='REVIEW'


def test_conditional_enrichment_does_not_conflict_with_unconditional_value(tmp_path):
    src,index,s5=fixture(tmp_path,texts=(CODE,CODE),known=True,fields=[field('linear_weight','1','kg/m',decimal_convention='integer')])
    other=saved(tmp_path,src,index,page=1,known=True,fields=[field('linear_weight','2','kg/m',decimal_convention='integer',conditions=[evidence('with reinforcement')])])
    cid=json.loads(s5.read_text())['candidates'][0]['id']
    result=exported(tmp_path,index,enrich(tmp_path,index,[s5,other],candidate_ids=[cid])['manifest'])
    props=[p for p in rows(result,'properties') if p['field']=='linear_weight']
    assert len(props)==2 and {p['conditional'] for p in props}=={'true','false'}
    assert all(json.loads(p['conflict_ids'])==[] for p in props)
    assert next(p for p in props if p['conditional']=='true')['condition_evidence_ids']!='[]'


def test_scope_evidence_coordinates_are_validated_before_transfer(tmp_path):
    src,index,s5=fixture(tmp_path,known=True)
    other=saved(tmp_path,src,index,page=1,known=True,fields=[field('linear_weight','2','kg/m',decimal_convention='integer')])
    raw=json.loads(s5.read_text());raw['identities'][0]['scope_evidence'][0]['evidence'][0]['box']=[-50,0,10,10];s5.write_text(json.dumps(raw))
    cid=raw['candidates'][0]['id'];s6=enrich(tmp_path,index,[s5,other],candidate_ids=[cid])
    result=exported(tmp_path,index,s6['manifest'])
    assert result['export_status']=='PARTIAL' and result['validation_errors']
    assert any(r['code'] in {'SCOPE_EVIDENCE_INVALID','INVALID_COORDINATE'} for r in rows(result,'review'))
    assert not any(r['kind']=='property_enrichment' and r['status']=='SUPPORTED' for r in rows(result,'relationships'))


def test_foreign_property_owner_is_quarantined(tmp_path):
    _,index,s5=fixture(tmp_path,fields=[field()])
    def mutate(raw):
        raw['candidates'][0]['page_id']='not-this-page'
        raw['identities'][0]['page_id']='not-this-page'
    result=exported(tmp_path,index,altered(tmp_path,index,s5,mutate))
    assert next(p for p in rows(result,'properties') if p['field']=='description')['accepted']=='false'
    assert any(r['code']=='FOREIGN_PROPERTY_EVIDENCE' for r in rows(result,'review'))


def test_dangling_property_owner_is_reported(tmp_path):
    _,index,s5=fixture(tmp_path,fields=[field()])
    s6=enrich(tmp_path,index,[s5]);raw=load(s6)
    bad=copy.deepcopy(json.loads(s5.read_text())['observations'][0]);bad['id']='orphan-observation';bad['candidate_id']='unknown-component'
    raw['source_observations'].append(bad);Path(s6['manifest']).write_text(json.dumps(raw))
    result=exported(tmp_path,index,s6['manifest'])
    assert result['validation_errors'] and any(r['code']=='PROPERTY_OWNER_MISSING' for r in rows(result,'review'))


def test_enrichment_prompt_and_model_identity_invalidate_lookup_cache(tmp_path,monkeypatch):
    src,index,s5=fixture(tmp_path,texts=(CODE,None));plan=window_plan(src,index)
    good=lambda ctx:lookup_response(ctx,[field('description','Reference label')])
    model=replace(MODEL,max_requests=1)
    first=e.run([s5],tmp_path/'stage6',index_path=index,plan=plan,client=Client(good),model=model)
    monkeypatch.setattr(e,'PROMPT',e.PROMPT+' Refined source evidence rule.')
    second=e.run([s5],tmp_path/'stage6',index_path=index,plan=plan,client=Client(good),model=model)
    assert second['lookups_completed']==1 and second['model_requests']==1
    third=e.run([s5],tmp_path/'stage6',index_path=index,plan=plan,client=Client(good),model=replace(model,identity='checkpoint@changed'))
    assert third['lookups_completed']==1 and third['model_requests']==1


def test_native_search_completeness_does_not_claim_unseen_visual_coverage(tmp_path):
    _,index,s5=fixture(tmp_path)
    out=load(enrich(tmp_path,index,[s5]))
    assert out['coverage']['native_index_status']=='COMPLETE'
    assert out['coverage']['visual_unsearched_pages']==[1,2]
    assert out['source_coverage_status']=='PARTIAL'


def test_detail_image_preserves_panel_pixels_and_aspect_ratio(tmp_path):
    first=tmp_path/'first.png';second=tmp_path/'second.png'
    Image.new('RGB',(90,30),'red').save(first)
    Image.new('RGB',(40,80),'blue').save(second)
    x.detail_image([{'path':str(first)},{'path':str(second)}],tmp_path/'image.png')
    with Image.open(tmp_path/'image.png') as output:
        assert output.size==(90,110)
        assert output.getpixel((89,29))==(255,0,0)
        assert output.getpixel((39,109))==(0,0,255)
        assert output.getpixel((40,109))==(255,255,255)

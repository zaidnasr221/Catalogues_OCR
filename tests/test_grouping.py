"""Synthetic saved Stage 2 exports: no catalogs or live inference."""
import hashlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import urllib.error

from PIL import Image, ImageDraw
import pymupdf as fitz
import pytest

from as_ocr import detection, grouping as group, indexer
from as_ocr.qwen import ModelConfig


def saved_page(tmp_path, labels, *, kinds=None, page=0, attempt='detection-a', types=None, reviews=True):
    directory=tmp_path/attempt; directory.mkdir()
    candidates=[]
    for n,text in enumerate(labels):
        cid=f'candidate-{n}'
        crop=directory/f'{cid}.png'
        size=(100+13*n,70+9*n)
        image=Image.new('RGB',size,'white')
        draw=ImageDraw.Draw(image); draw.rectangle((10,10,size[0]-10,size[1]-10),outline='black',width=2)
        image.save(crop)
        box=[20+160*n,50,140+160*n,170]
        sx,sy=size[0]/120,size[1]/120
        annotations=[{'ordinal':100*n+k,'text':label,'box':[box[0]+5,40,box[0]+100,55],
                      'status':'attached','owner_structural_box':box} for k,label in enumerate(text if isinstance(text,list) else [text]) if label]
        candidates.append({'id':cid,'label':f'C{n+1:03d}','document_id':'synthetic-doc','page_id':f'synthetic-doc:{page}',
                           'page_index':page,'source_path':'nonexistent-synthetic.pdf','box':box,'structural_box':box,
                           'kind':kinds[n] if kinds else 'assembly_candidate',
                           'view_type':types[n] if types else ('section' if n%2==0 else 'isometric'),
                           'status':'REVIEW' if reviews else 'PROPOSED',
                           'review_reasons':['Boundary uncertainty'] if reviews else [],
                           'method':'deterministic','annotations':annotations,'evidence':[],
                           'parent_id':None,'child_ids':[],'crop_path':str(crop.resolve()),
                           'crop_geometry':{'source_to_pixels':[sx,0,0,sy,-box[0]*sx,-box[1]*sy],
                                            'source_context_box':box,'pixel_size':list(size)}})
    manifest={'stage':2,'detector_version':'4','document_id':'synthetic-doc','page_id':f'synthetic-doc:{page}',
              'page_index':page,'source_path':'nonexistent-synthetic.pdf','candidates':candidates,
              'overview_geometry':{'source_context_box':[0,0,200+160*len(labels),220]},
              'annotation_audit':[a for c in candidates for a in c['annotations']]}
    path=directory/'manifest.json'; path.write_text(json.dumps(manifest))
    return path


def read_result(summary):
    assert summary['errors']==0,summary
    return json.loads(Path(summary['manifests'][0]).read_text())


def hashes(folder):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}


def test_explicit_detail_section_isometric_and_exact_source_name(tmp_path):
    labels=['Detail Q7 — Fixture joint — Section A','Detail Q7 — Fixture joint — Isometric']
    source=saved_page(tmp_path,labels)
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['views'])==2 and len(result['details'])==1
    detail=result['details'][0]
    assert detail['grouping_status']=='CONFIRMED'
    assert detail['name']=='Detail Q7 — Fixture joint' and not detail['generated_name']
    assert {t['exact_source_text'] for t in detail['source_titles']}==set(labels)
    assert detail['inherited_source_status']=='REVIEW' and not detail['complete']
    assert detail['review_status']=='REVIEW' and detail['completeness']=='NOT_ASSESSED'
    assert all(len(m['primary'])==1 for m in result['view_to_detail'].values())


def test_explicit_section_reference_links_unique_target(tmp_path):
    source=saved_page(tmp_path,['Section C-C','Isometric — see Section C-C'])
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==1 and result['details'][0]['grouping_status']=='CONFIRMED'
    assert result['details'][0]['grouping_evidence'][0]['type']=='explicit_section_reference'


def test_ambiguous_crop_text_and_view_words_do_not_become_detail_ids(tmp_path):
    source=saved_page(tmp_path,['DETAIL SECTION A','DETAIL SECTION B'])
    manifest=json.loads(source.read_text())
    for candidate in manifest['candidates']:
        candidate['annotations'].append({'text':'Detail Z4','ordinal':900,'box':[10,10,100,30],'status':'ambiguous'})
        candidate['evidence'].append({'table':'spans','ordinal':900,'text':'Detail Z4'})
    source.write_text(json.dumps(manifest))
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==2 and all(d['grouping_status']=='SINGLETON' for d in result['details'])


def test_duplicate_native_labels_do_not_create_competing_section_targets(tmp_path):
    source=saved_page(tmp_path,[['Section K-K','Section K-K'],'See Section K-K'])
    result=read_result(group.run([source],tmp_path/'out'))
    assert result['details'][0]['grouping_status']=='CONFIRMED'


def test_nearby_similar_views_remain_singletons(tmp_path):
    source=saved_page(tmp_path,['',''])
    manifest=json.loads(source.read_text())
    manifest['candidates'][1]['structural_box']=[135,50,255,170]
    source.write_text(json.dumps(manifest))
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==2
    assert all(d['grouping_status']=='SINGLETON' for d in result['details'])
    assert result['supporting_clues'] and not result['grouping_proposals'][0]['evidence']


def test_repeated_codes_and_weak_transitive_chain_do_not_group(tmp_path):
    source=saved_page(tmp_path,['P1200',['P1200','P2300'],'P2300'])
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==3
    assert all(len(d['view_ids'])==1 for d in result['details'])
    assert len([c for c in result['supporting_clues'] if c['shared_strings']])==2


@pytest.mark.parametrize('different_document',[False,True])
def test_same_identifier_on_separate_pages_never_deduplicates(tmp_path,different_document):
    first=saved_page(tmp_path,['Detail S9 — Section','Detail S9 — Isometric'],page=0,attempt='page-a')
    second=saved_page(tmp_path,['Detail S9 — Section','Detail S9 — Isometric'],page=0 if different_document else 1,attempt='page-b')
    if different_document:
        manifest=json.loads(second.read_text())
        manifest.update(document_id='other-synthetic-document',page_id='other-synthetic-document:0')
        for candidate in manifest['candidates']:
            candidate.update(document_id=manifest['document_id'],page_id=manifest['page_id'])
        second.write_text(json.dumps(manifest))
    run=group.run([first,second],tmp_path/'out')
    assert run['pages_completed']==2 and run['confirmed_multiview_groups']==2
    results=[json.loads(Path(p).read_text()) for p in run['manifests']]
    assert results[0]['details'][0]['id']!=results[1]['details'][0]['id']
    assert set(v['id'] for v in results[0]['views']).isdisjoint(v['id'] for v in results[1]['views'])


def test_more_than_four_views_paginate_without_dropping_panels(tmp_path):
    source=saved_page(tmp_path,[f'Detail T8 — Section {n}' for n in range(7)])
    result=read_result(group.run([source],tmp_path/'out',group.Config(composite_max=360,panel_max=120)))
    detail=result['details'][0]
    assert detail['grouping_status']=='CONFIRMED' and len(detail['view_ids'])==7
    assert len(detail['composites'])>1
    panels=[p for c in detail['composites'] for p in c['panels']]
    assert {p['view_id'] for p in panels}==set(detail['view_ids']) and len(panels)==7
    assert all(max(c['pixel_size'])<=360 for c in detail['composites'])


def test_profiles_metadata_and_mixed_containment_are_independent(tmp_path):
    source=saved_page(tmp_path,['','','','',''],kinds=['profile_candidate','document_metadata','mixed_content','unknown','assembly_candidate'])
    manifest=json.loads(source.read_text()); candidates=manifest['candidates']
    candidates[2]['child_ids']=[candidates[3]['id']]; candidates[3]['parent_id']=candidates[2]['id']
    source.write_text(json.dumps(manifest))
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['views'])==5 and len(result['details'])==1
    by_candidate={v['candidate_id']:v for v in result['views']}
    assert by_candidate['candidate-3']['parent_view_id']==by_candidate['candidate-2']['id']
    assert not by_candidate['candidate-2']['primary_detail_ids'] and not by_candidate['candidate-3']['primary_detail_ids']
    assert result['details'][0]['view_ids']==[by_candidate['candidate-4']['id']]
    assert by_candidate['candidate-0']['eligibility']=='STANDALONE_PROFILE'


def test_singleton_assembly_is_provisional_with_neutral_generated_name(tmp_path):
    source=saved_page(tmp_path,[''],reviews=False)
    details=read_result(group.run([source],tmp_path/'out'))['details']
    assert len(details)==1
    detail=details[0]
    assert detail['generated_name'] and detail['name']=='Detail — page 1 — group 1'
    assert detail['grouping_status']=='SINGLETON' and detail['assembly_status']=='PROVISIONAL'
    assert detail['review_status']=='REVIEW' and not detail['complete']


@pytest.mark.parametrize('prefix',['Detail # Z3','Detail: Z3','Detail No. Z3'])
def test_explicit_identifier_punctuation_and_singleton_source_title(tmp_path,prefix):
    label=prefix+' — Original fixture title — Section'
    source=saved_page(tmp_path,[label])
    detail=read_result(group.run([source],tmp_path/'out'))['details'][0]
    assert not detail['generated_name'] and detail['name']==prefix+' — Original fixture title'
    assert detail['source_titles'][0]['exact_source_text']==label


def test_conflicting_membership_proposals_stay_alternatives(tmp_path):
    source=saved_page(tmp_path,['Detail A',['Detail A','Detail B'],'Detail B'])
    result=read_result(group.run([source],tmp_path/'out'))
    alternatives=[d for d in result['details'] if d['membership_role']=='alternative']
    primary=[d for d in result['details'] if d['membership_role']=='primary']
    assert len(alternatives)==2 and all(d['grouping_status']=='REVIEW' for d in alternatives)
    assert len(primary)==3 and all(len(d['view_ids'])==1 for d in primary)
    assert all(len(v['primary_detail_ids'])==1 for v in result['views'])
    assert any(len(v['alternative_detail_ids'])==2 for v in result['views'])
    assert all(d['conflicting_proposal_ids'] for d in alternatives)


def test_source_titles_with_conflicting_scope_remain_review(tmp_path):
    source=saved_page(tmp_path,['Detail C2 — First title — Section','Detail C2 — Different title — Isometric'])
    details=read_result(group.run([source],tmp_path/'out'))['details']
    assert len([d for d in details if d['membership_role']=='primary'])==2
    detail=next(d for d in details if d['membership_role']=='alternative')
    assert detail['generated_name'] and detail['grouping_status']=='REVIEW'
    assert len(detail['source_titles'])==2


def test_scoped_title_proposal_preserves_exact_quote_and_scope_uncertainty(tmp_path):
    source=saved_page(tmp_path,['',''])
    manifest=json.loads(source.read_text())
    quote='Assembly overview — source caption'
    manifest['annotation_audit']=[{'ordinal':31,'text':quote,'box':[10,10,300,30],'status':'unassigned'}]
    manifest['grouping_evidence']=[{'type':'scoped_title','text':quote,'candidate_ids':['candidate-0','candidate-1']}]
    source.write_text(json.dumps(manifest))
    details=read_result(group.run([source],tmp_path/'out'))['details']
    assert len([d for d in details if d['membership_role']=='primary'])==2
    detail=next(d for d in details if d['membership_role']=='alternative')
    assert detail['generated_name'] and detail['source_titles'][0]['text']==quote
    assert detail['grouping_status']=='REVIEW' and detail['source_titles'][0]['source']['ordinal']==31


def test_ids_resume_version_config_and_upstream_invalidation(tmp_path,monkeypatch):
    source=saved_page(tmp_path,['Detail E6 — Section','Detail E6 — Isometric'])
    out=tmp_path/'out'
    first=group.run([source],out); original=read_result(first)
    before=hashes(out/'groupings')
    assert group.run([source],out)['pages_reused']==1
    forced=read_result(group.run([source],out,force=True))
    assert [v['id'] for v in forced['views']]==[v['id'] for v in original['views']]
    assert [d['id'] for d in forced['details']]==[d['id'] for d in original['details']]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in before.items())
    assert group.run([source],out,group.Config(columns=2))['pages_completed']==1
    with monkeypatch.context() as patch:
        patch.setattr(group,'VERSION','next-synthetic-logic')
        assert group.run([source],out)['pages_completed']==1
    manifest=json.loads(source.read_text()); manifest['candidates'][0]['review_reasons'].append('New upstream uncertainty')
    source.write_text(json.dumps(manifest))
    changed=read_result(group.run([source],out))
    assert changed['views'][0]['id']!=original['views'][0]['id']
    with Image.open(changed['views'][0]['crop_path']) as image:
        replacement=image.copy()
    replacement.putpixel((0,0),(100,50,0)); replacement.save(changed['views'][0]['crop_path'])
    assert group.run([source],out)['pages_completed']==1


def test_composite_aspect_ratio_transforms_and_original_crop_preservation(tmp_path):
    source=saved_page(tmp_path,['Detail F6 — Section','Detail F6 — Isometric'])
    before=hashes(source.parent)
    result=read_result(group.run([source],tmp_path/'out',group.Config(composite_max=512,panel_max=180)))
    for composite in result['details'][0]['composites']:
        with Image.open(composite['path']) as image:
            assert list(image.size)==composite['pixel_size'] and max(image.size)<=512
        for panel in composite['panels']:
            width,height=panel['source_crop_pixel_size']; x0,y0,x1,y1=panel['panel_box']
            assert abs((x1-x0)-(y1-y0)*width/height)<=1
            mapped=fitz.Rect(0,0,width,height)*fitz.Matrix(*panel['crop_pixels_to_composite'])
            assert list(mapped)==pytest.approx(panel['panel_box'])
            assert not panel['common_physical_scale'] and panel['source_to_composite']
            view=next(v for v in result['views'] if v['id']==panel['view_id'])
            source_projected=fitz.Rect(view['crop_geometry']['source_context_box'])*fitz.Matrix(*panel['source_to_composite'])
            assert list(source_projected)==pytest.approx(panel['panel_box'],abs=.001)
    assert hashes(source.parent)==before


def test_invalid_model_view_references_rejected_individually(tmp_path):
    source=saved_page(tmp_path,['',''])
    _,views,_=group.load_manifest(source)
    valid={'view_ids':[v['id'] for v in views],'observations':['Explicit matching marks visible'],'uncertainty':['Boundaries uncertain']}
    bad={**valid,'view_ids':[views[0]['id'],'invented-view']}
    accepted,rejected=group.validate_model_groups(json.dumps({'groups':[valid,bad]}),views)
    assert len(accepted)==1 and len(rejected)==1
    assert accepted[0]['strength']=='model'
    with pytest.raises(group.ModelError,match='Malformed'):
        group.validate_model_groups('not JSON',views)


def mock_groups(monkeypatch,view_ids):
    content=json.dumps({'groups':[{'view_ids':view_ids,'observations':['Proposed visual correspondence'],
                                   'uncertainty':['No independently established code/geometric equivalence']}]})
    reply={'model':'served-qwen','choices':[{'finish_reason':'stop','message':{'content':content}}]}
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:io.BytesIO(json.dumps(reply).encode()))


def test_mocked_hybrid_keeps_source_review_and_identity_sensitive_resume(tmp_path,monkeypatch):
    source=saved_page(tmp_path,['',''])
    _,views,_=group.load_manifest(source)
    mock_groups(monkeypatch,[v['id'] for v in views])
    model=ModelConfig(endpoint='http://mock/v1',model='served-qwen',identity='fixture@revision',max_requests=1,retries=0)
    cfg=group.Config(mode='hybrid')
    first=group.run([source],tmp_path/'out',cfg,model)
    result=read_result(first); detail=next(d for d in result['details'] if d['membership_role']=='alternative')
    assert len([d for d in result['details'] if d['membership_role']=='primary'])==2
    assert first['model_requests']==1 and detail['grouping_status']=='REVIEW'
    assert detail['inherited_source_status']=='REVIEW' and not detail['complete']
    assert group.run([source],tmp_path/'out',cfg,model)['pages_reused']==1
    changed=ModelConfig(endpoint='http://mock/v1',model='served-qwen',identity='fixture@new-revision',max_requests=1,retries=0)
    assert group.run([source],tmp_path/'out',cfg,changed)['pages_completed']==1
    with monkeypatch.context() as patch:
        patch.setattr(group,'GROUP_PROMPT',group.GROUP_PROMPT+' Changed synthetic instruction.')
        assert group.run([source],tmp_path/'out',cfg,model)['pages_completed']==1


def test_missing_model_identity_never_enables_live_calls_or_resume(tmp_path,monkeypatch):
    source=saved_page(tmp_path,['',''])
    def forbidden(*a,**k):
        raise AssertionError('No network calls permitted')
    monkeypatch.setattr('urllib.request.urlopen',forbidden)
    cfg=group.Config(mode='hybrid')
    model=ModelConfig(endpoint='http://mock/v1',model='served-qwen')
    first=group.run([source],tmp_path/'out',cfg,model)
    second=group.run([source],tmp_path/'out',cfg,model)
    assert first['model_requests']==second['model_requests']==0
    assert second['pages_reused']==0 and read_result(first)['model']['state']=='unavailable'


def test_model_failures_preserve_deterministic_results_and_retry(tmp_path,monkeypatch):
    source=saved_page(tmp_path,['',''])
    reply={'model':'served-qwen','choices':[{'finish_reason':'stop','message':{'content':'invalid json'}}]}
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:io.BytesIO(json.dumps(reply).encode()))
    model=ModelConfig(endpoint='http://mock/v1',model='served-qwen',identity='fixture@rev',max_requests=1,retries=0)
    cfg=group.Config(mode='hybrid')
    first=group.run([source],tmp_path/'out',cfg,model)
    assert first['model_failures']==1 and first['model_requests']==1
    assert len(read_result(first)['details'])==2
    assert group.run([source],tmp_path/'out',cfg,model)['pages_reused']==0


def test_mocked_invalid_model_references_keep_singletons_and_failure_evidence(tmp_path,monkeypatch):
    source=saved_page(tmp_path,['',''])
    _,views,_=group.load_manifest(source)
    mock_groups(monkeypatch,[views[0]['id'],'invented-view'])
    model=ModelConfig(endpoint='http://mock/v1',model='served-qwen',identity='fixture@rev',max_requests=1,retries=0)
    summary=group.run([source],tmp_path/'out',group.Config(mode='hybrid'),model)
    result=read_result(summary)
    assert summary['model_requests']==1 and summary['model_failures']==1
    assert result['model']['state']=='failed' and result['model']['failures']
    assert all(d['grouping_status']=='SINGLETON' for d in result['details'])
    assert not any(d['membership_role']=='alternative' for d in result['details'])


def test_model_budget_across_comparison_pages_is_bounded_and_explicit(tmp_path,monkeypatch):
    source=saved_page(tmp_path,['']*7)
    reply={'model':'served-qwen','choices':[{'finish_reason':'stop','message':{'content':'{"groups":[]}'}}]}
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:io.BytesIO(json.dumps(reply).encode()))
    model=ModelConfig(endpoint='http://mock/v1',model='served-qwen',identity='fixture@rev',max_requests=1,retries=0)
    cfg=group.Config(mode='hybrid',composite_max=360,panel_max=120)
    summary=group.run([source],tmp_path/'out',cfg,model)
    result=read_result(summary)
    assert summary['model_requests']==1 and summary['model_failures']==1
    assert result['model']['unexamined_view_ids'] and len(result['views'])==7
    assert len(result['details'])==7
    high=group.run([source],tmp_path/'high-budget',cfg,replace(model,max_requests=100))
    assert high['model_requests']<100 and high['model_failures']==0


def test_model_retries_count_toward_the_same_budget(tmp_path,monkeypatch):
    source=saved_page(tmp_path,['',''])
    calls=[]
    reply={'model':'served-qwen','choices':[{'finish_reason':'stop','message':{'content':'{"groups":[]}'}}]}
    def request(*a,**k):
        calls.append(True)
        if len(calls)==1:
            raise urllib.error.URLError('synthetic transient failure')
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr('urllib.request.urlopen',request)
    monkeypatch.setattr('as_ocr.qwen.time.sleep',lambda *a:None)
    model=ModelConfig(endpoint='http://mock/v1',model='served-qwen',identity='fixture@rev',max_requests=2,retries=1)
    summary=group.run([source],tmp_path/'out',group.Config(mode='hybrid'),model)
    assert summary['model_requests']==2 and summary['model_failures']==0 and len(calls)==2


def test_failed_attempt_preserves_previous_success_and_missing_exports_invalidate(tmp_path,monkeypatch):
    source=saved_page(tmp_path,[''])
    out=tmp_path/'out'; first=group.run([source],out)
    before=hashes(out/'groupings')
    def fail(*a,**k):
        raise OSError('Synthetic composite export failure')
    with monkeypatch.context() as patch:
        patch.setattr(group,'export_page',fail)
        assert group.run([source],out,force=True)['errors']==1
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in before.items())
    assert group.run([source],out)['pages_reused']==1
    result=read_result(first)
    Path(result['details'][0]['composites'][0]['path']).unlink()
    assert group.run([source],out)['pages_completed']==1
    with sqlite3.connect(out/'index.sqlite3') as db:
        assert db.execute("SELECT COUNT(*) FROM grouping_pages WHERE state='error'").fetchone()[0]==1
        assert db.execute('PRAGMA foreign_key_check').fetchall()==[]


def test_exact_attempt_lookup_additive_persistence_and_no_detection(tmp_path,monkeypatch):
    source=saved_page(tmp_path,[''])
    index=tmp_path/'index.sqlite3'
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE detection_pages(id TEXT,state TEXT,manifest TEXT)')
        db.execute('INSERT INTO detection_pages VALUES(?,?,?)',(source.parent.name,'complete',str(source)))
        before=db.execute('SELECT * FROM detection_pages').fetchall()
    paths=group.resolve_inputs([],['detection-a'],index)
    assert paths==[source.resolve()]
    def forbidden(*a,**k):
        raise AssertionError('Grouping must never rerun ingestion/detection')
    monkeypatch.setattr(indexer,'run',forbidden); monkeypatch.setattr(detection,'run',forbidden)
    assert group.run(paths,tmp_path/'out',index_path=index)['pages_completed']==1
    with sqlite3.connect(index) as db:
        assert db.execute('SELECT * FROM detection_pages').fetchall()==before
        assert db.execute('SELECT COUNT(*) FROM grouping_views').fetchone()[0]==1
    with pytest.raises(ValueError,match='No complete detection attempt'):
        group.resolve_inputs([],['obsolete-attempt'],index)


def test_cross_page_model_references_and_metadata_membership_rejected(tmp_path):
    source=saved_page(tmp_path,['',''],kinds=['assembly_candidate','document_metadata'])
    _,views,_=group.load_manifest(source)
    payload=json.dumps({'groups':[{'view_ids':[v['id'] for v in views], 'observations':['A claim'], 'uncertainty':[]}]})
    assert group.validate_model_groups(payload,views)[1]
    views[1]['kind']='assembly_candidate'; views[1]['page_index']=1
    assert group.validate_model_groups(payload,views)[1]

"""Synthetic original-source Stage 1–3 contracts; no catalog or live inference."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pymupdf as fitz
import pytest

from as_ocr import callouts as c, detection, grouping, indexer
from as_ocr.qwen import ModelConfig, ModelError

MODEL = ModelConfig('http://localhost:8003/v1','fake','checkpoint@immutable',retries=0,max_tokens=8192,max_requests=10)


def observation(raw='001.20 - A', box=(.1,.1,.3,.15), **kwargs):
    result = dict(leader_kind='uncertain',raw_text=raw,category='product_code',reading_state='exact',text_box=list(box),
                  native_candidate_ids=[],reading_evidence=['Article code label at leader origin'],
                  owner_view_id=None,association_method='unresolved',leader_state='missing',
                  leader_points=[],target_point=None,association_evidence=[],uncertainty=[])
    result.update(kwargs)
    return result


def response(*observations, coverage='complete', uncertainty=()):
    return json.dumps(dict(coverage=coverage,evidence=['All visible labels inspected'],
                           uncertainty=list(uncertainty),observations=list(observations)))


class Client:
    def __init__(self, *responses):
        self.config=MODEL
        self.responses = list(responses); self.requests = 0; self.response_models = {'fake'}
        self.contexts = []; self.schemas = []

    def request_json(self,image,prompt,context_text,*,response_schema):
        assert image.width <= 2048 and image.height <= 2048
        assert 'UNTRUSTED' in prompt
        self.contexts.append(json.loads(context_text)); self.schemas.append(response_schema)
        self.requests += 1
        if not self.responses:
            raise ModelError('Budget exhausted')
        result = self.responses.pop(0)
        if isinstance(result,Exception):
            raise result
        return result(self.contexts[-1]) if callable(result) else result


def fixture(tmp_path, *, text=None, rotation=0, scanned=False, boxes=None, no_detail=False):
    source = tmp_path/('source.png' if scanned else 'source.pdf')
    if scanned:
        Image.new('RGB',(300,240),'white').save(source)
    else:
        with fitz.open() as doc:
            page = doc.new_page(width=300,height=240)
            if text:
                page.insert_text((30,35),text,fontsize=10)
            page.set_rotation(rotation)
            doc.save(source)
    root = tmp_path/'stage1'
    indexer.run(source,root)
    index = root/'index.sqlite3'
    with sqlite3.connect(index) as db:
        db.row_factory = sqlite3.Row
        row = dict(db.execute('SELECT * FROM pages').fetchone())
    geometry = json.loads(row['geometry'])
    destination = tmp_path/'detection-one'; destination.mkdir()
    candidates = []
    for n,box in enumerate(boxes or [[10,10,290,230]]):
        image,mapping = detection.render(source,0,geometry,box,1200)
        crop = destination/f'c{n}.png'; image.save(crop)
        candidates.append(dict(id=f'c{n}',document_id=row['document_id'],page_id=row['id'],page_index=0,
               source_path=str(source),box=box,structural_box=box,kind='assembly_candidate',
               view_type='section',status='REVIEW',review_reasons=['Clipped Stage 2 label'],
               parent_id=None,child_ids=[],annotations=[],crop_path=str(crop),crop_geometry=mapping))
    manifest = dict(stage=2,detector_version='7',attempt_id=destination.name,document_id=row['document_id'],
                    page_id=row['id'],page_index=0,source_path=str(source),candidates=candidates,
                    page_decision='REVIEW',page_analysis={'review_reasons':['Uncertain page coverage']})
    if no_detail:
        manifest.update(candidates=[],page_decision='NO_DETAIL',skip_reason=['No eligible views'])
    path = destination/'manifest.json'; path.write_text(json.dumps(manifest))
    return path,index


def result(summary):
    return json.loads(Path(summary['manifest']).read_text())


def run(path,index,tmp_path,client,**kwargs):
    return c.run(path,tmp_path/'stage4',index_path=index,client=client,model=MODEL,
                 config=c.Config(context_fraction=0),**kwargs)


def test_exact_raw_codes_and_markers(tmp_path):
    path,index = fixture(tmp_path)
    codes = ['00012','001.20 - A','00 123/AB.02']
    markers = ['50 mm','item 2','Fig. 3','Page 7','4 pcs','20 x 50']
    observations = [observation(text,(.05,.05+i*.065,.4,.09+i*.065)) for i,text in enumerate(codes+markers)]
    client = Client(response(*observations))
    out = result(run(path,index,tmp_path,client))
    assert [o['raw_code'] for o in out['occurrences']] == codes
    assert out['occurrences'][1]['normalized_code'] == '001.20-A'
    assert len(out['markers']) == len(markers)
    assert {r['category'] for r in out['markers']} == {'dimension','item_marker','figure_reference','page_number','quantity'}
    assert all(o['status']=='REVIEW' and o['source_view_id'] is None and o['target_point'] is None for o in out['occurrences'])
    assert client.schemas == [c.Analysis.model_json_schema()]
    assert out['occurrences'][0]['inherited_review_reasons']
    assert Path(out['annotated_views'][0]['path']).is_file()


def test_same_code_different_label_occurrences(tmp_path):
    path,index = fixture(tmp_path)
    out = result(run(path,index,tmp_path,Client(response(observation('001'),observation('001',(.5,.5,.7,.55))))))
    assert len(out['occurrences']) == 2
    assert len({o['id'] for o in out['occurrences']}) == 2


def test_overlap_crops_reconcile_and_conflicting_readings(tmp_path):
    path,index = fixture(tmp_path,boxes=[[10,10,290,230],[10,10,290,230]])
    out = result(run(path,index,tmp_path,Client(response(observation('001')),response(observation('001')))))
    assert len(out['occurrences']) == 1
    assert len(out['occurrences'][0]['observed_in_view_ids']) == 2
    out = result(run(path,index,tmp_path,Client(response(observation('001')),response(observation('007'))),force=True))
    o = out['occurrences'][0]
    assert o['raw_code'] is None and o['reading_state'] == 'conflicting'
    assert {r['raw_text'] for r in o['readings']} == {'001','007'}


def test_ambiguous_duplicate_identity_is_review(tmp_path):
    path,index = fixture(tmp_path,boxes=[[10,10,290,230],[10,10,290,230]])
    out = result(run(path,index,tmp_path,Client(response(observation()),response(observation(box=(.2,.1,.4,.15))))))
    assert len(out['occurrences']) == 2
    assert all('Ambiguous duplicate occurrence identity' in o['review_reasons'] for o in out['occurrences'])


def test_native_candidates_are_fallible_exact_evidence(tmp_path):
    path,index = fixture(tmp_path,text='001.20 - A')
    def reply(context):
        native = context['native_candidates'][0]
        assert native['raw_text'] == '001.20 - A'
        return response(observation(native['raw_text'],native['text_box'],native_candidate_ids=[native['id']]))
    out = result(run(path,index,tmp_path,Client(reply)))
    o = out['occurrences'][0]
    assert o['raw_code']=='001.20 - A' and o['native_span_ids']==['span-0']
    def conflict(context):
        native = context['native_candidates'][0]
        return response(observation('007.20 - A',native['text_box'],native_candidate_ids=[native['id']]))
    o = result(run(path,index,tmp_path,Client(conflict),force=True))['occurrences'][0]
    assert o['raw_code'] is None and o['reading_state']=='conflicting'
    assert {r['raw_text'] for r in o['readings']} == {'001.20 - A','007.20 - A'}


def test_regex_never_proves_product_on_failure(tmp_path):
    path,index = fixture(tmp_path,text='000123')
    summary = run(path,index,tmp_path,Client(ModelError('timeout')))
    out = result(summary)
    assert summary['errors']==1 and out['state']=='FAILED' and not out['complete']
    assert out['occurrences']==[]
    report = json.loads(Path(out['review_path']).read_text())
    assert report['unresolved_views'][0]['native_candidates'][0]['raw_text']=='000123'


@pytest.mark.parametrize('scanned',[False,True])
def test_native_and_scanned_have_visual_path(tmp_path,scanned):
    path,index = fixture(tmp_path,text=None if scanned else '009',scanned=scanned)
    client = Client(response(observation('009')))
    out = result(run(path,index,tmp_path,client))
    assert out['model_requests']==1 and out['occurrences'][0]['raw_code']=='009'
    assert bool(client.contexts[0]['native_candidates']) is not scanned


@pytest.mark.parametrize('rotation',[0,90,180,270])
def test_rotated_coordinates_from_original(tmp_path,rotation):
    path,index = fixture(tmp_path,text='00012',rotation=rotation)
    def reply(context):
        native = context['native_candidates'][0]
        return response(observation('00012',native['text_box'],native_candidate_ids=[native['id']]))
    out = result(run(path,index,tmp_path,Client(reply)))
    box = out['occurrences'][0]['text_box']
    with sqlite3.connect(index) as db:
        native = json.loads(db.execute('SELECT data FROM spans').fetchone()[0])['bbox']
    assert box == pytest.approx(native,abs=.001)


def leader_reply(context, *, state='clear', points=None):
    owner = context['focus_view_id']
    clear = state=='clear'
    return response(observation('001',(.1,.1,.3,.15),owner_view_id=owner if clear else None,
        association_method='leader' if clear else 'unresolved',leader_state=state,
        leader_points=points or [[.3,.125],[.6,.5]],target_point=[.6,.5] if clear else None,
        association_evidence=['Visible leader runs from code label to the indicated profile corner']))


@pytest.mark.parametrize('state',['clear','crossed','shared','clipped','uncertain','missing'])
def test_visual_leader_and_unresolved_targets(tmp_path,state):
    path,index = fixture(tmp_path)
    o = result(run(path,index,tmp_path,Client(lambda ctx:leader_reply(ctx,state=state))))['occurrences'][0]
    if state == 'clear':
        assert o['target_point'] is not None and o['source_view_id']
        assert o['association_method']=='qwen_visual_leader' and not o['vector_evidence']
    else:
        assert o['target_point'] is None and o['leader_path'] is None
        assert o['source_view_id'] is None and o['association_method']=='unresolved'
    assert o['status']=='REVIEW'  # inherited uncertainty always retained


def test_indexed_vector_trace_and_crossing_branch():
    vectors = [(1,{'items':[['l',[10,10],[40,40]],['l',[40,40],[80,50]]]})]
    path,evidence,issue = c.vector_trace([[10,10],[40,40],[80,50]],vectors,2)
    assert path and not issue and len(evidence)==2
    for extra in ([['l',[20,40],[40,20]]],[['l',[40,40],[90,20]]]):
        path,evidence,issue = c.vector_trace([[10,10],[40,40],[80,50]],vectors+[(2,{'items':extra})],2)
        assert path is None and issue


def test_vector_geometry_used_by_actual_pipeline(tmp_path):
    path,index = fixture(tmp_path)
    # Context is [10,10,290,230]; normalized leader maps to these source points.
    with sqlite3.connect(index) as db:
        page = db.execute('SELECT id FROM pages').fetchone()[0]
        db.execute('INSERT INTO vectors VALUES(?,?,?)',(page,0,json.dumps({'items':[['l',[94,37.5],[178,120]]]})))
    o = result(run(path,index,tmp_path,Client(leader_reply)))['occurrences'][0]
    assert o['association_method']=='indexed_vector_leader_with_visual_interpretation'
    assert o['target_point']==[178,120] and o['vector_evidence']


def test_competing_views_cannot_own_same_endpoint(tmp_path):
    path,index = fixture(tmp_path,boxes=[[10,10,290,230],[10,10,290,230]])
    views = c.load_inputs(path)[1]
    out = result(run(path,index,tmp_path,Client(leader_reply),view_ids=[views[0]['id']]))
    o = out['occurrences'][0]
    assert o['target_point'] is None and any('competing' in r for r in o['review_reasons'])


def test_saved_stage3_context_is_not_association(tmp_path):
    path,index = fixture(tmp_path)
    s3 = grouping.run([path],tmp_path/'stage3',index_path=index)['manifests'][0]
    o = result(run(path,index,tmp_path,Client(response(observation('001'))),stage3=s3))['occurrences'][0]
    assert o['detail_ids'] and o['detail_context_only']
    assert o['target_point'] is None and o['status']=='REVIEW'
    assert any(r['stage']==3 for r in o['inherited_review_reasons'])
    saved = json.loads(Path(s3).read_text()); saved['upstream']['manifest_sha256']='wrong'
    Path(s3).write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='do not match'):
        run(path,index,tmp_path,Client(),stage3=s3)


@pytest.mark.parametrize('bad',[
    '{}', 'not json', response(observation(raw=None)),
    response(observation(text_box=[0,0,0,1])), response(observation(leader_state='crossed',association_method='leader')),
    response(observation(owner_view_id='imaginary')), response(observation(native_candidate_ids=['unknown'])),
    '{"coverage":"complete","coverage":"incomplete"}',
    response(observation()).replace('0.1','NaN',1),
    response(observation()).replace('"exact"','"exact", "injected": true',1),
])
def test_invalid_responses_fail_without_empty_success(tmp_path,bad):
    path,index = fixture(tmp_path)
    summary = run(path,index,tmp_path,Client(bad))
    assert summary['errors']==1
    out = result(summary)
    assert out['state']=='FAILED' and not out['complete']


def test_no_detail_skipped_without_source_or_inference(tmp_path):
    path,index = fixture(tmp_path,no_detail=True)
    (tmp_path/'source.pdf').unlink()
    client = Client()
    out = result(run(path,index,tmp_path,client))
    assert out['state']=='SKIPPED' and out['skip_reason']==['No eligible views']
    assert not client.requests and not out['occurrences'] and not out['annotated_views']


def test_no_detail_contradiction_rejected(tmp_path):
    path,index = fixture(tmp_path)
    m = json.loads(path.read_text()); m['page_decision']='NO_DETAIL'; path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='NO_DETAIL conflicts'):
        run(path,index,tmp_path,Client())


def test_partial_failure_preserves_success_and_resume(tmp_path):
    path,index = fixture(tmp_path,boxes=[[10,10,140,230],[160,10,290,230]])
    first = run(path,index,tmp_path,Client(response(observation('001')),ModelError('timeout')))
    assert first['errors']==1 and first['views_completed']==1
    old = {str(p):p.read_bytes() for p in (tmp_path/'stage4').rglob('*') if p.is_file()}
    second = run(path,index,tmp_path,Client(response(observation('002'))))
    assert second['views_reused']==1 and second['views_completed']==1
    assert all(Path(p).read_bytes()==raw for p,raw in old.items())
    assert len(result(second)['occurrences'])==2
    forced = run(path,index,tmp_path,Client(ModelError('fail'),ModelError('fail')),force=True)
    assert forced['errors']==2
    resumed = run(path,index,tmp_path,Client())
    assert resumed['views_reused']==2 and not resumed['errors']


def test_resume_ids_cache_invalidation_and_preservation(tmp_path,monkeypatch):
    path,index = fixture(tmp_path)
    client = Client(response(observation()))
    first = result(run(path,index,tmp_path,client))
    second = result(run(path,index,tmp_path,Client()))
    assert second['model_requests']==0
    assert first['occurrences'][0]['id']==second['occurrences'][0]['id']
    monkeypatch.setattr(c,'PROMPT',c.PROMPT+' Updated instructions.')
    changed = run(path,index,tmp_path,Client(response(observation())))
    assert changed['views_completed']==1
    with sqlite3.connect(index) as db:
        db.execute("UPDATE spans SET data=data WHERE 0")
    # Model identity is part of reuse; unknown checkpoint never reuses across runs.
    for model in (replace(MODEL,identity='checkpoint@new'),replace(MODEL,identity=None),replace(MODEL,max_tokens=4000)):
        summary = c.run(path,tmp_path/'stage4',index_path=index,client=Client(response(observation())),model=model,config=c.Config(context_fraction=0))
        assert summary['views_completed']==1
    artifact = Path(first['annotated_views'][0]['source_path']); artifact.write_bytes(b'broken')
    # The earlier source artifact is not silently overwritten by new exports.
    assert artifact.read_bytes()==b'broken'


def test_incomplete_coverage_is_not_complete_empty_result(tmp_path):
    path,index = fixture(tmp_path)
    out = result(run(path,index,tmp_path,Client(response(coverage='incomplete',uncertainty=['Clipped code cannot be read']))))
    assert out['state']=='PARTIAL' and not out['complete']


def test_source_identity_and_model_limits(tmp_path):
    path,index = fixture(tmp_path)
    client=Client(response(observation()))
    high=c.run(path,tmp_path/'high-budget',index_path=index,model=replace(MODEL,max_requests=100),client=client)
    assert high['model_requests']==1 and client.requests==1
    with pytest.raises(ValueError,match='positive integer'):
        replace(MODEL,max_requests=0)
    (tmp_path/'source.pdf').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Original source'):
        run(path,index,tmp_path,Client())


def test_distinct_crossed_or_shared_visual_paths_remain_review(tmp_path):
    path,index = fixture(tmp_path)
    def reply(ctx):
        owner=ctx['focus_view_id']
        return response(
            observation('001',(.1,.1,.2,.15),owner_view_id=owner,association_method='leader',leader_state='clear',
                        leader_points=[[.2,.125],[.7,.7]],target_point=[.7,.7],association_evidence=['Visible leader']),
            observation('002',(.7,.1,.8,.15),owner_view_id=owner,association_method='leader',leader_state='clear',
                        leader_points=[[.7,.125],[.2,.7]],target_point=[.2,.7],association_evidence=['Visible leader']))
    out=result(run(path,index,tmp_path,Client(reply)))
    assert len(out['occurrences'])==2
    assert all(o['target_point'] is None and any('cross or share' in r for r in o['review_reasons']) for o in out['occurrences'])


def test_explicit_on_component_location(tmp_path):
    path,index = fixture(tmp_path)
    def reply(ctx):
        return response(observation('001',owner_view_id=ctx['focus_view_id'],association_method='explicit_location',
                                    target_point=[.2,.125],association_evidence=['Code is printed directly on the indicated component']))
    o=result(run(path,index,tmp_path,Client(reply)))['occurrences'][0]
    assert o['association_method']=='qwen_explicit_location' and o['leader_path'] is None
    assert o['target_point'] is not None


def test_corrupt_cached_artifact_reprocesses(tmp_path):
    path,index=fixture(tmp_path)
    first=result(run(path,index,tmp_path,Client(response(observation()))))
    Path(first['annotated_views'][0]['source_path']).write_bytes(b'corrupt')
    summary=run(path,index,tmp_path,Client(response(observation())))
    assert summary['views_completed']==1 and summary['views_reused']==0


def test_upstream_and_index_changes_invalidate_cache(tmp_path):
    path,index=fixture(tmp_path,text='001')
    run(path,index,tmp_path,Client(response(observation())))
    saved=json.loads(path.read_text());saved['page_analysis']['review_reasons'].append('new uncertainty')
    path.write_text(json.dumps(saved))
    assert run(path,index,tmp_path,Client(response(observation())))['views_completed']==1
    with sqlite3.connect(index) as db:
        raw=json.loads(db.execute('SELECT data FROM spans').fetchone()[0]);raw['text']='007'
        db.execute('UPDATE spans SET data=?',(json.dumps(raw),))
    assert run(path,index,tmp_path,Client(response(observation())))['views_completed']==1


def test_source_context_extends_clipped_saved_boundary(tmp_path):
    path,index=fixture(tmp_path,boxes=[[50,50,150,150]])
    summary=c.run(path,tmp_path/'stage4',index_path=index,client=Client(response()),model=MODEL,config=c.Config(context_fraction=.1))
    box=result(summary)['annotated_views'][0]['geometry']['source_context_box']
    assert box==[20,20,180,180]
    assert json.loads(path.read_text())['candidates'][0]['box']==[50,50,150,150]


def test_unreadable_code_remains_null(tmp_path):
    path,index=fixture(tmp_path)
    o=result(run(path,index,tmp_path,Client(response(observation(None,reading_state='unreadable',category='uncertain_code')))))['occurrences'][0]
    assert o['raw_code'] is None and o['normalized_code'] is None and o['status']=='REVIEW'


def test_partial_numeric_reading_cannot_silently_replace_native(tmp_path):
    path,index=fixture(tmp_path,text='0012')
    def reply(context):
        native=context['native_candidates'][0]
        return response(observation('001',native['text_box'],native_candidate_ids=[native['id']]))
    o=result(run(path,index,tmp_path,Client(reply)))['occurrences'][0]
    assert o['raw_code'] is None and o['reading_state']=='conflicting'
    assert {r['raw_text'] for r in o['readings']}=={'001','0012'}


def test_truncated_transport_response_fails_and_records_reason(tmp_path):
    path,index=fixture(tmp_path)
    summary=run(path,index,tmp_path,Client(ModelError('Model output incomplete or truncated')))
    out=result(summary)
    assert out['state']=='FAILED' and not out['complete'] and not out['occurrences']
    report=json.loads(Path(out['review_path']).read_text())
    assert report['unresolved_views'][0]['reason']=='model_output_truncated'


def test_remaining_request_budget_does_not_invalidate_success(tmp_path):
    path,index=fixture(tmp_path)
    run(path,index,tmp_path,Client(response(observation())))
    summary=c.run(path,tmp_path/'stage4',index_path=index,client=Client(),
                  model=replace(MODEL,max_requests=3),config=c.Config(context_fraction=0))
    assert summary['views_reused']==1 and summary['model_requests']==0


def test_resume_retains_actual_response_model_provenance(tmp_path):
    path,index=fixture(tmp_path)
    run(path,index,tmp_path,Client(response(observation())))
    client=Client();client.response_models=set()
    out=result(run(path,index,tmp_path,client))
    assert out['actual_response_models']==['fake'] and out['model_requests']==0

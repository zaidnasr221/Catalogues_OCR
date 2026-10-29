"""Deterministic synthetic schema/transport/image contracts; no PDFs or live HTTP."""
from copy import deepcopy
from dataclasses import replace
import csv
import io
from itertools import product
import json
from pathlib import Path
import stat
import urllib.error

from PIL import Image, ImageDraw
import pytest

from as_ocr import callouts as c, components, detection, enrichment, exports, grouping, indexer, pipeline, provenance
from as_ocr.presentation import run as present
from as_ocr.qwen import QwenClient, ModelConfig, ModelError, capture_diagnostics, FAILED_RESPONSE_LIMIT
from test_callouts import observation, response
from test_components import component, field, analysis as component_analysis
from test_detection import Client as Detector, analysis, region

MODEL = ModelConfig('http://mock/v1', 'mock-model', 'synthetic@fixed', retries=0,
                    timeout=180, max_tokens=8192, max_requests=2)
IMAGE = Image.new('RGB', (120, 80), 'white')
MAPPING = {'source_to_pixels': [1,0,0,1,0,0], 'pixels_to_source': [1,0,0,1,0,0],
           'pixel_size': [100,100]}


@pytest.fixture(autouse=True)
def no_pdf_or_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('PDF processing or live HTTP is forbidden in this suite')
    monkeypatch.setattr(c.fitz, 'open', denied)
    monkeypatch.setattr('urllib.request.urlopen', denied)


class Reply(io.BytesIO):
    status = 200
    headers = {'X-Request-Id': 'server-request-42'}


def http_reply(text, *, finish='stop', usage=True):
    value = {'id':'chatcmpl-test', 'model':MODEL.model,
             'choices':[{'finish_reason':finish, 'message':{'content':text}}]}
    if usage:
        value['usage'] = {'prompt_tokens':101,'completion_tokens':23,'total_tokens':124}
    return Reply(json.dumps(value).encode())


def events(folder):
    return [json.loads(path.read_text()) for path in sorted(folder.glob('*.json'))]


@pytest.mark.parametrize('box', [[.1,.1,.3,1.0278699448529411],
    [-.00720714832042957,.1,.3,.2], [.1,.1,1.031140379644745,.2], [.3,.1,.1,.2], [0,0,0,1]])
def test_confirmed_invalid_boxes_stay_rejected(box):
    with pytest.raises(ValueError):c.parse_response(response(observation(text_box=box)))


@pytest.mark.parametrize('changes', [
    dict(association_method='leader',leader_state='clear',leader_points=[[.3,.1],[.4,.2]]),
    dict(association_method='explicit_location',owner_view_id='v1',target_point=[.4,.2]),
    dict(association_method='unresolved',owner_view_id='v1'),
    dict(association_method='leader',leader_state='crossed',owner_view_id='v1',
         target_point=[.4,.2],association_evidence=['Leader'],leader_points=[[.3,.1],[.4,.2]])])
def test_missing_or_ambiguous_association_remains_rejected(changes):
    with pytest.raises(ValueError):c.parse_response(response(observation(**changes)))


@pytest.mark.parametrize('state',['missing','crossed','shared','clipped','uncertain','clear'])
def test_unresolved_readable_codes_are_preserved(state):
    raw=observation('00012/AB',leader_state=state)
    saved=c.parse_response(response(raw)).observations[0].model_dump(mode='json')
    assert saved==raw and saved['owner_view_id'] is saved['target_point'] is None


@pytest.mark.parametrize('method',['leader','explicit_location'])
def test_supported_observation_serialization_is_unchanged(method):
    raw=observation(association_method=method,owner_view_id='v1',target_point=[.5,.5],
        association_evidence=['Visible source evidence'],leader_state='clear' if method=='leader' else 'missing',
        leader_points=[[.3,.1],[.5,.5]])
    # The old local validator allows points on explicit_location; do not tighten
    # accepted semantics merely because the prompt prefers an empty list there.
    assert c.parse_response(response(raw)).observations[0].model_dump(mode='json')==raw


def test_wire_branches_preserve_existing_local_association_semantics():
    # The pre-existing local validator is the compatibility oracle. Representing
    # its conditions on the wire must neither accept new claims nor reject old ones.
    for method,state,owner,target,evidence,points in product(
        ['unresolved','leader','explicit_location'],['clear','missing','crossed'],
        [None,'v1'],[None,[.4,.4]],[[],['Visible label']], [[],[[.2,.2],[.4,.4]]]):
        raw=observation(association_method=method,leader_state=state,owner_view_id=owner,
            target_point=target,association_evidence=evidence,leader_points=points)
        try:
            old=c.Observation.model_validate_json(json.dumps(raw)).model_dump(mode='json')
        except ValueError:
            with pytest.raises(ValueError):c.parse_response(response(raw))
        else:
            assert c.parse_response(response(raw)).observations[0].model_dump(mode='json')==old


def test_wire_schema_encodes_existing_association_and_coordinate_constraints():
    schema=c.Analysis.model_json_schema()
    assert len(schema['properties']['observations']['items']['anyOf'])==3
    definitions=schema['$defs']
    for name in ('UnresolvedObservation','LeaderObservation','ExplicitLocationObservation'):
        properties=definitions[name]['properties']
        assert all(item['minimum']==0 and item['maximum']==1 for item in properties['text_box']['prefixItems'])
        if name=='UnresolvedObservation':
            assert properties['owner_view_id']['type']==properties['target_point']['type']=='null'
        else:
            assert properties['owner_view_id']['type']=='string'
            assert properties['association_evidence']['minItems']==1
            assert all(item['minimum']==0 and item['maximum']==1 for item in properties['target_point']['prefixItems'])
    assert definitions['LeaderObservation']['properties']['leader_points']['minItems']==2
    assert definitions['LeaderObservation']['properties']['leader_state']['const']=='clear'
    assert definitions['ExplicitLocationObservation']['properties']['leader_state']['const']=='missing'


@pytest.mark.parametrize('method',['leader','explicit_location'])
@pytest.mark.parametrize('missing',['owner_view_id','target_point','association_evidence'])
def test_association_prerequisites_cannot_be_missing(method,missing):
    raw=observation(association_method=method,owner_view_id='v1',target_point=[.5,.5],
        association_evidence=['Visible source evidence'],leader_state='clear' if method=='leader' else 'missing',
        leader_points=[[.3,.1],[.5,.5]])
    raw[missing]=[] if missing=='association_evidence' else None
    with pytest.raises(ValueError):c.parse_response(response(raw))


def test_context_intersections_preserve_source_evidence_and_repeated_labels():
    candidates=[dict(id='span-8',raw_text='B',text_box=[10,95,20,102.78699448529411]),
                dict(id='span-9',raw_text='B',text_box=[30,95,40,102.78699448529411])]
    views=[dict(id='v1',kind='assembly_candidate',structural_box=[-2,-3,80,120])]
    original=deepcopy((candidates,views))
    context=c.request_context(views[0],views,candidates,MAPPING)
    assert context['native_candidates'][0]['text_box']==[.1,.95,.2,1.0]
    assert context['native_candidates'][0]['source_box']==candidates[0]['text_box']
    assert len(context['native_candidates'])==2
    assert context['views'][0]['structural_box']==[0,0,.8,1]
    assert context['views'][0]['clipped']
    assert (candidates,views)==original
    assert c.visible_context_box([101,10,110,20],MAPPING) is None


def test_stage5_clipped_context_never_changes_physical_membership():
    spans=[(1,{'text':'244431','bbox':[-8,20,3,40]})]
    occurrence=dict(id='o1',raw_code='244431',text_box=[-8,20,3,40],observed_in_view_ids=['v1'],
        target_point=[110,50],source_view_id='v1',association_method='leader',review_reasons=[])
    original=deepcopy((spans,occurrence))
    context=components.request_context({'id':'v1'},spans,[occurrence],MAPPING)
    assert context['native_spans'][0]['box']==[0,.2,.03,.4]
    label=context['occurrences'][0]
    assert label['target_point'] is None and label['target_visible'] is False
    assert label['source_target_point']==[110,50] and label['source_box']==[-8,20,3,40]
    assert (spans,occurrence)==original


def test_assembly_crop_retains_intersecting_native_code_without_claiming_ownership():
    bounds=[0,0,842,595]
    assembly=dict(model_id='assembly',kind='assembly_candidate',box=[436.25,490,700,592],
                  parent_ref=None,annotations=[],review_reasons=[])
    neighboring=dict(model_id='notes',kind='notes',box=[350,490,440,592],parent_ref=None)
    label=[419.15,577.61,453.97,591.50]
    audit=[dict(ordinal=13,box=label,owner_model_id=None,
                competing_model_ids=['assembly','notes'])]
    crop,reasons=detection.crop_context(assembly,[assembly,neighboring],bounds,8.42,audit)
    assert crop[0]<=label[0]-2 and crop[2]>=label[2]+2
    assert assembly['box']==[436.25,490,700,592]
    assert audit[0]['owner_model_id'] is None
    assert any('ownership remains unresolved' in reason for reason in reasons)


def test_metadata_competitor_does_not_expand_assembly_crop():
    assembly=dict(model_id='assembly',kind='assembly_candidate',box=[100,100,200,200],
                  parent_ref=None,annotations=[],review_reasons=[])
    metadata=dict(model_id='title',kind='document_metadata',box=[190,100,230,200],parent_ref=None)
    audit=[dict(ordinal=1,box=[190,150,215,160],owner_model_id=None,
                competing_model_ids=['assembly','title'])]
    crop,reasons=detection.crop_context(assembly,[assembly,metadata],[0,0,300,300],2,audit)
    assert crop[2]<=202 and not reasons


def test_native_control_symbols_flagged_without_guessing_the_glyph():
    spans=[{'text':'\u0094','bbox':[1,1,2,2]}, {'text':'\u0095','bbox':[2,1,3,2]},
           {'text':'244501','bbox':[3,1,8,2]}]
    assert indexer.flag_native_control_spans(spans)==2
    assert [span['text'] for span in spans]==['\u0094','\u0095','244501']
    assert all(span['review_reasons'] for span in spans[:2])
    assert 'review_reasons' not in spans[2]


def test_metadata_code_rejected_while_leader_review_retains_exact_reading():
    code=dict(raw_code='298257',category='product_code',reading_state='exact',
              readings=[],review_reasons=['Leader association remains uncertain'])
    assert provenance.supported_product_reading(code,['assembly_candidate'])
    assert not provenance.supported_product_reading(code,['document_metadata'])
    assert not provenance.supported_product_reading(dict(code,review_reasons=['Uncertain classification']))
    field=dict(field='code',accepted=True,normalized='298257',review_reasons=[])
    components.reject_metadata_codes({'kind':'document_metadata'},[field])
    assert field['accepted'] is False and field['normalized'] is None
    assert 'metadata' in field['review_reasons'][0].lower()


def test_success_diagnostics_include_usage_ids_status_without_raw_text(tmp_path,monkeypatch):
    captured=[]
    def fake(request,**kwargs):
        captured.append(request)
        return http_reply(response(observation()))
    monkeypatch.setattr('urllib.request.urlopen',fake)
    client=QwenClient(MODEL)
    with capture_diagnostics(client,tmp_path/'diag',stage=4,subject_id='v1'):
        raw=client.request_json(IMAGE,'PROMPT MUST NOT BE LOGGED',response_schema=c.Analysis.model_json_schema())
        c.parse_response(raw)
    event=events(tmp_path/'diag')[0]
    assert event['request_id']==captured[0].get_header('X-request-id')
    assert event['server_request_id']=='server-request-42' and event['response_id']=='chatcmpl-test'
    assert event['http_status']==200 and event['finish_reason']=='stop'
    assert event['usage']=={'prompt_tokens':101,'completion_tokens':23,'total_tokens':124}
    assert event['duration_seconds']>=0 and event['request_bytes']>event['png_bytes']>0
    assert event['image_size']==[120,80] and event['local_validation_status']=='PASSED'
    assert not list((tmp_path/'diag').glob('*.txt'))
    assert 'PROMPT MUST NOT BE LOGGED' not in json.dumps(event)
    assert client.requests==1


@pytest.mark.parametrize('raw,kind', [('{','JSONDecodeError'),
    (response(observation(text_box=[.1,.1,.3,1.0278699448529411])),'ValidationError'),
    (response(observation(association_method='leader',leader_state='clear',leader_points=[[.3,.1],[.4,.2]])),'ValidationError')])
def test_failed_generated_response_is_preserved_before_parsing(tmp_path,monkeypatch,raw,kind):
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:http_reply(raw))
    client=QwenClient(MODEL)
    with pytest.raises(ValueError),capture_diagnostics(client,tmp_path,stage=4,subject_id='D1'):
        c.parse_response(client.request_json(IMAGE,'not logged'))
    event=events(tmp_path)[0]
    assert event['error_type']==kind and event['response_status']=='local_validation_failed'
    path=Path(event['failed_response_path'])
    assert path.read_text()==raw and stat.S_IMODE(path.stat().st_mode)==0o600
    assert event['finish_reason']=='stop' and event['local_validation_status']=='FAILED'
    if kind=='ValidationError':assert event['validation_issues']


@pytest.mark.parametrize('phase',['awaiting_headers','reading_body','wrapped_timeout'])
def test_timeouts_are_classified_once_without_fabricated_response(tmp_path,monkeypatch,phase):
    calls=[]
    class SlowReply(Reply):
        def read(self,*args):raise TimeoutError('secret details must not be logged')
    def fake(*args,**kwargs):
        calls.append(1)
        if phase=='reading_body':return SlowReply()
        if phase=='wrapped_timeout':raise urllib.error.URLError(TimeoutError('private'))
        raise TimeoutError('private')
    monkeypatch.setattr('urllib.request.urlopen',fake)
    client=QwenClient(MODEL)
    with pytest.raises(ModelError),capture_diagnostics(client,tmp_path,stage=5,subject_id='D'):
        client.request_json(IMAGE,'private prompt')
    event=events(tmp_path)[0]
    expected='reading_body_timeout' if phase=='reading_body' else 'awaiting_headers_timeout'
    assert event['timeout_classification']==expected
    assert event['usage'] is None and event['local_validation_status']=='NOT_RUN'
    assert len(calls)==client.requests==1 and not list(tmp_path.glob('*.txt'))
    assert 'private' not in json.dumps(event) and 'secret details' not in json.dumps(event)


def test_truncation_and_bounded_failure_logging(tmp_path,monkeypatch):
    content='x'*(FAILED_RESPONSE_LIMIT+123)
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:http_reply(content,finish='length'))
    client=QwenClient(MODEL)
    with pytest.raises(ModelError,match='incomplete'),capture_diagnostics(client,tmp_path,stage=4,subject_id='v'):
        client.request_json(IMAGE,'prompt')
    event=events(tmp_path)[0]
    assert event['response_status']=='truncated' and event['finish_reason']=='length'
    assert event['failed_response_truncated'] and event['failed_response_logged_bytes']==FAILED_RESPONSE_LIMIT
    assert Path(event['failed_response_path']).stat().st_size==FAILED_RESPONSE_LIMIT


def test_complete_json_with_truncated_finish_reason_is_not_salvaged(tmp_path,monkeypatch):
    raw=response(observation())
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:http_reply(raw,finish='length'))
    client=QwenClient(MODEL)
    with pytest.raises(ModelError,match='incomplete'),capture_diagnostics(client,tmp_path,stage=4,subject_id='v'):
        client.request_json(IMAGE,'prompt')
    assert events(tmp_path)[0]['local_validation_status']=='NOT_RUN'
    assert Path(events(tmp_path)[0]['failed_response_path']).read_text()==raw
    assert client.requests==1


def test_failed_unicode_text_is_bounded_and_readable(tmp_path,monkeypatch):
    raw='x'+'部'*(FAILED_RESPONSE_LIMIT//3+1)
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:http_reply(raw))
    client=QwenClient(MODEL)
    with pytest.raises(ValueError),capture_diagnostics(client,tmp_path,stage=4,subject_id='v'):
        c.parse_response(client.request_json(IMAGE,'prompt'))
    event=events(tmp_path)[0]
    saved=Path(event['failed_response_path']).read_text()
    assert raw.startswith(saved) and len(saved.encode())==event['failed_response_logged_bytes']<=FAILED_RESPONSE_LIMIT
    assert event['failed_response_truncated']


@pytest.mark.parametrize('bad',[False,True])
def test_diagnostic_io_error_preserves_original_outcome(tmp_path,monkeypatch,bad):
    raw='{' if bad else response(observation())
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:http_reply(raw))
    client=QwenClient(MODEL)
    blocked=tmp_path/'not-a-directory';blocked.write_text('keep')
    def run():
        with capture_diagnostics(client,blocked/'diagnostics',stage=4,subject_id='v'):
            return c.parse_response(client.request_json(IMAGE,'prompt'))
    with pytest.warns(RuntimeWarning,match='could not be written'):
        if bad:
            with pytest.raises(json.JSONDecodeError):run()
        else:
            assert run().observations[0].raw_text=='001.20 - A'
    assert client.request_diagnostics[0]['diagnostics_write_error'] in {'NotADirectoryError','FileExistsError'}
    assert blocked.read_text()=='keep'


def test_failed_logging_redacts_credentials_and_keeps_existing_files(tmp_path,monkeypatch):
    key='sensitive-api-key'
    def fake(*a,**k):
        reply=http_reply('bad JSON '+key,usage=False)
        reply.headers={'X-Request-Id':key}
        return reply
    monkeypatch.setattr('urllib.request.urlopen',fake)
    (tmp_path/'existing.txt').write_text('keep')
    client=QwenClient(replace(MODEL,api_key=key))
    with pytest.raises(ValueError),capture_diagnostics(client,tmp_path,stage=5,subject_id='v'):
        components.parse_response(client.request_json(IMAGE,'prompt'))
    assert (tmp_path/'existing.txt').read_text()=='keep'
    assert key not in ''.join(path.read_text() for path in tmp_path.iterdir())
    assert events(tmp_path)[0]['usage'] is None


def test_http_error_body_is_never_logged(tmp_path,monkeypatch):
    def fake(*a,**k):
        raise urllib.error.HTTPError('http://private',504,'private',{'X-Request-Id':'server-timeout'},
            io.BytesIO(b'private credentials and payload'))
    monkeypatch.setattr('urllib.request.urlopen',fake)
    client=QwenClient(MODEL)
    with pytest.raises(ModelError),capture_diagnostics(client,tmp_path,stage=4,subject_id='v'):
        client.request_json(IMAGE,'prompt')
    event=events(tmp_path)[0]
    assert event['http_status']==504 and event['timeout_classification']=='http_timeout_status'
    assert event['server_request_id']=='server-timeout'
    assert not list(tmp_path.glob('*.txt')) and 'private' not in json.dumps(event)


def test_pipeline_timed_wrapper_keeps_diagnostics_and_counts(tmp_path,monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:http_reply(response(observation())))
    stats=pipeline.Stats();client=pipeline.TimedClient(QwenClient(MODEL),stats,directory=tmp_path/'transport',stage=4)
    with capture_diagnostics(client,tmp_path/'view',stage=4,subject_id='v'):
        c.parse_response(client.request_json(IMAGE,'prompt'))
    assert stats.requests==1 and stats.transport_failures==0
    assert len(events(tmp_path/'view'))==1 and not (tmp_path/'transport').exists()
    client.request_json(IMAGE,'prompt')
    assert events(tmp_path/'transport')[0]['local_validation_status']=='NOT_RUN'


def test_stage1_to7_and_public_csv_on_synthetic_image_only(tmp_path,monkeypatch):
    source=tmp_path/'synthetic.png'
    image=Image.new('RGB',(300,240),'white');draw=ImageDraw.Draw(image)
    draw.text((30,30),'001.20 - A\nOuter frame',fill='black');image.save(source)
    index_root=tmp_path/'stage1';indexer.run(source,index_root);db=index_root/'index.sqlite3'
    detected=detection.run(source,tmp_path/'stage2',detection.Config(256,512,400,verify_coverage=False),MODEL,db,
        client=Detector([analysis([region('detail',(.05,.05,.95,.95),kind='assembly_candidate')])],MODEL))
    s2=detected['manifests'][0]
    s3=grouping.run([s2],tmp_path/'stage3',index_path=db)['manifests'][0]
    def fake(request,**kwargs):
        body=json.loads(request.data);context=json.loads(body['messages'][1]['content'][0]['text'].split('\n',1)[1])
        schema=body['response_format']['json_schema']['schema']
        if 'observations' in schema['properties']:
            text=response(observation())
        else:
            text=component_analysis(component(context,[field('description','Outer frame')]))
        return http_reply(text)
    monkeypatch.setattr('urllib.request.urlopen',fake)
    s4=c.run(s2,tmp_path/'stage4',index_path=db,stage3=s3,model=MODEL)['manifest']
    config=replace(MODEL,max_requests=6,max_tokens=4096)
    s5=components.run(s3,s4,tmp_path/'stage5',index_path=db,model=config)['manifest']
    s6=enrichment.run([s5],tmp_path/'stage6',index_path=db,model=replace(config,max_requests=1))['manifest']
    s7=exports.run(s6,tmp_path/'stage7',index_path=db)
    publications=present([s7['artifacts']['summary']],output=tmp_path/'output',internal_root=tmp_path/'out')
    assert publications[0]['row_count']==1  # One selected Detail; unresolved callouts remain internal.
    with Path(publications[0]['csv_path']).open(encoding='utf-8-sig',newline='') as handle:
        public=list(csv.DictReader(handle))
    assert public[0]['detail_name']=='synthetic_01' and not public[0]['profile_id']
    assert list(public[0])==['detail_name','profile_count','profile_id','component_name','source_page']
    assert public[0]['profile_count']=='0'  # Source examined; no supported leader was extracted.
    trace=[json.loads(line) for line in Path(publications[0]['trace']).read_text().splitlines()][0]
    assert trace['association_status']=='NO_CONFIRMED_COMPONENTS' and trace['review_status']=='REVIEW'
    assert len(exports.TABLES)==8
    assert json.loads(Path(s4).read_text())['occurrences'][0]['raw_code']=='001.20 - A'
    assert json.loads(Path(s4).read_text())['occurrences'][0]['target_point'] is None
    assert events(next((tmp_path/'stage4/callouts').iterdir())/'diagnostics')[0]['local_validation_status']=='PASSED'
    assert events(next((tmp_path/'stage5/components').iterdir())/'diagnostics')[0]['local_validation_status']=='PASSED'

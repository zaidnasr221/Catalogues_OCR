"""CPU regression fixtures draw known physical lines; no model accuracy claims."""
from copy import deepcopy
from pathlib import Path
import csv
import json

from PIL import Image,ImageDraw
import pytest

from as_ocr import crop_evidence,leaders,line_evidence,public_contract,presentation,delivery,components,pipeline,exports

MAPPING=dict(source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0],pixel_size=[100,100],source_context_box=[0,0,100,100])
PATHS=[[[30,20],[60,45]],[[30,70],[60,60]]]


def fixture(tmp_path):
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    for path in PATHS:draw.line([tuple(p) for p in path],fill='black',width=1)
    draw.rectangle((55,35,85,70),outline='black')
    source=tmp_path/'original.png';image.save(source)
    detail=dict(id='d1',document_id='f'*64,page_id='p7',page_index=6,view_ids=['v1'],membership_role='primary')
    view=dict(id='v1',document_id=detail['document_id'],page_id='p7',page_index=6,kind='assembly_candidate',
        structural_box=[55,35,85,70],box=[10,10,95,90],crop_path=str(source),crop_geometry=MAPPING,
        crop_completeness=dict(status='COMPLETE',annotation_proofs=[dict(ownership_verified=True,
            annotation=dict(leader_points=path)) for path in PATHS]))
    records=[]
    for pass_name in ['primary','independent_audit']:
        for n,path in enumerate(PATHS):
            records.append(dict(leader_audit_pass=pass_name,leader_kind='component',raw_code='EXACT-001',category='product_code',
                leader_path=path,text_box=[10,path[0][1]-5,30,path[0][1]+5],target_point=path[-1],source_view_id='v1',
                association_evidence=['Visible leader from source label to this component'],
                proposed_geometry={'leader_state':'clear'},association_method='indexed_vector_leader_with_visual_interpretation'))
    result=dict(view_id='v1',leader_coverage='complete',leader_coverage_evidence=['Both external labels and lines examined'],
        leader_audit={'status':'COMPLETE'},source_image=str(source),mapping=MAPPING,records=records)
    vectors=[(n,dict(items=[['l',*path]])) for n,path in enumerate(PATHS)]
    return detail,view,result,vectors,image


def region():
    annotations=[]
    for path in PATHS:
        label=[10,path[0][1]-5,30,path[0][1]+5];target=[59,path[-1][1]-1,61,path[-1][1]+1]
        annotations.append(dict(box=[10,min(label[1],target[1]),61,max(label[3],target[3])],label_box=label,
            target_box=target,leader_points=path,association='leader',evidence='Actual source line'))
    return dict(model_id='a',box=[48,30,90,78],structural_box=[55,35,85,70],kind='assembly_candidate',
        evidence=[dict(context_annotations=annotations)],review_reasons=[])


def test_separate_same_id_leaders_are_counted_and_numbered(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);inventory=leaders.inventory(d,[v],[r],vec)
    assert leaders.count_state(inventory)==('2','COMPLETE')
    assert [o['number'] for o in inventory['observations'] if o['accepted']]==[1,2]
    leaders.write(tmp_path/'review',inventory,d,[v])
    assert all((tmp_path/'review'/name).is_file() for name in
        ['component_leaders_overlay.png','leader_observations.json','profile_count_review.md'])


def test_dimension_does_not_increase_count(tmp_path):
    d,v,r,vec,im=fixture(tmp_path)
    r['records'].append(dict(r['records'][0],leader_kind='dimension',category='dimension',leader_path=[[10,90],[90,90]]))
    result=leaders.inventory(d,[v],[r],vec)
    assert leaders.count_state(result)==('2','COMPLETE')
    assert result['observations'][-1]['status']=='EXCLUDED'


def test_component_dimension_classification_conflict_withholds_total(tmp_path):
    d,v,r,vec,im=fixture(tmp_path)
    r['records'][0]['category']='dimension'
    result=leaders.inventory(d,[v],[r],vec)
    assert leaders.count_state(result)[1]=='OBSERVED_PARTIAL'
    assert any(o['status']=='REVIEW' for o in result['observations'])


def test_agreeing_callout_passes_cannot_omit_a_known_detection_leader(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);r['records']=[r['records'][0],r['records'][2]]
    result=leaders.inventory(d,[v],[r],vec)
    assert result['observed_accepted_leaders']==1 and leaders.count_state(result)==('1','OBSERVED_PARTIAL')
    assert any('Detection component leader' in issue for issue in result['issues'])


def test_both_passes_misclassifying_known_component_as_dimension_withholds_count(tmp_path):
    d,v,r,vec,im=fixture(tmp_path)
    for i in [1,3]:r['records'][i].update(category='dimension',leader_kind='dimension')
    result=leaders.inventory(d,[v],[r],vec)
    assert leaders.count_state(result)[1]=='OBSERVED_PARTIAL' and not result['complete']


def test_saved_inventory_cannot_drop_a_detection_leader(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);record=leaders.inventory(d,[v],[r],vec)
    record['observations']=record['observations'][:1];record['verified_count']=1
    review=exports.Reviews()
    assert leaders.validate_saved([record],{'d1':d},{'v1':v},review)=={} and review.errors()


def test_duplicate_physical_line_not_double_counted(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);r['records'].append(deepcopy(r['records'][0]))
    result=leaders.inventory(d,[v],[r],vec)
    assert leaders.count_state(result)[0]=='2'
    assert len(result['observations'][0]['duplicate_detections'])==2


@pytest.mark.parametrize('color',['cyan','yellow'])
def test_colored_original_leader_pixels_are_not_lost_in_grayscale(tmp_path,color):
    d,v,r,vec,im=fixture(tmp_path)
    im=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(im)
    for path in PATHS:draw.line([tuple(p) for p in path],fill=color)
    im.save(r['source_image'])
    assert leaders.count_state(leaders.inventory(d,[v],[r],[]))[0]=='2'


@pytest.mark.parametrize('mutation',['incomplete','ambiguous','no_audit','unsupported_pixels','crop_review'])
def test_partial_evidence_withholds_public_count(tmp_path,mutation):
    d,v,r,vec,im=fixture(tmp_path)
    if mutation=='incomplete':r['leader_coverage']='incomplete'
    if mutation=='ambiguous':r['records'][0]['proposed_geometry']['leader_state']='crossed'
    if mutation=='no_audit':r['records']=r['records'][:2]
    if mutation=='unsupported_pixels':
        vec=[];Image.new('RGB',(100,100),'white').save(r['source_image'])
    if mutation=='crop_review':v['crop_completeness']['status']='REVIEW'
    result=leaders.inventory(d,[v],[r],vec)
    assert leaders.count_state(result)[1]=='OBSERVED_PARTIAL' and result['verified_count'] is None


def test_no_legacy_structural_count_fallback():
    assert leaders.count_state(dict(observed_count=12,profiles=[{}]*12,complete=True))[0]==''


def test_native_external_label_and_entire_leader_expand_context(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);candidate=region()
    review=crop_evidence.verify(candidate,[candidate],[0,0,100,100],[],vec,im,MAPPING)
    assert review['status']=='COMPLETE'
    assert review['corrected_context_box'][0]==10
    assert review['corrected_context_box'][1]==15
    assert review['initial_context_box']==[48,30,90,78]
    crop_evidence.write(tmp_path/'crop',im,MAPPING,candidate,review,im)
    assert (tmp_path/'crop/context_bbox_overlay.png').is_file()


def test_multiline_boundary_label_retained_without_guessed_ownership(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);candidate=region()
    spans=[(1,dict(text='FIRST LINE',bbox=[5,21,23,27])),(2,dict(text='FINAL-012345',bbox=[5,28,23,34]))]
    review=crop_evidence.verify(candidate,[candidate],[0,0,100,100],spans,vec,im,MAPPING)
    assert review['corrected_context_box'][0]==5
    assert review['status']=='REVIEW'
    assert any('ownership' in i for i in review['issues'])


def test_unenumerated_wholly_external_native_label_is_preserved(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);candidate=region();candidate['evidence']=[]
    review=crop_evidence.verify(candidate,[candidate],[0,0,100,100],[(1,dict(text='EXTERNAL',bbox=[9,15,30,25]))],vec,im,MAPPING)
    assert review['corrected_context_box'][0]<=9 and review['status']=='REVIEW'


def test_external_label_on_native_elbow_chain_is_preserved(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);candidate=region();candidate['evidence']=[]
    vec=[(0,dict(items=[['l',[30,20],[45,20]],['l',[45,20],[60,45]]]))]
    review=crop_evidence.verify(candidate,[candidate],[0,0,100,100],[(1,dict(text='OUTSIDE',bbox=[8,15,30,25]))],vec,im,MAPPING)
    assert review['corrected_context_box'][0]<=8 and review['status']=='REVIEW'


def test_adjacent_assemblies_competing_endpoint_never_accept(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);candidate=region()
    adjacent=dict(candidate,model_id='b',structural_box=[58,40,95,80])
    review=crop_evidence.verify(candidate,[candidate,adjacent],[0,0,100,100],[],vec,im,MAPPING)
    assert review['status']=='REVIEW'
    assert any('adjacent' in i for i in review['issues'])


def test_noninverse_crop_transform_is_review(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);candidate=region();mapping=dict(MAPPING,pixels_to_source=[1,0,0,1,5,0])
    assert crop_evidence.verify(candidate,[candidate],[0,0,100,100],[],vec,im,mapping)['status']=='REVIEW'


def test_unclear_name_and_cross_page_enrichment_remain_blank(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);o=dict(id='o',document_id=d['document_id'],page_id='p7',category='product_code',raw_code='EXACT-001')
    c=dict(id='c',document_id=d['document_id'],page_id='p6',occurrence_ids=['o'])
    assert public_contract.literal_name(o,[c],[dict(candidate_id='c',field='component_name',value_raw='Mullion',accepted=True)],v)==('',[])
    o.update(category='component_name',reading_state='unreadable',review_reasons=['Illegible'])
    assert public_contract.literal_name(o,[],[],v)==('',[])


def test_five_column_csv_roundtrip_blank_count_original_page(tmp_path):
    row=dict(zip(public_contract.COLUMNS,['sample_01','','EXACT-001','','7']))
    path=tmp_path/'detail.csv';delivery.write_csv(path,[row,row])
    assert delivery.read_csv(path)==[row,row]
    with path.open(encoding='utf-8-sig') as f:assert next(csv.reader(f))==list(public_contract.COLUMNS)
    assert 'profile_type' not in row and 'description' not in row


def test_disabled_old_inference_cannot_be_reenabled():
    with pytest.raises(ValueError,match='disabled'):components.Config(observed_profiles=True).validate()
    with pytest.raises(ValueError,match='disabled'):components.Config(paragraph_descriptions=True).validate()
    assert components.response_schema(True)['$defs']['Property']['properties']['field']['enum']==['component_name']


def test_changed_source_pixels_block_saved_count(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);inventory=leaders.inventory(d,[v],[r],vec)
    Image.new('RGB',(100,100),'white').save(r['source_image']);review=exports.Reviews()
    assert leaders.validate_saved([inventory],{'d1':d},{'v1':v},review)=={}
    assert review.errors()


def test_original_alupco_outlined_text_cannot_be_silently_complete():
    """Actual source vectors and rotation, not mocked source pixels or OCR."""
    import pymupdf as fitz
    import hashlib
    from as_ocr import detection
    saved=json.loads((Path(__file__).parent/'fixtures/alupco-page1-context.json').read_text())
    source=Path.cwd()/saved['source']
    if not source.is_file():pytest.skip('Original ALUPCO fixture PDF is not in this checkout')
    assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==saved['sha256']
    with fitz.open(source) as pdf:
        page=pdf[0];assert not page.get_text().strip()
        geometry=dict(source_to_display=list(page.rotation_matrix),source_rect=list(page.rect*page.derotation_matrix))
        vectors=list(enumerate(page.get_drawings(extended=True)))
    image,mapping=detection.render(source,0,geometry,geometry['source_rect'],1600)
    candidates=[r for r in saved['regions'] if r['kind']=='assembly_candidate']
    reviews=[crop_evidence.verify(r,saved['regions'],geometry['source_rect'],[],vectors,image,mapping) for r in candidates]
    assert len(reviews)==2 and all(r['status']=='REVIEW' for r in reviews)
    assert all(r['annotation_proofs'] and any(not p['ownership_verified'] for p in r['annotation_proofs']) for r in reviews)
    assert any(r['corrections'] for r in reviews)  # Source extents expose the crop gap.


def test_foreign_page_cannot_be_inserted_by_editing_public_table(tmp_path):
    from test_presentation import fixture,table
    summary=fixture(tmp_path);path=Path(json.loads(summary.read_text())['artifacts']['details'])
    with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    table(path,[dict(r,page_index='6') for r in rows])
    with pytest.raises(ValueError,match='source page'):presentation.project(summary)


def test_metadata_text_inside_an_assembly_crop_does_not_become_a_component_code(tmp_path):
    from test_presentation import fixture,update_callout,save
    import hashlib
    summary=fixture(tmp_path,code='HEADER-01');art=json.loads(summary.read_text())['artifacts']
    link=Path(art['enrichment']);s6=json.loads(link.read_text());source=s6['sources'][0]
    path=Path(source['stage3_path']);s3=json.loads(path.read_text());v=s3['views'][0]
    s3['views'].append(dict(v,id='metadata',kind='document_metadata',box=[0,0,20,20]))
    save(path,s3);source['stage3_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();save(link,s6)
    update_callout(summary,lambda data:data['occurrences'][0].update(text_box=[1,1,10,5]))
    assert all(not r['profile_id'] for r in presentation.project(summary)[3])


def test_cancellation_during_census_preserves_first_reading_without_completed_checkpoint(tmp_path):
    from as_ocr import callouts,grouping
    from test_callouts import fixture,Client,MODEL,observation,response
    s2,index=fixture(tmp_path);s3=grouping.run([s2],tmp_path/'grouping',index_path=index)['manifests'][0]
    first=json.loads(response(observation()));first.update(leader_coverage='complete',leader_coverage_evidence=['Initial census'])
    class CancelClient(Client):
        def request_json(self,*args,**kwargs):
            if self.requests:raise KeyboardInterrupt('cancelled census')
            return super().request_json(*args,**kwargs)
    with pytest.raises(KeyboardInterrupt):
        callouts.run(s2,tmp_path/'callouts',index_path=index,stage3=s3,model=MODEL,client=CancelClient(json.dumps(first)))
    drafts=list((tmp_path/'callouts').rglob('*-leader-draft.json'))
    assert len(drafts)==1
    draft=json.loads(drafts[0].read_text())
    assert draft['state']=='PARTIAL' and not draft['complete'] and draft['records'][0]['raw_code']=='001.20 - A'
    assert Path(draft['source_image']).is_file() and not list((tmp_path/'callouts').rglob('manifest.json'))
    import sqlite3
    with sqlite3.connect(index) as db:
        assert db.execute("SELECT count(*) FROM callout_view_attempts WHERE state='complete'").fetchone()[0]==0


def test_model_numeric_total_cannot_override_observed_lines(tmp_path):
    d,v,r,vec,im=fixture(tmp_path);r['profile_count']=999
    inventory=leaders.inventory(d,[v],[r],vec)
    assert leaders.count_state(inventory)[0]=='2'
    inventory['verified_count']=999
    assert leaders.count_state(inventory)[0]==''


@pytest.mark.parametrize('use_legend',[False,True])
def test_all_seven_stages_publish_actual_source_leaders_and_five_columns(tmp_path,use_legend):
    """Real PDF renders, actual stage APIs and saved handoffs; model is mocked."""
    import pymupdf as fitz
    from as_ocr import indexer,detection,grouping,callouts,enrichment
    from as_ocr.qwen import ModelConfig,ModelError
    from test_detection import analysis as detection_response,region as detection_region
    source=tmp_path/'drawing.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=100,height=100)
        page.draw_rect((55,35,85,70));page.draw_rect((60,40,80,65))
        for path in PATHS:
            page.draw_line(*path);page.insert_text((10,path[0][1]),'1' if use_legend else 'EXACT-001',fontsize=3)
        if use_legend:page.insert_text((10,82),'1 - EXACT-001 : Literal connector',fontsize=3)
        pdf.save(source)
    indexer.run(source,tmp_path/'stage1');db=tmp_path/'stage1/index.sqlite3'
    model=ModelConfig('http://mock/v1','synthetic','test@immutable',retries=0,max_requests=50)
    class Client:
        def __init__(self,fn):self.fn=fn;self.requests=0;self.response_models={'synthetic'};self.config=model
        def request_json(self,image,prompt,context_text='',**kw):
            self.requests+=1
            if self.requests>model.max_requests:raise ModelError('Budget exhausted')
            return json.dumps(self.fn(json.loads(context_text) if context_text else {}))
    annotation=region()['evidence'][0]['context_annotations']
    for a in annotation:
        for key in ['box','label_box','target_box']:a[key]=[x/100 for x in a[key]]
        a['leader_points']=[[x/100,y/100] for x,y in a['leader_points']]
    detected=detection.run(source,tmp_path/'stage2',detection.Config(256,512,400,context_margin=0,verify_coverage=False),model,db,
        client=Client(lambda ctx:detection_response([detection_region('detail',(.05,.1,.95,.9),structural_box=[.55,.35,.85,.7],context_annotations=annotation)])))
    s2=detected['manifests'][0]
    s3=grouping.run([s2],tmp_path/'stage3',index_path=db)['manifests'][0]
    def callout_reply(ctx):
        def point(p):return [(p[0]-5)/90,(p[1]-10)/80]
        observations=[]
        for path in PATHS:
            b=[10,path[0][1]-5,30,path[0][1]+5]
            observations.append(dict(raw_text='1' if use_legend else 'EXACT-001',category='item_marker' if use_legend else 'product_code',reading_state='exact',leader_kind='component',
                text_box=[*point(b[:2]),*point(b[2:])],native_candidate_ids=[],reading_evidence=['Exact source code'],
                owner_view_id=ctx['focus_view_id'],association_method='leader',leader_state='clear',
                leader_points=[point(p) for p in path],target_point=point(path[-1]),association_evidence=['Source line to component'],uncertainty=[]))
            if use_legend:
                observations[-1]['legend_link']=dict(marker='1',code='EXACT-001',name='Literal connector',
                    row_text='1 - EXACT-001 : Literal connector',row_box=[*point([5,77]),*point([85,85])],
                    code_box=[*point([12,78]),*point([32,83])],name_box=[*point([33,78]),*point([82,83])],
                    scope_text='Printed assembly key',scope_box=[0,0,1,1],evidence=['Same joint key explicitly maps marker 1'])
        return dict(coverage='complete',evidence=['Whole context examined'],uncertainty=[],observations=observations,
            leader_coverage='complete',leader_coverage_evidence=['Both original-source component leaders examined'])
    s4=callouts.run(s2,tmp_path/'stage4',index_path=db,stage3=s3,model=model,client=Client(callout_reply),
        config=callouts.Config(details_only=True,context_fraction=0))['manifest']
    saved=json.loads(Path(s4).read_text())
    assert leaders.count_state(saved['leader_observations'][0])==('2','COMPLETE')
    s5=components.run(s3,s4,tmp_path/'stage5',index_path=db,model=model,
        config=components.Config(details_only=True),client=Client(lambda ctx:dict(coverage='complete',evidence=['No explicit names'],uncertainty=[],scope=[],components=[])))['manifest']
    s6=enrichment.run([s5],tmp_path/'stage6',index_path=db,model=model)['manifest']
    s7=exports.run(s6,tmp_path/'stage7',index_path=db)
    published=delivery.publish(s7['artifacts']['summary'],output=tmp_path/'public');root=Path(published['root'])
    rows=delivery.read_csv(root/'all_details.csv')
    assert len(rows)==2 and rows[0]==rows[1]
    assert rows[0]['profile_count']=='2' and rows[0]['profile_id']=='EXACT-001'
    assert rows[0]['component_name']==('Literal connector' if use_legend else '') and rows[0]['source_page']=='1'
    folder=next((root/'details').iterdir())
    assert all((folder/name).is_file() for name in ['image.png','detail.csv','component_leaders_overlay.png','leader_observations.json','profile_count_review.md'])

"""Source-reference mechanics and additive prompt checks; no model requests."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

from PIL import Image,ImageDraw
import pytest

from as_ocr import detection,grouping as group,regions,reference_relations as refs
from test_grouping import saved_page,read_result


def add_mark(candidate,kind,role,marker,text,box=None,points=()):
    box=box or [candidate['structural_box'][0]+10,70,candidate['structural_box'][0]+40,80]
    candidate['annotations'].append(dict(status='attached',text=text,box=box))
    candidate.setdefault('reference_marks',[]).append(dict(kind=kind,role=role,marker=marker,
        text=text,box=box,leader_points=list(points),evidence=['Exact original reference mark']))


def pair(tmp_path,kind='enlargement',marker='X',caption='X 2:1'):
    source=saved_page(tmp_path,['',''])
    m=json.loads(source.read_text())
    add_mark(m['candidates'][0],kind,'source',marker,marker)
    add_mark(m['candidates'][1],kind,'caption',marker,caption)
    source.write_text(json.dumps(m))
    return source,m


def test_original_prompts_are_complete_unchanged_prefixes():
    expected=json.loads((Path(__file__).parent/'fixtures/original-prompt-prefixes.json').read_text())
    assert len(expected)>=13
    for qualified,spec in expected.items():
        module,name=qualified.rsplit('.',1)
        current=getattr(importlib.import_module(module),name)
        assert hashlib.sha256(current[:spec['length']].encode()).hexdigest()==spec['sha256'],qualified


@pytest.mark.parametrize('kind,marker,caption',[
    ('enlargement','X','X 2:1'),('enlargement','E1','Detail E1'),
    ('section','A','A - A'),('section','B','Schnitt B-B'),
])
def test_unique_source_verified_reference_groups_views(tmp_path,kind,marker,caption):
    source,m=pair(tmp_path,kind,marker,caption)
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==1
    detail=result['details'][0]
    assert detail['membership_role']=='primary' and detail['grouping_status']=='CONFIRMED'
    assert detail['inherited_source_status']=='REVIEW' and not detail['complete']
    assert any(e['type']=='source_verified_'+kind+'_reference' for e in detail['grouping_evidence'])
    assert Path(result['reference_relations_review']).is_file()


def test_bare_repeated_marker_does_not_group(tmp_path):
    source,m=pair(tmp_path,caption='X')
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==2


def test_model_hint_without_native_source_quote_remains_review(tmp_path):
    source,m=pair(tmp_path)
    m['candidates'][1]['annotations']=[];source.write_text(json.dumps(m))
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==2 and result['supporting_clues']


def test_competing_target_captions_do_not_choose_nearest(tmp_path):
    source=saved_page(tmp_path,['','','']);m=json.loads(source.read_text())
    add_mark(m['candidates'][0],'enlargement','source','X','X')
    for c in m['candidates'][1:]:add_mark(c,'enlargement','caption','X','X 2:1')
    source.write_text(json.dumps(m))
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==3 and all(d['grouping_status']=='SINGLETON' for d in result['details'])


def test_standard_hi_variants_do_not_merge_even_with_reference(tmp_path):
    source,m=pair(tmp_path)
    for candidate,text in zip(m['candidates'],['Standard','HI']):
        add_mark(candidate,'variant','caption',text,text,box=[candidate['box'][0]+5,130,candidate['box'][0]+60,140])
    source.write_text(json.dumps(m))
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==2
    assert any('variant' in ' '.join(c.get('observations',[])).lower() for c in result['supporting_clues'])


def test_variant_scope_downgrades_existing_conflicting_explicit_group(tmp_path):
    source=saved_page(tmp_path,['Detail A','Detail A']);m=json.loads(source.read_text())
    for candidate,text in zip(m['candidates'],['HI','SI']):
        add_mark(candidate,'variant','caption',text,text,box=[candidate['box'][0]+5,130,candidate['box'][0]+60,140])
    source.write_text(json.dumps(m))
    result=read_result(group.run([source],tmp_path/'out'))
    assert sum(d['membership_role']=='primary' for d in result['details'])==2
    assert any(d['membership_role']=='alternative' for d in result['details'])


def test_two_sections_of_one_main_view_form_one_group_not_overlapping_pairs(tmp_path):
    source=saved_page(tmp_path,['','','']);m=json.loads(source.read_text())
    add_mark(m['candidates'][0],'section','source','A','A')
    add_mark(m['candidates'][0],'section','source','B','B',box=[60,90,80,100])
    add_mark(m['candidates'][1],'section','caption','A','A-A')
    add_mark(m['candidates'][2],'section','caption','B','B-B')
    source.write_text(json.dumps(m))
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==1 and len(result['details'][0]['view_ids'])==3


def test_cross_page_reference_cannot_merge(tmp_path):
    source,m=pair(tmp_path);_,views,_=group.load_manifest(source)
    views[1]['page_index']=1;views[1]['page_id']='other'
    proposed,_=refs.proposals(views,[])
    assert not proposed


@pytest.mark.parametrize('ink', [True, False])
def test_external_reference_pointer_requires_continuous_original_pixels(tmp_path, ink):
    image=Image.new('RGB',(160,120),'white')
    if ink:ImageDraw.Draw(image).line([(15,40),(90,40)],fill='black',width=2)
    crop=tmp_path/'pointer.png';image.save(crop)
    view=dict(id='main',document_id='doc',page_index=0,kind='assembly_candidate',
              box=[0,0,160,120],structural_box=[80,20,140,100],crop_path=str(crop),
              crop_geometry=dict(source_to_pixels=[1,0,0,1,0,0]))
    mark=dict(box=[5,35,15,45],leader_points=[[15,40],[90,40]])
    assert refs.source_anchor(view,mark,[view]) is ink
    # A geometrically plausible pointer cannot choose between overlapping joints.
    other={**view,'id':'other'}
    assert not refs.source_anchor(view,mark,[view,other])


def test_assembly_step_and_alternative_code_are_context_only(tmp_path):
    source=saved_page(tmp_path,['','']);m=json.loads(source.read_text())
    for c in m['candidates']:
        add_mark(c,'assembly_step','caption','1','1')
        add_mark(c,'alternative_code','caption','FWS60','267446 (267504)',box=[c['box'][0]+5,100,c['box'][0]+100,110])
    source.write_text(json.dumps(m))
    result=read_result(group.run([source],tmp_path/'out'))
    assert len(result['details'])==2
    assert all(not c['membership_established'] for c in result['supporting_clues'])


def test_exploded_basis_and_reference_marks_map_to_original_coordinates():
    response=dict(decision='CANDIDATES',page_kind='engineering',evidence=['Visible exploded fastening axis'],
        coverage='complete',uncertainty=[],followups=[],regions=[dict(id='a',box=[0,0,1,1],
        structural_box=[.1,.1,.8,.8],kind='assembly_candidate',view_type='isometric',title=None,parent_id=None,
        evidence=['Two parts and explicit assembly axis'],uncertainty=[],incomplete=False,
        assembly_basis='exploded_assembly',reference_marks=[dict(kind='assembly_step',role='caption',marker='1',
            text='1',box=[.1,.1,.2,.2],evidence=['Printed step'],leader_points=[])])])
    parsed,_=regions.parse_response(json.dumps(response))
    mapping=dict(pixel_size=[100,100],pixels_to_source=[2,0,0,2,0,0])
    mapped=detection.mapped_regions(parsed,mapping,[0,0,200,200],'full')[0]
    assert mapped['kind']=='assembly_candidate' and mapped['reference_marks'][0]['box']==[20,20,40,40]
    assert any('Exploded' in s for s in mapped['review_reasons'])

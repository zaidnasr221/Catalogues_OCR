import json
from dataclasses import replace
from pathlib import Path

import pytest

from as_ocr import profile_observations as p


def mark(view='v1',box=(.1,.1,.3,.3)):
    return dict(view_id=view,box=list(box),explanation='Visible closed thin-walled metal section')


def profile(local='one',marks=None,candidates=None,category='Frame'):
    return dict(local_id=local,marks=marks or [mark()],candidate_ids=candidates or [],
        identity_evidence='A single connected extruded cross-section; hollow chambers are not separate profiles',
        profile_type=category,type_supported=True,type_evidence='The section forms the rigid supporting boundary of the assembly',review_reasons=[])


def classification(cid='c1',category='Frame',basis='explicit_label',quote='Frame',marks=None):
    return dict(candidate_id=cid,profile_type=category,type_supported=True,basis=basis,source_quote=quote,
                marks=marks or [],explanation='Functional type supported by the component label',review_reasons=[])


def analyze(profiles=(),types=(),coverage='complete',context=None):
    d=dict(id='d1',document_id='doc',page_id='page',page_index=6,view_ids=['v1','v2'])
    mapping=dict(source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0],pixel_size=[100,100])
    layout=[dict(view_id=v,image_box=[0,0,1,1],structural_box=[0,0,100,100],mapping=mapping) for v in d['view_ids']]
    ctx=context or dict(components=[dict(candidate_id='c1',profile_ids=['001'],component_names=['Frame'],
        targets=[dict(view_id='v1',source_point=[20,20])])])
    a=p.Analysis.model_validate_json(json.dumps(dict(profiles=list(profiles),component_types=list(types),coverage=coverage,
        no_profile_reason='Only glazing and sealant are visible' if not profiles else None,review_reasons=[])))
    return p.reconcile(a,d,ctx,layout,[0,0,100,100])


def test_multiple_profiles_repeated_id_and_missing_id_count_geometry():
    ctx=dict(components=[dict(candidate_id='c1',profile_ids=['001'],component_names=['Frame'],targets=[dict(view_id='v1',source_point=[20,20])]),
                         dict(candidate_id='c2',profile_ids=['001'],component_names=['Frame'],targets=[dict(view_id='v1',source_point=[60,20])]),
                         dict(candidate_id='c3',profile_ids=[],component_names=['Adapter'],targets=[dict(view_id='v1',source_point=[20,60])])])
    result=analyze([profile(candidates=['c1']),profile('two',[mark(box=(.5,.1,.7,.3))],['c2']),
                    profile('three',[mark(box=(.1,.5,.3,.7))],['c3'],'Adapter')],context=ctx)
    assert result['observed_count']==3
    assert [x['candidate_ids'] for x in result['profiles']]==[['c1'],['c2'],['c3']]


def test_unlabelled_profile_and_incomplete_review_detail_still_count():
    result=analyze([profile(),profile('two',[mark(box=(.5,.1,.7,.3))])],coverage='incomplete')
    assert result['observed_count']==2 and result['potentially_incomplete']
    assert all(not x['candidate_ids'] for x in result['profiles'])


def test_section_and_isometric_same_physical_profile_count_once():
    result=analyze([profile(marks=[mark(),mark('v2',(.5,.5,.9,.9))],candidates=['c1'])])
    assert result['observed_count']==1 and len(result['profiles'][0]['marks'])==2
    fragmented=analyze([profile(marks=[mark(),mark('v1',(.1,.3,.3,.5))])])
    assert fragmented['observed_count']==1  # Two regions can describe one physical part.


def test_duplicate_geometry_rejected_but_distinct_occurrences_retained():
    result=analyze([profile(),profile('duplicate'),profile('distinct',[mark(box=(.5,.1,.7,.3))])])
    assert result['observed_count']==2
    assert result['rejected_profiles'] and result['potentially_incomplete']


@pytest.mark.parametrize('category',['Gasket','Sealant','Glass / Infill','Fastener'])
def test_accessories_and_glass_not_counted_as_structural_profiles(category):
    result=analyze([profile(category=category)])
    assert result['observed_count']==0 and result['rejected_profiles']


def test_no_profiles_records_reason_and_zero():
    result=analyze()
    assert result['observed_count']==0 and result['review_reasons']
    assert not result['potentially_incomplete']
    observed=analyze([profile()])
    empty=analyze(coverage='incomplete')
    retained=p.retain_observed(observed,empty)
    assert retained['observed_count']==1 and retained['verification_disagreement']


def test_explicit_and_visual_types_and_undetermined_attempt_for_every_component():
    explicit=analyze(types=[classification()])
    assert explicit['component_types'][0]['profile_type']=='Frame'
    visual=analyze(types=[classification(category='Mullion',basis='visual_function',quote=None,marks=[mark()])])
    assert visual['component_types'][0]['profile_type']=='Mullion'
    invalid=analyze(types=[classification(category='Mullion',basis='visual_function',quote=None,marks=[mark(box=(.6,.6,.9,.9))])])
    assert invalid['component_types'][0]['profile_type']=='UNDETERMINED'
    assert analyze()['component_types'][0]['profile_type']=='UNDETERMINED'
    reviewed=classification();reviewed['review_reasons']=['Detail is provisional; full length is not visible']
    assert analyze(types=[reviewed])['component_types'][0]['profile_type']=='Frame'
    reviewed['type_supported']=False
    assert analyze(types=[reviewed])['component_types'][0]['profile_type']=='UNDETERMINED'


def test_identifier_alone_does_not_support_type_and_taxonomy_is_open():
    result=analyze(types=[classification(quote='001')])
    assert result['component_types'][0]['profile_type']=='UNDETERMINED'
    custom=dict(components=[dict(candidate_id='c1',profile_ids=['001'],component_names=['Load-bearing insert'],
        targets=[dict(view_id='v1',source_point=[20,20])])])
    result=analyze(types=[classification(category='Load-bearing insert',quote='Load-bearing insert')],context=custom)
    assert result['component_types'][0]['profile_type']=='Load-bearing insert'
    unknown_name=dict(components=[dict(candidate_id='c1',profile_ids=['001'],component_names=[],
        targets=[dict(view_id='v1',source_point=[20,20])])])
    combined=analyze(types=[classification(quote='Outer frame 001',marks=[mark()])],context=unknown_name)
    assert combined['component_types'][0]['profile_type']=='Frame'
    assert p.label_name('Outer frame 001',unknown_name['components'][0])=='Outer frame'
    assert p.label_name('001',unknown_name['components'][0]) is None
    assert analyze(types=[classification(category='6 x 12 mm')])['component_types'][0]['profile_type']=='UNDETERMINED'


def test_verified_physical_function_populates_type_when_label_has_only_identifier():
    from as_ocr.exports import Reviews
    record=analyze([profile(candidates=['c1'])],[classification(quote='001')])
    detail=dict(id='d1',document_id='doc',page_id='page',page_index=6,view_ids=['v1','v2'])
    views={v:dict(id=v,document_id='doc',page_id='page',view_type='section',box=[0,0,100,100]) for v in detail['view_ids']}
    candidate=dict(id='c1',memberships=[dict(view_id='v1',target_point=[20,20],status='SUPPORTED_BY_STAGE4')])
    observation=dict(id='code',candidate_id='c1',field='code',value_raw='001',accepted=True,conditions=[],conflicts_with=[])
    prop=dict(id='code',accepted=True,conditional=False,conflict_ids=[])
    checked=p.validate_saved([record],{'d1':detail},views,{'c1':candidate},{'code':observation},[prop],Reviews())
    assert checked['d1']['component_types'][0]['profile_type']=='Frame'
    assert checked['d1']['component_types'][0]['basis']=='visual_function'


@pytest.mark.parametrize('verification_fails',[False,True])
def test_removed_structural_inventory_mode_rejects_before_inference(tmp_path,verification_fails):
    from as_ocr import components
    with pytest.raises(ValueError,match='disabled'):components.Config(observed_profiles=True).validate()



def test_legacy_structural_inventory_cannot_create_public_rows_or_counts(tmp_path):
    from test_presentation import fixture
    from as_ocr import presentation
    summary=fixture(tmp_path,repeat=True);rows=presentation.project(summary)[3]
    assert len(rows)==3 and rows[0]==rows[1] and all(not r['profile_count'] for r in rows)


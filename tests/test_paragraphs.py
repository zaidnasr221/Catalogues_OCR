"""Paragraph ownership, source wording and component precedence contracts."""
import json
from pathlib import Path

import pytest

from as_ocr import paragraphs as p


def setup():
    views = [dict(id='v1', document_id='doc', page_id='page', page_index=6, box=[0,0,40,40]),
             dict(id='v2', document_id='doc', page_id='page', page_index=6, box=[60,0,100,40])]
    details = [dict(id='d1',view_ids=['v1']), dict(id='d2',view_ids=['v2'])]
    candidates = [dict(id=cid, observed_in_view_ids=[view], occurrence_ids=[cid], memberships=[dict(
        view_id=view, target_point=[10,10], status='SUPPORTED_BY_STAGE4')])
        for cid,view in [('c1','v1'),('c2','v1'),('c3','v2')]]
    facts = [dict(candidate_id=c['id'],profile_ids=[code] if code else [],component_names=[name])
             for c,code,name in zip(candidates,['A01','','B02'],['Spacer','Seal','Frame'])]
    mapping = dict(source_to_pixels=[1,0,0,1,0,0], pixels_to_source=[1,0,0,1,0,0], pixel_size=[100,100])
    return views,details,candidates,dict(components=facts),mapping


def paragraph(**kw):
    value = dict(text='A01 resists displacement of the glass.', text_kind='explanation', box=[.05,.6,.4,.8], native_span_ids=[],
        scope='component', detail_ids=['d1'],candidate_ids=['c1'],applies_to_all=False,
        basis='explicit_identifier', association_quote='A01',association_box=[.05,.6,.1,.65],
        explanation='The paragraph explicitly identifies A01.',review_reasons=[])
    value.update(kw)
    return value


def reconcile(*items, spans=()):
    views,details,candidates,ctx,mapping=setup()
    a=p.Analysis.model_validate_json(json.dumps(dict(coverage='complete',uncertainty=[],paragraphs=items)))
    return p.reconcile(a,ctx,views,details,candidates,mapping,[0,0,100,100],spans)


def test_native_multiline_outside_geometry_and_original_page():
    text='A01 resists displacement\nof the glass.'
    spans=[(1,dict(text='A01 resists displacement',bbox=[5,60,40,68])),
           (2,dict(text='of the glass.',bbox=[5,70,40,80]))]
    records,fields=reconcile(paragraph(text=text,native_span_ids=[1,2]),spans=spans)
    assert records[0]['accepted']
    assert fields[0]['value_raw']==text
    assert fields[0]['evidence'][0]['page_index']==6
    assert fields[0]['evidence'][0]['box'][1]==60
    bad,empty=reconcile(paragraph(text=text.replace('glass','frame'),native_span_ids=[1,2]),spans=spans)
    assert not empty and not bad[0]['accepted']


def test_component_precedence_and_name_only_general_fallback():
    general=paragraph(text='Install the assembly vertically.',scope='detail',candidate_ids=[],
        basis='structure',applies_to_all=True,association_quote='Assembly installation')
    records,fields=reconcile(general,paragraph())
    assert {o['candidate_id']:o['value_raw'] for o in fields}=={
        'c1':'A01 resists displacement of the glass.','c2':'Install the assembly vertically.'}
    assert records[0]['selected_candidate_ids']==['c2']


def test_explicit_name_supports_component_without_identifier():
    _,fields=reconcile(paragraph(text='Seal prevents water ingress.',candidate_ids=['c2'],
        basis='explicit_name',association_quote='Seal'))
    assert fields[0]['candidate_id']=='c2'


@pytest.mark.parametrize('change',[
    dict(detail_ids=['d2']), dict(candidate_ids=['c3']),dict(basis='structure'),
    dict(scope='unassigned',basis='ambiguous',detail_ids=[],candidate_ids=[]),
    dict(scope='detail',basis='structure',applies_to_all=False),
    dict(association_quote='',association_box=None),dict(text='A01'),
    dict(text='Double glass unit',text_kind='heading'),
    dict(text='Spacer A01 black anodized',text_kind='label'),
    dict(text='XA01 is a different component.',association_quote='XA01'),
])
def test_ambiguous_foreign_generic_or_label_description_withheld(change):
    records,fields=reconcile(paragraph(**change))
    assert not fields and not records[0]['accepted']
    assert records[0]['text'] and records[0]['review_reasons']


def test_neighboring_details_and_distinct_occurrences():
    _,fields=reconcile(paragraph(),paragraph(text='B02 supports the panel.',candidate_ids=['c3'],
        detail_ids=['d2'],association_quote='B02',box=[.6,.6,.95,.8]))
    assert {o['candidate_id'] for o in fields}=={'c1','c3'}
    assert all('B02' not in o['value_raw'] for o in fields if o['candidate_id']=='c1')


def test_multiple_paragraphs_preserve_order_without_duplicate_readings():
    _,fields=reconcile(paragraph(),paragraph(),paragraph(text='A01 also retains the gasket.',box=[.05,.85,.4,.95]))
    assert fields[0]['value_raw']=='A01 resists displacement of the glass.\n\nA01 also retains the gasket.'


def test_original_page_paragraph_cannot_transfer_even_with_verified_identity(tmp_path):
    from test_enrichment_exports import fixture,saved,field
    from as_ocr import enrichment
    src,index,target=fixture(tmp_path,texts=('001.20 - A','001.20 - A'),known=True)
    reference=saved(tmp_path,src,index,page=1,known=True,fields=[field('description','Source paragraph')])
    a=json.loads(target.read_text());b=json.loads(reference.read_text())
    observation=next(o for o in b['observations'] if o['field']=='description')
    observation['relation']={'kind':'page_paragraph'}
    matches,assignments=enrichment.match_sources(a['candidates'],a['identities'],b['candidates'],b['identities'],b['observations'])
    assert matches[0]['status']=='SUPPORTED'
    assert assignments==[]


def test_unreadable_page_recovers_source_paragraph_image_for_review_only(tmp_path,monkeypatch):
    from PIL import Image
    from test_callouts import Client
    views,details,candidates,_,mapping=setup()
    views[0]['structural_box']=[0,0,40,40]
    views[0]['crop_geometry']={'source_context_box':[0,0,40,100]}
    monkeypatch.setattr(p.detection,'render',lambda *a: (Image.new('RGB',(100,100),'white'),mapping))
    first=json.dumps(dict(coverage='incomplete',uncertainty=['Text too small'],paragraphs=[]))
    second=json.dumps(dict(coverage='complete',uncertainty=[],paragraphs=[paragraph()]))
    result,fields=p.extract('original.pdf',6,{'source_rect':[0,0,100,100]},[],views,details,candidates,[],tmp_path,Client(first,second))
    assert not fields and result['coverage']=='incomplete'
    assert Path(result['recovery_image']).is_file()
    assert result['paragraphs'][0]['scope']=='unassigned' and not result['paragraphs'][0]['accepted']


@pytest.mark.parametrize('agrees',[True,False])
def test_removed_csv_paragraph_mode_rejects_before_inference(tmp_path,agrees):
    from as_ocr import components
    from test_components import fixture,Client,MODEL
    stage3,stage4,index=fixture(tmp_path,target=True);client=Client()
    with pytest.raises(ValueError,match='disabled'):
        components.run(stage3,stage4,tmp_path/'new5',index_path=index,
            config=components.Config(details_only=True,paragraph_descriptions=True),client=client,model=MODEL)
    assert client.requests==0


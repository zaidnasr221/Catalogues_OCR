"""Stage 5 source/ownership contracts using synthetic saved Stage 3/4 outputs."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from as_ocr import callouts, components as c, grouping
from as_ocr.qwen import ModelConfig,ModelError
from test_callouts import fixture as source_fixture,Client,MODEL,observation,response


from as_ocr import presentation
from test_presentation import fixture as public_fixture, publish, read, table, update_callout, confirm_details

def evidence(quote,box=(.1,.18,.55,.23),method='visual_label',native=()):
    return dict(quote=quote,box=list(box),method=method,native_span_ids=list(native),review_reasons=[])


def field(name='description',value='Spacer holder',unit=None,**kwargs):
    result=dict(field=name,name_raw=None,value_raw=value,unit_raw=unit,category_value=None,
                meaning='unspecified',decimal_convention='unknown',evidence=[evidence(value+((' '+unit) if unit else ''))],
                conditions=[],review_reasons=[])
    result.update(kwargs)
    return result


def binding(code='001.20 - A',kind='code_label'):
    return dict(kind=kind,bounds=[.08,.08,.9,.9],code_evidence=evidence(code,(.1,.1,.3,.15)),
                evidence=[evidence(code+' Spacer holder',(.1,.1,.65,.24),'visual_table' if kind=='table_row' else 'visual_label')],
                explanation='The exact code and description share one visible label block/row')


def component(ctx,fields=(),kind='code_label',code=None,ids=None,local='part1'):
    occ=ctx['occurrences'][0] if ctx['occurrences'] else None
    code=code or (occ['code_raw'] if occ else '001.20 - A')
    return dict(local_id=local,code_raw=code,occurrence_ids=ids if ids is not None else ([occ['id']] if occ else []),
                relation=binding(code,kind),fields=list(fields),review_reasons=[])


def analysis(*components,scope=(),coverage='complete',uncertainty=()):
    return json.dumps(dict(coverage=coverage,evidence=['The selected code label and its explicit text were inspected'],
                           uncertainty=list(uncertainty),scope=list(scope),components=list(components)))


def test_detail_mode_excludes_unrequested_measurements_and_new_table_components(tmp_path):
    inputs = fixture(tmp_path, target=True)
    def answer(ctx):
        return analysis(component(ctx, [field('description', 'Spacer holder'),
                                       field('component_name', ctx['occurrences'][0]['code_raw']),
                                       field('material', 'aluminium')]),
                        component(ctx, [field('description', 'Unrelated table')],
                                  kind='table_row', code='999', ids=[], local='table'))
    result = c.run(inputs[0], inputs[1], tmp_path/'detail-fields', index_path=inputs[2],
                   client=Client(answer), model=replace(MODEL, max_requests=6), config=c.Config(details_only=True))
    saved = json.loads(Path(result['manifest']).read_text())
    assert result['errors'] == 0
    assert all(o['field'] in {'code','component_name','category','role','description'}
               for o in saved['observations'])
    assert all(candidate['code_raw'] != '999' for candidate in saved['candidates'])
    assert any(o['field'] == 'component_name' and not o['accepted'] and
               'An identifier is not a component name' in o['review_reasons']
               for o in saved['observations'])


def fixture(tmp_path,*,codes=('001.20 - A',),target=False,boxes=None,text=None,rotation=0,scanned=False, category='product_code'):
    s2,index=source_fixture(tmp_path,boxes=boxes,text=text,rotation=rotation,scanned=scanned)
    s3=grouping.run([s2],tmp_path/'stage3',index_path=index)['manifests'][0]
    def reply(ctx):
        obs=[]
        for i,code in enumerate(codes):
            kwargs={}
            if target:
                kwargs=dict(owner_view_id=ctx['focus_view_id'],association_method='leader',leader_state='clear',
                            leader_points=[[.3,.125+i*.2],[.6,.5+i*.2]],target_point=[.6,.5+i*.2],
                            association_evidence=['The label leader terminates on the component'])
            obs.append(observation(code,(.1,.1+i*.2,.3,.15+i*.2),category=category,**kwargs))
        return response(*obs)
    count=len(boxes) if boxes else 1
    s4=callouts.run(s2,tmp_path/'stage4',index_path=index,stage3=s3,client=Client(*([reply]*count)),model=MODEL,
                    config=callouts.Config(context_fraction=0))['manifest']
    return s3,s4,index


def run(f,tmp_path,client,**kwargs):
    s3,s4,index=f
    return c.run(s3,s4,tmp_path/'stage5',index_path=index,client=client,model=replace(MODEL,max_requests=6,max_tokens=4096),
                 config=c.Config(context_fraction=0),**kwargs)


def result(summary):
    return json.loads(Path(summary['manifest']).read_text())


def observations(out,name):
    return [o for o in out['observations'] if o['field']==name]


def test_exact_codes_repeated_occurrences_and_unknown_scope(tmp_path):
    codes=('00 123/AB.02','00 123/AB.02','001.20 - A')
    f=fixture(tmp_path,codes=codes)
    out=result(run(f,tmp_path,Client(analysis())))
    assert [v['code_raw'] for v in out['candidates']]==list(codes)
    assert len({v['id'] for v in out['candidates']})==3 and len(out['identities'])==3
    assert out['candidates'][0]['code_normalized']=='00123/AB.02'
    assert all(i['scope']=={'manufacturer':None,'system':None,'edition':None} for i in out['identities'])
    assert all(v['fields']['material'] is None and v['fields']['category']=='unknown' for v in out['candidates'])
    assert all(v['memberships'][0]['target_point'] is None for v in out['candidates'])
    assert len({v['occurrence_ids'][0] for v in out['candidates']})==3


def test_null_target_cannot_be_upgraded_by_model(tmp_path):
    f=fixture(tmp_path)
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[field('material','Aluminium')],kind='stage4_target')))))
    o=observations(out,'material')[0]
    assert not o['accepted'] and o['normalized'] is None
    candidate=out['candidates'][0]
    assert candidate['fields']['material'] is None
    assert candidate['memberships'][0]['status']=='UNRESOLVED'
    assert any('Null/unresolved' in r for r in o['review_reasons'])


def test_explicit_label_fields_are_candidate_observations_not_membership(tmp_path):
    f=fixture(tmp_path)
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[field(),field('material','EPDM')])))))
    candidate=out['candidates'][0]
    assert candidate['fields']['description']['value']=='Spacer holder'
    assert candidate['fields']['material']['value']=='EPDM'
    assert candidate['memberships'][0]['target_point'] is None
    assert candidate['memberships'][0]['status']=='UNRESOLVED' and candidate['status']=='REVIEW'
    assert all(o['evidence'][0]['document_id'] and o['evidence'][0]['view_id'] for o in out['observations'])
    assert candidate['inherited_review_reasons']
    assert Path(out['review_path']).read_text().startswith('# Stage 5 review')
    assert Path(out['annotated_views'][0]['path']).exists()


def test_explicit_table_row_without_callout_supports_property_only(tmp_path):
    f=fixture(tmp_path,codes=())
    def reply(ctx):
        return analysis(component(ctx,[field('linear_weight','2500','g/m',decimal_convention='integer')],kind='table_row'))
    out=result(run(f,tmp_path,Client(reply)))
    candidate=out['candidates'][0]
    assert not candidate['occurrence_ids'] and not candidate['memberships']
    assert candidate['fields']['linear_weight']['normalized']=={'value':'2.5','unit':'kg/m'}
    assert observations(out,'linear_weight')[0]['relation']['kind']=='table_row'


def test_row_geometry_must_demonstrate_field_ownership(tmp_path):
    f=fixture(tmp_path,codes=())
    def reply(ctx):
        part=component(ctx,[field('material','EPDM')],kind='table_row')
        part['relation']['bounds']=[.08,.08,.4,.16]
        return analysis(part)
    out=result(run(f,tmp_path,Client(reply)))
    assert not observations(out,'material')[0]['accepted']
    assert out['candidates'][0]['fields']['material'] is None


def test_new_component_requires_explicit_row(tmp_path):
    f=fixture(tmp_path,codes=())
    summary=run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[field()],kind='code_label'))))
    assert summary['errors']==1 and result(summary)['state']=='FAILED'


@pytest.mark.parametrize('field_name,raw,unit,convention,expected',[
    ('dimension','2.5','cm','dot',('25','mm')),
    ('dimension','1','m','integer',('1000','mm')),
    ('dimension','2','in','integer',('50.8','mm')),
    ('linear_weight','2500','g/m','integer',('2.5','kg/m')),
    ('linear_weight','2,5','kg/m','comma',('2.5','kg/m')),
    ('linear_weight','0.02','kg/mm','dot',('20','kg/m')),
    ('inertia','120000','mm⁴','integer',('12','cm4')),
    ('inertia','0.5','cm^4','dot',('0.5','cm4')),
    ('section_modulus','12000','mm³','integer',('12','cm3')),
    ('area','250','mm2','integer',('2.5','cm2')),
])
def test_deterministic_unit_conversions(field_name,raw,unit,convention,expected):
    normalized,reason=c.normalize_value(field_name,raw,unit,convention)
    assert reason is None and normalized==dict(zip(('value','unit'),expected))


@pytest.mark.parametrize('raw,unit,convention',[
    ('1,234','kg/m','comma'),('1.234','kg/m','dot'),('1,2','kg/m','unknown'),
    ('1,2','kg/m','dot'),('NaN','kg/m','dot'),('Infinity','kg/m','dot'),
    ('-3','kg/m','integer'),('2.5','mm','dot'),('2 500','kg/m','integer'),
    ('2,500.0','kg/m','dot'),('1e3','kg/m','integer'),('2',None,'integer'),
])
def test_ambiguous_incompatible_nonfinite_values_rejected(raw,unit,convention):
    normalized,reason=c.normalize_value('linear_weight',raw,unit,convention)
    assert normalized is None and reason


def test_unknown_values_and_finish_not_material(tmp_path):
    f=fixture(tmp_path)
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[
        field('material','Aluminium',evidence=[evidence('black anodized*')]),
        field('material','black anodized',evidence=[evidence('black anodized')]),
        field('surface_finish','black anodized')])))))
    candidate=out['candidates'][0]
    assert candidate['fields']['material'] is None and candidate['fields']['category']=='unknown'
    assert candidate['fields']['surface_finish']['value']=='black anodized'
    assert not any(o['accepted'] for o in observations(out,'material'))


@pytest.mark.parametrize('meaning',['assembly_dimension','stock_length','cut_length','glass_thickness','joint_dimension','unspecified'])
def test_assembly_and_other_dimensions_not_assigned_to_profile(tmp_path,meaning):
    f=fixture(tmp_path,target=True)
    numeric=field('dimension','50','mm',name_raw='width',meaning=meaning,decimal_convention='integer',
                  evidence=[evidence('width 50 mm')])
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[numeric],kind='stage4_target')))))
    assert not observations(out,'dimension')[0]['accepted']
    assert out['candidates'][0]['fields']['dimensions']==[]


def test_explicit_component_dimension_and_axes_remain_distinct(tmp_path):
    f=fixture(tmp_path,target=True)
    fields=[field('dimension','50','mm',name_raw='width',meaning='component_dimension',decimal_convention='integer',evidence=[evidence('width 50 mm')]),
            field('inertia','12','cm4',name_raw='Ix',decimal_convention='integer',evidence=[evidence('Ix 12 cm4')]),
            field('section_modulus','6','cm3',name_raw='Wx',decimal_convention='integer',evidence=[evidence('Wx 6 cm3')])]
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,fields,kind='stage4_target')))))
    assert all(o['accepted'] for o in observations(out,'dimension')+observations(out,'inertia')+observations(out,'section_modulus'))
    assert observations(out,'inertia')[0]['normalized']['unit']=='cm4'
    assert observations(out,'section_modulus')[0]['normalized']['unit']=='cm3'


def test_conditional_properties_and_conflicts_preserved(tmp_path):
    f=fixture(tmp_path)
    fields=[field('linear_weight','2','kg/m',decimal_convention='integer'),
            field('linear_weight','3','kg/m',decimal_convention='integer'),
            field('linear_weight','4','kg/m',decimal_convention='integer',conditions=[evidence('with reinforcement')])]
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,fields)))))
    weights=observations(out,'linear_weight')
    assert len(weights)==3
    assert all(o['conflicts_with'] for o in weights[:2]) and not weights[2]['conflicts_with']
    assert weights[2]['conditions'][0]['quote']=='with reinforcement'
    assert out['candidates'][0]['fields']['linear_weight'] is None
    assert {o['value_raw'] for o in weights}=={'2','3','4'}


def test_conditional_wording_not_silently_unconditional(tmp_path):
    f=fixture(tmp_path)
    value=field('inertia','12','cm4',name_raw='Ix',decimal_convention='integer',evidence=[evidence('reinforced Ix 12 cm4')])
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[value])))))
    assert not observations(out,'inertia')[0]['accepted']


def test_verified_scope_can_share_identity_without_erasing_occurrences(tmp_path):
    f=fixture(tmp_path,codes=('001','001'))
    scope=[dict(field=key,value_raw=value,evidence=[evidence(key.title()+': '+value)],review_reasons=[])
           for key,value in [('manufacturer','SourceCo'),('system','System Q'),('edition','Edition 2020')]]
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(scope=scope))))
    assert len(out['candidates'])==2 and len(out['identities'])==1
    assert out['identities'][0]['scope_status']=='SOURCE_VERIFIED'
    assert len({v['occurrence_ids'][0] for v in out['candidates']})==2


def test_scope_conflict_and_unknown_scope_never_guess(tmp_path):
    f=fixture(tmp_path,codes=('001','001'))
    scope=[dict(field='manufacturer',value_raw=value,evidence=[evidence('Manufacturer: '+value)],review_reasons=[]) for value in ['A','B']]
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(scope=scope))))
    assert len(out['identities'])==2
    assert all(i['scope']['manufacturer'] is None for i in out['identities'])


def test_source_document_identity_prevents_code_entity_merging(tmp_path):
    first=tmp_path/'one';first.mkdir();second=tmp_path/'two';second.mkdir()
    f1=fixture(first,codes=('001',));f2=fixture(second,codes=('001',),text='different document')
    out1=result(run(f1,first,Client(analysis())));out2=result(run(f2,second,Client(analysis())))
    assert out1['identities'][0]['id']!=out2['identities'][0]['id']


@pytest.mark.parametrize('scanned',[False,True])
def test_native_and_scanned_source_evidence_paths(tmp_path,scanned):
    f=fixture(tmp_path,text=None if scanned else 'EPDM',scanned=scanned)
    def reply(ctx):
        spans=ctx['native_spans']
        ev=evidence('EPDM')
        if not scanned:
            assert spans
            ev=evidence('EPDM',spans[0]['box'],'native_text',[spans[0]['id']])
        value=field('material','EPDM',evidence=[ev])
        part=component(ctx,[value])
        if not scanned:
            part['relation']['bounds']=[min(.08,ev['box'][0]),min(.08,ev['box'][1]),max(.9,ev['box'][2]),max(.9,ev['box'][3])]
        return analysis(part)
    out=result(run(f,tmp_path,Client(reply)))
    o=observations(out,'material')[0]
    assert o['evidence'][0]['method']==('visual_label' if scanned else 'native_text')
    assert o['accepted']


def test_visual_native_conflict_preserved(tmp_path):
    f=fixture(tmp_path,text='Steel')
    def reply(ctx):
        return analysis(component(ctx,[field('material','EPDM',evidence=[evidence('EPDM',ctx['native_spans'][0]['box'])])]))
    out=result(run(f,tmp_path,Client(reply)))
    o=observations(out,'material')[0]
    assert not o['accepted'] and o['evidence'][0]['native_readings']==['Steel']


def test_rotated_source_field_bounds(tmp_path):
    f=fixture(tmp_path,text='EPDM',rotation=90)
    def reply(ctx):
        span=ctx['native_spans'][0]
        return analysis(component(ctx,[field('material','EPDM',evidence=[evidence('EPDM',span['box'],'native_text',[span['id']])])]))
    out=result(run(f,tmp_path,Client(reply)))
    with sqlite3.connect(f[2]) as db:
        expected=json.loads(db.execute('SELECT data FROM spans').fetchone()[0])['bbox']
    assert observations(out,'material')[0]['evidence'][0]['box']==pytest.approx(expected,abs=.001)


@pytest.mark.parametrize('bad',[
    '{}','not JSON','{"coverage":"complete","coverage":"incomplete"}',
    analysis().replace('"complete"','NaN'),
    json.dumps(dict(coverage='complete',evidence=[],uncertainty=[],scope=[],components=[])),
])
def test_invalid_responses_fail_not_empty_success(tmp_path,bad):
    f=fixture(tmp_path)
    summary=run(f,tmp_path,Client(bad));out=result(summary)
    assert summary['errors']==1 and out['state']=='FAILED' and not out['complete']
    assert out['candidates'][0]['code_raw']=='001.20 - A'
    assert out['candidates'][0]['fields']['material'] is None


def test_truncation_and_failure_reasons_preserve_codes(tmp_path):
    f=fixture(tmp_path)
    out=result(run(f,tmp_path,Client(ModelError('Model output incomplete or truncated'))))
    assert out['failures'][0]['reason']=='model_output_truncated' and out['state']=='FAILED'
    assert out['candidates'][0]['code_raw']=='001.20 - A'


def test_unknown_evidence_or_occurrence_references_fail(tmp_path):
    f=fixture(tmp_path)
    def bad(ctx):
        return analysis(component(ctx,[field()],ids=['not-an-occurrence']))
    assert run(f,tmp_path,Client(bad))['errors']==1
    def bad_native(ctx):
        return analysis(component(ctx,[field(evidence=[evidence('Spacer holder',method='native_text',native=[9000])])]))
    assert run(f,tmp_path,Client(bad_native))['errors']==1


def test_partial_failure_resume_force_and_output_preservation(tmp_path):
    f=fixture(tmp_path,boxes=[[10,10,140,230],[160,10,290,230]])
    good=lambda ctx:analysis(component(ctx,[field()]))
    first=run(f,tmp_path,Client(good,ModelError('timeout')))
    assert first['views_completed']==1 and first['errors']==1
    old={str(p):p.read_bytes() for p in (tmp_path/'stage5').rglob('*') if p.is_file()}
    second=run(f,tmp_path,Client(good))
    assert second['views_reused']==1 and second['views_completed']==1
    assert all(Path(p).read_bytes()==raw for p,raw in old.items())
    assert run(f,tmp_path,Client(ModelError('fail'),ModelError('fail')),force=True)['errors']==2
    assert run(f,tmp_path,Client())['views_reused']==2


def test_resume_invalidation_model_prompt_input_and_budget(tmp_path,monkeypatch):
    f=fixture(tmp_path)
    good=lambda ctx:analysis(component(ctx,[field()]))
    first=result(run(f,tmp_path,Client(good)))
    client=Client();client.response_models=set()
    out=result(c.run(f[0],f[1],tmp_path/'stage5',index_path=f[2],client=client,
                    model=replace(MODEL,max_requests=1,max_tokens=4096),config=c.Config(context_fraction=0)))
    assert out['model_requests']==0 and out['actual_response_models']==['fake']
    assert out['candidates'][0]['id']==first['candidates'][0]['id']
    monkeypatch.setattr(c,'PROMPT',c.PROMPT+' Updated instructions.')
    assert run(f,tmp_path,Client(good))['views_completed']==1
    changed=json.loads(Path(f[1]).read_text());changed['complete']=False;Path(f[1]).write_text(json.dumps(changed))
    assert run(f,tmp_path,Client(good))['views_completed']==1
    for identity in ['checkpoint@changed',None]:
        summary=c.run(f[0],f[1],tmp_path/'stage5',index_path=f[2],client=Client(good),
                      model=replace(MODEL,max_requests=6,max_tokens=4096,identity=identity),config=c.Config(context_fraction=0))
        assert summary['views_completed']==1


def test_corrupted_evidence_artifact_invalidates_cache(tmp_path):
    f=fixture(tmp_path)
    out=result(run(f,tmp_path,Client(analysis())))
    Path(out['annotated_views'][0]['source_path']).write_bytes(b'corrupt')
    assert run(f,tmp_path,Client(analysis()))['views_completed']==1


def test_model_budget_and_upstream_provenance_validation(tmp_path):
    f=fixture(tmp_path)
    high=c.run(f[0],f[1],tmp_path/'high-budget',index_path=f[2],model=replace(MODEL,max_requests=100),
               client=Client(analysis()))
    assert high['model_requests']==1
    with pytest.raises(ValueError,match='positive integer'):
        replace(MODEL,max_requests=0)
    saved=json.loads(Path(f[1]).read_text());saved['upstream']['stage3']['sha256']='wrong';Path(f[1]).write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='different Stage 3'):
        run(f,tmp_path,Client())


def test_detail_response_schema_and_empty_additions_preserve_all_baselines(tmp_path):
    f=fixture(tmp_path,codes=('235546','235546','50SG01'))
    client=Client(analysis())
    summary=c.run(f[0],f[1],tmp_path/'compact',index_path=f[2],client=client,
        model=replace(MODEL,max_requests=100),config=c.Config(details_only=True))
    saved=result(summary)
    assert len(saved['candidates'])==3
    assert [o['value_raw'] for o in saved['observations'] if o['field']=='code'].count('235546')==2
    schema=client.schemas[0]
    assert schema['properties']['scope']['maxItems']==0
    assert schema['$defs']['Property']['properties']['field']['enum']==['component_name']
    assert 'material' in c.response_schema(False)['$defs']['Property']['properties']['field']['enum']
    assert client.requests==1


def test_ambiguous_numeric_observation_not_asserted(tmp_path):
    f=fixture(tmp_path)
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[field('linear_weight','1,234','kg/m',decimal_convention='comma')])))))
    weight=observations(out,'linear_weight')[0]
    assert weight['value_raw']=='1,234' and weight['normalized'] is None and not weight['accepted']
    assert out['candidates'][0]['fields']['linear_weight'] is None


def test_unit_word_substrings_are_not_unit_evidence(tmp_path):
    f=fixture(tmp_path)
    value=field('dimension','50','m',name_raw='minimum',meaning='component_dimension',
                decimal_convention='integer',evidence=[evidence('minimum 50')])
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[value])))))
    assert not observations(out,'dimension')[0]['accepted']


def test_table_matching_code_does_not_assign_to_unresolved_occurrence(tmp_path):
    f=fixture(tmp_path)
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[field('linear_weight','2','kg/m',decimal_convention='integer')],kind='table_row')))))
    assert len(out['candidates'])==2
    callout=next(v for v in out['candidates'] if v['occurrence_ids'])
    row=next(v for v in out['candidates'] if not v['occurrence_ids'])
    assert callout['fields']['linear_weight'] is None
    assert row['fields']['linear_weight']['normalized']=={'value':'2','unit':'kg/m'}
    assert row['related_occurrence_ids']==callout['occurrence_ids']
    assert row['identity_id']!=callout['identity_id']


def test_code_label_at_other_occurrence_location_is_not_ownership(tmp_path):
    f=fixture(tmp_path)
    def reply(ctx):
        part=component(ctx,[field('material','EPDM')])
        part['relation']['code_evidence']['box']=[.7,.7,.8,.75]
        return analysis(part)
    out=result(run(f,tmp_path,Client(reply)))
    assert not observations(out,'material')[0]['accepted']


def test_native_text_does_not_prove_ownership(tmp_path):
    f=fixture(tmp_path,text='EPDM')
    def reply(ctx):
        span=ctx['native_spans'][0]
        part=component(ctx,[field('material','EPDM')])
        part['relation']['evidence']=[evidence('EPDM',span['box'],'native_text',[span['id']])]
        return analysis(part)
    out=result(run(f,tmp_path,Client(reply)))
    assert not observations(out,'material')[0]['accepted']


def test_unresolved_footnote_not_silently_unconditional(tmp_path):
    f=fixture(tmp_path)
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[field('surface_finish','black anodized',evidence=[evidence('black anodized*')])])))))
    assert not observations(out,'surface_finish')[0]['accepted']
    assert out['candidates'][0]['fields']['surface_finish'] is None


def test_failed_view_exports_source_context_without_property_success(tmp_path):
    f=fixture(tmp_path)
    out=result(run(f,tmp_path,Client(ModelError('timeout'))))
    assert out['state']=='FAILED' and not out['complete']
    assert out['annotated_views'][0]['status'].startswith('FAILED')
    assert Path(out['annotated_views'][0]['path']).is_file()
    assert set(o['field'] for o in out['observations'])=={'code'}


def test_membership_review_does_not_replace_field_validation(tmp_path):
    f=fixture(tmp_path)
    def reply(ctx):
        part=component(ctx,[field()]);part['review_reasons']=['Drawing membership remains unresolved']
        return analysis(part)
    out=result(run(f,tmp_path,Client(reply)))
    assert observations(out,'description')[0]['accepted']
    assert out['candidates'][0]['status']=='REVIEW'
    assert out['candidates'][0]['memberships'][0]['status']=='UNRESOLVED'


def test_shared_quote_footnote_does_not_invalidate_unqualified_description(tmp_path):
    f=fixture(tmp_path)
    prop=field('description','Spacer holder',evidence=[evidence('Spacer holder black anodized*')])
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,[prop])))))
    assert observations(out,'description')[0]['accepted']


def test_glass_thickness_requires_explicit_glass_category(tmp_path):
    f=fixture(tmp_path)
    props=[field('dimension','6','mm',meaning='glass_thickness',name_raw='thickness',decimal_convention='integer',evidence=[evidence('thickness 6 mm')]),
           field('category','Glass',category_value='glass')]
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(component(ctx,props)))))
    assert observations(out,'dimension')[0]['accepted']
    assert out['candidates'][0]['fields']['category']['value']=='glass'


def test_view_title_and_year_are_not_verified_system_or_edition(tmp_path):
    f=fixture(tmp_path)
    scope=[dict(field=name,value_raw=value,evidence=[evidence(value)],review_reasons=[])
           for name,value in [('system','Spandrel glass unit'),('edition','2011')]]
    out=result(run(f,tmp_path,Client(lambda ctx:analysis(scope=scope))))
    assert not any(s['accepted'] for s in out['scope_observations'])
    assert out['identities'][0]['scope']=={'manufacturer':None,'system':None,'edition':None}


def test_aggregation_revalidates_cached_scope_without_guessing():
    scope=[dict(field='system',value_raw='Spandrel glass unit',accepted=True,review_reasons=[],
                evidence=[dict(quote='Spandrel glass unit',view_id='v1')])]
    c.validate_scope(scope)
    assert not scope[0]['accepted'] and scope[0]['review_reasons']


# Component-name provenance and public field contracts.
@pytest.mark.parametrize('target', [True, False])
def test_name_occurrence_through_export(tmp_path, target):
    f = fixture(tmp_path, codes=('Joint EPDM',), target=target, category='component_name')
    out = result(run(f, tmp_path, Client(analysis())))
    names = [o for o in out['observations'] if o['field'] == 'component_name']
    assert len(names) == 1 and names[0]['accepted'] is True
    assert names[0]['value_raw'] == 'Joint EPDM'
    assert not any(o['accepted'] for o in out['observations'] if o['field'] == 'code')
    from test_enrichment_exports import enrich, exported
    stage6 = enrich(tmp_path, f[2], [out['review_path'].replace('review.md', 'manifest.json')])
    stage7 = exported(tmp_path, f[2], stage6['manifest'])
    # Export must preserve the name as an independently supported field.
    import csv
    summary = stage7
    with Path(summary['artifacts']['properties']).open() as handle:
        props = list(csv.DictReader(handle))
    assert any(p['field'] == 'component_name' and p['accepted'] == 'true' for p in props)
    assert summary['validation_errors'] == 0
    published = presentation.run([summary['artifacts']['summary']], output=tmp_path/'public', internal_root=tmp_path/'internal')[0]
    public = read(published)
    assert any(r['component_name'] == 'Joint EPDM' for r in public) if target else all(not r['component_name'] for r in public)
    assert all(not r['profile_id'] for r in public)



@pytest.mark.parametrize('code,name', [('001.20 - A','Joint EPDM'), ('','Joint EPDM'), ('001.20 - A',''), ('','')])
def test_public_name_and_id_combinations(tmp_path, code, name):
    summary = public_fixture(tmp_path, code=code, repeat=True)
    artifacts = json.loads(summary.read_text())['artifacts']
    import csv
    def rows(key):
        with Path(artifacts[key]).open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))
    if name:
        props, evidence, candidates = rows('properties'), rows('evidence'), rows('components')
        props.append(dict(props[0], id='pn', field='component_name', value_raw=name, evidence_ids='["en"]'))
        evidence.append(dict(evidence[0], id='en', quote=name))
        candidates[0]['property_ids'] = json.dumps([p['id'] for p in props])
        for key, values in [('properties',props),('evidence',evidence),('components',candidates)]:
            table(Path(artifacts[key]), values)
    output = read(publish(tmp_path,[summary])[0])
    selected = output[:2]
    assert len(selected) == 2
    assert all(r['profile_id'] == code and r['component_name'] == '' for r in selected)


def test_model_name_field_requires_existing_ownership(tmp_path):
    f = fixture(tmp_path, target=False)
    out = result(run(f,tmp_path,Client(lambda ctx: analysis(component(ctx,[field('component_name','Joint EPDM')],kind='stage4_target')))))
    assert not next(o for o in out['observations'] if o['field']=='component_name')['accepted']


def test_callout_schema_accepts_separate_name_category():
    from test_callouts import observation, response
    parsed = callouts.parse_response(response(observation('Joint EPDM', category='component_name')))
    assert parsed.observations[0].category == 'component_name'


@pytest.mark.parametrize('supported', [True, False])
def test_id_and_exact_name_property_keep_independent_evidence(tmp_path, supported):
    f = fixture(tmp_path, target=True)
    def reply(ctx):
        value = field('component_name', 'Dichtung ÉPDM')
        if not supported:
            value['evidence'][0]['quote'] = 'Unrelated nearby label'
        return analysis(component(ctx, [value], kind='stage4_target'))
    out = result(run(f, tmp_path, Client(reply)))
    assert any(o['field']=='code' and o['accepted'] for o in out['observations'])
    name = next(o for o in out['observations'] if o['field']=='component_name')
    assert name['accepted'] == supported
    assert name['value_raw'] == 'Dichtung ÉPDM'


def test_name_only_count_is_unknown_without_geometry(tmp_path):
    summary = public_fixture(tmp_path, code='Joint EPDM', repeat=True)
    artifacts = json.loads(summary.read_text())['artifacts']
    import csv
    with Path(artifacts['properties']).open(encoding='utf-8-sig') as handle:
        props = list(csv.DictReader(handle))
    for prop in props:
        if prop['field']=='code': prop['field']='component_name'
    table(Path(artifacts['properties']), props)
    update_callout(summary, lambda saved: [o.update(category='component_name') for o in saved['occurrences']])
    confirm_details(summary)
    rows = read(publish(tmp_path,[summary])[0])
    named = [r for r in rows if r['component_name']]
    assert len(named)==2
    assert all(r['profile_count']=='' and r['profile_id']=='' for r in named)


@pytest.mark.parametrize('text', ['(07)','(9)','(01,02,03)','(01, 02, 03)'])
def test_standalone_item_numbers_never_become_profile_ids(tmp_path,text):
    assert callouts.non_code_category(text)=='item_marker'
    summary=public_fixture(tmp_path,code=text)
    rows=read(publish(tmp_path,[summary])[0])
    assert not any(row['profile_id'] for row in rows)


@pytest.mark.parametrize('text', ['244501','001.20 - A','CW(07)-123','A(9)','(1234)'])
def test_numbers_inside_real_identifier_candidates_are_preserved(text):
    assert callouts.non_code_category(text) is None

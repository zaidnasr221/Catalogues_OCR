"""Generic catalog mechanics, source evidence and negative fixtures; no live model."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

from PIL import Image,ImageDraw
import pytest
import pymupdf as fitz

from as_ocr import catalog_lookup as lookup,local_legend,public_contract,callouts,leaders,line_evidence
from as_ocr import presentation,reference_relations as refs,grouping


def test_all_previous_prompts_remain_exact_prefixes():
    expected=json.loads((Path(__file__).parent/'fixtures/mx-prompt-prefixes.json').read_text())
    assert len(expected)==22
    for qualified,spec in expected.items():
        module,name=qualified.rsplit('.',1)
        value=getattr(importlib.import_module('as_ocr.'+module),name)
        assert hashlib.sha256(value[:spec['length']].encode()).hexdigest()==spec['sha256'],qualified


def scopes(section='Roof system',variant=''):
    return [dict(kind=k,text=v,verified=True,box=[0,0,100,10],method='native_quote')
        for k,v in [('system','Generic system'),('section',section),('variant',variant)] if v]


def unit(text='| Reference | Description | Quantity |\n|---|---|---|\n| ZX-205 | Literal gasket | 2 |'):
    return dict(kind='table',markdown=text,complete=True,issues=[],source_box=[0,20,100,90],
        source_audit=dict(complete=True,issues=[],missing_regions=[],unsupported_text=[]))


def catalog(tmp_path):
    source=tmp_path/'source.pdf'
    with fitz.open() as pdf:
        pdf.new_page();pdf.new_page();pdf.save(source)
    doc=hashlib.sha256(source.read_bytes()).hexdigest()
    image=tmp_path/'table.png';Image.new('RGB',(100,100),'white').save(image)
    table=unit();table.update(source_image=str(image),source_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    pages=[dict(document_id=doc,source_page=1,scope_evidence=scopes(),units=[]),
        dict(document_id=doc,source_page=2,scope_evidence=scopes(),units=[table])]
    manifest=tmp_path/'ocr.json';manifest.write_text(json.dumps(dict(document_id=doc,source_path=str(source),pages=pages)))
    return manifest,doc


def test_cross_page_names_require_literal_code_scope_and_audited_source(tmp_path):
    manifest,doc=catalog(tmp_path);snapshot=presentation.Snapshot()
    index=lookup.build(manifest,snapshot,doc)
    name,evidence,reasons=lookup.lookup(index,'ZX-205',1,[0,0,90,90])
    assert name=='Literal gasket' and not reasons
    assert evidence[0]['source_page']==2 and evidence[0]['row_number']==1
    assert evidence[0]['code_column']==1 and evidence[0]['name_column']==2
    assert lookup.lookup(index,'zx-205',1,[0,0,90,90])[0]==''
    assert lookup.lookup(index,'ZX-999',1,[0,0,90,90])[0]==''
    snapshot.verify()


@pytest.mark.parametrize('change',['section','variant','missing_scope','conflict','unaudited','tampered_image','foreign_doc'])
def test_cross_page_name_negative_cases(tmp_path,change):
    manifest,doc=catalog(tmp_path);saved=json.loads(manifest.read_text())
    if change=='section':saved['pages'][1]['scope_evidence']=scopes('Other section')
    if change=='variant':saved['pages'][1]['scope_evidence']=scopes(variant='Left handed')
    if change=='missing_scope':saved['pages'][0]['scope_evidence']=[]
    if change=='conflict':
        saved['pages'][1]['units'].append(deepcopy(saved['pages'][1]['units'][0]))
        saved['pages'][1]['units'][1]['markdown']=saved['pages'][1]['units'][1]['markdown'].replace('Literal gasket','Different name')
    if change=='unaudited':saved['pages'][1]['units'][0]['source_audit']['complete']=False
    if change=='tampered_image':Path(saved['pages'][1]['units'][0]['source_image']).write_bytes(b'changed')
    if change=='foreign_doc':saved['document_id']='b'*64
    manifest.write_text(json.dumps(saved))
    if change in {'tampered_image','foreign_doc'}:
        with pytest.raises(ValueError):lookup.build(manifest,presentation.Snapshot(),doc)
    else:
        index=lookup.build(manifest,presentation.Snapshot(),doc)
        assert not lookup.lookup(index,'ZX-205',1,[0,0,90,90])[0]


def test_reference_cell_may_contain_code_and_literal_name_without_quantity_transfer():
    table=unit('| Reference | Quantity |\n|---|---|\n| AB033 Glazing bead | 2 per transom |')
    entries=list(lookup.rows(table))
    assert len(entries)==1 and entries[0]['code']=='AB033' and entries[0]['name']=='Glazing bead'
    assert not list(lookup.rows(unit('| Code | Quantity |\n|---|---|\n| AB033 | 2 per transom |')))


def test_scopes_cannot_be_invented_or_read_at_another_location():
    mapping=dict(pixel_size=[100,100],pixels_to_source=[1,0,0,1,0,0])
    claim=dict(kind='system',text='Printed system',box=[0,0,.5,.1])
    spans=[(0,dict(text='Printed system',bbox=[0,0,50,10]))]
    assert lookup.verify_scopes([claim],mapping,[0,0,100,100],spans,[])
    assert not lookup.verify_scopes([{**claim,'text':'Invented'}],mapping,[0,0,100,100],spans,[])
    assert not lookup.verify_scopes([{**claim,'box':[.5,.5,1,.6]}],mapping,[0,0,100,100],spans,[])


def test_local_callout_takes_precedence_and_count_never_changes(tmp_path):
    manifest,doc=catalog(tmp_path);index=lookup.build(manifest,presentation.Snapshot(),doc)
    rows=[dict(profile_id='ZX-205',component_name=name,profile_count='7',source_page='1') for name in ['', 'Local name']]
    traces=[dict(source_structural_box=[0,0,90,90],reasons=[]) for _ in rows]
    lookup.apply(rows,traces,index)
    assert [r['component_name'] for r in rows]==['Literal gasket','Local name']
    assert all(r['profile_count']=='7' for r in rows)


def legend_records():
    view=dict(id='v',kind='assembly_candidate',box=[0,0,100,100])
    link=dict(marker='3',code='GEN-147',name='Literal connector',row_text='3 - GEN-147 : Literal connector',
        row_box=[5,60,70,75],code_box=[12,60,30,70],name_box=[32,60,69,70],
        scope_text='Printed assembly key',scope_box=[0,0,100,100],evidence=['Scoped key explicitly identifies marker 3'])
    record=dict(category='item_marker',reading_state='exact',reading_review_reasons=[],reading_clipped=False,
        raw_code='3',text_box=[0,10,15,20],source_view_id='v',page_index=4,
        association_method='qwen_visual_leader',leader_kind='component',leader_path=[[15,15],[60,40]],
        legend_link=link,review_reasons=[])
    return view,[dict(deepcopy(record),leader_audit_pass=p) for p in ['primary','independent_audit']]


def test_local_legend_resolves_code_and_name_but_keeps_original_marker():
    view,records=legend_records();local_legend.resolve(records,view)
    assert all(r['raw_code']=='GEN-147' and r['category']=='product_code' for r in records)
    assert records[0]['legend_resolution']['marker']=='3'
    assert local_legend.name(records[0])=='Literal connector'
    assert public_contract.literal_name(records[0],[],[],view)==('Literal connector',[])


@pytest.mark.parametrize('case',['no_audit','conflict','outside_scope','no_marker','bad_code'])
def test_local_legend_does_not_guess_unverified_or_ambiguous_rows(case):
    view,records=legend_records()
    if case=='no_audit':records=records[:1]
    if case=='conflict':records[1]['legend_link']['code']='GEN-999'
    if case=='outside_scope':
        for r in records:r['legend_link']['scope_box']=[0,0,50,50]
    if case=='no_marker':
        for r in records:r['legend_link']['marker']='8'
    if case=='bad_code':
        for r in records:r['legend_link']['code']='(03)'
    local_legend.resolve(records,view)
    assert not any(r.get('legend_resolution') for r in records)


def test_same_label_two_independent_rays_are_not_a_shared_stem(tmp_path):
    from test_leader_correction import fixture
    d,v,r,_,im=fixture(tmp_path)
    paths=[[[30,20],[60,45]],[[30,20],[60,60]]]
    im=Image.new('RGB',(100,100),'white')
    for path in paths:ImageDraw.Draw(im).line([tuple(p) for p in path],fill='black')
    im.save(r['source_image'])
    for i,o in enumerate(r['records']):
        o.update(leader_path=paths[i%2],text_box=[10,15,30,25],target_point=paths[i%2][-1])
    v['crop_completeness']['annotation_proofs']=[dict(ownership_verified=True,annotation=dict(leader_points=p)) for p in paths]
    inventory=leaders.inventory(d,[v],[r])
    assert leaders.count_state(inventory)==('2','COMPLETE')
    assert len({o['label_id'] for o in inventory['observations'] if o['accepted']})==1
    assert line_evidence.conflict([[30,20],[40,30],[60,45]],[[30,20],[40,30],[60,60]],[10,15,30,25])


def test_numeric_detail_locators_provide_context_without_union(tmp_path):
    from test_grouping import saved_page
    from test_fabrication_relations import add_mark
    source=saved_page(tmp_path,['','']);saved=json.loads(source.read_text())
    saved['candidates'][0]['kind']='legend_notes'
    add_mark(saved['candidates'][0],'detail_locator','source','2','2')
    add_mark(saved['candidates'][1],'detail_locator','caption','2','Detail 2')
    source.write_text(json.dumps(saved))
    _,views,_=grouping.load_manifest(source)
    proposed,clues=refs.proposals(views,[])
    assert not proposed
    relation=next(c for c in clues if c.get('relation_type')=='detail_locator_context')
    assert relation['source_quote_verified'] and not relation['membership_established']


def test_generic_variant_labels_prevent_unsafe_merge(tmp_path):
    from test_grouping import saved_page
    from test_fabrication_relations import add_mark
    source=saved_page(tmp_path,['','']);saved=json.loads(source.read_text())
    for c,text in zip(saved['candidates'],['Left hand','Right hand']):add_mark(c,'variant','caption',text,text)
    source.write_text(json.dumps(saved));_,views,_=grouping.load_manifest(source)
    assert not refs.same_variant(views)


def test_name_only_detail_is_published_while_unlabeled_detail_is_skipped(tmp_path):
    from as_ocr import delivery
    from test_presentation import prepared,update_callout
    summary=prepared(tmp_path)
    def mutate(s4):
        s4['occurrences'][0].update(category='component_name',raw_code='Literal name only')
    update_callout(summary,mutate)
    published=delivery.publish(summary,output=tmp_path/'public')
    rows=delivery.read_csv(Path(published['root'])/'all_details.csv')
    assert len(rows)==1 and rows[0]['component_name']=='Literal name only' and not rows[0]['profile_id']
    assert rows[0]['detail_name']=='catalog_01'


def test_delivery_can_fill_name_from_another_source_page(tmp_path):
    from as_ocr import delivery,indexer
    from test_presentation import prepared,save
    # Rebind the synthetic saved graph to an actual PDF hash; all saved handoffs
    # are rehashed after rebinding, as in a real catalog run.
    source=tmp_path/'actual.pdf'
    with fitz.open() as pdf:
        pdf.new_page().insert_text((30,30),'Generic system - Roof system')
        p=pdf.new_page();p.insert_text((30,30),'ZX205 Literal gasket')
        pdf.save(source)
    doc=indexer.fingerprint(source)
    summary=prepared(tmp_path,code='ZX205');original=json.loads(summary.read_text())['selected_sources'][0]['document_id']
    root=summary.parent
    for path in root.rglob('*.json'):
        path.write_text(path.read_text().replace(original,doc),encoding='utf-8')
    for path in root.glob('*.csv'):
        path.write_text(path.read_text(encoding='utf-8-sig').replace(original,doc),encoding='utf-8-sig')
    s=json.loads(summary.read_text());s['selected_sources'][0]['source_path']=str(source)
    enrichment=Path(s['artifacts']['enrichment']);e=json.loads(enrichment.read_text())
    for ref in e['sources']:
        for n in [3,4,5]:ref[f'stage{n}_sha256']=hashlib.sha256(Path(ref[f'stage{n}_path']).read_bytes()).hexdigest()
    save(enrichment,e);save(summary,s)
    image=tmp_path/'table-source.png'
    with fitz.open(source) as pdf:pdf[1].get_pixmap().save(image)
    table=unit('| Code | Name |\n|---|---|\n| ZX205 | Literal gasket |')
    table.update(source_image=str(image),source_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    manifest=tmp_path/'ocr.json'
    save(manifest,dict(document_id=doc,source_path=str(source),pages=[
        dict(document_id=doc,source_page=1,scope_evidence=scopes(),units=[]),
        dict(document_id=doc,source_page=2,scope_evidence=scopes(),units=[table])]))
    result=delivery.publish(summary,output=tmp_path/'public',catalog_manifest=manifest)
    rows=delivery.read_csv(Path(result['root'])/'all_details.csv')
    assert len(rows)==1 and rows[0]['component_name']=='Literal gasket'
    traces=[json.loads(line) for line in (Path(result['root'])/'_internal/rows.jsonl').read_text().splitlines()]
    assert traces[0]['catalog_lookup']['evidence'][0]['source_page']==2
    assert traces[0]['name_source']=='catalog_table' and rows[0]['source_page']=='1'


def test_skipping_and_reversed_saved_results_preserve_source_order(tmp_path):
    from as_ocr import delivery
    from test_presentation import prepared,update_callout
    summary=prepared(tmp_path,code='GEN205')
    def mutate(s4):
        s4['occurrences'].append(dict(s4['occurrences'][0],id='second',raw_code='GEN206',
            source_view_id='v2',target_point=[5,105]))
    update_callout(summary,mutate)
    result=delivery.publish(summary,output=tmp_path/'public')
    rows=delivery.read_csv(Path(result['root'])/'all_details.csv')
    assert [(r['detail_name'],r['profile_id']) for r in rows]==[('catalog_01','GEN205'),('catalog_02','GEN206')]


def test_compact_ocr_records_verified_scope_and_original_table_pixels(tmp_path,monkeypatch,portable_checkpoint_io):
    from as_ocr import page_markdown as md
    from test_markdown_compact import Client,reading,audit
    monkeypatch.setattr(md,'render_text_region',lambda *a:(Image.new('RGB',(100,100),'white'),
        dict(pixel_size=[100,100],source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0])))
    layout=dict(coverage='complete',uncertainty=[],regions=[
        dict(kind='heading',box=[0,0,1,.2],rotation=0),dict(kind='table',box=[0,.2,1,1],rotation=0)],
        scope=[dict(kind='system',text='Generic system',box=[0,0,.8,.08]),
               dict(kind='section',text='Roof family',box=[0,.08,.8,.16])])
    client=Client(layout,reading('Generic system\nRoof family'),audit(),
        reading('| Reference | Description |\n|---|---|\n| GEN205 | Exact name |'),audit(),audit())
    result=md.extract_page_compact(Path('mock.pdf'),dict(id='p',document_id='doc',page_index=0),
        dict(source_rect=[0,0,100,100]),[],tmp_path/'ocr',client)
    assert lookup.scope_values(result['scope_evidence'])['section']=='roof family'
    table=result['units'][1]
    assert table['source_image_sha256']==hashlib.sha256(Path(table['source_image']).read_bytes()).hexdigest()
    assert list(lookup.rows(table))[0]['name']=='Exact name'

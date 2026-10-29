"""Presentation fixtures contain only CSV/JSON: no PDF rendering or inference."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from as_ocr import presentation as p
from as_ocr import delivery, exports, grouping
from PIL import Image


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def table(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root, *, catalog='A', filename='catalog.pdf', partial=False, code='001.20 - A',
            repeat=False, empty=False, unlinked=False, bad_code=False, description='Outer frame',
            category='frame', system='FW 50+', provider='Provider A'):
    folder = root / 'out' / catalog
    folder.mkdir(parents=True)
    doc = hashlib.sha256(catalog.encode()).hexdigest()
    page = doc + ':0'
    evidence = [dict(id='ec', document_id=doc, page_id=page, page_index='0', quote=code,
                     box='[1,1,10,5]', view_ids='["v1"]')]
    props = []
    for field, value in [('code', code), ('description', description), ('category', category)]:
        if not value:
            continue
        eid = 'e-' + field
        evidence.append(dict(evidence[0], id=eid, quote=value))
        props.append(dict(id='p-'+field, component_id='c1', field=field, value_raw=value,
            accepted='true', source_accepted='true', validation_status='SOURCE_SUPPORTED',
            conditional='false', conflict_ids='[]', evidence_ids=json.dumps([eid]),
            applies_to_component_ids='["c1"]', review_ids='[]'))
    if provider:
        evidence.append(dict(evidence[0], id='ep', quote='Manufacturer: '+provider))
    if system:
        evidence.append(dict(evidence[0], id='es', quote='System: '+system))
    details = [dict(id='d1', document_id=doc, page_id=page, page_index='0', name='Old neutral name',
                    status='PROVISIONAL', completeness='NOT_ASSESSED', view_ids='["v1"]', review_ids='[]'),
               dict(id='d2', document_id=doc, page_id=page, page_index='0', name='Other Detail',
                    status='PROVISIONAL', completeness='NOT_ASSESSED', view_ids='["v2"]', review_ids='[]')]
    occurrences, members = [], []
    for n in range(2 if repeat else 1):
        oid = 'o'+str(n+1)
        occurrences.append(dict(id=oid, document_id=doc, page_id=page, raw_code=code,
            category='product_code', reading_state='exact', review_reasons=['Uncertain classification'] if bad_code else [],
            source_view_id=None if unlinked else 'v1', target_point=None if unlinked else [5,5],
            association_method='unresolved' if unlinked else 'leader', association_evidence=['Visible leader']))
        members.append(dict(id='m'+str(n), detail_id='' if unlinked else 'd1', component_id='c1',
            occurrence_id=oid, view_id='' if unlinked else 'v1', target_point='' if unlinked else '[5,5]',
            association_status='UNRESOLVED' if unlinked else 'SUPPORTED_VIEW_IN_PROVISIONAL_DETAIL',
            context_detail_ids='["d1"]' if unlinked else '[]', review_ids='[]'))
    components = [dict(id='c1', document_id=doc, page_id=page, code_raw=code, category=category or 'unknown',
        property_ids=json.dumps([r['id'] for r in props]), scope=json.dumps({'system':system or None,'manufacturer':provider or None}),
        scope_evidence_ids=json.dumps((['es'] if system else [])+(['ep'] if provider else [])), status='REVIEW', review_ids='[]')]
    for prop in props:
        prop['relation_id']='r-'+prop['id']
    relations=[dict(id=prop['relation_id'],status='SUPPORTED',data=json.dumps({'kind':'page_paragraph' if prop['field']=='description' else 'code_label'})) for prop in props]
    rows = dict(relationships=relations,details=details, detail_components=[] if empty else members, components=components,
                properties=props, evidence=evidence, review=[])
    artifacts = {}
    for name, records in rows.items():
        path = folder/(name+'.csv')
        table(path, records, fields=list(members[0]) if name=='detail_components' else ['id'] if name=='review' else None)
        artifacts[name] = str(path)
    paths = []
    # Deliberately reverse input order: geometry, not source iteration, names Details.
    for did, y in [('d2',100),('d1',10)]:
        path = folder/(did+'.json')
        save(path, dict(stage=7, detail_id=did, source_views=[dict(id='v1' if did=='d1' else 'v2',source_box=[0,y,20,y+10])]))
        paths.append(str(path))
    s3=folder/'stage3.json'
    source_views=[dict(id='v1',document_id=doc,page_id=page,kind='assembly_candidate',structural_box=[0,0,20,20]),
                  dict(id='v2',document_id=doc,page_id=page,kind='assembly_candidate',structural_box=[0,100,20,120])]
    save(s3,dict(stage=3,views=source_views,details=[dict(d,page_index=0,view_ids=json.loads(d['view_ids'])) for d in details]))
    s5=folder/'stage5.json';save(s5,dict(stage=5,candidates=[],observations=[]))
    s4 = folder/'stage4.json'
    save(s4, dict(stage=4, upstream={'document_id':doc}, occurrences=[] if empty else occurrences))
    s6 = folder/'stage6.json'
    save(s6, dict(stage=6, sources=[dict(document_id=doc,stage3_path=str(s3),stage3_sha256=hashlib.sha256(s3.read_bytes()).hexdigest(),stage5_path=str(s5),stage5_sha256=hashlib.sha256(s5.read_bytes()).hexdigest(),stage4_path=str(s4),stage4_sha256=hashlib.sha256(s4.read_bytes()).hexdigest())]))
    artifacts.update(detail_results=paths, enrichment=str(s6))
    summary = folder/'run-summary.json'
    save(summary, dict(stage=7, run_id='export-'+catalog, export_status='PARTIAL' if partial else 'COMPLETE',
        complete=not partial, execution_status='PARTIAL' if partial else 'COMPLETE',
        source_coverage_status='PARTIAL' if partial else 'COMPLETE', acceptance_status='REVIEW',
        validation_errors=0, selected_sources=[dict(document_id=doc,source_path='/unopened/'+filename)], artifacts=artifacts))
    return summary


def publish(root, summaries):
    return p.run(summaries, output=root/'output', internal_root=root/'out')


def read(publication):
    with Path(publication['csv_path']).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        assert tuple(reader.fieldnames)==p.COLUMNS
        return list(reader)


def traces(publication):
    return [json.loads(line) for line in Path(publication['trace']).read_text().splitlines()]


def confirm_details(summary):
    path=Path(json.loads(summary.read_text())['artifacts']['details'])
    with path.open(encoding='utf-8-sig',newline='') as handle:
        records=list(csv.DictReader(handle))
    table(path,[dict(row,status='CONFIRMED',completeness='COMPLETE') for row in records])


def update_callout(summary, mutation):
    s=json.loads(summary.read_text()); path=Path(s['artifacts']['enrichment']); s6=json.loads(path.read_text())
    source=s6['sources'][0]; s4path=Path(source['stage4_path']); s4=json.loads(s4path.read_text())
    mutation(s4);save(s4path,s4);source['stage4_sha256']=hashlib.sha256(s4path.read_bytes()).hexdigest();save(path,s6)


def test_multiple_catalogs_same_filename_are_isolated(tmp_path):
    a=fixture(tmp_path,catalog='A');b=fixture(tmp_path,catalog='B',system='System Q')
    out=publish(tmp_path,[b,a])
    assert len({r['csv_path'] for r in out})==2
    assert {f.suffix for f in (tmp_path/'output').iterdir()}=={'.csv'}
    assert len(list((tmp_path/'output').iterdir()))==2
    assert out[0]['catalog_id']!=out[1]['catalog_id']
    assert all(traces(item)[0]['catalog_name']=='catalog.pdf' for item in out)
    assert {traces(item)[0]['system_name'] for item in out}=={''}  # Neutral naming does not import catalog scope.


def test_exact_five_column_schema_and_internal_identity(tmp_path):
    pub=publish(tmp_path,[fixture(tmp_path,filename='Original Catalog.pdf')])[0]
    row=read(pub)[0];trace=traces(pub)[0]
    assert list(row)==['detail_name','profile_count','profile_id','component_name','source_page']
    assert row['profile_count']=='' and row['detail_name']=='Original Catalog_01'
    assert trace['catalog_name']=='Original Catalog.pdf' and trace['row_key']==[pub['catalog_id'],'d1','o1']



def test_previous_public_schema_can_be_replaced_without_changing_internal_trace(tmp_path):
    summary=fixture(tmp_path);first=publish(tmp_path,[summary])[0]
    trace=Path(first['trace']);before=trace.read_bytes()
    # A previously published v1 CSV identifies its catalog directly.
    table(Path(first['csv_path']),[dict(catalog_id=first['catalog_id'],detail_id='d1',
        catalog_file='catalog.pdf',page_number='1')])
    previous=Path(first['csv_path']).read_bytes()
    with pytest.raises(ValueError,match='Historical schema'):publish(tmp_path,[summary])
    assert Path(first['csv_path']).read_bytes()==previous and trace.read_bytes()==before


def test_unknown_public_file_is_not_overwritten(tmp_path):
    summary=fixture(tmp_path);first=publish(tmp_path,[summary])[0];path=Path(first['csv_path'])
    changed=path.read_bytes().replace(b'001.20 - A',b'HAND EDITED');path.write_bytes(changed)
    with pytest.raises(ValueError,match='filename collision'):publish(tmp_path,[summary])
    assert path.read_bytes()==changed



def test_multiple_distinct_source_callouts_in_one_detail(tmp_path):
    summary=fixture(tmp_path)
    update_callout(summary,lambda s:s['occurrences'].append(dict(s['occurrences'][0],id='o2',raw_code='002/B',target_point=[6,5])))
    rows=read(publish(tmp_path,[summary])[0])
    assert [r['profile_id'] for r in rows[:2]]==['001.20 - A','002/B']
    assert rows[0]['detail_name']==rows[1]['detail_name']=='catalog_01'
    assert all('profile_type' not in r and 'description' not in r for r in rows)



def test_dangling_detail_is_rejected(tmp_path):
    summary=fixture(tmp_path);s=json.loads(summary.read_text());path=Path(s['artifacts']['detail_components'])
    with path.open(encoding='utf-8-sig',newline='') as handle:members=list(csv.DictReader(handle))
    members[0]['detail_id']='absent';table(path,members)
    with pytest.raises(ValueError,match='Dangling Detail'):publish(tmp_path,[summary])


def test_details_repeated_source_callouts_and_trace(tmp_path):
    pub=publish(tmp_path,[fixture(tmp_path,repeat=True)])[0];rows=read(pub);audit=traces(pub)
    assert len(rows)==3 and rows[0]==rows[1] and rows[0]['profile_id']=='001.20 - A'
    assert all(r['profile_count']=='' for r in rows)
    assert [t['row_key'][2] for t in audit]==['o1','o2','']
    assert [t['csv_row_number'] for t in audit]==[2,3,4]
    assert audit[-1]['association_status']=='NO_CONFIRMED_COMPONENTS'



def test_empty_details_and_unresolved_context_are_not_lost(tmp_path):
    pub=publish(tmp_path,[fixture(tmp_path,unlinked=True)])[0];rows=read(pub)
    assert len(rows)==2 and all(not r['profile_id'] and not r['profile_count'] for r in rows)
    assert pub['unassigned_memberships']



def test_unassigned_repeated_codes_remain_internal_without_invented_membership(tmp_path):
    pub=publish(tmp_path,[fixture(tmp_path,unlinked=True,repeat=True)])[0]
    assert len(read(pub))==2 and all(not r['profile_id'] for r in read(pub))
    assert len(pub['unassigned_memberships'])==2



def test_all_empty_details(tmp_path):
    summary=fixture(tmp_path,empty=True)
    rows=read(publish(tmp_path,[summary])[0])
    assert len(rows)==2 and all(not r['profile_id'] for r in rows)


def test_legacy_type_and_description_are_absent_from_public_csv(tmp_path):
    rows=read(publish(tmp_path,[fixture(tmp_path,category='',description='',system='')])[0])
    assert all('profile_type' not in r and 'description' not in r for r in rows)
    assert rows[0]['profile_id']=='001.20 - A'



@pytest.mark.parametrize('value',['000123','001.20 - A','00/01-A','00000000000000001234'])
def test_exact_codes_are_strings(tmp_path,value):
    assert read(publish(tmp_path,[fixture(tmp_path,code=value)])[0])[0]['profile_id']==value


@pytest.mark.parametrize('value',['=1+1','+CMD','-001','@SUM(A1)','\t=CMD',' \r=CMD','＝CMD','\ufeff=CMD','\u200b@CMD'])
def test_formula_safety_preserves_raw_in_trace(tmp_path,value):
    pub=publish(tmp_path,[fixture(tmp_path,code=value,description=value)])[0]
    assert read(pub)[0]['profile_id']=='' and traces(pub)[0]['withheld']['profile_id']==value
    assert all(not p.unsafe(v) for r in read(pub) for v in r.values())



def test_unicode_multiline_quotes_and_bom(tmp_path):
    text='"Gasket", Δ\nعربي';summary=fixture(tmp_path)
    update_callout(summary,lambda s:s['occurrences'][0].update(category='component_name',raw_code=text))
    pub=publish(tmp_path,[summary])[0]
    assert Path(pub['csv_path']).read_bytes().startswith(b'\xef\xbb\xbf')
    assert read(pub)[0]['component_name']==text



def test_metadata_code_not_promoted_even_if_stage7_accepted_it(tmp_path):
    rows=read(publish(tmp_path,[fixture(tmp_path,code='F-45',bad_code=True)])[0])
    assert all(not r['profile_id'] for r in rows)



def test_unresolved_metadata_code_stays_internal(tmp_path):
    publication=publish(tmp_path,[fixture(tmp_path,unlinked=True,code='F-45',bad_code=True)])[0]
    assert len(read(publication))==2
    manifest=json.loads(Path(publication['trace']).with_name('manifest.json').read_text())
    assert len(manifest['unassigned_memberships'])==1


def test_saved_stage7_code_flow_and_metadata_counterexample(tmp_path):
    root=Path(__file__).resolve().parents[1]
    saved=root/'out/1.)FW 50+SG'
    connections=list((saved/'3d connections').glob('*/stage_7/partial/*/run-summary.json'))
    gasket=list((saved/'3d gasket pg 1').glob('*/stage_7/partial/*/run-summary.json'))
    if not connections or not gasket:
        pytest.skip('Saved integration snapshots are unavailable in this checkout')
    _,_,_,rows,traces,_,_,_=p.project(connections[0])
    related=[(r,t) for r,t in zip(rows,traces) if r['profile_id'] in ('298257','298019')]
    assert len(related)==1 and related[0][0]['profile_id']=='298257'
    assert related[0][0]['detail_name'].startswith('Detail P0001') and related[0][1]['association_status']=='SUPPORTED'
    assert p.project(connections[0])[6]  # Unassigned observations remain internal.
    assert all(t['review_status']=='REVIEW' for r,t in zip(rows,traces) if r['profile_id'] in ('298257','298019'))
    assert len({tuple(t['row_key']) for t in traces})==len(traces)
    _,_,_,gasket_rows,_,_,_,_=p.project(gasket[0])
    assert len(gasket_rows)==4 and all(r['profile_id']!='F-45' for r in gasket_rows)
    publications=p.run([connections[0],gasket[0]], output=tmp_path/'output',
                       internal_root=tmp_path/'internal')
    assert len(publications)==2 and len(list((tmp_path/'output').glob('*.csv')))==2
    public_rows=[read(item) for item in publications]
    assert all(row['profile_id']!='298019' for table in public_rows for row in table)
    assert all(row['profile_id']!='F-45' for table in public_rows for row in table)


def test_legacy_descriptions_never_form_public_values(tmp_path):
    rows=read(publish(tmp_path,[fixture(tmp_path,description='Legacy paragraph text')])[0])
    assert all('description' not in r for r in rows)



def test_unsafe_catalog_filename_and_system(tmp_path):
    publication=publish(tmp_path,[fixture(tmp_path,filename='=catalog.pdf',system='@BAD')])[0]
    row=read(publication)[0]
    internal=traces(publication)[0]
    assert internal['catalog_name']=='=catalog.pdf' and internal['system_name']==''  # Raw internal provenance is preserved.
    assert all(not p.unsafe(value) for value in row.values())


def complete_counts(summary):
    saved=json.loads(summary.read_text());doc=saved['selected_sources'][0]['document_id']
    inventory={did:dict(document_id=doc,page_id=doc+':0',page_index=0,observed_count=0,profiles=[],component_types=[],coverage='complete',potentially_incomplete=False,no_profile_reason='No structural sections visible') for did in ('d1','d2')}
    path=summary.parent/'inventory.json';save(path,inventory);saved['artifacts']['profile_observations']=str(path);save(summary,saved)


def test_failed_publication_preserves_previous_csv(tmp_path,monkeypatch):
    summary=fixture(tmp_path);pub=publish(tmp_path,[summary])[0];path=Path(pub['csv_path']);before=path.read_bytes()
    update_callout(summary,lambda s:s['occurrences'][0].update(raw_code='UPDATED-002'))
    def fail(*args):raise OSError('simulated publication failure')
    monkeypatch.setattr(p.os,'replace',fail)
    with pytest.raises(OSError):publish(tmp_path,[summary])
    assert path.read_bytes()==before and not list((tmp_path/'out').glob('.present-*'))



def test_partial_never_overwrites_complete(tmp_path,monkeypatch):
    summary=fixture(tmp_path);original_project=p.project
    # Exercise transaction naming independently of leader verification. This
    # fixture simulates a previously accepted complete publication.
    def complete(*args,**kwargs):
        parts=list(original_project(*args,**kwargs));parts[-1]=False;return tuple(parts)
    monkeypatch.setattr(p,'project',complete);pub=publish(tmp_path,[summary])[0]
    path=Path(pub['csv_path']);before=path.read_bytes();monkeypatch.setattr(p,'project',original_project)
    partial=publish(tmp_path,[summary])[0]
    assert partial['csv_path'].endswith('__PARTIAL.csv') and Path(partial['csv_path'])!=path
    assert path.read_bytes()==before and len(list((tmp_path/'output').iterdir()))==2



def test_stable_bytes_trace_and_existing_artifacts_unchanged(tmp_path):
    summary=fixture(tmp_path,repeat=True)
    before={p_:p_.read_bytes() for p_ in summary.parent.iterdir() if p_.is_file()}
    first=publish(tmp_path,[summary])[0];data=Path(first['csv_path']).read_bytes()
    second=publish(tmp_path,[summary])[0]
    assert first==second and Path(second['csv_path']).read_bytes()==data
    assert all(path.read_bytes()==value for path,value in before.items())
    assert not list((tmp_path/'out').glob('.present-*'))


def test_duplicate_occurrence_rows_do_not_duplicate_output(tmp_path):
    summary=fixture(tmp_path);s=json.loads(summary.read_text());path=Path(s['artifacts']['detail_components'])
    with path.open(encoding='utf-8-sig',newline='') as f:members=list(csv.DictReader(f))
    members.append(dict(members[0],id='extra-observation'));table(path,members)
    publication=publish(tmp_path,[summary])[0]
    assert len(read(publication))==2
    trace=json.loads(Path(publication['trace']).read_text().splitlines()[0])
    assert len(trace['membership_ids'])==2


def test_changed_provenance_rejected_without_publication(tmp_path):
    summary=fixture(tmp_path)
    (summary.parent/'stage4.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):publish(tmp_path,[summary])
    assert not (tmp_path/'output').exists()


def test_wrong_target_does_not_upgrade_context(tmp_path):
    summary=fixture(tmp_path)
    update_callout(summary,lambda data:data['occurrences'][0].update(source_view_id=None,target_point=None))
    publication=publish(tmp_path,[summary])[0];rows=read(publication)
    assert rows[0]['profile_id']==''
    assert traces(publication)[0]['association_status']=='NO_CONFIRMED_COMPONENTS'


def test_missing_callout_provenance_is_partial(tmp_path):
    summary=fixture(tmp_path);(summary.parent/'stage4.json').unlink()
    publication=publish(tmp_path,[summary])[0]
    assert publication['publication_status']=='PARTIAL'
    assert all(not r['profile_id'] for r in read(publication))


def test_one_snapshot_per_catalog_and_separate_roots(tmp_path):
    summary=fixture(tmp_path)
    with pytest.raises(ValueError,match='one snapshot'):publish(tmp_path,[summary,summary])
    with pytest.raises(ValueError,match='separate'):p.run([summary],output=tmp_path/'out/output',internal_root=tmp_path/'out')


def test_legacy_schema_unchanged():
    from as_ocr import exports
    assert len(exports.TABLES)==8
    assert exports.TABLES['details']==['id','document_id','page_id','page_index','name','status','grouping_status','completeness','view_ids','composites','review_ids']
    assert 'catalog_details' not in exports.TABLES


def test_cli_returns_partial_status(tmp_path):
    summary=fixture(tmp_path,partial=True)
    assert p.main([str(summary),'--output',str(tmp_path/'output'),'--internal-root',str(tmp_path/'out')])==1


def test_original_source_page_is_not_detail_ordinal(tmp_path):
    summary=fixture(tmp_path);art=json.loads(summary.read_text())['artifacts'];path=Path(art['details'])
    with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    table(path,[dict(r,page_index='109') for r in rows])
    link=Path(art['enrichment']);s6=json.loads(link.read_text());source=s6['sources'][0]
    group=Path(source['stage3_path']);s3=json.loads(group.read_text())
    for d in s3['details']:d['page_index']=109
    for v in s3['views']:v['page_index']=109
    save(group,s3);source['stage3_sha256']=hashlib.sha256(group.read_bytes()).hexdigest();save(link,s6)
    result=read(publish(tmp_path,[summary])[0])
    assert all(r['source_page']=='110' and r['detail_name'] in {'catalog_01','catalog_02'} for r in result)



def test_detail_only_publication_keeps_unassigned_evidence_internal(tmp_path):
    summary=fixture(tmp_path, unlinked=True)
    publication=p.run([summary],output=tmp_path/'output',internal_root=tmp_path/'out',details_only=True)[0]
    assert len(read(publication))==2 and all(r['source_page']=='1' for r in read(publication))
    assert all(not t['row_key'][2] for t in traces(publication))
    assert publication['unassigned_memberships']


# Detail folder delivery and source-image publication contracts.
def prepared(tmp_path, **kwargs):
    from as_ocr import leaders
    summary=fixture(tmp_path,**kwargs);saved=json.loads(summary.read_text());folder=summary.parent
    s3=folder/'stage3.json';s2=folder/'stage2.json';s5=folder/'stage5.json'
    save(s2,dict(stage=2,page_index=0,candidates=[],page_analysis={'coverage':'incomplete','review_reasons':['Unresolved boundary']}))
    grouped=json.loads(s3.read_text());grouped['upstream']={'manifest_path':str(s2)}
    for v in grouped['views']:
        v['crop_path']=str(folder/'image.png');v['crop_geometry']=dict(source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0])
    save(s3,grouped);component=json.loads(s5.read_text());component['state']='PARTIAL';save(s5,component)
    Image.new('RGB',(123,57),'white').save(folder/'image.png')
    results=[]
    for old in saved['artifacts']['detail_results']:
        result=json.loads(Path(old).read_text());did=result['detail_id'];dest=folder/did;dest.mkdir()
        result_path=dest/'result.json';save(result_path,result);(dest/'image.png').write_bytes((folder/'image.png').read_bytes())
        d=next(d for d in grouped['details'] if d['id']==did)
        leaders.write(dest,dict(contract=leaders.CONTRACT,document_id=d['document_id'],page_id=d['page_id'],page_index=0,
            detail_id=did,complete=False,coverage='incomplete',observations=[],issues=['Fixture has no verified leaders']),d,grouped['views'])
        results.append(str(result_path))
    saved['artifacts']['detail_results']=results;save(summary,saved)
    s6path=Path(saved['artifacts']['enrichment']);s6=json.loads(s6path.read_text())
    s6['sources'][0].update(stage3_sha256=hashlib.sha256(s3.read_bytes()).hexdigest(),stage5_sha256=hashlib.sha256(s5.read_bytes()).hexdigest())
    save(s6path,s6);return summary


def test_delivery_preserves_occurrences_skips_unlabeled_details_and_repeat_merge(tmp_path):
    summary = prepared(tmp_path, repeat=True)
    document = json.loads(summary.read_text())['selected_sources'][0]['document_id']
    delivery.review_stage(tmp_path/'public'/delivery.identifier('catalog.pdf', document), 4)
    result = delivery.publish(summary, output=tmp_path/'public')
    root = Path(result['root'])
    rows = delivery.read_csv(root/'all_details.csv')
    assert len(rows) == 2
    assert rows[0] == rows[1]  # Distinct source occurrences must survive.
    skipped=json.loads((root/'_internal/publication.json').read_text())['skipped_details']
    assert len(skipped)==1 and skipped[0]['reason']=='NO_SUPPORTED_COMPONENT_CODE_OR_NAME'
    original = (root/'all_details.csv').read_bytes()
    assert delivery.merge(root) == 2
    assert (root/'all_details.csv').read_bytes() == original
    delivery.publish(summary, output=tmp_path/'public')
    assert (root/'all_details.csv').read_bytes() == original
    for path in (root/'details').glob('*/image.png'):
        with Image.open(path) as image:
            assert image.size == (123, 57)
    assert len(list((root/'stages').glob('*/review.md'))) == 7
    assert len(list((root/'details').rglob('leader_observations.json')))==1
    assert 'incomplete' in (root/'stages/stage_02/review.md').read_text()


def test_merge_rejects_missing_foreign_or_changed_detail_files(tmp_path):
    root = Path(delivery.publish(prepared(tmp_path), output=tmp_path/'public')['root'])
    path = next((root/'details').glob('*/detail.csv'))
    original = path.read_bytes()
    path.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='changed'):
        delivery.merge(root)
    path.write_bytes(original)
    manifest_path = root/'_internal/publication.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['details'][path.parent.name]['document_id'] = 'foreign'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Foreign'):
        delivery.merge(root)


def test_final_image_uses_original_crop_pixels_instead_of_review_thumbnail(tmp_path):
    original = tmp_path/'crop.png'
    thumbnail = tmp_path/'review.png'
    Image.new('RGB', (987, 1012), 'red').save(original)
    Image.new('RGB', (512, 624), 'white').save(thumbnail)
    composites = [{'path':str(thumbnail), 'panels':[
        {'source_crop':str(original),'source_crop_sha256':grouping.file_digest(original)}]}]
    exports.detail_image(composites, tmp_path/'image.png')
    with Image.open(tmp_path/'image.png') as actual, Image.open(original) as expected:
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()


def test_confirmed_empty_source_has_header_only_merged_csv(tmp_path):
    result = delivery.publish_empty(tmp_path, 'source.pdf', 'a'*64, complete=True)
    assert result['status'] == 'NO_DETAIL'
    assert delivery.read_csv(tmp_path/'all_details.csv') == []
    assert delivery.merge(tmp_path) == 0

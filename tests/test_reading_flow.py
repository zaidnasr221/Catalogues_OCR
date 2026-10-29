"""Saved JSON/SQLite replay and CPU contracts: PDF opening and HTTP are forbidden."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sqlite3

import pytest

from as_ocr import callouts, components, enrichment, exports, grouping, presentation, provenance
from as_ocr.qwen import ModelConfig
from test_presentation import fixture, confirm_details, publish, read, save, table, traces, update_callout

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT/'out1/1.)FW 50+SG/3d gasket pg 1/gasket-controlled-20261026T070559Z'


@pytest.fixture(autouse=True)
def no_pdf_or_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('No PDF processing or network is allowed')
    monkeypatch.setattr(callouts.fitz, 'open', denied)
    monkeypatch.setattr('urllib.request.urlopen', denied)


def record(code='244501', view='crop1', **changes):
    value=dict(document_id='doc', page_id='page', page_index=0, category='product_code',
        raw_code=code, normalized_code=code, text_box=[1,1,20,10], reading_state='exact',
        readings=[dict(raw_text=code,method='indexed_native',span_id='span1')],
        reading_method='indexed_native', reading_evidence=['Saved source label'],
        native_span_ids=['span1'], observed_in_view_ids=[view], source_view_id='v1',
        detail_ids=['d1'], target_point=[25,25], leader_path=None,
        association_method='qwen_explicit_location', association_evidence=['Visible label on component'],
        vector_evidence=[], proposed_geometry={'leader_state':'missing'},
        inherited_review_reasons=[], review_reasons=[])
    value.update(changes)
    return value


def reconcile(*records):
    return callouts.reconcile(records, {'document_id':'doc'})


def test_clipped_reading_preserves_full_code_and_raw_conflicting_evidence():
    full=record(); clipped=record('24450', 'crop2', reading_clipped=True,
        source_view_id=None,target_point=None,association_method='unresolved')
    merged,=reconcile(full,clipped)
    assert merged['raw_code']=='244501' and provenance.supported_product_reading(merged)
    assert {r['raw_text'] for r in merged['readings']}=={'244501','24450'}
    assert merged['source_view_id']=='v1' and merged['target_point']==[25,25]
    assert full==record()  # Input evidence remains immutable.


def test_real_reading_conflict_is_not_resolved_by_third_agreeing_crop():
    merged,=reconcile(record(),record('244502','crop2'),record(view='crop3'))
    assert merged['raw_code'] is None and merged['reading_state']=='conflicting'
    assert not provenance.supported_product_reading(merged)


@pytest.mark.parametrize('change', [dict(target_point=[40,40]),dict(source_view_id='v2'),
    dict(source_view_id=None,target_point=None,review_reasons=['Indicated endpoint has competing source views'])])
def test_real_association_conflict_preserves_reading_but_not_membership(change):
    merged,=reconcile(record(),record(view='crop2',**change))
    assert merged['source_view_id'] is None and merged['target_point'] is None
    assert provenance.supported_product_reading(merged)


@pytest.mark.parametrize('other', [record(view='crop2',native_span_ids=['span2']),
    record(),record(view='crop2',text_box=[30,1,49,10],native_span_ids=['span2'])])
def test_repeated_code_never_establishes_occurrence_identity(other):
    merged=reconcile(record(),other)
    expected=1 if other==record() else 2
    assert len(merged)==expected and len({r['id'] for r in merged})==expected


@pytest.mark.parametrize('reason,accepted', [('Component association unresolved: missing',True),
    ('Ambiguous duplicate occurrence identity',True),('Unknown reading concern',False),
    ('Drawing number or component code',False),('One character is uncertain',False)])
def test_reading_acceptance_is_independent_but_conservative(reason,accepted):
    item=record(review_reasons=[reason])
    assert provenance.supported_product_reading(item)==accepted


def test_metadata_inside_assembly_crop_and_dates_cannot_be_products():
    assert provenance.metadata_source(record(), ['assembly_candidate'], [[0,0,25,15]])
    assert not provenance.supported_product_reading(record(), ['assembly_candidate'], [[0,0,25,15]])
    assert callouts.non_code_category('27.01.2006')=='other'


def test_legacy_without_geometry_does_not_infer_count_from_repeated_ids(tmp_path):
    summary=fixture(tmp_path,repeat=True);confirm_details(summary)
    rows=read(publish(tmp_path,[summary])[0])
    assert rows[0]==rows[1] and rows[0]['profile_count']==''
    assert rows[2]['profile_count']==''


@pytest.mark.parametrize('options', [dict(partial=True),dict(unlinked=True)])
def test_legacy_without_leaders_withholds_count(tmp_path,options):
    summary=fixture(tmp_path,**options);confirm_details(summary)
    rows=read(publish(tmp_path,[summary])[0])
    assert rows[0]['profile_count']==''


def test_supported_name_is_not_replaced_with_default_classification(tmp_path):
    summary=fixture(tmp_path,category='Dichtungsprofil',description='')
    rows=read(publish(tmp_path,[summary])[0])
    assert 'profile_type' not in rows[0] and 'description' not in rows[0]


def test_supported_view_without_detail_stays_unassigned(tmp_path):
    summary=fixture(tmp_path,unlinked=True)
    path=Path(json.loads(summary.read_text())['artifacts']['detail_components'])
    with path.open(encoding='utf-8-sig',newline='') as handle:
        members=list(csv.DictReader(handle))
    members[0].update(association_status='SUPPORTED_VIEW_ONLY',view_id='v3',target_point='[5,5]')
    table(path,members)
    update_callout(summary,lambda data:data['occurrences'][0].update(
        source_view_id='v3',target_point=[5,5],association_method='leader'))
    rows=read(publish(tmp_path,[summary])[0])
    assert all(not r['profile_id'] for r in rows)


def saved_readings():
    s3path=next(SAVED.glob('stage_3/groupings/*/manifest.json'))
    s4path=next(SAVED.glob('stage_4/callouts/*/manifest.json'))
    s2,views,details,s4,upstream=components.load_inputs(s3path,s4path)
    metadata=[v['box'] for v in views if v['kind']=='document_metadata']
    kinds={v['id']:v['kind'] for v in views}
    raw=[]
    for path in sorted(s4path.parent.glob('*-view.json')):
        for item in json.loads(path.read_text())['records']:
            if provenance.metadata_source(item,[kinds[v] for v in item['observed_in_view_ids']],metadata):
                item['source_role']='document_metadata'
                if item['category'] in {'product_code','uncertain_code'}:
                    item['category']='other'
            raw.append(item)
    return s3path,s4path,views,details,s4,callouts.reconcile(raw,s4['upstream'])


def test_saved_gasket_codes_survive_reconciliation_and_stage5():
    _,_,views,details,s4,records=saved_readings()
    accepted=[r for r in records if provenance.supported_product_reading(r)]
    assert {'244501','244431','244429','244962'} <= {r['raw_code'] for r in accepted}
    drawings=[r for r in accepted if r['source_view_id']]
    assert [r['raw_code'] for r in drawings].count('244501')==2
    assert {'244431','244429'} <= {r['raw_code'] for r in drawings}
    for item in accepted:
        view=next(v for v in views if v['id'] in item['observed_in_view_ids'])
        candidate,fields=components.baseline(item,view,details,s4)
        assert fields[0]['accepted'] and fields[0]['value_raw']==item['raw_code']
        if not item['source_view_id']:
            assert candidate['memberships'][0]['status']=='UNRESOLVED'
    assert not any(r['raw_code']=='K17353' for r in accepted)


def replay_saved_gasket(destination, public_output=None):
    """Replay deterministic stages only; preserve incomplete saved inference coverage."""
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    s3path,s4path,views,details,s4,records=saved_readings()
    original5=next(SAVED.glob('stage_5/components/*/manifest.json'))
    s5=json.loads(original5.read_text())
    note={'mode':'CPU_SAVED_EVIDENCE_REPLAY','model_requests':0,
          'source_stage4':str(s4path),'source_stage5':str(original5),
          'limitations':'No new extraction; saved timeouts and incomplete coverage remain.'}
    s4.update(run_id='cpu-reading-replay',occurrences=[r for r in records if r['category'] in {'product_code','uncertain_code'}],
              markers=[r for r in records if r['category'] not in {'product_code','uncertain_code'}],cpu_replay=note)
    new4=destination/'stage_4/manifest.json';save(new4,s4)
    _,views,details,s4,upstream=components.load_inputs(s3path,new4)
    candidates=[];fields=[]
    for item in s4['occurrences']:
        for vid in item['observed_in_view_ids']:
            view=next(v for v in views if v['id']==vid)
            candidate,observations=components.baseline(item,view,details,s4)
            candidates.append(candidate);fields.extend(observations)
    candidates,fields,identities=components.summarize(candidates,fields,[])
    s5.update(run_id='cpu-reading-replay',component_version=components.VERSION,upstream=upstream,
              candidates=candidates,observations=fields,identities=identities,scope_observations=[],cpu_replay=note)
    review=destination/'stage_5/review.md';review.parent.mkdir(parents=True,exist_ok=True)
    components.write_review(review,s5);s5['review_path']=str(review)
    new5=destination/'stage_5/manifest.json';save(new5,s5)
    dbpath=destination/'stage_1/index.sqlite3';dbpath.parent.mkdir()
    source=next(SAVED.glob('stage_1/index.sqlite3'))
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as old, sqlite3.connect(dbpath) as new:
        old.backup(new)
    s6=enrichment.run([new5],destination/'stage_6',index_path=dbpath,
        model=ModelConfig('http://mock/v1','mock','cpu@replay',max_requests=1,retries=0))
    s7=exports.run(s6['manifest'],destination/'stage_7',index_path=dbpath)
    publications=presentation.run([s7['artifacts']['summary']],
        output=public_output or destination.parent/(destination.name+'-csv'),internal_root=destination)
    save(destination/'replay-summary.json',dict(note,stage6=s6,stage7=s7,publications=publications))
    return publications[0]


def test_saved_evidence_through_stages4_to7_and_seven_column_csv(tmp_path):
    publication=replay_saved_gasket(tmp_path/'internal')
    rows=read(publication)
    assert {'244501','244431','244429'} <= {r['profile_id'] for r in rows}
    assert all(r['profile_count']=='' and 'profile_type' not in r and 'description' not in r for r in rows)
    assigned=[r for r in rows if r['detail_name'] and r['profile_id']]
    assert len(assigned)==4 and [r['profile_id'] for r in assigned].count('244501')==2
    assert not any(r['profile_id']=='244962' for r in rows)  # Unassigned source evidence remains internal.
    assert not any(r['profile_id'] in {'K17353','F45','F-45','27.01.2006'} for r in rows)
    assert len({tuple(t['row_key']) for t in traces(publication)})==len(rows)

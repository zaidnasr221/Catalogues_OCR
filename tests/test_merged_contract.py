"""User-approved contract and LangGraph scheduling; CPU only, no Qwen requests."""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import get_context
import threading
import time

import pytest

from as_ocr import leaders, public_contract, pipeline
from as_ocr.graph import page_results
from test_leader_correction import fixture
from PIL import Image, ImageDraw


def test_missing_detection_leader_exports_extracted_count_with_review(tmp_path):
    d,v,r,vec,im=fixture(tmp_path)
    r['records']=[r['records'][0],r['records'][2]]
    record=leaders.inventory(d,[v],[r],vec)
    assert leaders.count_state(record)==('1','OBSERVED_PARTIAL')
    assert not record['complete'] and record['issues']


@pytest.mark.parametrize('category,text', [('component_name','EPDM Gasket'), ('product_code','001')])
def test_single_pass_name_or_code_counts_without_complete_census(tmp_path,category,text):
    d,v,r,vec,im=fixture(tmp_path)
    r['records']=[dict(r['records'][0],category=category,raw_code=text)]
    record=leaders.inventory(d,[v],[r],vec)
    assert leaders.count_state(record)==('1','OBSERVED_PARTIAL')


def test_ten_distinct_leaders_with_names_or_repeated_codes_count_ten(tmp_path):
    d,v,r,vec,im=fixture(tmp_path)
    v['structural_box']=[55,5,85,95]
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    paths=[[[30,10+8*i],[60,10+8*i]] for i in range(10)]
    for path in paths:draw.line([tuple(p) for p in path],fill='black',width=1)
    image.save(r['source_image'])
    template=r['records'][0];r['records']=[]
    for pass_name in ['primary','independent_audit']:
        for i,path in enumerate(paths):
            r['records'].append(dict(template,leader_path=path,target_point=path[-1],
                text_box=[10,path[0][1]-3,30,path[0][1]+3],leader_audit_pass=pass_name,
                category='component_name' if i%2 else 'product_code',raw_code='Gasket' if i%2 else '001'))
    v['crop_completeness']['annotation_proofs']=[dict(ownership_verified=True,annotation=dict(leader_points=p)) for p in paths]
    vectors=[(i,dict(items=[['l',*p]])) for i,p in enumerate(paths)]
    record=leaders.inventory(d,[v],[r],vectors)
    assert leaders.count_state(record)==('10','COMPLETE')


def test_graph_overlaps_jobs_with_one_shared_cap_and_preserves_all_results():
    lock=threading.Lock();active=0;peak=0
    def work(task):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        time.sleep(.02)
        with lock:active-=1
        return task*2
    with ThreadPoolExecutor(max_workers=32) as pool:
        results=list(page_results(pool,work,range(40),32,lambda:None))
    assert 1<peak<=32
    assert sorted(results)==[(i,i*2) for i in range(40)]


def square(value):
    return value*value


def test_graph_runs_actual_spawned_processes():
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        assert sorted(page_results(pool,square,[2,3,4],2,lambda:None))==[(2,4),(3,9),(4,16)]


def test_graph_failure_cancels_workload():
    cancelled=[]
    def fail(task):raise ValueError('failed page')
    with ThreadPoolExecutor(max_workers=2) as pool, pytest.raises(ValueError,match='failed page'):
        list(page_results(pool,fail,[1],2,lambda:cancelled.append(True)))
    assert cancelled==[True]


def test_five_column_contract_and_default_http_cap():
    assert public_contract.COLUMNS==('detail_name','profile_count','profile_id','component_name','source_page')
    assert pipeline._parser().parse_args(['example.pdf']).max_in_flight==32

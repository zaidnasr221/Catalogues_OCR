"""Focused local/mock HTTP checks; no PDFs, external services or GPU calls."""
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from multiprocessing import get_context
from pathlib import Path
import threading
import time

import httpx
from PIL import Image
import pytest

from as_ocr import async_inference as ai, performance, progress
from as_ocr.qwen import ModelConfig, ModelError, QwenClient, capture_diagnostics


def envelope(model='fixture',finish='stop',content='{"code":"00042"}'):
    return dict(model=model,choices=[dict(finish_reason=finish,message=dict(content=content))],
                usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15))


@contextmanager
def server(replies=None,delay=.03,ocr=False):
    state=dict(active=0,peak=0,requests=0,connections=set())
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with lock:
                state['active']+=1;state['peak']=max(state['peak'],state['active'])
                n=state['requests'];state['requests']+=1;state['connections'].add(self.client_address)
            status,payload,pause=(replies[min(n,len(replies)-1)] if replies else (200,envelope(),delay))
            if ocr:
                fields=request['response_format']['json_schema']['schema']['properties']
                content=(dict(coverage='complete',uncertainty=[],regions=[dict(kind='prose',box=[0,0,1,1],rotation=0)]) if 'regions' in fields else
                    dict(markdown='32496',complete=True,unreadable=[]) if 'markdown' in fields else
                    dict(complete=True,missing_regions=[],issues=[],unsupported_text=[]))
                payload=envelope(content=json.dumps(content))
            try:
                time.sleep(pause);raw=json.dumps(payload).encode()
                self.send_response(status);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass
            finally:
                with lock:state['active']-=1
        def log_message(self,*args):pass
    instance=ThreadingHTTPServer(('127.0.0.1',0),Handler);instance.daemon_threads=True
    thread=threading.Thread(target=instance.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{instance.server_port}/v1',state
    finally:instance.shutdown();instance.server_close();thread.join(2)


def model(endpoint,**kwargs):
    return replace(ModelConfig(endpoint=endpoint,model='fixture',timeout=2,retries=0,max_requests=4),**kwargs)


def read(client,directory,*,stage=4,page=7):
    progress.start(stage,page,'Fixture')
    with capture_diagnostics(client,directory,stage=stage,subject_id='fixture-read'):
        raw=client.request_json(Image.new('RGB',(64,64),'white'),'read',response_schema={'type':'object'})
        value=json.loads(raw)
        if value!={'code':'00042'}:raise ValueError('Local fixture contract rejected')
    return value


def _process_job(task):
    endpoint,directory,ordinal,*stage=task
    return read(QwenClient(model(endpoint)),Path(directory)/str(ordinal),page=ordinal,stage=stage[0] if stage else 4)


@pytest.mark.parametrize('limit',[1,2,4])
def test_shared_limit_including_spawned_processes(tmp_path,limit):
    with server(delay=.08) as (endpoint,state),ai.Workload(limit,tmp_path/'perf') as workload:
        with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn'),**ai.worker_options()) as pool:
            tasks=[(endpoint,str(tmp_path/'diagnostics'),n) for n in range(1,9)]
            results=[f.result() for _,f in performance.bounded_results(pool,_process_job,tasks,4)]
        assert all(x=={'code':'00042'} for x in results)
        assert 1<=state['peak']<=limit
        if limit>1:assert state['peak']>1
        assert workload.state.limit.active.value==0
        assert workload.state.limit.peak.value<=limit
    events=[json.loads(line) for p in (tmp_path/'perf').glob('*.jsonl') for line in p.read_text().splitlines()]
    assert len(events)==8
    assert all(e['stage']==4 and 1<=e['page']<=8 and e['purpose']=='fixture-read' for e in events)
    assert all(e['local_validation_status']=='PASSED' and e['max_in_flight']==limit for e in events)


def test_async_pool_real_overlap_and_event_loop_responsive():
    async def check():
        limit=ai.SharedLimit(2);active=0;peak=0
        async def handle(request):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            await asyncio.sleep(.03)
            active-=1
            return httpx.Response(200,json=envelope())
        pool=ai.AsyncHTTPPool(limit,transport=httpx.MockTransport(handle))
        ticks=[]
        async def ticker():
            for _ in range(5):await asyncio.sleep(.005);ticks.append(1)
        try:
            await asyncio.gather(ticker(),*(pool.request('http://fixture/v1/chat/completions',b'{}',{},1,{}) for _ in range(4)))
            assert peak==2 and len(ticks)==5
        finally:await pool.aclose()
    asyncio.run(check())


def test_target_32_overlaps_and_applies_backpressure():
    async def check():
        limit=ai.SharedLimit(32);active=0;peak=0
        async def handle(request):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            await asyncio.sleep(.05)
            active-=1
            return httpx.Response(200,json=envelope())
        pool=ai.AsyncHTTPPool(limit,transport=httpx.MockTransport(handle))
        try:
            await asyncio.gather(*(pool.request('http://fixture/v1/chat/completions',b'{}',{},2,{}) for _ in range(40)))
            assert peak==32 and limit.peak.value==32 and limit.active.value==0
        finally:await pool.aclose()
    asyncio.run(check())


@pytest.mark.parametrize('maximum',[0,33,True,1.5])
def test_invalid_target_concurrency(maximum):
    with pytest.raises(ValueError):ai.SharedLimit(maximum)


def test_cancel_waiting_and_running_requests_releases_permits():
    async def check():
        limit=ai.SharedLimit(1);entered=asyncio.Event()
        async def handle(request):
            entered.set();await asyncio.sleep(20);return httpx.Response(200,json=envelope())
        pool=ai.AsyncHTTPPool(limit,transport=httpx.MockTransport(handle))
        running=asyncio.create_task(pool.request('http://fixture',b'{}',{},30,{}))
        await entered.wait()
        waiting=asyncio.create_task(pool.request('http://fixture',b'{}',{},30,{}))
        await asyncio.sleep(.01);waiting.cancel();running.cancel()
        result=await asyncio.gather(waiting,running,return_exceptions=True)
        assert all(isinstance(x,asyncio.CancelledError) for x in result)
        assert limit.active.value==0 and limit.permits.acquire(False)
        limit.permits.release();await pool.aclose()
    asyncio.run(check())


def test_keepalive_reused_and_sync_interface_preserved(tmp_path):
    with server(delay=0) as (endpoint,state),ai.Workload(2,tmp_path/'perf'):
        client=QwenClient(model(endpoint))
        assert read(client,tmp_path/'one')==read(client,tmp_path/'two')
        assert client.requests==2 and len(state['connections'])==1


@pytest.mark.parametrize('status,expected',[(400,1),(408,1),(429,2),(500,2),(504,2)])
def test_http_retry_policy_preserved(tmp_path,status,expected):
    with server([(status,{},0),(200,envelope(),0)]) as (endpoint,state),ai.Workload(1,tmp_path/'perf'):
        client=QwenClient(model(endpoint,retries=1,max_requests=2))
        if expected==1:
            with pytest.raises(ModelError):read(client,tmp_path/'diag')
        else:assert read(client,tmp_path/'diag')=={'code':'00042'}
        assert client.requests==state['requests']==expected


def test_timeout_retry_and_classification(tmp_path):
    # Allow Windows connection/timer overhead while still forcing the first timeout.
    windows=__import__('os').name=='nt'
    with server([(200,envelope(),.6 if windows else .1),(200,envelope(),0)]) as (endpoint,state),ai.Workload(1,tmp_path/'perf') as workload:
        client=QwenClient(model(endpoint,timeout=.25 if windows else .025,retries=1,max_requests=2))
        assert read(client,tmp_path/'diag')=={'code':'00042'}
        assert client.requests==2 and client.request_diagnostics[0]['timeout_classification']
        assert workload.state.limit.active.value==0


@pytest.mark.parametrize('payload,match',[(envelope(model='wrong'),'differs'),(envelope(finish='length'),'truncated'),([], 'Malformed')])
def test_envelope_rejections_do_not_semantically_retry(tmp_path,payload,match):
    with server([(200,payload,0)]) as (endpoint,state),ai.Workload(2,tmp_path/'perf'):
        client=QwenClient(model(endpoint,retries=2))
        with pytest.raises(ModelError,match=match):read(client,tmp_path/'diag')
        assert client.requests==state['requests']==1


def test_local_validation_and_failed_content_redaction(tmp_path):
    with server([(200,envelope(content='{"bad":"private-api-key"}'),0)]) as (endpoint,_),ai.Workload(1,tmp_path/'perf'):
        with pytest.raises(ValueError):read(QwenClient(model(endpoint,api_key='private-api-key')),tmp_path/'diag')
    events=[json.loads(p.read_text()) for p in (tmp_path/'diag').glob('*.json')]
    assert events[0]['response_status']=='local_validation_failed'
    assert 'private-api-key' not in ''.join(p.read_text() for p in tmp_path.rglob('*.txt'))
    perf=[json.loads(line) for p in (tmp_path/'perf').glob('*.jsonl') for line in p.read_text().splitlines()]
    assert perf[0]['local_validation_status']=='FAILED'


def test_total_stage2_budget_shared_by_processes(tmp_path):
    with server(delay=0) as (endpoint,state),ai.Workload(2,tmp_path/'perf'):
        ai.set_stage2_budget(1)
        read(QwenClient(model(endpoint)),tmp_path/'first',stage=2)
        rejected=QwenClient(model(endpoint))
        with pytest.raises(ModelError,match='budget exhausted'):
            read(rejected,tmp_path/'second',stage=2)
        assert state['requests']==1 and rejected.requests==0


def test_stage2_total_budget_is_not_multiplied_by_worker_count(tmp_path):
    with server(delay=.01) as (endpoint,state),ai.Workload(2,tmp_path/'perf'):
        ai.set_stage2_budget(3)
        with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn'),**ai.worker_options()) as pool:
            futures=[pool.submit(_process_job,(endpoint,str(tmp_path/'diag'),n,2)) for n in range(6)]
            accepted=0
            for future in futures:
                try:future.result();accepted+=1
                except ModelError as exc:assert 'budget exhausted' in str(exc)
        assert accepted==state['requests']==3
    summary=performance.summarize(tmp_path/'perf',1)
    assert summary['requests']==summary['budget_exhaustions']==3


def test_sync_bridge_cancellation_is_logged_and_does_not_wait_for_http_timeout(tmp_path):
    with server(delay=.3) as (endpoint,state),ai.Workload(1,tmp_path/'perf') as workload:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(read,QwenClient(model(endpoint,timeout=20)),tmp_path/'diag')
            deadline=time.monotonic()+2
            while not state['requests'] and time.monotonic()<deadline:time.sleep(.005)
            workload.state.limit.cancel.set()
            with pytest.raises(KeyboardInterrupt):future.result(timeout=2)
    summary=performance.summarize(tmp_path/'perf',1)
    assert summary['cancelled_attempts']==1


def test_bounded_submission_and_completion_order():
    produced=[]
    def tasks():
        for i in range(6):produced.append(i);yield i
    def work(i):time.sleep(.05 if i==0 else .001);return i
    with ThreadPoolExecutor(max_workers=2) as pool:
        iterator=performance.bounded_results(pool,work,tasks(),2)
        task,future=next(iterator)
        assert task==future.result()==1 and produced==[0,1]
        results=[task]+[f.result() for _,f in iterator]
        assert sorted(results)==list(range(6))


def test_empty_corrupt_and_foreign_checkpoint_rejected(tmp_path):
    folder=tmp_path/'page';folder.mkdir();(folder/'result.json').write_text('')
    with pytest.raises(ValueError,match='Empty'):performance.validate_markdown(folder,'source',7)
    performance.atomic_json(folder/'result.json',{'document_id':'other','source_page':7,'status':'COMPLETE'})
    performance.atomic_write(folder/'page.md','# Original PDF page 7\n\n**OCR status: COMPLETE**\nText')
    with pytest.raises(ValueError):performance.validate_markdown(folder,'source',7)


def test_summary_does_not_invent_server_queue_time(tmp_path):
    performance.atomic_write(tmp_path/'requests-1.jsonl',json.dumps(dict(network_started=True,response_status='truncated',
        attempt=2,in_flight=2,timeout_classification=None,usage=dict(completion_tokens=5,prompt_tokens=10)))+'\n'+'broken\n')
    result=performance.summarize(tmp_path,2,[{'status':'PARTIAL'}])
    assert result['pages_per_minute']==30 and result['requests_per_second']==.5
    assert result['retries']==result['truncations']==1 and result['corrupt_log_records']
    assert result['server_queue_time_seconds'] is None and not result['accuracy_verified']


def test_independent_selected_page_jobs_and_durable_output(tmp_path):
    import pymupdf
    from as_ocr import indexer,page_markdown as md
    source=tmp_path/'synthetic.pdf'
    with pymupdf.open() as document:
        for _ in range(3):document.new_page(width=200,height=200).insert_text((20,20),'32496')
        document.save(source)
    indexer.run(source,tmp_path/'index',pages=[1,3])
    with server(delay=.01,ocr=True) as (endpoint,state),ai.Workload(2,tmp_path/'perf'):
        result=md.run_compact(source,tmp_path/'index/index.sqlite3',tmp_path/'public',tmp_path/'evidence',
            [3,1],model(endpoint,max_requests=8),workers=2)
    assert [r['source_page'] for r in result['pages']]==[1,3]
    assert result['complete'] and result['requests']==8
    assert not (tmp_path/'public/pages/page_0002.md').exists()
    for page in [1,3]:
        saved,text=performance.validate_markdown(tmp_path/'evidence'/f'page_{page:04d}',indexer.fingerprint(source),page,md.compact_contract())
        assert '32496' in text and saved['status']=='COMPLETE'
    with pytest.raises(FileExistsError):
        md.run_compact(source,tmp_path/'index/index.sqlite3',tmp_path/'public',tmp_path/'evidence',[1],model(endpoint))

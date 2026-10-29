"""Terminal benchmarks; explicit small source selections or a local synthetic server.

No service control. Every invocation requires a new output directory. Synthetic
results measure scheduling/transport only, never GPU or extraction accuracy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from multiprocessing import get_context
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

from . import async_inference, indexer, page_markdown, performance, pipeline
from .qwen import ModelConfig, QwenClient, capture_diagnostics


def _get(url, config):
    headers={'Authorization':'Bearer '+config.api_key} if config.api_key else {}
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=5) as reply:
        return reply.read(1024*1024).decode()


def resource_snapshot(config):
    result={'at':time.time(),'server_queue_time_seconds':None}
    base=config.endpoint.removesuffix('/v1').rstrip('/')
    try:
        text=_get(base+'/metrics',config)
        result['server_metrics']=[line for line in text.splitlines() if not line.startswith('#') and any(
            term in line for term in ('num_requests_running','num_requests_waiting','cache_usage_perc',
                                     'request_success_total','prompt_tokens_total','generation_tokens_total'))]
        result['metrics_available']=True
    except Exception as exc:
        result.update(metrics_available=False,metrics_error=type(exc).__name__)
    try:
        result['host_memory']=[line for line in Path('/proc/meminfo').read_text().splitlines()
                               if line.startswith(('MemTotal:','MemAvailable:','SwapFree:'))]
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,power.draw',
                            '--format=csv,noheader'],capture_output=True,text=True,timeout=5)
        result['gpu']=gpu.stdout.strip() if gpu.returncode==0 else 'UNAVAILABLE'
    except (OSError,subprocess.TimeoutExpired):
        result['gpu']='UNAVAILABLE'
    return result


class Sampler:
    def __init__(self, config, path, interval):
        self.config,self.path,self.interval=config,Path(path),interval
        self.stop=threading.Event()
    def sample(self):
        snapshot=resource_snapshot(self.config)
        with self.path.open('a') as handle:handle.write(json.dumps(snapshot)+'\n')
    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.interval)
    def __enter__(self):
        self.thread=threading.Thread(target=self.run,name='benchmark-resources',daemon=True)
        self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(12)
        if not self.thread.is_alive():self.sample()


def _synthetic_job(task):
    from PIL import Image
    from . import progress
    number,model,directory=task
    progress.start(0,number,'Synthetic transport only')
    client=QwenClient(model)
    with capture_diagnostics(client,Path(directory)/f'page_{number:04d}',stage=0,subject_id='synthetic'):
        raw=client.request_json(Image.new('RGB',(128,128),'white'),'Synthetic transport test')
        if json.loads(raw)!={'identifier':'00042'}:raise ValueError('Synthetic response mismatch')
    return dict(source_page=number,status='SYNTHETIC_VALIDATED')


def _synthetic_server():
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            with self.server.observation_lock:
                state=self.server.observation
                state['active']+=1;state['requests']+=1;state['peak']=max(state['peak'],state['active'])
            try:
                time.sleep(.04)
                body=json.dumps(dict(model='synthetic',choices=[dict(finish_reason='stop',
                    message=dict(content=json.dumps(dict(identifier='00042'))))],
                    usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15))).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
            finally:
                with self.server.observation_lock:self.server.observation['active']-=1
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.daemon_threads=True
    server.observation={'active':0,'peak':0,'requests':0}
    server.observation_lock=threading.Lock()
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server


def compare_content(output, expectations, source_hash, pages):
    if expectations is None:
        return dict(status='REVIEW',reason='No source-reviewed oracle supplied; accuracy unmeasured')
    oracle=json.loads(Path(expectations).read_text())
    if oracle.get('source_sha256')!=source_hash:raise ValueError('Oracle belongs to a different source')
    results=[]
    for page,checks in oracle['pages'].items():
        if int(page) not in pages:raise ValueError('Oracle page outside benchmark selection')
        text=(Path(output)/'pages'/f'page_{int(page):04d}.md').read_text().split('\n[REVIEW:',1)[0]
        for value in checks.get('contains',[]):
            results.append(dict(page=int(page),kind='contains',value=value,passed=value in text))
        for value,count in checks.get('occurrences',{}).items():
            boundary=r'[\w.,]' if re.fullmatch(r'[\d.,]+',value) else r'\w'
            actual=len(re.findall(r'(?<!'+boundary+r')'+re.escape(value)+r'(?!'+boundary+r')',text))
            results.append(dict(page=int(page),kind='occurrences',value=value,expected=count,actual=actual,passed=actual==count))
        if 'tables' in checks:
            actual=[dict(headers=t['headers'],rows=t['rows']) for t in page_markdown.markdown_tables(text)]
            results.append(dict(page=int(page),kind='tables',passed=actual==checks['tables'],actual=actual))
    return dict(status='PASS' if results and all(x['passed'] for x in results) else 'FAIL' if results else 'REVIEW',
                checks=results,checked_pages=sorted(map(int,oracle['pages'])),human_accuracy_verified=False)


def run(args):
    if args.concurrency not in (1,2,4) or not 1<=args.workers<=4:raise ValueError('Benchmark supports concurrency 1,2,4 and workers 1-4')
    if args.output.exists():raise ValueError('Benchmark output must be absent; never resume historical results')
    if not args.synthetic and (not args.pdf or not args.pdf.is_file() or args.pdf.suffix.lower()!='.pdf' or
            not args.pages or min(args.pages)<1 or len(args.pages)>10 or len(set(args.pages))!=len(args.pages)):
        raise ValueError('Supply one PDF and 1-10 unique original pages; full catalogs are forbidden')
    if args.transport=='sync' and args.workers!=args.concurrency:
        raise ValueError('Legacy sync comparison uses workers=concurrency (no shared admission transport)')
    config=ModelConfig(endpoint=os.getenv('AS_OCR_QWEN_ENDPOINT'),model=os.getenv('AS_OCR_QWEN_MODEL'),
        identity=os.getenv('AS_OCR_QWEN_IDENTITY'),api_key=os.getenv('AS_OCR_QWEN_API_KEY'),
        timeout=180,max_tokens=1536,retries=0,max_requests=20)
    if args.dry_run:
        return dict(dry_run=True,workload=args.workload,transport=args.transport,concurrency=args.concurrency,
                    workers=args.workers,pages=args.pages,model=config.public(),output=str(args.output))
    server=None
    if args.synthetic:
        server=_synthetic_server()
        config=ModelConfig(endpoint=f'http://127.0.0.1:{server.server_port}/v1',model='synthetic',
                           timeout=5,max_tokens=64,retries=0,max_requests=1)
    else:
        QwenClient(config)  # config validation without inference
        if not config.endpoint or not config.model:raise ValueError('Configure Qwen endpoint and model')
        _get(config.endpoint.removesuffix('/v1').rstrip('/')+'/health',config)
        listing=json.loads(_get(config.endpoint.rstrip('/')+'/models',config))
        if config.model not in {x['id'] for x in listing.get('data',[])}:raise ValueError('Configured model unavailable')
    args.output.mkdir(parents=True,exist_ok=False)
    internal=args.output/'_internal';internal.mkdir()
    started=time.monotonic();records=[];logdir=internal/'performance';public=args.output
    metadata=dict(workload='synthetic' if args.synthetic else args.workload,transport=args.transport,concurrency=args.concurrency,workers=args.workers,
        synthetic=args.synthetic,pages=args.pages,model=config.public(),contract=page_markdown.compact_contract(),
        native_fast_path=args.prefer_native,accuracy_verified=False)
    if not args.synthetic:metadata.update(source=str(args.pdf.resolve()),source_sha256=indexer.fingerprint(args.pdf))
    performance.atomic_json(internal/'invocation.json',metadata)
    performance.atomic_json(internal/'benchmark-state.json',dict(status='RUNNING',complete=False))
    try:
        with (Sampler(config,internal/'resources.jsonl',args.metrics_interval) if not args.synthetic else nullcontext()):
            if args.workload=='pipeline' and not args.synthetic:
                values=[str(args.pdf),'--pages',*map(str,args.pages),'--focused-test','--fresh',
                    '--output-root',str(args.output/'catalog'),'--run-id','benchmark','--page-workers',str(args.workers),
                    '--markdown-engine','compact','--markdown-max-requests','20','--markdown-after-extraction',
                    '--stage2-retries','0','--vision-retries','0']
                if args.transport=='async':values+=['--async-inference','--max-in-flight',str(args.concurrency)]
                if args.prefer_native:values+=['--markdown-prefer-native']
                result=pipeline.run_pipeline(pipeline._parser().parse_args(values))
                logdir=Path(result['outputs']['run_dir'])/'performance'
                public=Path(result['outputs']['public_root'])
                records=result.get('page_markdown',{}).get('pages',[])
                metadata['pipeline_status']=result['status']
                metadata['effective_limits']=result.get('effective_limits',{})
                metadata['model_scope']='Readiness defaults; use effective_limits and request events for per-stage settings'
                performance.atomic_json(internal/'invocation.json',metadata)
            else:
                with (async_inference.Workload(args.concurrency,logdir) if args.transport=='async' else nullcontext()):
                    if args.synthetic:
                        tasks=[(n,config,internal/'diagnostics') for n in range(1,13)]
                        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn'),
                                **async_inference.worker_options()) as pool:
                            for _,future in performance.bounded_results(pool,_synthetic_job,tasks,args.workers,async_inference.cancel):
                                records.append(future.result())
                    else:
                        indexer.run(args.pdf,internal/'stage_1',pages=args.pages,force=True)
                        result=page_markdown.run_compact(args.pdf,internal/'stage_1/index.sqlite3',args.output,
                            internal/'ocr',args.pages,config,page_markdown.CompactConfig(prefer_native=args.prefer_native),args.workers)
                        records=result['pages']
            elapsed=time.monotonic()-started
        summary=performance.summarize(logdir,elapsed,records,synthetic=args.synthetic)
        if args.transport=='sync':
            # Legacy capture files are the measurement source; preserve originals.
            logs=[p for p in args.output.rglob('*.json') if 'diagnostics' in p.parts]
            logdir=internal/'legacy-performance';logdir.mkdir()
            events={};rank={'FAILED':3,'PASSED':2,'NOT_RUN':1}
            for path in logs:
                event=json.loads(path.read_text())
                if 'request_id' not in event:continue
                prior=events.get(event['request_id'])
                if prior and rank.get(prior.get('local_validation_status'),0)>rank.get(event.get('local_validation_status'),0):continue
                events[event['request_id']]=event
            with (logdir/'requests-legacy.jsonl').open('w') as handle:
                for event in events.values():
                    event.setdefault('network_started',True)
                    handle.write(json.dumps(event)+'\n')
            summary=performance.summarize(logdir,elapsed,records,synthetic=args.synthetic)
        if records and not summary['requests']:
            raise ValueError('Missing request measurements; benchmark cannot be reported as valid')
        if server:
            summary['synthetic_server_observation']=dict(server.observation)
        summary.update(metadata,performance_logs=str(logdir),
            quality=compare_content(public,args.expectations,metadata.get('source_sha256'),args.pages or []) if not args.synthetic else
                dict(status='NOT_MEASURED',reason='Synthetic HTTP, no OCR/GPU/source accuracy measurement'))
        if not args.synthetic and indexer.fingerprint(args.pdf)!=metadata['source_sha256']:raise ValueError('Source changed during benchmark')
        performance.atomic_json(args.output/'summary.json',summary)
        lines=['# Inference benchmark','',f'- Synthetic: {args.synthetic}',f'- Transport: {args.transport}',
               f'- Concurrency: {args.concurrency}; page workers: {args.workers}',
               '- Source accuracy is not certified by elapsed time or processing completion.','',
               '```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','']
        performance.atomic_write(args.output/'review.md','\n'.join(lines))
        performance.atomic_json(internal/'benchmark-state.json',dict(status='FINISHED',complete=True,
            extraction_complete=all(r.get('status')=='COMPLETE' for r in records) if not args.synthetic else None))
        return summary
    finally:
        if not (args.output/'summary.json').is_file():
            performance.atomic_json(internal/'benchmark-state.json',dict(status='INTERRUPTED_OR_FAILED',
                complete=False,elapsed_seconds=time.monotonic()-started,selected_pages=args.pages))
        if server:server.shutdown();server.server_close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf',type=Path,nargs='?')
    parser.add_argument('--pages',type=int,nargs='+')
    parser.add_argument('--concurrency',type=int,choices=[1,2,4],required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workload',choices=['markdown','pipeline'],default='markdown')
    parser.add_argument('--transport',choices=['async','sync'],default='async')
    parser.add_argument('--prefer-native',action='store_true')
    parser.add_argument('--expectations',type=Path)
    parser.add_argument('--metrics-interval',type=float,default=5)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.metrics_interval<=60:parser.error('Metrics interval must be 1-60 seconds')
    try:
        result=run(args)
    except KeyboardInterrupt:
        print('Interrupted: completed evidence preserved; benchmark incomplete.');return 130
    except Exception as exc:
        reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
        print('Benchmark failed: '+reason+'; no service settings changed.');return 1
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if result.get('quality',{}).get('status')=='FAIL' or result.get('page_statuses',{}).get('FAILED') or result.get('corrupt_log_records') else 0


if __name__=='__main__':raise SystemExit(main())

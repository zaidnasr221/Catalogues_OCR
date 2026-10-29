"""Durable, source-verified 20-page experiment. Invoke from tmux with uv --env-file.

Uses the maintained pipeline parser/API; no extraction rules are reimplemented.
Review generation is CPU-only and can be repeated while extraction is running.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import time

import pymupdf as fitz
from PIL import Image, ImageDraw

from as_ocr import async_inference as ai, benchmark, delivery, detection, indexer, performance, pipeline
from as_ocr.qwen import ModelConfig, QwenClient, capture_diagnostics, request_metadata
from as_ocr.provenance import code_fingerprint

SOURCE = Path('/home/domapp/Z_AiWork/As_ocr/Data/Alupco/50SG ALUPCO PDF COMBINED.pdf')
SHA = 'd8173d47de9d732d527df7485aa3c0f1bbfcc4d6f5d7ebe6b7aea73ecd211e58'
PAGES = list(range(1, 21))
SMOKE_PAGES = [1, 7, 17]
CONTAINER = 'as-ocr-visual64-qwen-vision-1'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default if default is not None else {}


def state(root, phase, **values):
    performance.atomic_json(root/'experiment-state.json', dict(phase=phase, at=now(), pid=os.getpid(), **values))
    with (root/'experiment-events.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(phase=phase,at=now(),pid=os.getpid(),**values))+'\n')
    print(json.dumps(dict(at=now(), phase=phase, **values)), flush=True)


def verify_source():
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'P0':
        raise ValueError('Only branch P0 is authorized')
    if indexer.fingerprint(SOURCE) != SHA:
        raise ValueError('Source PDF hash changed')
    with fitz.open(SOURCE) as document:
        if len(document) != 75 or document.needs_pass:
            raise ValueError('Unexpected source page count or password requirement')


def model():
    return ModelConfig(endpoint=os.getenv('AS_OCR_QWEN_ENDPOINT', 'http://127.0.0.1:8003/v1'),
        model=os.getenv('AS_OCR_QWEN_MODEL', 'qwen-catalog-vision'),
        identity=os.getenv('AS_OCR_QWEN_IDENTITY'), api_key=os.getenv('AS_OCR_QWEN_API_KEY'),
        timeout=180, max_tokens=128, retries=0, max_requests=1)


class ExperimentSampler(benchmark.Sampler):
    def sample(self):
        snapshot=benchmark.resource_snapshot(self.config)
        state=getattr(ai,'_state',None)
        if state is not None:
            with state.limit.active.get_lock():snapshot['client_in_flight']=state.limit.active.value
            with state.limit.peak.get_lock():snapshot['client_peak_in_flight']=state.limit.peak.value
            snapshot['client_permit_limit']=state.limit.maximum
        try:
            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            snapshot['gpu_compute_process_memory_mib']=gpu.stdout.strip() if gpu.returncode==0 else 'UNAVAILABLE'
        except (OSError,subprocess.TimeoutExpired):snapshot['gpu_compute_process_memory_mib']='UNAVAILABLE'
        with self.path.open('a') as handle:handle.write(json.dumps(snapshot)+'\n')


@contextmanager
def record_progress(root,name):
    """Persist the existing progress events without changing stage scheduling."""
    original=pipeline.progress.Reporter.consume
    path=root/(name+'-page-events-'+str(time.time_ns())+'.jsonl')
    wall_origin=time.time()-time.monotonic()
    with path.open('x') as handle:
        def consume(reporter,event):
            saved=dict(event,at_utc=datetime.fromtimestamp(wall_origin+event['at'],timezone.utc).isoformat())
            handle.write(json.dumps(saved)+'\n');handle.flush()
            return original(reporter,event)
        pipeline.progress.Reporter.consume=consume
        try:yield
        finally:pipeline.progress.Reporter.consume=original


def health(config):
    base=config.endpoint.removesuffix('/v1').rstrip('/')
    benchmark._get(base+'/health', config)
    listing=json.loads(benchmark._get(config.endpoint.rstrip('/')+'/models', config))
    if config.model not in {item['id'] for item in listing.get('data', [])}:
        raise ValueError('Configured model unavailable')
    return listing


def server_record(root):
    result=json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER], text=True))[0]
    # Deliberately exclude environment and possible credentials.
    safe=dict(at=now(), command=result['Config']['Cmd'], entrypoint=result['Config']['Entrypoint'],
        image=result['Config']['Image'], image_id=result['Image'], state=result['State'],
        restart_count=result['RestartCount'],mounts=[{k:m[k] for k in ('Source','Destination','RW')} for m in result['Mounts']],
        active_client_model=model().public(),memory_limit=result['HostConfig']['Memory'], restart_policy=result['HostConfig']['RestartPolicy'])
    performance.atomic_json(root/'server.json', safe)
    return safe


def prepare(root):
    verify_source()
    if (root/'source.json').exists():
        prior=read(root/'source.json')
        if prior.get('sha256')!=SHA or prior.get('selected_pages')!=PAGES:
            raise ValueError('Existing source evidence is incompatible')
    else:
        performance.atomic_json(root/'source.json', dict(path=str(SOURCE), sha256=SHA, bytes=SOURCE.stat().st_size,
            pages=75, selected_pages=PAGES, requested_concurrency=32, at=now()))
    server_record(root)
    result=indexer.run(SOURCE, root/'source_index', pages=PAGES, preview_max=1400)
    performance.atomic_json(root/'source-index-summary.json', result)
    if result.get('errors'):
        raise ValueError('Source indexing failed')
    with sqlite3.connect(root/'source_index/index.sqlite3') as db:
        db.row_factory=sqlite3.Row
        rows=db.execute('SELECT * FROM pages WHERE document_id=? ORDER BY page_index', (SHA,)).fetchall()
        contact=Image.new('RGB', (5*280,4*235), '#dedede')
        for row in rows:
            folder=root/'review'/f'page_{row["page_index"]+1:04d}'/'stage_1'
            folder.mkdir(parents=True, exist_ok=True)
            geometry, spans, _=detection.load_page(db,row)
            performance.atomic_json(folder/'native-text.json', dict(source_page=row['page_index']+1,
                document_id=SHA, status=row['status'], rotation=row['rotation'], geometry=geometry,
                signals=json.loads(row['signals']), spans=[dict(ordinal=i, **s) for i,s in spans]))
            performance.atomic_write(folder/'native-text.txt', '\n'.join(s['text'] for _,s in spans) or
                '[No PDF-native text spans. Visual OCR required.]\n')
            with Image.open(root/'source_index'/row['preview']) as image:
                original=image.convert('RGB'); original.save(folder/'original.png')
                overlay=original.copy(); draw=ImageDraw.Draw(overlay)
                transform=fitz.Matrix(*geometry['source_to_display'])*fitz.Matrix(*geometry['display_to_preview'])
                for i,span in spans:
                    box=list(fitz.Rect(span['bbox'])*transform)
                    draw.rectangle(box, outline='red', width=2); draw.text((box[0],box[1]),str(i),fill='red')
                draw.rectangle((0,0,overlay.width,22), fill='white')
                draw.text((4,4), f'Original page {row["page_index"]+1} | native spans: {len(spans)} | indexing {row["status"]}', fill='black')
                overlay.save(folder/'native-overlay.png')
                original.thumbnail((270,205))
                x=(row['page_index']%5)*280;y=(row['page_index']//5)*235
                contact.paste(original,(x,y+24));ImageDraw.Draw(contact).text((x+4,y+4),f'Original page {row["page_index"]+1}',fill='black')
        contact.save(root/'review/contact-sheet.png')
        comparisons=[]
        for p in (1,10,20):
            row=next(r for r in rows if r['page_index']==p-1)
            geometry,_,_=detection.load_page(db,row);bounds=geometry['source_rect']
            before=time.monotonic()
            a,ma=detection.render(SOURCE,p-1,geometry,bounds,1536)
            b,mb=detection.render(SOURCE,p-1,geometry,bounds,1536)
            old=time.monotonic()-before
            before=time.monotonic();c,mc=detection.render(SOURCE,p-1,geometry,bounds,1536);new=time.monotonic()-before
            comparisons.append(dict(source_page=p, baseline_renders=2, optimized_renders=1,
                baseline_seconds=old, optimized_seconds=new,
                exact_pixels_and_mapping=a.tobytes()==b.tobytes()==c.tobytes() and ma==mb==mc))
        performance.atomic_json(root/'safety/render-benchmark.json',dict(kind='CPU render only; no inference speedup claim',pages=comparisons))
    state(root,'PREPARED', selected_pages=PAGES)


def request_statistics(directory, elapsed):
    summary=performance.summarize(directory,elapsed)
    events=[json.loads(line) for path in Path(directory).glob('requests-*.jsonl') for line in path.read_text().splitlines()]
    durations=sorted(e['duration_seconds'] for e in events if e.get('network_started') and 'duration_seconds' in e)
    if not durations:
        durations=sorted(e['network_duration_seconds'] for e in events if 'network_duration_seconds' in e)
    summary['durations_seconds']=dict(mean=statistics.mean(durations),median=statistics.median(durations),
        p95=durations[min(len(durations)-1, int(.95*len(durations)))],maximum=max(durations)) if durations else None
    return summary


def metric_summary(path):
    snapshots=[json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []
    values={key:[] for key in ('num_requests_running','num_requests_waiting','cache_usage','preempt')}
    for snapshot in snapshots:
        for line in snapshot.get('server_metrics',[]):
            for key in values:
                if key in line:
                    try: values[key].append(float(line.rsplit(' ',1)[-1]))
                    except ValueError: pass
    return dict(samples=len(snapshots), peaks={k:max(v) if v else None for k,v in values.items()},
        unavailable=[k for k,v in values.items() if not v])


def probe(root,followup=False):
    verify_source();config=replace(model(),max_tokens=256) if followup else model();health(config);server=server_record(root)
    command=server['command'];seqs=int(command[command.index('--max-num-seqs')+1])
    series='probes-response256' if followup else 'probes'
    if (root/series).exists(): raise ValueError('Probe series already exists; preserve it')
    results=[]
    schema={'type':'object','properties':{'text':{'type':'string'}},'required':['text'],'additionalProperties':False}
    for cap in ((8,16,32) if followup else (2,4,8,16,32)):
        health(config)
        folder=root/series/f'c{cap}';folder.mkdir(parents=True)
        jobs=32 if followup else max(4,cap)
        # Same 1536px page size as detection. Simple readings benchmark transport,
        # not full extraction quality or long-response saturation.
        images=[]
        with fitz.open(SOURCE) as doc:
            for p in (1,7,12,20):
                page=doc[p-1];pix=page.get_pixmap(matrix=fitz.Matrix(1536/max(page.rect.width,page.rect.height),1536/max(page.rect.width,page.rect.height)),alpha=False)
                images.append(Image.frombytes('RGB',(pix.width,pix.height),pix.samples))
        def one(number):
            client=QwenClient(config)
            with request_metadata(page=(1,7,12,20)[number%4],purpose='controlled_short_probe'), capture_diagnostics(client,folder/'diagnostics',stage=0,subject_id=str(number)):
                raw=client.request_json(images[number%4], 'Read ONE clearly visible short heading or label literally. Return JSON with text only. Never invent text; return an empty string if unreadable.',response_schema=schema)
                value=json.loads(raw)
                if set(value)!={'text'} or not isinstance(value['text'],str): raise ValueError('Probe schema failed')
            return value
        state(root,'PROBING',requested_concurrency=32,testing=cap,server_max_num_seqs=seqs)
        started=time.monotonic();failures=[]
        with benchmark.Sampler(config,folder/'resources.jsonl',1), ai.Workload(cap,folder/'performance') as workload:
            with ThreadPoolExecutor(max_workers=cap) as pool:
                for number,future in performance.bounded_results(pool,one,range(jobs),cap,ai.cancel):
                    try: performance.atomic_json(folder/f'reading-{number:03d}.json',future.result())
                    except Exception as exc: failures.append(dict(job=number,error=type(exc).__name__+': '+str(exc)[:200]))
            peak=workload.state.limit.peak.value
        elapsed=time.monotonic()-started
        summary=request_statistics(folder/'performance',elapsed)
        metrics=metric_summary(folder/'resources.jsonl')
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        stable=not failures and not summary['timeouts'] and not summary['truncations'] and available>16*1024**3 and summary['durations_seconds'] is not None and summary['durations_seconds']['maximum']<150
        result=dict(cap=cap,jobs=jobs,peak_client_in_flight=peak,summary=summary,metrics=metrics,
            failures=failures,stable=stable,mem_available_bytes=available,server_max_num_seqs=seqs)
        performance.atomic_json(folder/'summary.json',result);results.append(result)
        performance.atomic_json(root/(series+'-results.json'),dict(requested_concurrency=32,max_tokens=config.max_tokens,levels=results))
        performance.atomic_json(root/'probe-results.json',dict(requested_concurrency=32,series=series,max_tokens=config.max_tokens,levels=results))
        print(json.dumps(result),flush=True)
        if not stable: break
        # Sampler context has closed and every request completed before next cap.
    safe=[r for r in results if r['stable'] and r['cap']<=seqs]
    if not safe: raise ValueError('No empirically safe probe level; full extraction blocked')
    effective=max(r['cap'] for r in safe)
    if followup:
        best=max(r['summary']['requests_per_second'] for r in safe)
        effective=max(r['cap'] for r in safe if r['summary']['requests_per_second']>=best*.95)
    performance.atomic_json(root/'concurrency-choice.json',dict(requested=32,effective=effective,
        page_workers=effective,server_max_num_seqs=seqs,
        reason='Measured throughput fallback: choose the highest stable tested cap within 5% of best comparable short-probe throughput. Requested target remains 32. Short probes do not prove safety of long OCR/JSON requests; smoke and resource monitoring remain required. Twenty selected page jobs can generate at most twenty simultaneous page requests, even with a 32-permit ceiling.',
        probe_only=True,actual_full_run_peak=None))
    state(root,'PROBES_FINISHED',requested_concurrency=32,effective_concurrency=effective)


def pipeline_args(root, name, pages, *, resume=False):
    choice=read(root/'concurrency-choice.json')
    if not choice: raise ValueError('Controlled probes required first')
    settings=read(root/'run-config.json')
    budgets=settings.get('budgets',{})
    if not budgets:
        budgets=dict(markdown=50,detection=160,grouping=20,callouts=100,components=100,
            profile_inventory=100,profile_verification=100,paragraphs=20,enrichment=50)
        settings=dict(requested_max_num_seqs=64,requested_concurrency=32,budgets=budgets,
            scope=dict(markdown='page',detection='stage run',grouping='page',callouts='page',components='page',
                profile_inventory='page across Details',profile_verification='page across Details',paragraphs='page',enrichment='stage run'),
            note='Manual ceilings only; no adaptive budgets. Change before a NEW experiment; preserve this configuration and evidence when resuming.')
        performance.atomic_json(root/'run-config.json',settings)
    argv=[str(SOURCE),'--run-id',name,'--output-root',str(root/name),'--pages',*map(str,pages),
        '--async-inference','--max-in-flight',str(choice['effective']),
        '--page-workers',str(choice['page_workers']),'--live-progress',
        '--grouping-mode','deterministic','--markdown-after-extraction','--markdown-engine','compact',
        '--markdown-max-requests',str(budgets['markdown']),'--stage2-max-requests',str(budgets['detection']),
        '--stage3-max-requests',str(budgets['grouping']),'--stage4-max-requests',str(budgets['callouts']),
        '--stage5-max-requests',str(budgets['components']),'--profile-inventory-max-requests',str(budgets['profile_inventory']),
        '--profile-verification-max-requests',str(budgets['profile_verification']),
        '--paragraph-max-requests',str(budgets['paragraphs']),'--stage6-max-requests',str(budgets['enrichment']),
        '--stage2-retries','1','--vision-retries','1',
        '--stage2-timeout',str(settings.get('timeouts',{}).get('stage2',180)),
        '--vision-timeout',str(settings.get('timeouts',{}).get('vision',180))]
    argv+=['--resume'] if resume else ['--fresh']
    result=pipeline._parser().parse_args(argv)
    result._invocation_argv=argv
    return result


def execute(root, name, pages, resume=False):
    verify_source();health(model());server_record(root)
    cpu=read(root/'safety/cpu-gate.json')
    if cpu.get('status')!='PASS' or cpu.get('code_fingerprint')!=code_fingerprint():
        raise ValueError('Passing CPU gate for this exact pipeline code is required')
    args=pipeline_args(root,name,pages,resume=resume)
    # Parser/API is the same one used by as-ocr pipeline, with supported options.
    invocation=root/(name+'-invocation.json')
    if invocation.exists():
        prior=root/'safety'/(name+'-invocation-'+str(time.time_ns())+'.json')
        performance.atomic_write(prior,invocation.read_text())
    performance.atomic_json(invocation,dict(argv=args._invocation_argv,
        command=shlex.join(['uv','run','--env-file','.env','--frozen','--no-sync','as-ocr','pipeline',*args._invocation_argv]),pages=pages,
        requested_concurrency=32,effective=args.max_in_flight,workers=args.page_workers,at=now(),resume=resume))
    state(root,name.upper()+'_RUNNING',pages=pages,requested_concurrency=32,effective=args.max_in_flight)
    started=time.monotonic()
    with ExperimentSampler(model(),root/(name+'-resources.jsonl'),3), record_progress(root,name):
        summary=pipeline.run_pipeline(args)
    verify_source()
    performance.atomic_json(root/(name+'-outcome.json'),dict(status=summary['status'],
        elapsed_seconds=time.monotonic()-started,summary_path=summary['outputs']['run_summary'],
        public_root=summary['outputs']['public_root'],run_dir=summary['outputs']['run_dir']))
    state(root,name.upper()+'_FINISHED',status=summary['status'])
    return summary


def smoke_gate(root, summary):
    stages=summary.get('stages',{})
    # PARTIAL/REVIEW content remains visible; missing handoffs and transport
    # failures block expansion. Human accuracy is never inferred from execution.
    reasons=[]
    for n in range(1,8):
        item=stages.get(f'stage_{n}',{})
        if not item or item.get('status') in {'FAILED','NOT_RUN','BLOCKED'}:
            reasons.append(f'Stage {n} missing/failed/blocked')
    public=Path(summary['outputs']['public_root'])
    csv=public/'all_details.csv'
    if not csv.is_file(): reasons.append('Merged delivery CSV missing')
    else:
        try:
            rows=delivery.read_csv(csv)
            if any(int(r['source_page']) not in set(SMOKE_PAGES) for r in rows): reasons.append('Foreign source page in CSV')
        except ValueError as exc: reasons.append(str(exc))
    md=summary.get('page_markdown',{}).get('pages',[])
    if {r['source_page'] for r in md}!=set(SMOKE_PAGES): reasons.append('Smoke Markdown page inventory differs')
    if any(r['status']=='FAILED' for r in md): reasons.append('Smoke OCR failed')
    for p in md:
        if not (public/'pages'/f'page_{p["source_page"]:04d}.md').is_file(): reasons.append('Page Markdown missing')
    if not list((Path(summary['outputs']['run_dir'])/'stage_3').rglob('composite-*.png')):
        reasons.append('Smoke produced no Detail composite; representative Assembly handoff untested')
    performance.atomic_json(root/'smoke-gate.json',dict(status='BLOCKED' if reasons else 'PASS_WITH_REVIEW',
        reasons=reasons,accuracy_verified=False,scope='Pages 1, 7 and 17; execution/inventory gate, manual image review also required'))
    return reasons


def recover_smoke(root):
    """Continue supported Stages 5-7/OCR from verified Stage 1-4 evidence.

    The only inference change is compact formatting and additional-field reading.
    No previous source, attempt, checkpoint, image or delivery is rewritten.
    """
    from multiprocessing import get_context
    from visual_pipeline_review import manifest_page
    verify_source();health(model())
    cpu=read(root/'safety/cpu-gate.json')
    if cpu.get('status')!='PASS' or cpu.get('code_fingerprint')!=code_fingerprint():
        raise ValueError('CPU gate must pass for the corrected response code')
    args=pipeline_args(root,'smoke-recovery',SMOKE_PAGES)
    public=args.output_root.resolve()/delivery.identifier(SOURCE.name,SHA)
    run=pipeline._source_paths(SOURCE,args.output_root,'smoke-recovery')
    if run.exists():raise ValueError('Recovery evidence already exists; preserve it')
    run.mkdir(parents=True)
    dirs={n:run/f'stage_{n}' for n in range(1,8)}
    for directory in dirs.values():directory.mkdir()
    prior=next((root/'smoke').glob('*__*'))/'_internal/runs/smoke'
    index=dirs[1]/'index.sqlite3'
    with sqlite3.connect((prior/'stage_1/index.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as before, sqlite3.connect(index) as after:
        before.backup(after)
    shutil.copytree(prior/'stage_1/previews',dirs[1]/'previews')
    selections={3:{},4:{}}
    for stage in (3,4):
        for path in sorted((prior/f'stage_{stage}').rglob('manifest.json'),key=lambda p:p.stat().st_mtime_ns):
            saved=read(path);page=manifest_page(saved)
            if page in SMOKE_PAGES:selections[stage][page]=path
    proof=[]
    for page in SMOKE_PAGES:
        s3,s4=selections[3][page],selections[4][page]
        # Existing supported API checks both identities, hashes, source and views.
        inputs=pipeline.components.load_inputs(s3,s4)
        s2=Path(read(s3)['upstream']['manifest_path'])
        if read(s2).get('document_id')!=SHA or read(s2).get('page_index')!=page-1 or read(s2).get('page_decision')=='FAILED':
            raise ValueError('Incompatible source/page/detection handoff')
        for stage,path in ((2,s2),(3,s3),(4,s4)):
            shutil.copytree(path.parent,dirs[stage]/f'inherited-page-{page:04d}')
            proof.append(dict(stage=stage,source_page=page,path=str(path),sha256=indexer.fingerprint(path),
                reason='Exact source-verified successful/REVIEW Stage 1-4 evidence; detector/grouping/callout rules unchanged. No model inference cache is reused for corrected Stage 5 or independent OCR.'))
    performance.atomic_json(run/'inherited-source-evidence.json',proof)
    summary=dict(status='PARTIAL',source=dict(path=str(SOURCE),sha256=SHA),
        pages=dict(page_count=75,selected=3,original_pages=SMOKE_PAGES),stages={},
        outputs=dict(run_dir=str(run),public_root=str(public),run_summary=str(run/'run-summary.json')))
    for n in range(1,5):
        summary['stages'][f'stage_{n}']=dict(status='COMPLETE' if n==1 else 'PARTIAL',
            duration_seconds=None,execution_status='REUSED_SOURCE_VERIFIED',
            note='Actual completed source evidence copied with provenance; earlier processing times remain in original smoke logs.')
    state(root,'SMOKE_RECOVERY_RUNNING',pages=SMOKE_PAGES,effective_concurrency=args.max_in_flight)
    performance.atomic_json(root/'smoke-recovery-invocation.json',dict(at=now(),pages=SMOKE_PAGES,
        reuse_proof=str(run/'inherited-source-evidence.json'),command='Supported components.run, enrichment.run, exports.run, delivery.publish, page_markdown.run_compact APIs; exact Stage 1-4 handoffs listed in inherited-source-evidence.json'))
    started=time.monotonic()
    with ExperimentSampler(model(),root/'smoke-recovery-resources.jsonl',3),record_progress(root,'smoke-recovery'), \
            ai.Workload(args.max_in_flight,run/'performance'), \
            pipeline.progress.Reporter(SOURCE,75,args.page_workers,run/'progress.log'):
        stage_clock=time.monotonic();outputs=[]
        component_model=pipeline._model(args,max_requests=args.stage5_max_requests,
            timeout_seconds=args.vision_timeout,max_tokens=4096,retries=args.vision_retries)
        tasks=[(5,(selections[3][p],selections[4][p],dirs[5]),
            dict(index_path=index,model=component_model,config=pipeline._component_config(args),force=True,complete_views=True),
            dirs[5]/'diagnostics',p) for p in SMOKE_PAGES]
        pipeline.progress.plan(5,SMOKE_PAGES)
        with sqlite3.connect(index) as db:pipeline.components.migrate(db)
        with ProcessPoolExecutor(max_workers=min(args.page_workers,3),mp_context=get_context('spawn'),
                **ai.worker_options(pipeline.progress.worker_options())) as pool:
            for task,future in performance.bounded_results(pool,pipeline._page_worker,tasks,min(args.page_workers,3),ai.cancel):
                packed=future.result()
                if packed['error']:raise RuntimeError(packed['error'])
                outputs.append(packed['result'])
        manifests=sorted((Path(o['manifest']) for o in outputs),key=lambda p:manifest_page(read(p)))
        failed=[p for p in manifests if read(p).get('state')=='FAILED']
        summary['stages']['stage_5']=dict(status='FAILED' if failed else 'PARTIAL' if any(read(p).get('complete') is False for p in manifests) else 'COMPLETE',
            duration_seconds=time.monotonic()-stage_clock,outputs=dict(manifests=list(map(str,manifests))))
        performance.atomic_json(run/'run-summary.json',summary)
        if failed:raise ValueError('Corrected component extraction still FAILED: '+str(failed))
        stage_clock=time.monotonic()
        enrichment_model=pipeline._model(args,max_requests=args.stage6_max_requests,
            timeout_seconds=args.vision_timeout,max_tokens=4096,retries=args.vision_retries)
        enriched=pipeline.enrichment.run(manifests,dirs[6],index_path=index,
            config=pipeline.enrichment.Config(False,args.max_hit_windows,args.image_max),model=enrichment_model,force=True)
        summary['stages']['stage_6']=dict(status='PARTIAL' if enriched.get('errors') else 'COMPLETE',
            duration_seconds=time.monotonic()-stage_clock,outputs=dict(manifest=enriched['manifest']))
        stage_clock=time.monotonic()
        exported=pipeline.exports.run(enriched['manifest'],dirs[7],index_path=index,force=True)
        publication=delivery.publish(exported['artifacts']['summary'],output=args.output_root)
        summary['stages']['stage_7']=dict(status='PARTIAL' if exported.get('export_status')!='COMPLETE' else 'COMPLETE',
            duration_seconds=time.monotonic()-stage_clock,outputs=dict(summary=exported['artifacts']['summary']))
        performance.atomic_json(run/'run-summary.json',summary)
        stage_clock=time.monotonic()
        markdown_model=pipeline._model(args,max_requests=args.markdown_max_requests,timeout_seconds=args.vision_timeout,max_tokens=8192,retries=0)
        pipeline.progress.plan(1,SMOKE_PAGES)
        summary['page_markdown']=pipeline.page_markdown.run_compact(SOURCE,index,public,run/'page_ocr-recovery',SMOKE_PAGES,
            markdown_model,pipeline.page_markdown.CompactConfig(max_requests=args.markdown_max_requests),workers=min(args.page_workers,3))
        summary['markdown_wall_seconds']=time.monotonic()-stage_clock
    verify_source();summary['elapsed_seconds']=time.monotonic()-started
    performance.atomic_json(run/'run-summary.json',summary)
    performance.atomic_json(root/'smoke-recovery-outcome.json',dict(status=summary['status'],elapsed_seconds=summary['elapsed_seconds'],
        public_root=str(public),run_dir=str(run),summary_path=str(run/'run-summary.json')))
    reasons=smoke_gate(root,summary)
    state(root,'SMOKE_RECOVERY_FINISHED',status=summary['status'],gate='BLOCKED' if reasons else 'PASS_WITH_REVIEW')
    if reasons:raise ValueError('Recovery smoke gate blocked: '+'; '.join(reasons))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','probe','probe-followup','smoke','smoke-resume','smoke-recovery','run','resume','review','status','monitor','stop'])
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.action in {'review','status','monitor'}:
        from visual_pipeline_review import generate
        generated=generate(root)
        print(json.dumps(dict(phase=read(root/'experiment-state.json'),mode=generated['mode'],run_dir=generated['run_dir'],
            page_statuses=[p for p in generated['page_statuses'] if not (generated['mode'] or '').startswith('smoke') or p['source_page'] in SMOKE_PAGES])),flush=True)
        if args.action=='status':
            action=read(root/'active-job.json').get('action','run')
            logs=sorted(root.glob(action+'-terminal*.log'),key=lambda p:p.stat().st_mtime_ns)
            if logs:print('\n'.join(logs[-1].read_text().splitlines()[-8:]),flush=True)
        if args.action=='monitor':
            while True:
                time.sleep(20)
                generate=importlib.reload(sys.modules['visual_pipeline_review']).generate
                generate(root)
                if read(root/'experiment-state.json').get('phase') in {'FULL_FINISHED','STOPPED','BLOCKED'}:break
        return
    if args.action=='stop':
        record=read(root/'active-job.json');pid=record.get('pid')
        if not pid:raise ValueError('No recorded active process')
        # Verify Linux start ticks to avoid signalling a reused PID.
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        if stat[21]!=record['start_ticks']:raise ValueError('Recorded PID identity no longer matches')
        performance.atomic_write(root/'STOP_REQUESTED',now()+'\n')
        os.kill(pid,signal.SIGINT)
        return
    with (root/'experiment.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'STOP_REQUESTED').exists():
            if args.action not in {'resume','smoke-resume','smoke-recovery'}:raise ValueError('Safe stop requested; only explicit resume/recovery may continue')
            (root/'STOP_REQUESTED').rename(root/('STOP_ACKNOWLEDGED-'+str(time.time_ns())))
        performance.atomic_json(root/'active-job.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
            action=args.action,started=now()))
        try:
            if args.action=='prepare':prepare(root)
            elif args.action in {'probe','probe-followup'}:probe(root,followup=args.action=='probe-followup')
            elif args.action in {'smoke','smoke-resume'}:
                summary=execute(root,'smoke',SMOKE_PAGES,resume=args.action=='smoke-resume');reasons=smoke_gate(root,summary)
                if reasons:raise ValueError('Smoke gate blocked: '+'; '.join(reasons))
            elif args.action=='smoke-recovery':recover_smoke(root)
            else:
                gate=read(root/'smoke-gate.json')
                if gate.get('status')!='PASS_WITH_REVIEW':raise ValueError('Smoke handoff gate must pass before the 20-page run')
                execute(root,'full',PAGES,resume=args.action=='resume')
        except KeyboardInterrupt:
            state(root,'STOPPED',status='INCOMPLETE',reason='SIGINT; saved stage evidence preserved')
            return 130
        except BaseException as exc:
            state(root,'BLOCKED',status='INCOMPLETE',error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            from visual_pipeline_review import generate
            generate(root)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

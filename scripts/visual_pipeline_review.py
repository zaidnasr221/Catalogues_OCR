"""Offline diagnostic viewer and truthful inventory from saved stage evidence only."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import statistics

from as_ocr import delivery, performance, presentation


def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (OSError,ValueError): return default if default is not None else {}


def manifest_page(value):
    page=value.get('page_index',value.get('upstream',{}).get('page_index'))
    if isinstance(page,int):return page+1
    for key in ('stage3','stage2','stage4'):
        upstream=value.get('upstream',{}).get(key,{})
        if isinstance(upstream,dict):
            if isinstance(upstream.get('page_index'),int):return upstream['page_index']+1
            path=upstream.get('path',upstream.get('manifest_path'))
            if path and Path(path).is_file():
                result=manifest_page(read(path))
                if result:return result
    return None


def existing_paths(value, root):
    found=set()
    def walk(item):
        if isinstance(item,dict):
            for v in item.values():walk(v)
        elif isinstance(item,list):
            for v in item:walk(v)
        elif isinstance(item,str) and item.startswith(str(root)) and len(item)<4096:
            p=Path(item)
            if p.is_file():found.add(p)
    walk(value)
    return found


def timings(events):
    durations=sorted(e['duration_seconds'] for e in events if isinstance(e.get('duration_seconds'),(float,int)))
    return dict(requests=len(events),mean=statistics.mean(durations) if durations else None,
        median=statistics.median(durations) if durations else None,
        p95=durations[min(len(durations)-1,int(.95*len(durations)))] if durations else None,
        timeouts=sum(bool(e.get('timeout_classification')) for e in events),
        retries=sum(e.get('attempt',1)>1 for e in events),
        truncations=sum(e.get('finish_reason')=='length' for e in events),
        failures=sum(e.get('response_status')!='received' or e.get('local_validation_status')=='FAILED' for e in events),
        tokens=dict(prompt=sum((e.get('usage') or {}).get('prompt_tokens',0) for e in events),
            completion=sum((e.get('usage') or {}).get('completion_tokens',0) for e in events)))


def generate(root):
    root=Path(root).resolve();review=root/'review';review.mkdir(exist_ok=True)
    outcome=read(root/'full-outcome.json');smoke=read(root/'smoke-outcome.json')
    mode='full' if (root/'full').exists() else 'smoke-recovery' if (root/'smoke-recovery').exists() else 'smoke' if (root/'smoke').exists() else None
    public=next((root/mode).glob('*__*'),None) if mode else None
    run=next((public/'_internal/runs').glob('*'),None) if public and (public/'_internal/runs').exists() else None
    summary=read(run/'run-summary.json') if run else {}
    state=read(run/'run.json') if run else {}
    source=read(root/'source.json');experiment=read(root/'experiment-state.json')
    pages={p:{str(n):dict(status='NOT_RUN',notes=['No saved stage evidence yet'],files=[],images=[],details=[],evidence={})
        for n in range(1,9)} for p in range(1,21)}
    def url(path):return Path(os.path.relpath(path,review)).as_posix()
    def apply(p,stage,status,evidence,files=(),notes=()):
        if p not in pages:return
        paths=set(files)|existing_paths(evidence,root)
        item=pages[p][str(stage)]
        item.update(status=status,notes=list(notes),evidence=evidence,
            files=sorted(url(f) for f in paths if f.is_file() and f.suffix not in {'.sqlite3','.pyc'}),
            images=sorted(url(f) for f in paths if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg'}))
    for p in pages:
        folder=review/f'page_{p:04d}'/'stage_1'
        native=read(folder/'native-text.json')
        if native:apply(p,1,'COMPLETE',native,folder.glob('*'),['Source-verified original-page indexing; visual OCR is listed separately.'])
    detection={};groupings={}
    if run:
        for stage in (2,3,4,5):
            manifests=sorted((run/f'stage_{stage}').rglob('manifest.json'),key=lambda f:f.stat().st_mtime_ns)
            for path in manifests:
                saved=read(path);p=manifest_page(saved)
                if not p:continue
                files=list(path.parent.rglob('*'));files=[f for f in files if f.is_file() and not f.name.startswith('.')]
                if stage==2:
                    detection[p]=saved
                    status='FAILED' if saved.get('page_decision')=='FAILED' else 'PARTIAL' if saved.get('page_decision')=='REVIEW' else 'COMPLETE'
                    notes=[saved.get('page_decision','Unknown')]+saved.get('page_analysis',{}).get('review_reasons',[])
                elif stage==3:
                    groupings[p]=saved
                    status='PARTIAL' if any(d.get('grouping_review_reasons') or d.get('review_status')=='REVIEW' for d in saved.get('details',[])) else 'COMPLETE'
                    notes=['Grouping is provisional. Inspect primary and alternative memberships; execution does not establish assembly correctness.']
                else:
                    status=saved.get('state','PARTIAL')
                    if status not in {'FAILED','COMPLETE','PARTIAL'}:status='PARTIAL'
                    notes=['Per-view evidence is retained; inspect unresolved relations and inherited REVIEW flags.']
                apply(p,stage,status,saved,files,notes)
        for p,saved in detection.items():
            for n in (3,4,5):
                if pages[p][str(n)]['status']=='NOT_RUN':
                    if saved.get('page_decision')=='NO_DETAIL':
                        apply(p,n,'NOT_RUN',dict(reason='NO_DETAIL: downstream stage intentionally inapplicable',source_detection=saved),notes=['NO_DETAIL, supported eligibility decision. No fabricated empty success.'])
                    elif saved.get('page_decision')=='FAILED':
                        apply(p,n,'BLOCKED',{},notes=['Assembly detection failed; no downstream evidence.'])
        for stage in (6,7):
            paths=list((run/f'stage_{stage}').rglob('manifest.json' if stage==6 else 'run-summary.json'))
            for path in paths:
                saved=read(path)
                applicable=set()
                if stage==6:
                    for item in saved.get('sources',[]):
                        if isinstance(item.get('page_index'),int):applicable.add(item['page_index']+1)
                        for key in ('stage5_path','stage4_path','stage3_path'):
                            if item.get(key):
                                p=manifest_page(read(item[key]));
                                if p:applicable.add(p)
                    applicable.update(groupings)
                else:applicable.update(groupings)
                files=[f for f in path.parent.rglob('*') if f.is_file()]
                status='FAILED' if saved.get('execution_status')=='FAILED' else 'COMPLETE' if saved.get('complete') and saved.get('acceptance_status')!='REVIEW' else 'PARTIAL'
                for p in applicable:
                    apply(p,stage,status,saved,files,['Aggregate evidence: filter by Detail/source IDs. Missing or unsupported transfers remain withheld.'])
        for path in sorted(run.glob('page_ocr-*/page_*/result.json'),key=lambda f:f.stat().st_mtime_ns):
            saved=read(path);p=saved.get('source_page')
            if isinstance(p,int):
                files=[f for f in path.parent.rglob('*') if f.is_file()]
                md=public/'pages'/f'page_{p:04d}.md'
                if md.exists():files.append(md)
                apply(p,8,saved.get('status','FAILED'),saved,files,saved.get('issues',[]))
        for path in run.glob('page_ocr-*/page_*-failure.json'):
            saved=read(path);p=saved.get('source_page')
            if isinstance(p,int):apply(p,8,'FAILED',saved,[path],saved.get('issues',[]))
    for p,groups in groupings.items():
        details=groups.get('details',[])
        for item in pages[p].values():
            item['details']=[dict(id=d['id'],views=d.get('view_ids',[]),role=d.get('membership_role'),
                reasons=d.get('grouping_review_reasons',[])) for d in details]
    publication=read(public/'_internal/publication.json') if public else {}
    identity_issues=[];validation=dict(status='NOT_RUN',accuracy_verified=False,issues=identity_issues)
    if public:
        merged=public/'all_details.csv'
        if merged.is_file():
            try:
                rows=delivery.read_csv(merged);expected=[]
                for did,detail in sorted(publication.get('details',{}).items(),key=lambda pair:(pair[1]['source_page'],pair[1].get('order',0),pair[0])):
                    folder=public/'details'/did;path=folder/'detail.csv'
                    current=delivery.read_csv(path);expected+=current
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=detail['csv_sha256']:identity_issues.append(did+': CSV hash differs')
                    if not (folder/'image.png').is_file() or not (folder/'image.png').stat().st_size:identity_issues.append(did+': image missing')
                    if detail['document_id']!=source.get('sha256'):identity_issues.append(did+': foreign document')
                    if any(r['source_page']!=str(detail['source_page']) or r['detail_name']!=detail['detail_name'] for r in current):identity_issues.append(did+': source/name mismatch')
                    p=detail['source_page']
                    if p in pages:
                        item=pages[p]['7'];item['files']+=list(map(url,[path,folder/'image.png']));item['images'].append(url(folder/'image.png'))
                if rows!=expected:identity_issues.append('Merged CSV differs from ordered per-Detail rows')
                if any(int(r['source_page']) not in range(1,21) for r in rows):identity_issues.append('CSV contains page outside selected scope')
                validation.update(status='PARTIAL',rows=len(rows),details=len(publication.get('details',{})),
                    automated_identity_checks='FAIL' if identity_issues else 'PASS',
                    note='Schema, source identity, image existence, owned CSV hashes, original page/name, and ordered merge checked. Human glyph, grouping, physical counts, paragraph and classification review remains required.')
                stage7=publication.get('stage7_summary')
                if stage7 and Path(stage7).exists():
                    _,_,_,projected,traces,_,_,_=presentation.project(stage7,details_only=True)
                    validation['source_projection_rows']=len(projected)
                    validation['trace_records']=len(traces)
                    validation['field_statuses']=dict(Counter(t.get('status','REVIEW') for t in traces))
            except (ValueError,KeyError,TypeError,OSError) as exc:
                identity_issues.append(type(exc).__name__+': '+str(exc));validation['automated_identity_checks']='FAIL'
        # Display trace and validation files for each selected page without inventing fields.
        for p in pages:
            for filename in ('_internal/rows.jsonl','_internal/occurrence-audit.json','all_details.csv'):
                path=public/filename
                if path.is_file():pages[p]['7']['files'].append(url(path))
    visual=read(root/'accuracy-review.json')
    validation['source_visual_validation']=('FAILED' if visual.get('issues') else 'REVIEW') if mode==visual.get('scope') else 'PENDING_FOR_THIS_RUN'
    validation['source_visual_issues']=visual.get('issues',[]) if mode==visual.get('scope') else []
    validation['prior_smoke_accuracy_risks']=visual.get('issues',[]) if mode!=visual.get('scope') else []
    performance.atomic_json(root/'validation.json',validation)
    all_statuses=[]
    for p,stages in pages.items():
        all_statuses.append(dict(source_page=p,**{('markdown' if n=='8' else 'stage_'+n):v['status'] for n,v in stages.items()},
            detection_decision=detection.get(p,{}).get('page_decision','NOT_RUN')))
        for n,item in stages.items():
            folder=review/f'page_{p:04d}'/('markdown_ocr' if n=='8' else 'stage_'+n)
            folder.mkdir(parents=True,exist_ok=True)
            performance.atomic_json(folder/'status.json',dict(source_page=p,stage=n,**item))
    performance.atomic_json(root/'stage-status.json',dict(updated=datetime.now(timezone.utc).isoformat(),mode=mode,pages=all_statuses))
    # Small, complete 20-row CSV is useful even before any inference finishes.
    import io
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(all_statuses[0]));writer.writeheader();writer.writerows(all_statuses)
    performance.atomic_write(root/'stage-status.csv',stream.getvalue())
    events=[]
    if run:
        for path in (run/'performance').glob('requests-*.jsonl'):
            for line in path.read_text().splitlines():
                try:events.append(json.loads(line))
                except ValueError:pass # live trailing write is re-read on next refresh
    attempts=[e for e in events if e.get('network_started')]
    bystage=defaultdict(list);bypage=defaultdict(list)
    for e in attempts:
        stage='markdown' if e.get('stage')==1 else str(e.get('stage','unknown'))
        bystage[stage].append(e)
        if e.get('page'):bypage[str(e['page'])].append(e)
    bybudget=defaultdict(list)
    for e in attempts:
        bybudget[e.get('budget_scope','markdown_page' if e.get('stage')==1 else 'stage_'+str(e.get('stage','unknown')))].append(e)
    invocation=read(root/((mode or 'full')+'-invocation.json'))
    configured=read(root/'run-config.json')
    policies={'markdown':(1,None),'detection':(2,None),'grouping':(3,None),'callouts':(4,None),
        'components':(5,'component_extraction_page'),'profile_inventory':(5,'physical_profile_inventory_page'),
        'profile_verification':(5,'profile_verification_page'),'paragraphs':(5,'paragraph_extraction_page'),'enrichment':(6,None)}
    consumption={}
    for name,(stage,scope) in policies.items():
        relevant=[e for e in attempts if e.get('stage')==stage and (scope is None or e.get('budget_scope')==scope)]
        current=[e for e in relevant if e.get('started_at','')>=invocation.get('at','')]
        consumption[name]=dict(configured_budget=configured.get('budgets',{}).get(name),
            scope=configured.get('scope',{}).get(name),current_invocation_http_attempts=len(current),
            per_page_http_attempts=dict(Counter(str(e.get('page','UNKNOWN')) for e in current)),
            earlier_preserved_attempts=len(relevant)-len(current),
            note='Zero attempts does not establish a completed stage. Read the stage-status inventory.')
    worker_durations=[];starts={}
    for path in sorted(root.glob((mode or 'full')+'-page-events-*.jsonl')):
        for line in path.read_text().splitlines():
            try:e=json.loads(line)
            except ValueError:continue
            key=(e.get('pid'),e.get('stage'),e.get('page'))
            if e.get('event')=='start':starts[key]=e
            elif e.get('event')=='done' and key in starts:
                before=starts.pop(key)
                worker_durations.append(dict(stage=e.get('stage'),source_page=e.get('page'),
                    duration_seconds=e['at']-before['at'],started_at=before['at_utc'],finished_at=e['at_utc'],
                    ok=e.get('ok'),reused=e.get('reused'),reason=e.get('reason')))
    resource_peaks=[]
    resource_file=root/((mode or 'full')+'-resources.jsonl')
    if resource_file.exists():
        for line in resource_file.read_text().splitlines():
            try:sample=json.loads(line)
            except ValueError:continue
            if 'client_peak_in_flight' in sample:resource_peaks.append(sample['client_peak_in_flight'])
    metrics=dict(stage={k:timings(v) for k,v in bystage.items()},page={k:timings(v) for k,v in bypage.items()},
        budget_scopes={k:dict(**timings(v),configured_limits=sorted({e['configured_max_requests'] for e in v if 'configured_max_requests' in e})) for k,v in bybudget.items()},
        manual_budget_consumption=consumption,
        measured_page_worker_durations=worker_durations,
        timing_note='Page-worker intervals use saved original progress start/done timestamps where available. They include HTTP waits. Stage barriers and scheduling waits are included only in whole-run wall time. Independent Markdown has per-page duration_seconds in its result.json. Earlier smoke attempts predate this extra event recording; their page worker wall times are unavailable.',
        all=timings(attempts),observed_peak_client_in_flight=max([e.get('in_flight',0) for e in attempts]+resource_peaks,default=None),
        actual_effective_limit=read(root/'concurrency-choice.json').get('effective'),
        server_running_peak=None,server_waiting_peak=None,kv_cache_pressure=None,
        note='This server currently exposes no vLLM scheduler/KV metrics; null means unavailable, not zero. Request time includes server queue and computation; client admission wait is distinct.')
    performance.atomic_json(root/'performance-summary.json',metrics)
    source_text=json.dumps(source,ensure_ascii=False,indent=2)
    choice=read(root/'concurrency-choice.json');probes=read(root/'probe-results.json');server=read(root/'server.json')
    config=read(root/'run-config.json');cpu=read(root/'safety/cpu-gate.json');diagnosis=read(root/'runtime-diagnosis.json')
    comparison=read(root/'component-response-comparison.json')
    if mode=='smoke-recovery':smoke=read(root/'smoke-recovery-outcome.json')
    lines=['# Performance and accuracy review','',f'Updated: {datetime.now(timezone.utc).isoformat()}',
        f'Experiment: {experiment.get("phase","PREPARING")}. Active evidence: {mode or "source indexing only"}.',
        '**INCOMPLETE / accuracy REVIEW** until all selected pages, every applicable handoff, Markdown coverage, CSV identities and source visuals have been checked.',
        '', '## Source and scope','', '```json',source_text,'```','',
        'Only original pages 1–20 are authorized for the full experiment. Smoke/probe output is isolated and never reused as fresh extraction evidence.',
        '', '## Actual model and concurrency','',
        'Requested concurrency: **32**. Chosen inference permits: '+str(choice.get('effective','NOT_CHOSEN'))+'. Page workers: '+str(choice.get('page_workers','NOT_CHOSEN'))+'.',
        'Client HTTP peak in full/smoke workload: '+str(metrics['observed_peak_client_in_flight'])+'.',
        '```json',json.dumps(server,indent=2),'```','',
        'Running sequence capacity differs from client HTTP in-flight admission. A separate 64-sequence container was started only after verifying idle resources and stopping the otherwise unused four-sequence container. The latter remains preserved as a fallback. GPU-memory fraction 0.30, container memory limit 48 GiB, context 32768 and batch-token limit 8192 were unchanged. The healthy 64-sequence server has not been restarted.',
        metrics['note'],'',
        '## Runtime diagnosis and CPU gate','', '```json',json.dumps(dict(cpu_gate=cpu,runtime_diagnosis=diagnosis),indent=2),'```','',
        'The 370-test pass and the subsequent missing-review-file shell failure were different processes. Corrected-budget regressions were investigated and retained in their original logs; the final gate above identifies the passing code fingerprint.','',
        '## Manual request budgets','',
        'All ceilings are positive finite integers, accept 10/20/50/100 or higher, include retries, and do not force extra inference. No ceiling scales automatically with page/view count or retries. Pipeline defaults are Markdown 5, detection 160 total (fixed replacement for former automatic budget), grouping 2/page, callouts 10/page, components 6/page, profile inventory 6/page, profile verification 6/page, paragraphs 6/page, enrichment 10/run. The user\'s general ModelConfig default 48 is preserved. Deterministic grouping makes zero model requests even when its allowance is positive.',
        '', '| Work | Pipeline option | Scope | This experiment |', '|---|---|---|---:|']
    for key,option in [('markdown','--markdown-max-requests'),('detection','--stage2-max-requests'),('grouping','--stage3-max-requests'),('callouts','--stage4-max-requests'),('components','--stage5-max-requests'),('profile_inventory','--profile-inventory-max-requests'),('profile_verification','--profile-verification-max-requests'),('paragraphs','--paragraph-max-requests'),('enrichment','--stage6-max-requests')]:
        lines.append(f'| {key} | `{option}` | {config.get("scope",{}).get(key,"NOT_CONFIGURED")} | {config.get("budgets",{}).get(key,"NOT_CONFIGURED")} |')
    lines += ['', 'Exact supported pipeline invocation:','```bash',read(root/((mode or 'full')+'-invocation.json')).get('command','Not started yet.'),'```','',
        'Change options directly through `as-ocr pipeline`, or edit `run-config.json` before starting a new experiment through this terminal runner. Preserve the same settings for resume. Stage-specific standalone CLIs also accept `--max-model-requests`; components exposes the three independent substage budgets.','',
        '## Controlled short concurrency probes','',
        'These read one short literal label from source images. They establish overlap, transport health and short-response pressure, not end-to-end extraction accuracy or sustained long-output throughput.',
        '', '| Limit | Jobs | Peak client | Seconds | Requests/s | Failures | Max request s | Stable |', '|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in probes.get('levels',[]):
        s=r['summary'];lines.append(f'| {r["cap"]} | {r["jobs"]} | {r["peak_client_in_flight"]} | {s["elapsed_seconds"]:.2f} | {s["requests_per_second"]:.3f} | {s["failed_attempts"]} | {(s.get("durations_seconds") or {}).get("maximum")} | {r["stable"]} |')
    lines+=['',choice.get('reason','Probes pending.'),'','## Request and stage measurements','',
        '```json',json.dumps(metrics,indent=2),'```','',
        'End-to-end outcome:','```json',json.dumps(outcome or smoke,indent=2),'```','',
        'Stage wall times (not sums of overlapping request durations):','```json',json.dumps({k:{f:v.get(f) for f in ('status','duration_seconds','started_at','finished_at','model')} for k,v in summary.get('stages',{}).items()},indent=2),'```','',
        'Resource evidence: `probes/c*/resources.jsonl`, `probes-response256/c*/resources.jsonl`, `smoke-resources.jsonl`, `full-resources.jsonl`. GB10 total dedicated-memory accounting is unsupported; GPU compute-process memory (MiB), host MemAvailable and GPU utilization are sampled separately. A measured EngineCore process observation was 35,318 MiB; Docker RAM is a different measurement. Server scheduler/preemption/KV counters are unavailable with this active configuration. Startup logs report 240,298 KV-cache tokens and 7.33 maximum full-32768-token contexts; this is not a limit of seven short requests.',
        '', '## Targeted changes and measured limits','',
        '- Existing reusable async HTTP pool and shared cross-process bounded semaphore now accept 1–32 permits. More than four page workers requires async inference; scheduling retains bounded submitted futures and cancellation.',
        '- Stage 2 initial detection and full-page coverage audit share identical rendered pixels and mapping. Every coverage request and follow-up remains. CPU-only measured render comparison is in `safety/render-benchmark.json`; no model speedup is inferred from it.',
        '- Measured Stage 5 truncation recovery: compact JSON, a generation schema restricted to the exact fields already retained by Detail mode, and additional-information requests avoid restating known codes as names. All Stage 4 baseline occurrences and unknown/review evidence remain. Same-source individual component requests changed from truncated 4096-token responses at ~382 s to validated 1150 tokens/~115 s (page 7) and 963 tokens/~96 s (page 17). This is a provisional single-repeat request comparison, not a whole-pipeline accuracy or speedup claim.',
        '- Full pipeline explicitly selects existing compact region OCR with a manual 50-attempt page ceiling, 2048px targeted regions, saved source crops, independent source audits, page coverage, rotations and split-table recovery. No native text exists on these 20 source pages, so native shortcuts would not help this source.',
        '- Existing general ModelConfig max_requests=48 edit is preserved; Stage 3/4 local defaults explicitly retain ten attempts. Fixture budgets now explicitly match their stage contracts.',
        '- Empty/foreign/hash-mismatched compact checkpoints remain rejected. OCR COMPLETE/PARTIAL/FAILED reflects saved evidence. Partial scopes are retained. No image links are inserted into Markdown.',
        '- No comparable before/after GPU extraction experiment has been completed. The follow-up probes use the same 32 source readings at caps 8/16/32 and a fixed 256-token response ceiling. They establish short-request throughput only; long extraction accuracy and performance remain separate.',
        '- No Tesseract binary, pytesseract, EasyOCR, RapidOCR or PaddleOCR backend is installed. No new OCR dependency was installed.',
        '', 'Source-matched component measurements:','```json',json.dumps(comparison,indent=2),'```',
        '', '## Source validation and accuracy','', '```json',json.dumps(validation,indent=2),'```','',
        'Unverified component-leader counts remain blank; names and IDs require associated original callouts. Public CSVs have five columns; type prediction and paragraph descriptions are disabled. Independent repeated occurrences are retained. JSON evidence includes accepted/rejected relationships and withheld-value reasons.',
        '', '## All selected pages','', '| Original page | S1 | S2 | S3 | S4 | S5 | S6 | S7 | Markdown |', '|---:|---|---|---|---|---|---|---|---|']
    for row in all_statuses:lines.append('| '+str(row['source_page'])+' | '+' | '.join(row['stage_'+str(n)] for n in range(1,8))+' | '+row['markdown']+' |')
    lines+=['', '## Next corrections, in priority order','',
        '1. Inspect failed/partial detection coverage and uncertain group memberships against original-page images before trusting omitted or merged assemblies.',
        '2. Inspect exact callout glyphs, independent occurrences, target/leader relations and physical profile counts against the saved crops and overlays.',
        '3. Review Markdown missing scopes, fine annotations and every table against original region images; recover only demonstrated failures in a new isolated experiment.',
        '4. Resolve unsupported component/paragraph/reference associations using same-page source evidence, preserving withheld fields.',
        '5. Consider enabling scheduler/KV telemetry during an independently authorized future service maintenance window; do not restart this run for counters.',
        '6. Recommend pages 21–30 or 21–40 only after complete 20-page timing, failure rate, visual coverage, CSV source checks and resource stability justify it. No expansion is automatic.',
        '', 'Local visual navigation: [review/index.html](review/index.html). Status inventory: [stage-status.csv](stage-status.csv). All raw stage artifacts are linked from the viewer.', '']
    performance.atomic_write(root/'PERFORMANCE_REVIEW.md','\n'.join(lines))
    # Embed data/text because file:// browsers prohibit fetch() of local JSON.
    texts={}
    for stages in pages.values():
        for item in stages.values():
            for relative in item['files']:
                path=(review/relative).resolve()
                if path.suffix in {'.md','.txt','.csv'} and path.stat().st_size<512*1024:
                    texts[relative]=path.read_text(encoding='utf-8-sig',errors='replace')
    # Aggregate Stage 6/7 objects are shared across source pages. Embed each once
    # so an offline 20-page review does not duplicate the entire evidence graph.
    shared={};identities={};navigation={}
    for p,stages in pages.items():
        navigation[p]={}
        for n,item in stages.items():
            identity=id(item['evidence'])
            if identity not in identities:
                key=str(len(shared));identities[identity]=key;shared[key]=item['evidence']
            navigation[p][n]={k:v for k,v in item.items() if k!='evidence'}
            navigation[p][n]['evidence_key']=identities[identity]
    data=json.dumps(dict(pages=navigation,evidence=shared,texts=texts,experiment=experiment,mode=mode),ensure_ascii=False).replace('<','\\u003c')
    page=r'''<!doctype html><meta charset="utf-8"><title>OCR_Detaila stage evidence</title>
<style>body{font:15px system-ui;margin:18px;color:#17212b}select,button{padding:7px;margin:4px}header{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #bbb}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%;background:#eee}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f6;padding:12px;max-height:650px;overflow:auto}.status{font-weight:bold;padding:8px;background:#ffe8b7}a{overflow-wrap:anywhere}details{margin:8px 0}iframe{width:100%;height:500px;border:1px solid #ccc}</style>
<header><b>OCR_Detaila — original source and saved stage evidence</b><br>
<label>Original page <select id="page"></select></label><label>Stage <select id="stage"></select></label>
<label>Detail <select id="detail"></select></label><button onclick="location.reload()">Reload saved progress</button>
<a href="../PERFORMANCE_REVIEW.md">Performance report</a> · <a href="../stage-status.csv">All 20 page statuses</a> · <a href="contact-sheet.png">Source contact sheet</a>
<div id="phase"></div></header><div id="status" class="status"></div><pre id="notes"></pre>
<div class="grid"><section><h3>Original source page</h3><a id="originalLink"><img id="original"></a></section><section><h3>Processed overlay / region / final image</h3><select id="imageSelect"></select><a id="processedLink"><img id="processed"></a><p id="noImage"></p></section></div>
<h3>Extracted text / Markdown / CSV</h3><select id="textSelect"></select><pre id="text"></pre>
<details open><summary>Errors, REVIEW reasons, identities, coordinates, accepted/rejected relations and raw evidence</summary><pre id="evidence"></pre></details>
<details><summary>All stage artifacts (click for full original file)</summary><ul id="files"></ul></details>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);let item;
for(let n=1;n<=20;n++)$('page').add(new Option(n,n));
['Index and native text','Assembly detection and coverage','Section / Isometric grouping','Callout occurrences and targets','Components / physical profiles / same-page prose','Reference enrichment','Validation / Detail images / seven-column CSV','Independent full-page Markdown OCR'].forEach((x,i)=>$('stage').add(new Option((i+1)+' — '+x,i+1)));
function options(id,values){$(id).replaceChildren();values.forEach(v=>$(id).add(new Option(v,v)));}
function showImage(){let v=$('imageSelect').value;$('processed').src=v||'';$('processed').hidden=!v;$('processedLink').href=v||'#';$('noImage').textContent=v?'':'No processed image saved for this stage. See explicit status and evidence.';}
function showText(){$('text').textContent=DATA.texts[$('textSelect').value]||'No extracted text saved for this stage.';}
function draw(keepDetail=false){let p=$('page').value,s=$('stage').value;item=DATA.pages[p][s];$('status').textContent='Original page '+p+' | stage '+s+' | '+item.status;$('notes').textContent=item.notes.join('\n');
let original='page_'+p.padStart(4,'0')+'/stage_1/original.png';$('original').src=original;$('originalLink').href=original;
if(!keepDetail){$('detail').replaceChildren(new Option('All Details',''));item.details.forEach(d=>$('detail').add(new Option(d.id+' ('+d.role+')',d.id)));}
let did=$('detail').value;let selected=item.images;if(did){const d=item.details.find(d=>d.id===did);const subset=selected.filter(v=>v.includes(did)||d?.views.some(id=>v.includes(id)));if(subset.length)selected=subset;}
options('imageSelect',selected);showImage();options('textSelect',item.files.filter(v=>v in DATA.texts));showText();$('evidence').textContent=JSON.stringify({selected_detail:did||null,details:item.details,evidence:DATA.evidence[item.evidence_key]},null,2);
$('files').replaceChildren();item.files.forEach(v=>{const li=document.createElement('li'),a=document.createElement('a');a.href=v;a.textContent=v;li.append(a);$('files').append(li)});
$('phase').textContent='Experiment '+DATA.experiment.phase+' | evidence '+(DATA.mode||'source only')+' | refreshed '+DATA.experiment.at+' | Accuracy requires source review.';}
$('page').onchange=()=>draw();$('stage').onchange=()=>draw();$('detail').onchange=()=>draw(true);$('imageSelect').onchange=showImage;$('textSelect').onchange=showText;draw();</script>'''
    performance.atomic_write(review/'index.html',page.replace('__DATA__',data))
    return dict(mode=mode,run_dir=str(run) if run else None,page_statuses=all_statuses)

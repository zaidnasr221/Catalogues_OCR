"""Bounded scheduling, durable checkpoints and measured performance summaries."""
from __future__ import annotations

from concurrent.futures import wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import uuid


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def bounded_results(executor, function, tasks, maximum, cancel=None):
    """At most maximum submitted futures; yield completion order for durable saves."""
    if type(maximum) is not int or maximum<1:
        raise ValueError('Submission window must be positive')
    tasks = iter(tasks)
    pending = {}
    def fill():
        while len(pending)<maximum:
            try:
                task=next(tasks)
            except StopIteration:
                break
            pending[executor.submit(function,task)]=task
    fill()
    try:
        while pending:
            done,_ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
            for future in done:
                task=pending.pop(future)
                yield task,future
            fill()
    except BaseException:
        if cancel:
            cancel()
        for future in pending:
            future.cancel()
        raise


def validate_markdown(folder, document_id, page, contract=None):
    folder=Path(folder)
    try:
        raw=(folder/'result.json').read_text()
        if not raw.strip():
            raise ValueError('Empty OCR checkpoint')
        result=json.loads(raw)
        text=(folder/'page.md').read_text()
        if (not text.strip() or result['document_id']!=document_id or
                result['source_page']!=page or result['status'] not in {'COMPLETE','PARTIAL','FAILED'} or
                (contract is not None and result['contract']!=contract) or
                result.get('output_sha256')!=hashlib.sha256(text.encode()).hexdigest() or
                not text.startswith(f'# Original PDF page {page}\n') or
                f'**OCR status: {result["status"]}**' not in text):
            raise ValueError('OCR checkpoint source/status/hash mismatch')
        return result,text
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('Corrupt or missing OCR checkpoint: '+type(exc).__name__) from exc


def summarize(directory, elapsed, pages=(), *, synthetic=False):
    events=[];corrupt=[]
    for path in sorted(Path(directory).glob('requests-*.jsonl')):
        for number,line in enumerate(path.read_text().splitlines(),1):
            try:events.append(json.loads(line))
            except ValueError:corrupt.append(dict(file=str(path),line=number))
    attempts=[e for e in events if e.get('network_started')]
    completion=sum((e.get('usage') or {}).get('completion_tokens',0) for e in attempts)
    prompt=sum((e.get('usage') or {}).get('prompt_tokens',0) for e in attempts)
    statuses={}
    for row in pages:statuses[row['status']]=statuses.get(row['status'],0)+1
    failures=sum(e.get('response_status')!='received' or e.get('local_validation_status')=='FAILED' for e in attempts)
    return dict(elapsed_seconds=round(elapsed,6),page_outcomes=len(pages),page_statuses=statuses,
        pages_per_minute=len(pages)*60/elapsed if elapsed and not synthetic else None,
        synthetic_jobs_per_minute=len(pages)*60/elapsed if elapsed and synthetic else None,
        requests=len(attempts),requests_per_second=len(attempts)/elapsed if elapsed else None,
        prompt_tokens=prompt,completion_tokens=completion,
        completion_tokens_per_wall_second=completion/elapsed if elapsed and completion else None,
        usage_reported_requests=sum(bool(e.get('usage')) for e in attempts),
        timeouts=sum(bool(e.get('timeout_classification')) for e in attempts),
        retries=sum(e.get('attempt',1)>1 for e in attempts),
        truncations=sum(e.get('response_status')=='truncated' for e in attempts),
        budget_exhaustions=sum(e.get('response_status')=='budget_exhausted' for e in events),
        cancelled_attempts=sum(e.get('response_status')=='cancelled' for e in events),
        validation_passed=sum(e.get('local_validation_status')=='PASSED' for e in attempts),
        validation_not_run=sum(e.get('local_validation_status')=='NOT_RUN' for e in attempts),
        failed_attempts=failures,failure_rate=failures/len(attempts) if attempts else None,
        retry_rate=sum(e.get('attempt',1)>1 for e in attempts)/len(attempts) if attempts else None,
        observed_peak_in_flight=max((e['in_flight'] for e in attempts if 'in_flight' in e),default=None),
        corrupt_log_records=corrupt,synthetic=synthetic,accuracy_verified=False,
        server_queue_time_seconds=None)

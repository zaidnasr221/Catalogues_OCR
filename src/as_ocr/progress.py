"""Opt-in event-driven terminal progress; never determines extraction outcomes."""
from __future__ import annotations

from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
import os
import queue
import signal
import sys
import threading
import time

_sink = None
_cancel = None
_context = {}


def configure(sink, cancel):
    global _sink, _cancel, _context
    _sink, _cancel, _context = sink, cancel, {}


def worker_options():
    return {'initializer': configure, 'initargs': (_sink, _cancel)} if _sink is not None else {}


def cancelled():
    return _cancel is not None and _cancel.is_set()


def context():
    return dict(_context)


def emit(event, **values):
    if _sink is not None:
        _sink.put(dict(_context, event=event, pid=os.getpid(), at=time.monotonic(), **values))


def plan(stage, pages):
    emit('plan', planned_stage=stage, pages=list(pages))


def start(stage, page, operation):
    _context.update(stage=stage, page=page)
    emit('start', operation=operation)


def operation(stage, page, text):
    _context.update(stage=stage, page=page)
    emit('operation', operation=text)


def done(stage, page, *, ok=True, failed=None, details=None, callouts=None, empty=False, reused=False, reason=''):
    _context.update(stage=stage, page=page)
    emit('done', ok=ok, failed=not ok if failed is None else failed,
         details=details, callouts=callouts, empty=empty, reused=reused, reason=reason)


def request_start():
    emit('request_start')


def request_end(seconds, error=None, timeouts=0):
    emit('request_end', seconds=seconds, error=error, timeouts=timeouts)


class Reporter:
    def __init__(self, source, total, workers, path, *, interval=30, stream=None):
        self.source, self.total, self.workers = Path(source).name, total, workers
        self.path, self.interval = Path(path), interval
        self.stream = stream or sys.stdout
        self.started = time.monotonic()
        self.active, self.plans = {}, {}
        self.finished = defaultdict(set)
        self.completed, self.failed, self.incomplete = set(), set(), set()
        self.details, self.callouts = {}, {}
        self.errors = self.timeouts = 0
        self.stage = 1
        self.ctx = get_context('spawn')
        self.events, self.cancel = self.ctx.Queue(), self.ctx.Event()
        self.thread = None
        self.log = None
        self.previous_signal = None

    def write(self, text):
        line = text+'\n'
        self.stream.write(line); self.stream.flush()
        if self.log:
            self.log.write(line); self.log.flush()

    def location(self, event):
        page = event.get('page')
        return f'Page {page}/{self.total}' if page is not None else 'Catalog'

    def consume(self, event):
        kind, pid = event['event'], event['pid']
        if kind == 'plan':
            self.stage = max(self.stage, event['planned_stage'])
            self.plans[event['planned_stage']] = set(event['pages'])
            self.write(f'[Stage {self.stage}] {self.source} — scheduled {len(event["pages"])} source-page tasks' if self.stage != 6 else f'[Stage 6] {self.source} — catalog enrichment')
            return
        stage, page = event.get('stage', self.stage), event.get('page')
        prefix = f'[Stage {stage}] {self.location(event)} [worker {pid}]'
        self.stage = max(self.stage, stage)
        if kind in {'start','operation'}:
            self.active[pid] = event
            self.write(f'{prefix} — {event["operation"]}')
        elif kind == 'request_start':
            self.active[pid] = dict(event, request_at=event['at'], operation='awaiting VLM response')
            self.write(f'{prefix} — VLM request started; server queue/inference split unavailable')
        elif kind == 'request_end':
            active = self.active.get(pid)
            if active:
                active.pop('request_at', None)
                active['operation'] = 'validating response / saving source artifacts'
            self.errors += bool(event.get('error'))
            self.timeouts += event.get('timeouts', 0)
            status = 'FAILED: '+event['error'] if event.get('error') else 'response received (local validation follows)'
            self.write(f'{prefix} — VLM {status}; request elapsed={event["seconds"]:.1f}s')
        elif kind == 'done':
            active = self.active.get(pid)
            if active and active.get('stage') == stage and active.get('page') == page:
                self.active.pop(pid, None)
            self.finished[stage].add(page)
            if not event['ok'] and page is not None:
                (self.failed if event.get('failed',True) else self.incomplete).add(page)
            if event.get('empty') or stage == 7 and event['ok']:
                if page is not None: self.completed.add(page)
            if event.get('details') is not None: self.details[page] = event['details']
            if event.get('callouts') is not None: self.callouts[page] = event['callouts']
            status = 'finished' if event['ok'] else 'FAILED' if event.get('failed',True) else 'PARTIAL / review required'
            suffix = ' (reused)' if event.get('reused') else ''
            if event.get('empty'): suffix += '; confirmed no eligible Details'
            if event.get('details') is not None: suffix += f'; Details={event["details"]}'
            if event.get('callouts') is not None: suffix += f'; Callouts={event["callouts"]}'
            if event.get('reason'): suffix += '; '+event['reason']
            self.write(f'{prefix} — {status}{suffix}')

    def summary(self, now=None):
        now = now if now is not None else time.monotonic()
        planned = self.plans.get(self.stage, set(range(1,self.total+1)))
        active_pages = {e.get('page') for e in self.active.values() if e.get('stage')==self.stage}
        finished = self.finished[self.stage]
        pending = planned - finished - active_pages
        assignments = []
        for pid,e in sorted(self.active.items()):
            text = f'{pid}:S{e.get("stage")}/P{e.get("page") if e.get("page") is not None else "catalog"} {e.get("operation", "working")}'
            if e.get('request_at') is not None: text += f' (request wait {now-e["request_at"]:.1f}s)'
            assignments.append(text)
        self.write(f'[Status elapsed={now-self.started:.1f}s] {self.source} | Stage {self.stage} outcomes={len(finished & planned)}/{len(planned)} | '
                   f'pages completed={len(self.completed-self.failed-self.incomplete)}/{len(self.plans.get(1,set(range(1,self.total+1))))} selected | active={sorted(p for p in active_pages if p is not None)} | '
                   f'pending this stage={len(pending)} | Details={sum(self.details.values())} | Callouts={sum(self.callouts.values())} | '
                   f'failed pages={sorted(self.failed)} | incomplete/review pages={sorted(self.incomplete)} | request errors={self.errors}, timeouts={self.timeouts} | '
                   f'workers: {"; ".join(assignments) or "none active"}')

    def monitor(self):
        deadline = time.monotonic()+self.interval
        while True:
            try:
                event = self.events.get(timeout=max(.01, deadline-time.monotonic()))
                if event is None: break
                self.consume(event)
            except queue.Empty:
                pass
            if time.monotonic() >= deadline:
                self.summary()
                deadline = time.monotonic()+self.interval
        self.summary()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.path.open('x', encoding='utf-8')
        configure(self.events, self.cancel)
        self.previous_signal = signal.getsignal(signal.SIGINT)
        def interrupt(signum, frame):
            self.cancel.set()
            from . import async_inference
            async_inference.cancel()
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, interrupt)
        self.write(f'[Start] {self.source} | total pages={self.total} | page workers={self.workers} | status interval={self.interval}s')
        self.thread = threading.Thread(target=self.monitor, name='pipeline-progress', daemon=True)
        self.thread.start()
        plan(1, range(1,self.total+1))
        return self

    def __exit__(self, kind, value, trace):
        if kind is not None: self.cancel.set()
        self.events.put(None)
        self.thread.join()
        signal.signal(signal.SIGINT, self.previous_signal)
        configure(None, None)
        self.events.close(); self.events.join_thread()
        self.log.close()

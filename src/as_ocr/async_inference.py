"""Opt-in pooled asyncio HTTP; one shared admission limit across page processes.

PDF/PIL work and stage validation stay in their existing worker processes. Each
process reuses one event loop and AsyncClient; only network I/O runs on that loop.
Retry policy stays exclusively in QwenClient, never in the HTTP transport.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout, CancelledError as FutureCancelled
from contextlib import asynccontextmanager
from dataclasses import dataclass
import io
import json
from multiprocessing import get_context
from multiprocessing.util import Finalize
from pathlib import Path
import threading
import time
import urllib.error

import httpx

RESPONSE_LIMIT = 1024 * 1024
_state = None
_transport = None
_transport_lock = threading.Lock()


class AdmissionError(RuntimeError):
    pass


class SharedLimit:
    """Spawn-safe permits and counters, shared by the coordinator and all workers."""
    def __init__(self, maximum):
        if type(maximum) is not int or not 1 <= maximum <= 32:
            raise ValueError('Inference concurrency must be between 1 and 32')
        ctx = get_context('spawn')
        self.maximum = maximum
        self.permits = ctx.BoundedSemaphore(maximum)
        self.active = ctx.Value('i', 0)
        self.peak = ctx.Value('i', 0)
        self.stage2_remaining = ctx.Value('q', -1)
        self.cancel = ctx.Event()

    @asynccontextmanager
    async def slot(self, event):
        started = time.monotonic()
        acquired = False
        try:
            # Nonblocking acquisition avoids orphaned executor threads obtaining
            # a permit after their awaiting coroutine has already been cancelled.
            while not acquired:
                if self.cancel.is_set():
                    raise asyncio.CancelledError()
                acquired = self.permits.acquire(False)
                if not acquired:
                    await asyncio.sleep(.005)
            if self.cancel.is_set():
                raise asyncio.CancelledError()
            if event.get('stage') == 2:
                with self.stage2_remaining.get_lock():
                    if self.stage2_remaining.value == 0:
                        raise AdmissionError('Model request budget exhausted; analysis unfinished')
                    if self.stage2_remaining.value > 0:
                        self.stage2_remaining.value -= 1
            with self.active.get_lock():
                self.active.value += 1
                count = self.active.value
            with self.peak.get_lock():
                self.peak.value = max(self.peak.value, count)
            event.update(client_admission_wait_seconds=round(time.monotonic()-started, 6),
                         in_flight=count, max_in_flight=self.maximum, network_started=True)
            try:
                yield
            finally:
                with self.active.get_lock():
                    self.active.value -= 1
        finally:
            if acquired:
                self.permits.release()


class AsyncHTTPPool:
    """Genuine async transport, also usable directly by independent async callers."""
    def __init__(self, limit, *, transport=None):
        self.limit = limit
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=limit.maximum,
                               max_keepalive_connections=limit.maximum, keepalive_expiry=60),
            transport=transport, follow_redirects=False, trust_env=False)

    async def request(self, endpoint, body, headers, timeout, event):
        async with self.limit.slot(event):
            phase = 'awaiting_headers'
            network_started=time.monotonic()
            try:
                # A total HTTP deadline bounds trickling/partial responses too;
                # local admission wait is measured separately, outside this limit.
                async with asyncio.timeout(timeout):
                    async with self.client.stream('POST', endpoint, content=body,
                                                  headers=headers, timeout=timeout) as response:
                        event['headers_received_seconds'] = round(time.monotonic()-network_started,6)
                        if response.status_code>=300:
                            return response.status_code,dict(response.headers),b''
                        phase = 'reading_body'
                        chunks = bytearray()
                        async for chunk in response.aiter_bytes():
                            chunks.extend(chunk[:RESPONSE_LIMIT+1-len(chunks)])
                            if len(chunks) > RESPONSE_LIMIT:
                                break
                        return response.status_code, dict(response.headers), bytes(chunks)
            except (TimeoutError, httpx.TimeoutException) as exc:
                event['timeout_classification'] = phase + '_timeout'
                raise TimeoutError('Inference HTTP deadline exceeded') from exc
            except httpx.RequestError as exc:
                # No URL/body/header values in outward-facing error text.
                raise urllib.error.URLError(type(exc).__name__) from exc
            finally:
                event['network_duration_seconds']=round(time.monotonic()-network_started,6)

    async def aclose(self):
        await self.client.aclose()


@dataclass
class State:
    limit: SharedLimit
    directory: str


class _LoopTransport:
    def __init__(self, state):
        self.state = state
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever,
                                       name='inference-http', daemon=True)
        self.thread.start()
        async def create():
            return AsyncHTTPPool(state.limit)
        self.pool = asyncio.run_coroutine_threadsafe(create(), self.loop).result(10)

    def request(self, endpoint, body, headers, timeout, event):
        event['_network_clock'] = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(
            self.pool.request(endpoint, body, headers, timeout, event), self.loop)
        try:
            while True:
                try:
                    return future.result(.1)
                except FutureTimeout:
                    if future.done():
                        raise  # HTTP deadline, rather than a polling timeout
                    if self.state.limit.cancel.is_set():
                        raise KeyboardInterrupt('Inference workload cancelled')
                except FutureCancelled:
                    raise KeyboardInterrupt('Inference request cancelled') from None
        except BaseException:
            future.cancel()
            raise
        finally:
            event.pop('_network_clock', None)

    def close(self):
        async def close():
            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self.pool.aclose()
        try:
            asyncio.run_coroutine_threadsafe(close(), self.loop).result(5)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(5)
            if not self.thread.is_alive():
                self.loop.close()


def shutdown():
    global _transport
    if _transport is not None:
        transport, _transport = _transport, None
        transport.close()


def configure(state):
    global _state
    shutdown()
    _state = state


def _initialize(prior, prior_args, state):
    if prior:
        prior(*prior_args)
    configure(state)
    Finalize(None, shutdown, exitpriority=10)


def worker_options(base=None):
    base = base or {}
    if _state is None:
        return base
    return dict(initializer=_initialize,
                initargs=(base.get('initializer'), base.get('initargs', ()), _state))


class Workload:
    def __init__(self, maximum, directory):
        self.state = State(SharedLimit(maximum), str(directory))

    def __enter__(self):
        if _state is not None:
            raise RuntimeError('Nested inference workloads are not supported')
        Path(self.state.directory).mkdir(parents=True, exist_ok=True)
        configure(self.state)
        return self

    def __exit__(self, kind, value, trace):
        if kind is not None:
            self.state.limit.cancel.set()
        configure(None)


def set_stage2_budget(maximum):
    if _state is not None:
        with _state.limit.stage2_remaining.get_lock():
            _state.limit.stage2_remaining.value = maximum


def enabled():
    return _state is not None


def diagnostics_directory():
    return Path(_state.directory)/'validation' if _state is not None else None


def cancel():
    if _state is not None:
        _state.limit.cancel.set()


def open_response(endpoint, body, headers, timeout, event):
    """Synchronous compatibility boundary; CPU/PDF workers never own an async loop."""
    global _transport
    with _transport_lock:
        if _transport is None:
            _transport = _LoopTransport(_state)
    status, response_headers, raw = _transport.request(endpoint, body, headers, timeout, event)
    response_headers = httpx.Headers(response_headers)
    if status >= 400 or 300 <= status < 400:
        raise urllib.error.HTTPError(endpoint, status, 'Inference HTTP error', response_headers, io.BytesIO())
    response = io.BytesIO(raw)
    response.status = status
    response.headers = response_headers
    return response


def record(event):
    """One append-only performance stream per process, after local validation."""
    if _state is None:
        return
    import os
    safe = {k: v for k, v in event.items() if not k.startswith('_')}
    path = Path(_state.directory) / f'requests-{os.getpid()}.jsonl'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(safe, ensure_ascii=False, allow_nan=False)+'\n')

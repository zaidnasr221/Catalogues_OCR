"""Bounded synchronous adapter for an externally managed vision-capable Qwen server."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid
import warnings

from PIL import Image

@dataclass(frozen=True)
class ModelConfig:
    endpoint: str | None = None
    model: str | None = None
    identity: str | None = None  # operator-verified checkpoint AND revision, not alias
    timeout: float = 60
    max_tokens: int = 2048
    retries: int = 1
    max_requests: int = 48
    api_key: str | None = None
    temperature: float = 0

    def __post_init__(self):
        if type(self.max_requests) is not int or self.max_requests <= 0:
            raise ValueError('max_requests must be a finite positive integer')

    def public(self):
        result = asdict(self)
        result.pop('api_key')
        return result


class ModelError(RuntimeError):
    pass


@dataclass
class RequestBudget:
    """Manual attempt allowance for one scope; retries consume attempts.

    Convert the remaining allowance to the shared client's absolute counter
    while using a scope. The configured allowance never changes, and raising
    it never creates extra work. Inventory and verification use separate ledgers.
    """
    maximum: int
    name: str
    used: int = 0

    def __post_init__(self):
        if type(self.maximum) is not int or self.maximum <= 0:
            raise ValueError('Request budget must be a finite positive integer')

    @contextmanager
    def scope(self, client):
        if self.used >= self.maximum:
            raise ModelError('Model request budget exhausted; '+self.name+' unfinished')
        transport=client
        while hasattr(transport, '_client'):
            transport=transport._client
        before=transport.requests
        original=transport.config
        public_original=client.config
        scoped=replace(original,max_requests=before+self.maximum-self.used)
        transport.config=scoped
        client.config=scoped
        try:
            with request_metadata(budget_scope=self.name,configured_max_requests=self.maximum,
                                  budget_used_before=self.used):
                yield
        finally:
            self.used += transport.requests-before
            transport.config=original
            client.config=public_original

    def public(self):
        return dict(max_requests=self.maximum,requests=self.used,remaining=max(0,self.maximum-self.used),scope=self.name)


FAILED_RESPONSE_LIMIT = 64 * 1024
_metadata = ContextVar('inference_metadata',default={})


@contextmanager
def request_metadata(**values):
    token=_metadata.set(dict(_metadata.get(),**values))
    try:yield
    finally:_metadata.reset(token)


def _diagnostic_text(value, config, limit=200):
    if not isinstance(value, str):
        return None
    return (value.replace(config.api_key, '[REDACTED]') if config.api_key else value)[:limit]


@contextmanager
def capture_diagnostics(client, directory, *, stage, subject_id, validate=True):
    """Include local parsing in diagnostics; supports the pipeline's timed adapter.

    Test transports need no extra interface. No request payloads, HTTP error
    bodies, prompts or credentials are logged. Only failed generated text is kept.
    """
    transport = getattr(client, '_client', client)
    if not isinstance(transport, QwenClient):
        yield
        return
    previous = transport._diagnostic_scope
    scope = {'directory': Path(directory), 'stage': stage, 'subject_id': subject_id, 'events': []}
    transport._diagnostic_scope = scope
    try:
        yield
    except BaseException as exc:
        if scope['events']:
            event, _ = scope['events'][-1]
            if event['response_status'] == 'received':
                event.update(response_status='local_validation_failed', local_validation_status='FAILED',
                             error_type=type(exc).__name__)
                if hasattr(exc, 'errors'):
                    issues = exc.errors(include_input=False, include_context=False, include_url=False)
                    event['validation_issues'] = [
                        {'field': [_diagnostic_text(part, transport.config, 100) if isinstance(part, str) else part for part in e['loc']],
                         'type': e['type']} for e in issues[:100]]
                    event['validation_issue_count'] = len(issues)
                    event['validation_issues_truncated'] = len(issues) > 100
        raise
    else:
        if scope['events']:
            event, _ = scope['events'][-1]
            if event['response_status'] == 'received' and validate:
                event['local_validation_status'] = 'PASSED'
    finally:
        transport._diagnostic_scope = previous
        for event, content in scope['events']:
            try:
                folder = scope['directory']
                folder.mkdir(parents=True, exist_ok=True, mode=0o700)
                if event['response_status'] != 'received' and content is not None:
                    raw = content.encode('utf-8')
                    # Generated text may quote source content; redact a configured
                    # credential even if an endpoint unexpectedly echoed it.
                    if transport.config.api_key:
                        raw = raw.replace(transport.config.api_key.encode(), b'[REDACTED]')
                    bounded = raw[:FAILED_RESPONSE_LIMIT].decode('utf-8', errors='ignore').encode('utf-8')
                    path = folder / (event['request_id'] + '.response.txt')
                    with path.open('xb') as handle:
                        os.chmod(path, 0o600)
                        handle.write(bounded)
                    event.update(failed_response_path=str(path), failed_response_original_bytes=len(raw),
                                 failed_response_logged_bytes=len(bounded),
                                 failed_response_truncated=len(raw) > FAILED_RESPONSE_LIMIT,
                                 failed_response_sha256=hashlib.sha256(raw).hexdigest())
                path = folder / (event['request_id'] + '.json')
                with path.open('x', encoding='utf-8') as handle:
                    os.chmod(path, 0o600)
                    json.dump(event, handle, ensure_ascii=False, allow_nan=False, indent=2)
                    handle.write('\n')
            except OSError as exc:
                # A diagnostic disk failure must not hide the original parsing/
                # transport failure or turn a validated response into a failure.
                event['diagnostics_write_error'] = type(exc).__name__
                warnings.warn('Inference diagnostics could not be written: '+type(exc).__name__, RuntimeWarning)
            _performance_record(event)


def _performance_record(event):
    from . import async_inference
    try:
        async_inference.record(event)
    except (OSError,ValueError,TypeError) as exc:
        warnings.warn('Performance log could not be written: '+type(exc).__name__, RuntimeWarning)


class QwenClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.requests = 0
        self.response_models: set[str] = set()
        self.request_diagnostics: list[dict] = []
        self._diagnostic_scope = None
        if config.identity and ('@' not in config.identity or not all(config.identity.rsplit('@',1)) or config.identity == config.model):
            raise ValueError('Model identity must be checkpoint@immutable-revision, not a served alias')
        if config.endpoint:
            url = urllib.parse.urlsplit(config.endpoint)
            if url.scheme not in {'http','https'} or not url.netloc or url.username or url.password or url.query or url.fragment:
                raise ValueError('Endpoint must be an HTTP(S) API base URL without credentials/query/fragment')
        if not (0 <= config.retries <= 3 and 0 < config.timeout <= 600 and
                0 < config.max_tokens <= 32768 and type(config.max_requests) is int and config.max_requests > 0 and
                math.isfinite(config.temperature) and 0 <= config.temperature <= 2):
            raise ValueError('Invalid model timeout/token/retry/request limits')

    def request_json(self, image: Image.Image, prompt: str, context_text: str = '', *,
                     response_schema: dict | None = None):
        """Shared bounded vision transport; callers validate their stage schema."""
        cfg = self.config
        serialization_started = time.monotonic()
        if not cfg.endpoint or not cfg.model:
            raise ModelError('Qwen endpoint and served model alias are not configured; model check NOT RUN')
        if max(image.size) > 2048:
            raise ModelError('Request image exceeds pixel budget')
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='PNG')
        if buffer.tell() > 8 * 1024 * 1024:
            raise ModelError('Request image exceeds encoded byte budget')
        data_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
        payload = {'model': cfg.model, 'max_tokens': cfg.max_tokens, 'temperature': cfg.temperature,
                          'messages': [{'role': 'system', 'content': prompt+'\nEmit compact JSON without indentation or whitespace outside string values. Preserve original whitespace inside source-text strings, using JSON escapes for line breaks.'},
                                       {'role': 'user', 'content': [
                                           {'type': 'text', 'text': f'Image size: {image.width} x {image.height} pixels.'+('\n'+context_text if context_text else '')},
                                           {'type': 'image_url', 'image_url': {'url': data_url}}]}]}
        if response_schema is not None:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'stage2_regions', 'schema': response_schema, 'strict': True}}
        body = json.dumps(payload).encode()
        serialization_seconds = time.monotonic() - serialization_started
        endpoint = cfg.endpoint.rstrip('/') + '/chat/completions'
        headers = {'Content-Type': 'application/json'}
        if cfg.api_key:
            headers['Authorization'] = f'Bearer {cfg.api_key}'
        for attempt in range(cfg.retries + 1):
            if self.requests >= cfg.max_requests:
                raise ModelError('Model request budget exhausted; analysis unfinished')
            self.requests += 1
            started = time.monotonic()
            event = {'request_id': uuid.uuid4().hex, 'server_request_id': None,
                     'response_id': None, 'started_at': datetime.now(timezone.utc).isoformat(),
                     'attempt': attempt + 1, 'http_status': None, 'response_status': 'pending',
                     'local_validation_status': 'NOT_RUN', 'finish_reason': None, 'usage': None,
                     'timeout_classification': None, 'phase': 'awaiting_headers',
                     'image_size': list(image.size), 'png_bytes': buffer.tell(),
                     'request_bytes': len(body), 'timeout_seconds': cfg.timeout,
                     'max_tokens': cfg.max_tokens, 'serialization_seconds': round(serialization_seconds, 6)}
            event['configured_max_requests']=cfg.max_requests
            from . import async_inference, progress
            event.update(progress.context())
            event.update(_metadata.get())
            if self._diagnostic_scope is not None:
                event.update(stage=self._diagnostic_scope['stage'])
                event.setdefault('purpose',self._diagnostic_scope['subject_id'])
                subject = str(self._diagnostic_scope['subject_id']).split(':')
                if len(subject)>2 and len(subject[0])==64 and subject[1].isdigit():
                    event.setdefault('page',int(subject[1])+1)
            event.setdefault('purpose','vision_request')
            event['network_started'] = False
            content = None
            try:
                request = urllib.request.Request(endpoint, data=body,
                    headers={**headers, 'X-Request-Id': event['request_id']})
                if async_inference.enabled():
                    response_context = async_inference.open_response(endpoint, body, request.headers, cfg.timeout, event)
                else:
                    event['network_started'] = True
                    response_context = urllib.request.urlopen(request, timeout=cfg.timeout)
                with response_context as response:
                    event['http_status'] = getattr(response, 'status', None) or 200
                    response_headers = getattr(response, 'headers', {})
                    server_id = response_headers.get('X-Request-Id')
                    event['server_request_id'] = _diagnostic_text(server_id, cfg)
                    event.setdefault('headers_received_seconds',round(time.monotonic()-started,6))
                    event['phase'] = 'reading_body'
                    raw = response.read(1024 * 1024 + 1)
                event['response_bytes'] = len(raw)
                event['phase'] = 'decoding_envelope'
                if len(raw) > 1024 * 1024:
                    raise ModelError('Response exceeds byte budget')
                reply = json.loads(raw)
                if not isinstance(reply,dict):
                    raise ValueError('Expected response object')
                event['response_id'] = _diagnostic_text(reply.get('id'), cfg)
                usage = reply.get('usage')
                if isinstance(usage, dict):
                    event['usage'] = {k: v for k, v in usage.items()
                        if k in {'prompt_tokens', 'completion_tokens', 'total_tokens'}
                        and isinstance(v, int) and not isinstance(v, bool) and v >= 0} or None
                actual = reply.get('model')
                event['response_model'] = _diagnostic_text(actual, cfg)
                if not isinstance(actual, str) or not actual:
                    raise ModelError('Endpoint did not report response model identity/alias')
                self.response_models.add(actual)
                if actual != cfg.model:
                    raise ModelError('Response model differs from configured alias; verify identity')
                choice = reply['choices'][0]
                if not isinstance(choice,dict):
                    raise ValueError('Expected completion object')
                candidate = choice.get('message', {})
                if isinstance(candidate, dict) and isinstance(candidate.get('content'), str):
                    content = candidate['content']
                event['finish_reason'] = _diagnostic_text(choice.get('finish_reason'), cfg)
                if choice.get('finish_reason') != 'stop':
                    raise ModelError('Model output incomplete or truncated')
                if not isinstance(content,str):
                    raise ValueError('Response content must be text')
                event.update(response_status='received', phase='completed')
                return content
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                event.update(response_status='transport_error', error_type=type(exc).__name__)
                if isinstance(exc, urllib.error.HTTPError):
                    event.update(http_status=exc.code, response_status='http_error')
                    event['server_request_id'] = _diagnostic_text(
                        exc.headers.get('X-Request-Id') if exc.headers else None, cfg)
                    if exc.code in {408, 504}:
                        event['timeout_classification'] = 'http_timeout_status'
                elif isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(getattr(exc, 'reason', None), (TimeoutError, socket.timeout)):
                    if not event.get('timeout_classification'):
                        event['timeout_classification'] = event['phase'] + '_timeout'
                transient = not isinstance(exc, urllib.error.HTTPError) or exc.code in {429, 500, 502, 503, 504}
                if not transient or attempt == cfg.retries:
                    # Do not persist URLs/headers/body that might contain credentials.
                    raise ModelError(f'Qwen request failed: {type(exc).__name__}') from exc
                event['duration_seconds'] = round(time.monotonic() - started, 6)
                event['retry_delay_seconds'] = min(2, .25 * (2 ** attempt))
                time.sleep(event['retry_delay_seconds'])
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                event.update(response_status='malformed_envelope', error_type=type(exc).__name__)
                raise ModelError(f'Malformed model response: {type(exc).__name__}') from exc
            except ModelError as exc:
                event.update(response_status='truncated' if str(exc) == 'Model output incomplete or truncated'
                             else 'invalid_envelope', error_type=type(exc).__name__)
                raise
            except async_inference.AdmissionError as exc:
                self.requests -= 1  # admission rejection is not an HTTP attempt
                event.update(response_status='budget_exhausted',error_type='AdmissionError')
                raise ModelError(str(exc)) from exc
            except KeyboardInterrupt:
                event.update(response_status='cancelled',error_type='KeyboardInterrupt')
                raise
            except Exception as exc:
                event.update(response_status='request_error', error_type=type(exc).__name__)
                raise
            finally:
                event.setdefault('duration_seconds', round(time.monotonic() - started, 6))
                self.request_diagnostics.append(event)
                if self._diagnostic_scope is not None:
                    scope = self._diagnostic_scope
                    event.update(stage=scope['stage'], subject_id=scope['subject_id'])
                    scope['events'].append((event, content))
                else:
                    _performance_record(event)
        raise ModelError('Qwen request failed')

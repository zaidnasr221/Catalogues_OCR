"""Synthetic deployment checks; never start Docker or contact a model."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from as_ocr.qwen import ModelError


def module(name):
    spec=importlib.util.spec_from_file_location('vision_'+name,Path(__file__).parents[1]/'server'/(name+'.py'))
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_low_unified_memory_refuses_before_exec(monkeypatch,capsys):
    start=module('start')
    monkeypatch.setattr(start.Path,'read_text',lambda p:'MemTotal: 134217728 kB\nMemAvailable: 8388608 kB\n')
    monkeypatch.setattr(start.os,'execvp',lambda *a:pytest.fail('Must not initialize model under memory pressure'))
    assert start.main(['/model','--gpu-memory-utilization','0.30'])==78
    assert 'Existing services were left untouched' in capsys.readouterr().err


def test_missing_memory_accounting_refuses_startup(monkeypatch):
    start=module('start')
    monkeypatch.setattr(start.Path,'read_text',lambda p:'MemTotal: 134217728 kB\n')
    assert start.main(['/model','--gpu-memory-utilization','0.30'])==78


def test_admitted_start_preserves_exact_vllm_arguments(monkeypatch):
    start=module('start')
    monkeypatch.setattr(start.Path,'read_text',lambda p:'MemTotal: 134217728 kB\nMemAvailable: 100663296 kB\n')
    calls=[]
    monkeypatch.setattr(start.os,'execvp',lambda *a:calls.append(a))
    args=['/model','--gpu-memory-utilization','0.30','--max-num-seqs','1']
    start.main(args)
    assert calls==[('vllm',['vllm','serve',*args])]


def test_synthetic_checks_use_real_schema_and_do_not_disclose_identifier(monkeypatch,capsys):
    verify=module('verify')
    identifier='F29AB807'
    monkeypatch.setattr(verify.secrets,'token_hex',lambda n:identifier)
    payload={'decision':'CANDIDATES','page_kind':'engineering','evidence':['Visible drawing'],
             'coverage':'complete','uncertainty':[],'followups':[],'regions':[{
                 'id':'one','box':[.1,.2,.7,.8],'structural_box':None,'kind':'assembly_candidate',
                 'view_type':'section','title':None,'parent_id':None,
                 'evidence':['Visible connection section'],'uncertainty':[],'incomplete':False}]}
    class Client:
        config=SimpleNamespace(retries=0,max_requests=6)
        requests=0
        def request_json(self,image,prompt,context_text='', *, response_schema=None):
            assert identifier not in prompt
            self.requests+=1
            return json.dumps({'identifier':identifier} if self.requests==1 else payload)
    client=Client()
    verify.verify(client)
    assert client.requests==2
    assert identifier not in capsys.readouterr().out


def test_incorrect_image_identifier_stops_before_region_request(monkeypatch):
    verify=module('verify')
    class Client:
        config=SimpleNamespace(retries=0,max_requests=6)
        requests=0
        def request_json(self,*a):
            self.requests+=1
            return '{"identifier":"incorrect"}'
    client=Client()
    with pytest.raises(ModelError,match='mismatch'):
        verify.verify(client)
    assert client.requests==1


def test_synthetic_verification_rejects_unbounded_retry_configuration():
    verify=module('verify')
    client=SimpleNamespace(config=SimpleNamespace(retries=1,max_requests=6))
    with pytest.raises(ValueError,match='retries=0'):
        verify.verify(client)

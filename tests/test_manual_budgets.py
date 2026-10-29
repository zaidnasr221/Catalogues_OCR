"""Manual request ceilings are independent of work quantity and HTTP concurrency."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from as_ocr import components, page_markdown, pipeline, async_inference
from as_ocr.qwen import ModelConfig, ModelError, QwenClient, RequestBudget
from test_async_inference import server, model


@pytest.mark.parametrize('maximum',[1,10,20,50,100,10000])
def test_positive_manual_budgets_have_no_arbitrary_ceiling(maximum):
    assert ModelConfig(max_requests=maximum).max_requests==maximum
    page_markdown.CompactConfig(max_requests=maximum).validate()
    components.Config(profile_inventory_max_requests=maximum,
        profile_verification_max_requests=maximum,paragraph_max_requests=maximum).validate()


@pytest.mark.parametrize('maximum',[0,-1,True,1.5,float('inf')])
def test_invalid_manual_budgets_are_rejected(maximum):
    with pytest.raises(ValueError):ModelConfig(max_requests=maximum)
    with pytest.raises(ValueError):RequestBudget(maximum,'test')
    with pytest.raises(ValueError):page_markdown.CompactConfig(max_requests=maximum).validate()
    with pytest.raises(ValueError):components.Config(paragraph_max_requests=maximum).validate()


OPTIONS=['--markdown-max-requests','--stage2-max-requests','--stage3-max-requests',
    '--stage4-max-requests','--stage5-max-requests','--profile-inventory-max-requests',
    '--profile-verification-max-requests','--paragraph-max-requests','--stage6-max-requests']


@pytest.mark.parametrize('option',OPTIONS)
@pytest.mark.parametrize('bad',['0','-1'])
def test_cli_rejects_nonpositive_budgets(option,bad):
    with pytest.raises(SystemExit):pipeline._parser().parse_args(['x.pdf',option,bad])


def test_cli_overrides_and_model_propagation_are_independent(tmp_path):
    pdf=tmp_path/'source.pdf';pdf.write_bytes(b'%PDF-1.4\n')
    argv=[str(pdf),'--dry-run','--async-inference','--max-in-flight','32','--page-workers','32','--grouping-mode','hybrid',
          '--qwen-endpoint','http://fixture/v1','--qwen-model','fixture']
    for n,option in enumerate(OPTIONS):argv.extend([option,str(50+n)])
    args=pipeline._parser().parse_args(argv)
    result=pipeline.run_pipeline(args);limits=result['effective_limits']
    assert limits['max_in_flight']==32 and limits['page_workers']==32
    assert limits['page_markdown']['max_requests_per_page']==50
    assert limits['stage_2']['max_requests']==51
    assert limits['stage_3']['max_requests_per_page']==52
    assert limits['stage_4']['max_requests_per_page']==53
    assert limits['stage_5']['max_requests_per_page']==54
    assert limits['stage_5']['profile_inventory_max_requests_per_page']==55
    assert limits['stage_5']['profile_verification_max_requests_per_page']==56
    assert limits['stage_5']['paragraph_max_requests_per_page']==57
    assert limits['stage_6']['max_requests_total']==58
    cfg=pipeline._component_config(args)
    assert (cfg.profile_inventory_max_requests,cfg.profile_verification_max_requests,cfg.paragraph_max_requests)==(55,56,57)
    for budget in (args.stage2_max_requests,args.stage3_max_requests,args.stage4_max_requests,args.stage5_max_requests,args.stage6_max_requests):
        cfg=pipeline._model(args,max_requests=budget,timeout_seconds=60,max_tokens=512,retries=1)
        assert QwenClient(cfg).config.max_requests==budget
    args.max_in_flight=2
    assert pipeline.run_pipeline(args)['effective_limits']['stage_5']['max_requests_per_page']==54


def test_defaults_and_bounded_automatic_detection_budget():
    args=pipeline._parser().parse_args(['x.pdf'])
    assert (args.markdown_max_requests,args.stage3_max_requests,args.stage4_max_requests,args.stage5_max_requests,args.stage6_max_requests)==(5,2,10,6,10)
    assert (args.profile_inventory_max_requests,args.profile_verification_max_requests,args.paragraph_max_requests)==(6,6,6)
    assert ModelConfig().max_requests==48 # preserve the user's existing edit
    assert pipeline._stage2_budget(args,1)==8
    assert pipeline._stage2_budget(args,75)==600
    assert pipeline._view_budget(Path('unused'),100,3,True,extra=500)==100


def test_scoped_manual_limits_stop_and_do_not_force_additional_requests(tmp_path):
    with server(delay=.001) as (endpoint,observed):
        client=QwenClient(model(endpoint,max_requests=6))
        image=Image.new('RGB',(64,64),'white')
        inventory=RequestBudget(1,'inventory_page')
        verification=RequestBudget(100,'verification_page')
        with inventory.scope(client):client.request_json(image,'read')
        with pytest.raises(ModelError,match='budget exhausted'):
            with inventory.scope(client):client.request_json(image,'read')
        # A different independent budget remains available after inventory's cap.
        with verification.scope(client):client.request_json(image,'verify')
        assert inventory.used==1 and verification.used==1
        assert observed['requests']==2 and client.requests==2
        assert client.config.max_requests==6
        assert verification.public()['remaining']==99


def test_retries_consume_manual_attempts_and_restore_client_configuration():
    with server(replies=[(500,{},.001),(200,{},.001)]) as (endpoint,observed):
        client=QwenClient(model(endpoint,max_requests=100,retries=1))
        budget=RequestBudget(1,'callouts_page')
        with pytest.raises(ModelError):
            with budget.scope(client):client.request_json(Image.new('RGB',(64,64),'white'),'read')
        assert observed['requests']==1 and budget.used==1
        assert client.config.max_requests==100


def test_scope_restores_configuration_on_cancellation():
    client=SimpleNamespace(config=ModelConfig(max_requests=6),requests=0)
    budget=RequestBudget(100,'paragraphs')
    with pytest.raises(KeyboardInterrupt):
        with budget.scope(client):
            client.requests+=1
            raise KeyboardInterrupt()
    assert client.config.max_requests==6 and budget.used==1


def test_worker_tasks_receive_exact_component_and_substage_budgets(tmp_path,monkeypatch):
    args=pipeline._parser().parse_args(['x.pdf','--page-workers','4','--stage5-max-requests','100',
        '--profile-inventory-max-requests','55','--profile-verification-max-requests','56','--paragraph-max-requests','57'])
    submitted=[]
    class Future:
        def result(self):return {'result':{},'stats':{},'error':None}
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,fn,tasks):
            submitted.extend(tasks)
            return [Future().result() for t in tasks]
    monkeypatch.setattr(pipeline,'ProcessPoolExecutor',Pool)
    dirs={'stage_5':tmp_path/'s5'}
    list(pipeline._page_jobs(5,{0:{'manifest':'m4'}},args,dirs,tmp_path/'index.sqlite3',{'0':{'manifest':'m3'}}))
    keywords=submitted[0][2]
    assert keywords['model'].max_requests==100
    assert keywords['config'].profile_inventory_max_requests==55
    assert keywords['config'].profile_verification_max_requests==56
    assert keywords['config'].paragraph_max_requests==57

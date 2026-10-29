"""NEW_VOCR filename naming, output schema and finite detection budgets."""
from pathlib import Path
import csv
import hashlib
import json

import pytest

from as_ocr import pipeline,public_contract,presentation
from test_presentation import fixture,table,save


@pytest.mark.parametrize('filename,ordinal,expected',[
    ('catalog.pdf',1,'catalog_01'),('catalog.pdf',12,'catalog_12'),
    ('FWS 50.pdf',101,'FWS 50_101'),('كتالوج.pdf',2,'كتالوج_02'),
])
def test_filename_detail_name(filename,ordinal,expected):
    assert public_contract.detail_name(filename,ordinal)==expected


@pytest.mark.parametrize('ordinal',[0,-1,True,1.5])
def test_invalid_ordinal_rejected(ordinal):
    with pytest.raises(ValueError):public_contract.detail_name('catalog.pdf',ordinal)


def test_five_columns_and_repeated_components_keep_one_detail_name(tmp_path):
    rows=presentation.project(fixture(tmp_path,repeat=True))[3]
    assert [r['detail_name'] for r in rows]==['catalog_01','catalog_01','catalog_02']
    assert all(list(r)==['detail_name','profile_count','profile_id','component_name','source_page'] for r in rows)


def test_detail_sequence_continues_across_pages_and_input_order(tmp_path):
    summary=fixture(tmp_path);art=json.loads(summary.read_text())['artifacts']
    path=Path(art['details'])
    with path.open(encoding='utf-8-sig',newline='') as handle:details=list(csv.DictReader(handle))
    second=next(d for d in details if d['id']=='d2');second['page_index']='11';second['page_id']+='-later'
    table(path,list(reversed(details)))
    link=Path(art['enrichment']);s6=json.loads(link.read_text());source=s6['sources'][0]
    path=Path(source['stage3_path']);s3=json.loads(path.read_text())
    for d in s3['details']:
        if d['id']=='d2':d.update(page_index=11,page_id=second['page_id'])
    for v in s3['views']:
        if v['id']=='v2':v.update(page_index=11,page_id=second['page_id'])
    s3['details'].reverse();save(path,s3)
    source['stage3_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();save(link,s6)
    rows=presentation.project(summary)[3]
    assert [(r['detail_name'],r['source_page']) for r in rows]==[('catalog_01','1'),('catalog_02','12')]


@pytest.mark.parametrize('pages',[1,75,208,230,144,126,783])
def test_detection_budget_covers_full_page_followups_coverage_and_retries(pages):
    args=pipeline._parser().parse_args(['catalog.pdf'])
    assert pipeline._stage2_budget(args,pages)==pages*8
    args.stage2_retries=0;args.stage2_max_followups=0
    assert pipeline._stage2_budget(args,pages)==pages*2
    args.stage2_max_requests=7
    assert pipeline._stage2_budget(args,pages)==7


def test_unknown_dry_run_budget_is_not_a_fake_fixed_total():
    args=pipeline._parser().parse_args(['catalog.pdf'])
    assert pipeline._stage2_budget(args,0) is None

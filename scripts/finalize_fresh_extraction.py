"""Reconcile literal labels from this fresh run, then validate/deliver without OCR."""
import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
import uuid

from as_ocr import components,delivery,detection,enrichment,exports,grouping,indexer,profile_observations
from as_ocr.qwen import ModelConfig,QwenClient


def main():
    raise RuntimeError("Historical physical-profile/type/description recipe is disabled. Use the current five-column as-ocr pipeline/deliver contract; preserve existing outputs.")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--verify-geometry',action='store_true')
    parser.add_argument('--fresh-geometry',action='store_true',help='Reobserve geometry from original PDF and verify it; never reuse an inventory')
    parser.add_argument('--run-id')
    args=parser.parse_args()
    project=Path.cwd().resolve();output=project/'output'
    audit=json.loads((output/'_internal/cleanup-audit.json').read_text())
    manifest=args.manifest.resolve()
    if output not in manifest.parents or not audit['recreated_empty']:
        raise ValueError('Only this freshly cleaned output may be reconciled')
    saved=json.loads(manifest.read_text())
    if not saved['settings']['configuration'].get('observed_profiles'):
        raise ValueError('A fresh source geometry inventory is required')
    s2,views,details,_,upstream=components.load_inputs(saved['upstream']['stage3']['path'],saved['upstream']['stage4']['path'])
    for link in ('stage3','stage4'):
        if output not in Path(upstream[link]['path']).resolve().parents:
            raise ValueError('An earlier extraction source is forbidden')
    components.check_upstream(upstream,views,Path(s2['source_path']))
    root=next(p for p in manifest.parents if p.parent==output)
    if args.run_id and not re.fullmatch('[A-Za-z0-9_-]+',args.run_id):raise ValueError('Unsafe run identifier')
    folder=root/'_internal'/(args.run_id or ('fresh-geometry' if args.fresh_geometry else 'fresh-verified' if args.verify_geometry else 'fresh-finalized'))
    folder.mkdir(exist_ok=False)
    if args.verify_geometry or args.fresh_geometry:
        model=ModelConfig(endpoint=os.environ['AS_OCR_QWEN_ENDPOINT'],model=os.environ['AS_OCR_QWEN_MODEL'],
            identity=os.environ.get('AS_OCR_QWEN_IDENTITY'),api_key=os.environ.get('AS_OCR_QWEN_API_KEY'),
            timeout=360,max_tokens=8192,retries=0,max_requests=len(details)*(2 if args.fresh_geometry else 1))
        client=QwenClient(model)
        with sqlite3.connect(args.index) as db:
            db.row_factory=sqlite3.Row;row=db.execute('SELECT * FROM pages WHERE id=?',(s2['page_id'],)).fetchone()
            geometry,_,_=detection.load_page(db,row)
        if args.fresh_geometry:
            saved['profile_observations']=profile_observations.extract(Path(s2['source_path']),s2['page_index'],geometry,
                details,views,saved['candidates'],saved['observations'],folder,client)
        else:
            saved['profile_observations']=profile_observations.verify_existing(Path(s2['source_path']),s2['page_index'],geometry,
                details,views,saved['candidates'],saved['observations'],saved['profile_observations'],folder,client)
        saved['geometry_verification']=dict(model=model.public(),requests=client.requests)
        saved['model_requests']+=client.requests
        components.check_upstream(upstream,views,Path(s2['source_path']))
    fields=profile_observations.recover_label_fields(saved['profile_observations'],saved['candidates'],saved['observations'],details)
    cs,obs,ids=components.summarize(saved['candidates'],saved['observations']+fields,saved['scope_observations'])
    saved.update(run_id=uuid.uuid4().hex,candidates=cs,observations=obs,identities=ids,review_path=str(folder/'review.md'))
    saved['settings']['profile_contract']=profile_observations.contract()
    saved['fresh_source_processing']=dict(path=str(manifest),sha256=grouping.file_digest(manifest),
        note='Literal name/code label reconciliation from this fresh extraction; zero additional OCR requests')
    result=folder/'manifest.json';components.write_review(folder/'review.md',saved);result.write_text(indexer.dumps(saved))
    s6=enrichment.run([result],folder/'stage_6',index_path=args.index,model=ModelConfig(max_requests=0),force=True)
    s7=exports.run(s6['manifest'],folder/'stage_7',index_path=args.index,force=True)
    published=delivery.publish(s7['artifacts']['summary'])
    print(json.dumps(dict(delivery=published,stage5=str(result),validation_errors=s7['validation_errors'])),flush=True)


if __name__=='__main__':main()

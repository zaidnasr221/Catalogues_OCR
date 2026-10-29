"""Bounded paragraph-only validation using saved, verified component extraction."""
from pathlib import Path
import argparse
import copy
import json
import os
import sqlite3
import uuid

from as_ocr import components, delivery, detection, enrichment, exports, grouping, indexer, paragraphs
from as_ocr.qwen import ModelConfig, QwenClient


def main():
    raise RuntimeError("Historical physical-profile/type/description recipe is disabled. Use the current five-column as-ocr pipeline/deliver contract; preserve existing outputs.")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage5',type=Path)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--verify-only',action='store_true',help='Reuse saved page association, independently reread its source paragraphs')
    args=parser.parse_args()
    saved=json.loads(args.stage5.read_text())
    s2,views,details,s4,upstream=components.load_inputs(saved['upstream']['stage3']['path'],saved['upstream']['stage4']['path'])
    root=args.output/delivery.identifier(Path(s2['source_path']).name,s2['document_id'])
    folder=(root/'_internal'/args.run_id).resolve();folder.mkdir(parents=True,exist_ok=False)
    index=folder/'index.sqlite3'
    with sqlite3.connect(args.index.resolve().as_uri()+'?mode=ro',uri=True) as origin,sqlite3.connect(index) as target:
        origin.backup(target)
    model=ModelConfig(endpoint=os.environ['AS_OCR_QWEN_ENDPOINT'],model=os.environ['AS_OCR_QWEN_MODEL'],
        identity=os.environ.get('AS_OCR_QWEN_IDENTITY'),api_key=os.environ.get('AS_OCR_QWEN_API_KEY'),
        timeout=360,max_tokens=8192,retries=0,max_requests=2)
    candidates=copy.deepcopy(saved['candidates'])
    observations=[copy.deepcopy(o) for o in saved['observations'] if o['field']!='description']
    for o in observations:
        o['conflicts_with']=[]
    with sqlite3.connect(index) as db:
        db.row_factory=sqlite3.Row
        row=db.execute('SELECT * FROM pages WHERE id=?',(s2['page_id'],)).fetchone()
        geometry,spans,signature=detection.load_page(db,row)
    components.check_upstream(upstream,views,Path(s2['source_path']))
    client=QwenClient(model)
    result,fields=paragraphs.extract(Path(s2['source_path']),s2['page_index'],geometry,spans,
        views,details,candidates,observations,folder,client,prior=saved['paragraph_analysis'] if args.verify_only else None,
        profile_results=saved.get('profile_observations',[]))
    components.check_upstream(upstream,views,Path(s2['source_path']))
    candidates,observations,identities=components.summarize(candidates,observations+fields,saved['scope_observations'])
    saved.update(run_id=uuid.uuid4().hex,candidates=candidates,observations=observations,identities=identities,
        paragraph_analysis=result,review_path=str(folder/'review.md'),model_requests=client.requests)
    saved['settings']['paragraph_contract']=paragraphs.contract()
    saved['settings']['configuration']['paragraph_descriptions']=True
    saved['settings']['paragraph_model']=model.public()
    saved['reused_component_evidence']={'path':str(args.stage5.resolve()),'sha256':grouping.file_digest(args.stage5)}
    if result['coverage']!='complete' or result['uncertainty'] or result.get('review_required'):
        saved.update(state='PARTIAL',complete=False)
    components.write_review(folder/'review.md',saved)
    manifest=folder/'manifest.json';manifest.write_text(indexer.dumps(saved))
    s6=enrichment.run([manifest],folder/'stage_6',index_path=index,model=ModelConfig(max_requests=0),force=True)
    s7=exports.run(s6['manifest'],folder/'stage_7',index_path=index,force=True)
    result=delivery.publish(s7['artifacts']['summary'],output=args.output)
    print(json.dumps(dict(delivery=result,stage5=str(manifest),validation_errors=s7['validation_errors'])),flush=True)


if __name__=='__main__':
    main()

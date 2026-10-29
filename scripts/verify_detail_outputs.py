"""Continue explicitly reviewed regression pages through the existing Stages 4–7."""
from pathlib import Path
import argparse
import json
import os
import sqlite3
from as_ocr import callouts, components, enrichment, exports, delivery
from as_ocr.qwen import ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--source-contains', required=True)
    parser.add_argument('--pages', type=int, nargs='+', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--reuse-stage4', type=Path)
    parser.add_argument('--additional-stage5', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    records = [r for r in json.loads(args.results.read_text())
               if args.source_contains in r['source'] and r['page'] in args.pages]
    if {r['page'] for r in records} != set(args.pages) or len({r['source'] for r in records}) != 1:
        raise ValueError('Select completed pages from exactly one source PDF')
    source = Path(records[0]['source'])
    s2 = json.loads(Path(records[0]['detection']['manifests'][0]).read_text())
    root = args.output/delivery.identifier(source.name, s2['document_id'])
    work = root/'_internal'/args.run_id
    work.mkdir(parents=True, exist_ok=False)
    index = work/'index.sqlite3'
    with sqlite3.connect(Path(records[0]['index']).as_uri()+'?mode=ro', uri=True) as origin, sqlite3.connect(index) as target:
        origin.backup(target)
    def model(budget, tokens=8192):
        return ModelConfig(endpoint=os.environ['AS_OCR_QWEN_ENDPOINT'],
            model=os.environ['AS_OCR_QWEN_MODEL'], identity=os.environ.get('AS_OCR_QWEN_IDENTITY'),
            api_key=os.environ.get('AS_OCR_QWEN_API_KEY'), timeout=360,
            max_tokens=tokens, retries=0, max_requests=budget)
    if args.reuse_stage4 and len(records) != 1:
        raise ValueError('A saved Stage 4 manifest must select exactly one source page')
    saved = [str(p.resolve()) for p in args.additional_stage5]
    for record in records:
        stage2 = record['detection']['manifests'][0]
        stage3 = record['grouping']['manifests'][0]
        grouping = json.loads(Path(stage3).read_text())
        count = len({v for d in grouping['details'] for v in d['view_ids']})
        print(f'{source.name} original page {record["page"]}: {count} Detail views', flush=True)
        stage4 = ({'manifest':str(args.reuse_stage4.resolve()), 'reused':True} if args.reuse_stage4 else
            callouts.run(stage2, work/'stage_4', stage3=stage3, index_path=index,
                config=callouts.Config(details_only=True), model=model(max(10, count)),
                complete_views=True, force=True))
        print('Stage 4 '+json.dumps(stage4), flush=True)
        delivery.review_stage(root, 4, [stage4['manifest']], outcome=stage4)
        stage5 = components.run(stage3, stage4['manifest'], work/'stage_5', index_path=index,
            config=components.Config(details_only=True, paragraph_descriptions=False, observed_profiles=False), model=model(max(6, count+2+2*len(grouping['details']))),
            complete_views=True, force=True)
        print('Stage 5 '+json.dumps(stage5), flush=True)
        saved.append(stage5['manifest'])
        (work/'completed.json').write_text(json.dumps(saved, indent=2))
    stage6 = enrichment.run(saved, work/'stage_6', index_path=index,
                            model=model(1), force=True)
    stage7 = exports.run(stage6['manifest'], work/'stage_7', index_path=index, force=True)
    print('Stage 7 '+json.dumps(stage7), flush=True)
    result = delivery.publish(stage7['artifacts']['summary'], output=args.output)
    (work/'delivery.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

"""Explicit six-page regression; never runs a complete catalog."""
from pathlib import Path
import argparse
import json
import os
import fitz
from as_ocr import indexer, detection, grouping
from as_ocr.qwen import ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--source-contains')
    parser.add_argument('--pages', type=int, nargs='+')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    cases = [('Data/Alupco/ALUPCO50SG CAT.(3rd ED 2011).pdf', [7, 9, 12, 23]),
             ('Data/1.)FW 50+SG/SECTION SG bind.pdf', [1]),
             ('Data/1.)FW 50+SG/FW 50+SGconnectionbind.pdf', [1])]
    model = ModelConfig(endpoint=os.environ['AS_OCR_QWEN_ENDPOINT'],
        model=os.environ['AS_OCR_QWEN_MODEL'], identity=os.environ.get('AS_OCR_QWEN_IDENTITY'),
        api_key=os.environ.get('AS_OCR_QWEN_API_KEY'), timeout=360,
        max_tokens=8192, retries=0, max_requests=8)
    records = []
    for number, (filename, pages) in enumerate(cases):
        if args.source_contains and args.source_contains not in filename:
            continue
        if args.pages:
            pages = [p for p in pages if p in args.pages]
        if not pages:
            continue
        source = Path(filename).resolve()
        folder = args.root / f'case-{number}'
        indexer.run(source, folder/'stage_1', max_pages=max(pages), preview_max=1200)
        index = folder/'stage_1/index.sqlite3'
        with fitz.open(source) as pdf:
            total = len(pdf)
        for page in pages:
            print(f'{source.name}: original page {page}/{total}', flush=True)
            result = detection.run(source, folder/'stage_2', model=model,
                index_path=index, pages=[page], force=True)
            record = dict(source=str(source), page=page, total_pages=total,
                          index=str(index.resolve()), detection=result)
            if result['manifests']:
                record['grouping'] = grouping.run(result['manifests'], folder/'stage_3',
                                                  index_path=index, force=True)
            records.append(record)
            (args.root/'results.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

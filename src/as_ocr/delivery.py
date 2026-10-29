"""Readable Detail delivery from the existing, validated Stage 7 evidence graph."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from PIL import Image

from . import presentation,leaders,traceability


STAGES = {
    1: ('Page indexing and full-page OCR', 'Index original PDF pages and independently transcribe visible text to Markdown.'),
    2: ('Assembly detection', 'Find eligible assemblies and audit complete-page coverage.'),
    3: ('Grouping and cropping', 'Group only supported related views and retain annotation context.'),
    4: ('Callout extraction', 'Read individual occurrences and verify their drawing associations.'),
    5: ('Component information', 'Verify literal same-callout component names and identifiers; no type or paragraph inference.'),
    6: ('Evidence reconciliation', 'Retain the existing evidence dependencies and coverage checks.'),
    7: ('Validation and delivery', 'Validate component evidence and publish Detail images and CSVs.'),
}


def identifier(filename, sha):
    stem = re.sub('[^A-Za-z0-9_-]+', '_', Path(filename).stem).strip('_')[:80] or 'pdf'
    return stem + '__' + sha


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, presentation.COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(presentation.COLUMNS):
            raise ValueError('Unexpected Detail CSV columns: ' + str(path))
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError('Malformed Detail CSV: ' + str(path))
    if any((r['profile_count'] and not re.fullmatch(r'0|[1-9][0-9]*',r['profile_count'])) or
           not re.fullmatch(r'[1-9][0-9]*',r['source_page']) for r in rows):
        raise ValueError('Detail CSV needs an verified leader count or withheld value, and original 1-based page')
    if any(presentation.unsafe(value) for row in rows for value in row.values()):
        raise ValueError('Unsafe public CSV field')
    return rows


def merge(root):
    """Read only the owned per-Detail CSVs; keep identical distinct occurrences."""
    root = Path(root)
    manifest = json.loads((root/'_internal/publication.json').read_text())
    expected = manifest['details']
    paths = sorted((root/'details').glob('*/detail.csv'))
    if {p.parent.name for p in paths} != set(expected):
        raise ValueError('Detail CSV inventory differs from publication manifest')
    paths.sort(key=lambda p: (expected[p.parent.name]['source_page'],
                             expected[p.parent.name].get('order', 0), p.parent.name))
    rows = []
    for path in paths:
        detail = expected[path.parent.name]
        if detail['document_id'] != manifest['document_id']:
            raise ValueError('Foreign PDF in Detail inventory')
        current = read_csv(path)
        if len({r['profile_count'] for r in current})>1:
            raise ValueError('Detail rows disagree on observed profile count')
        if manifest.get('contract_version') in {2,3,4,5} and (any(r['profile_count']!=detail['profile_count'] for r in current) or
                any(r['profile_count']=='0' and detail['profile_count_status'] not in {'VERIFIED_ZERO','OBSERVED_PARTIAL'} for r in current)):
            raise ValueError('Detail count differs from validated component-leader observation state')
        if not current or any(r['source_page'] != str(detail['source_page']) or
                              r['detail_name'] != detail['detail_name'] for r in current):
            raise ValueError('Detail name/source page mismatch')
        if presentation.digest(path.read_bytes()) != detail['csv_sha256']:
            raise ValueError('Detail CSV changed after validated publication')
        rows.extend(current)
    temporary = root/'_internal/all_details.tmp'
    write_csv(temporary, rows)
    if read_csv(temporary) != rows:
        raise ValueError('Merged CSV does not match all per-Detail rows')
    os.replace(temporary, root/'all_details.csv')
    return len(rows)


def publish_empty(root, source, document_id, *, complete):
    """Explicit successful NO_DETAIL processing still gets a header-only CSV."""
    root = Path(root)
    details = root/'details'
    if details.exists() and any(details.iterdir()):
        raise ValueError('New empty detection conflicts with an existing Detail publication; review required')
    details.mkdir(parents=True, exist_ok=True)
    internal = root/'_internal'
    internal.mkdir(exist_ok=True)
    manifest = dict(contract_version=5,profile_count_contract=leaders.COMPONENT_COUNT_CONTRACT,document_id=document_id, source_filename=Path(source).name,
                    details={}, status='NO_DETAIL' if complete else 'PARTIAL / NO_DETAIL on selected pages')
    (internal/'publication.json').write_text(json.dumps(manifest, indent=2))
    merge(root)
    return dict(root=str(root), details=0, rows=0, status=manifest['status'])


def review_stage(root, number, manifests=(), *, outcome=None):
    """One readable report per actual stage, preserving observed uncertainty."""
    folder = Path(root)/'stages'/f'stage_{number:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    name, purpose = STAGES[number]
    lines = [f'# Stage {number}: {name}', '', purpose, '',
             'Processing completion is not an accuracy certification.', '']
    if outcome is not None:
        lines += ['## Execution outcome', '']
        values = {**outcome, **outcome.get('public', {})}
        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                lines.append(f'- {key.replace("_", " ")}: {value}')
        if outcome.get('original_pages'):
            lines.append('- Original PDF pages: '+', '.join(map(str, outcome['original_pages'])))
        if outcome.get('page_failures'):
            lines.append('- Failures: '+json.dumps(outcome['page_failures'],ensure_ascii=False))
        if outcome.get('model'):
            lines.append('- Model execution: '+json.dumps(outcome['model'],ensure_ascii=False))
        if outcome.get('page_inventory'):
            lines += ['', '### Processed, skipped, failed and incomplete original pages', '']
            for state, pages in outcome['page_inventory'].items():
                lines.append(f'- {state}: {pages}')
        for page in outcome.get('page_markdown',{}).get('pages',[]):
            lines.append(f'- Page {page["source_page"]} Markdown OCR: {page["status"]}; '+ '; '.join(page.get('issues',[])))
            for request in page.get('requests',[]):
                if not Path(request['image']).is_file():continue
                (folder/'review_images').mkdir(exist_ok=True)
                target=f'page-{page["source_page"]:04d}-{request["id"]}.png'
                shutil.copyfile(request['image'],folder/'review_images'/target)
                lines += ['',f'![Original page {page["source_page"]} {request["id"]}](review_images/{target})','']
        trace_path=Path(outcome.get('field_trace','/nonexistent'))
        if trace_path.is_file():
            lines += ['', '### Public field and occurrence decisions', '']
            for line in trace_path.read_text().splitlines():
                trace=json.loads(line)
                lines.append(f'- {trace["row_key"]}: '+ '; '.join(trace.get('reasons',[])))
            audit_path=trace_path.with_name('occurrence-audit.json')
            if audit_path.is_file():lines.append(traceability.markdown(json.loads(audit_path.read_text())))
        lines.append('')
    if not manifests:
        lines += ['No saved stage manifest supplied. Results are not validated.', '']
    for ordinal, filename in enumerate(manifests):
        path = Path(filename)
        saved = json.loads(path.read_text())
        lines += ['## Input and results', '', f'Input evidence: `{path}`', '']
        if saved.get('reference_relations_review'):
            reference=Path(saved['reference_relations_review'])
            if reference.is_file():
                target=folder/f'reference-relations-{ordinal:03d}.md'
                shutil.copyfile(reference,target)
                lines+=['',f'[Source fabrication reference review]({target.name})','']
        source = saved.get('upstream', saved)
        if 'page_index' in source:
            lines += [f'Original source PDF page: {source["page_index"] + 1}', '']
        for key in ('page_decision', 'state', 'complete', 'completeness', 'execution_status',
                    'source_coverage_status', 'acceptance_status', 'counts'):
            if key in saved:
                lines += [f'- {"Internal evidence table rows" if key=="counts" else key}: {saved[key]}']
        if 'observed_profile_occurrences' in saved:
            lines += [f'- Observed physical profile occurrences: {saved["observed_profile_occurrences"]}']
        for key in ('candidates', 'views', 'details', 'occurrences', 'markers', 'observations'):
            if key in saved:
                lines += [f'- {key}: {len(saved[key])}']
        issues = []
        def collect(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ('review_reasons','grouping_review_reasons','uncertainty','failures',
                               'unresolved_views','limitations','skip_reason') and child:
                        issues.append(f'{key}: {json.dumps(child, ensure_ascii=False)}')
                    elif key not in ('proposed_candidates', 'provenance'):
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        collect(saved)
        if number == 7 and saved.get('artifacts', {}).get('review'):
            with Path(saved['artifacts']['review']).open(encoding='utf-8-sig', newline='') as handle:
                issues.extend(f'{r["severity"]} {r["code"]}: {r["message"]}'
                              for r in csv.DictReader(handle))
        if saved.get('review_path') and Path(saved['review_path']).suffix == '.json':
            collect(json.loads(Path(saved['review_path']).read_text()))
        analysis = saved.get('page_analysis', {})
        if analysis:
            audit = analysis.get('coverage_validation', {})
            lines += ['', f'Audit execution: {audit.get("state", "not run")}. '
                      f'Assembly coverage: {analysis.get("coverage", "unknown")}.', '']
        lines += ['', '### Failed, incomplete or review items', '']
        lines += ['- '+issue for issue in dict.fromkeys(issues)] or [
            'No issues recorded by this stage; source accuracy still requires visual inspection.']
        images = []
        if number == 2:
            images = list(saved.get('coverage_review_paths', {}).values())
            images += [saved[k] for k in ('context_path','overview_path') if saved.get(k)]
        elif number == 3:
            images = [c['path'] for d in saved.get('details', []) for c in d.get('composites', [])]
            lines += ['', '### Group membership', '']
            lines += [f'- {d["id"]}: views {", ".join(d.get("view_ids", []))}; '
                      f'{d.get("grouping_status", "REVIEW")}; {d.get("membership_role", "unresolved")} membership' for d in saved.get('details', [])]
        elif number in (4, 5):
            images = [v['path'] for v in saved.get('annotated_views', [])]
            for record in saved.get('leader_observations',[]):
                if record.get('overlay_path'):images.append(record['overlay_path'])
                count,status=leaders.count_state(record)
                lines += ['',f'Detail {record["detail_id"]}: verified leaders={count or "WITHHELD"}; {status}.','']
            paragraph = saved.get('paragraph_analysis') or {}
            if paragraph.get('review_image'):
                images.append(paragraph['review_image'])
            if paragraph.get('reading_image'):
                images.append(paragraph['reading_image'])
            if paragraph.get('recovery_image'):
                images.append(paragraph['recovery_image'])
            if paragraph:
                records=paragraph.get('paragraphs', [])
                lines += ['', '### Original-page paragraphs', '',
                          f'Paragraph regions: {len(records)}; associated: {sum(bool(r["accepted"]) for r in records)}; '
                          f'requiring review: {sum(not r["accepted"] for r in records)}.', '']
                for n, item in enumerate(paragraph.get('paragraphs', []), 1):
                    lines += [f'- P{n}: {item["scope"]}; '+('associated' if item['accepted'] else 'REVIEW')+
                              '; Details: '+', '.join(item['detail_ids'])+'; '+json.dumps(item['text'],ensure_ascii=False)]
                for reading in paragraph.get('readings', []):
                    lines += [f'- P{reading["paragraph_index"]+1} independent OCR: '+
                              ('agrees' if reading['agrees'] else 'REVIEW; disagrees/uncertain')+'; '+
                              json.dumps(reading['text'],ensure_ascii=False)]
        lines += ['', '### Visual evidence', '']
        for index, image in enumerate(images):
            target = f'{ordinal + 1:03d}-{index + 1:03d}-{Path(image).name}'
            (folder/'review_images').mkdir(exist_ok=True)
            shutil.copyfile(image, folder/'review_images'/target)
            lines += [f'![{Path(image).stem}](review_images/{target})', '']
        if not images:
            lines += ['No additional review image recorded for this stage.', '']
    if outcome is not None:
        selected=set(outcome.get('original_pages',[]))
        processed=set()
        for filename in manifests:
            saved=json.loads(Path(filename).read_text())
            page=saved.get('page_index',saved.get('upstream',{}).get('page_index'))
            if isinstance(page,int):processed.add(page+1)
        lines += ['', '## Original-page execution inventory', '',
                  f'- Selected: {sorted(selected)}', f'- Saved page manifests: {sorted(processed)}',
                  f'- No page manifest (skipped, failed, incomplete or aggregate-stage evidence): {sorted(selected-processed)}',
                  '- Aggregate-stage evidence and explicit skip reasons above distinguish these cases.', '']
    (folder/'review.md').write_text('\n'.join(lines)+'\n')


def publish(summary_path, *, output=Path('output'), catalog_manifest=None):
    summary, catalog, filename, rows, traces, snapshot, _, partial = presentation.project(
        summary_path, details_only=True)
    catalog_index=None
    for row,trace in zip(rows,traces):
        if not trace.get('profile_count_verified'):
            trace['observed_profile_count']=row['profile_count']
            row['profile_count']=''
    if catalog_manifest:
        from . import catalog_lookup
        catalog_index=catalog_lookup.build(catalog_manifest,snapshot,catalog)
        catalog_lookup.apply(rows,traces,catalog_index)
        for row,trace in zip(rows,traces):
            if presentation.unsafe(row['component_name']):
                trace['withheld']['component_name']=row['component_name'];row['component_name']=''
                trace['reasons'].append('UNSAFE_CSV_component_name');partial=True
    root = Path(output).resolve()/identifier(filename, catalog)
    root.mkdir(parents=True, exist_ok=True)
    internal = root/'_internal'
    internal.mkdir(exist_ok=True)
    prior = internal/'publication.json'
    if prior.exists() and json.loads(prior.read_text())['document_id'] != catalog:
        raise ValueError('Publication belongs to another PDF')
    if prior.exists() and json.loads(prior.read_text()).get('contract_version')!=5:
        raise ValueError('Historical publication contract is preserved; select a fresh output root for the five-column contract')
    if any((root/name).exists() for name in ('details','incomplete_details')) and not prior.exists():
        raise ValueError('Refusing to replace an unowned Detail directory')
    staging = internal/('publication-'+uuid.uuid4().hex)
    staging.mkdir()
    (staging/'details').mkdir()
    (staging/'incomplete_details').mkdir()
    (staging/'_internal').mkdir()
    grouped = {}
    for row, trace in zip(rows, traces):
        grouped.setdefault(trace['row_key'][1], []).append(row)
    order = {did: index for index, did in enumerate(grouped)}
    inventory = {};observed_counts={}; exported_traces=[]; next_csv_row=2;skipped=[]
    for trace in traces:trace['exported_csv_row_number']=None
    saved_results=[(path,snapshot.json(path)) for path in summary['artifacts']['detail_results']]
    for path,saved in sorted(saved_results,key=lambda item:order.get(item[1]['detail_id'],len(order))):
        did = saved['detail_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', did):
            raise ValueError('Unsafe Detail identifier')
        selected = grouped.pop(did)
        # Unsupported observations are technical evidence, not CSV components.
        geometry_keys={t['row_key'][2] for t in traces if t['row_key'][1]==did and t.get('observed_profile_id')}
        verified = [r for r,t in zip(selected,[t for t in traces if t['row_key'][1]==did])
                    if r['profile_id'] or r['component_name'] or t['row_key'][2] in geometry_keys]
        if not verified:
            skipped.append(dict(detail_id=did,source_page=saved.get('page_index',int(selected[0]['source_page'])-1)+1,
                reason='NO_SUPPORTED_COMPONENT_CODE_OR_NAME',
                source_result=path,review_status='REVIEW',
                profile_count_status=next(t['profile_count_status'] for t in traces if t['row_key'][1]==did),
                absence_of_components_established=False))
            review_folder=staging/'incomplete_details'/did
            review_folder.mkdir()
            (review_folder/'image.png').write_bytes(snapshot.bytes(Path(path).with_name('image.png')))
            for name in ('component_leaders_overlay.png','leader_observations.json','profile_count_review.md'):
                (review_folder/name).write_bytes(snapshot.bytes(Path(path).with_name(name)))
            skipped[-1]['review_image']='incomplete_details/'+did+'/image.png'
            (review_folder/'review.json').write_text(json.dumps(skipped[-1],indent=2)+'\n')
            continue
        exported_name=presentation.public_contract.detail_name(filename,len(inventory)+1)
        for row in verified:row['detail_name']=exported_name
        for r,t in zip(selected,[t for t in traces if t['row_key'][1]==did]):
            if r['profile_id'] or r['component_name'] or t.get('observed_profile_id'):
                t['exported_csv_row_number']=next_csv_row
                exported_traces.append(dict(t,csv_row_number=next_csv_row))
                next_csv_row+=1
        public_id=f'detail_{len(inventory)+1:03d}'
        folder = staging/'details'/public_id
        folder.mkdir()
        image_path = Path(path).with_name('image.png')
        image_data = snapshot.bytes(image_path)
        (folder/'image.png').write_bytes(image_data)
        write_csv(folder/'detail.csv', verified)
        for name in ('component_leaders_overlay.png','leader_observations.json','profile_count_review.md'):
            (folder/name).write_bytes(snapshot.bytes(Path(path).with_name(name)))
        inventory[public_id] = dict(detail_id=did,document_id=catalog, order=order[did], detail_name=verified[0]['detail_name'],
            source_page=int(verified[0]['source_page']),
            profile_count=verified[0]['profile_count'],profile_count_status=next(t['profile_count_status'] for t in traces if t['row_key'][1]==did),
            csv_sha256=presentation.digest((folder/'detail.csv').read_bytes()))
        observed_counts[did]=int(verified[0]['profile_count']) if verified[0]['profile_count'] else None
    if grouped:
        raise ValueError('Missing Stage 7 image/result for projected Detail')
    manifest = dict(contract_version=5,profile_count_contract=leaders.COMPONENT_COUNT_CONTRACT,document_id=catalog, source_filename=filename, details=inventory,
                    skipped_details=skipped,catalog_name_source=str(catalog_manifest) if catalog_manifest else None,
                    stage7_summary=str(Path(summary_path).resolve()),
                    status='PARTIAL / REVIEW' if partial else 'REVIEW', source_files=snapshot.files)
    (staging/'_internal/publication.json').write_text(json.dumps(manifest, indent=2))
    (staging/'_internal/rows.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces))
    merged_rows = merge(staging)
    occurrence_audit=traceability.from_snapshot(summary,snapshot,exported_traces)
    traceability.write(staging/'_internal/occurrence-audit.json',occurrence_audit)
    s6path = summary['artifacts']['enrichment']
    s6 = snapshot.json(s6path)
    manifests = {n: [] for n in STAGES}
    manifests[6] = [s6path]
    manifests[7] = [summary_path]
    for source in s6['sources']:
        for number in (3, 4, 5):
            manifests[number].append(source[f'stage{number}_path'])
        s3 = snapshot.json(source['stage3_path'])
        manifests[2].append(s3['upstream']['manifest_path'])
    for number, paths in manifests.items():
        review_stage(staging, number, paths, outcome=(
            {'source':filename, 'original_pages':sorted({d['source_page'] for d in inventory.values()}),
             'scope':'Selected evidence only; full-document completeness is not established'} if number == 1 else None))
    validated=snapshot.json(summary['artifacts']['leader_observations']) if summary['artifacts'].get('leader_observations') else {}
    final_review=staging/'stages/stage_07'
    with (final_review/'review.md').open('a') as handle:
        handle.write('\n## Verified component leader counts\n\n')
        for did,record in validated.items():
            count,status=leaders.count_state(record)
            handle.write(f'- {did}: leader_diagnostic_count={count or "WITHHELD"}; {status}.\n')
    with (staging/'stages/stage_07/review.md').open('a') as handle:
        handle.write('\n## Detected Details with incomplete component extraction\n\n')
        for item in skipped:handle.write('- '+json.dumps(item)+'\n')
        handle.write('\n## Catalog name source evidence\n\n')
        for trace in traces:
            if trace.get('catalog_lookup'):
                handle.write('- '+json.dumps(dict(row_key=trace['row_key'],lookup=trace['catalog_lookup']),ensure_ascii=False)+'\n')
        handle.write('\n'+traceability.markdown(occurrence_audit))
        handle.write('\n## Missing or withheld public fields\n\n')
        for row, trace in zip(rows,traces):
            missing=[key for key,value in row.items() if not value or value=='UNDETERMINED']
            if missing:
                handle.write(f'- {trace["row_key"]}: {", ".join(missing)}; '+ '; '.join(trace['reasons'])+'\n')
        handle.write(f'\nDelivery: {len(inventory)} Details, {merged_rows} CSV rows.\n'
                     'Merged CSV was read back and compared with every per-Detail CSV.\n'
                     'CSV profile_count counts extracted associated components; leader verification is reported separately.\n')
        handle.write(f'Extracted associated components: {sum(v for v in observed_counts.values() if v is not None)}; '
                     f'withheld counts: {sum(v is None for v in observed_counts.values())}.\n')
    snapshot.verify()
    if catalog_index:
        from .indexer import fingerprint
        if fingerprint(catalog_index['source_path'])!=catalog:raise ValueError('Catalog source changed before publication')
    (staging/'_internal/publication.json').write_text(json.dumps(manifest,indent=2)+'\n')
    history = internal/('previous-'+uuid.uuid4().hex)
    history.mkdir()
    moved, installed = [], []
    try:
        # Stage reviews can already exist from an in-progress first pipeline run.
        for name in ('details','incomplete_details','stages','all_details.csv'):
            if (root/name).exists():
                (root/name).rename(history/name)
                moved.append(name)
        if prior.exists():
            shutil.copyfile(prior, history/'publication.json')
        for name in ('details','incomplete_details','stages','all_details.csv'):
            (staging/name).rename(root/name)
            installed.append(name)
        os.replace(staging/'_internal/publication.json', prior)
        os.replace(staging/'_internal/rows.jsonl', internal/'rows.jsonl')
        os.replace(staging/'_internal/occurrence-audit.json', internal/'occurrence-audit.json')
    except OSError:
        for name in reversed(installed):
            (root/name).rename(staging/name)
        for name in moved:
            (history/name).rename(root/name)
        if (history/'publication.json').exists():
            shutil.copyfile(history/'publication.json', prior)
        raise
    return dict(root=str(root), details=len(inventory), incomplete_details=len(skipped),rows=merged_rows, status=manifest['status'])


# Selected-page contract reporting belongs with delivery validation.
def focused_report(summary):
    root=Path(summary['outputs']['public_root'])
    internal=root/'_internal';internal.mkdir(parents=True,exist_ok=True)
    pages=summary.get('pages',{}).get('original_pages',[])
    selected=set(pages)
    checks={name:dict(status='REVIEW',reasons=['Requires selected source-page visual validation'])
            for name in ('detection','grouping','callouts','counts','component_names','markdown_ocr','csv_integrity')}
    failed=summary.get('pages',{}).get('failed_pages_one_based',[])
    if failed:checks['detection']=dict(status='FAIL',reasons=['Failed original pages: '+str(failed)])
    markdown=summary.get('page_markdown',{}).get('pages',[])
    states={r['source_page']:r['status'] for r in markdown}
    checks['markdown_ocr']=dict(status='FAIL' if set(states)!=selected or 'FAILED' in states.values()
        else 'REVIEW' if any(s!='COMPLETE' for s in states.values()) else 'PASS',
        reasons=['Transcription execution states: '+str(states),'Text accuracy has not been source-validated'])
    csv_path=root/'all_details.csv'
    if csv_path.is_file():
        try:
            rows=read_csv(csv_path)
            if any(int(r['source_page']) not in selected for r in rows):
                raise ValueError('CSV row refers to an unselected original page')
            publication=json.loads((internal/'publication.json').read_text())
            merged=[]
            for key,detail in sorted(publication['details'].items(),key=lambda item:(item[1]['source_page'],item[1].get('order',0),item[0])):
                path=root/'details'/key/'detail.csv'
                if presentation.digest(path.read_bytes())!=detail['csv_sha256']:
                    raise ValueError('Per-Detail CSV changed')
                merged.extend(read_csv(path))
            if rows!=merged:raise ValueError('all_details.csv differs from the ordered per-Detail CSVs')
            checks['csv_integrity']=dict(status='PASS',reasons=['Five fields, page selection, hashes and exact merged rows validated'])
            for field,check in [('profile_count','counts'),('component_name','component_names')]:
                missing=sum(not r[field] or r[field]=='UNDETERMINED' for r in rows)
                checks[check]['reasons'].insert(0,f'{missing}/{len(rows)} rows have missing or unresolved {field}')
        except (ValueError,KeyError,OSError) as exc:
            checks['csv_integrity']=dict(status='FAIL',reasons=[str(exc)])
    else:
        checks['csv_integrity']=dict(status='FAIL',reasons=['No final CSV was published'])
    result=dict(source=summary.get('source'),original_pages=pages,checks=checks,
        selected_page_completion=summary.get('selected_page_completion','INCOMPLETE'),
        full_catalog_completion=summary.get('full_catalog_completion','INCOMPLETE'),
        accuracy_verified=False,failures=summary.get('failures',[]))
    (internal/'focused-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    stage=root/'stages/stage_07';stage.mkdir(parents=True,exist_ok=True)
    with (stage/'review.md').open('a',encoding='utf-8') as handle:
        handle.write('\n## Selected-page contract checks\n\n')
        for name,check in checks.items():
            handle.write(f'- {name}: {check["status"]}; '+ '; '.join(check['reasons'])+'\n')
        handle.write('\nSelected-page processing: '+result['selected_page_completion']+
                     '; full catalog: '+result['full_catalog_completion']+'. Extraction accuracy is unverified.\n')
    return result



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path, nargs='?')
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--catalog-ocr',type=Path,help='Saved original-page OCR manifest for scoped catalog names')
    parser.add_argument('--merge', type=Path, help='Rebuild only all_details.csv from a published PDF folder')
    args = parser.parse_args(argv)
    if args.merge:
        print(f'Merged {merge(args.merge)} rows')
    elif args.summary:
        print(json.dumps(publish(args.summary, output=args.output,catalog_manifest=args.catalog_ocr)))
    else:
        parser.error('Supply a Stage 7 summary or --merge PDF_FOLDER')

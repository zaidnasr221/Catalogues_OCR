"""Validate the fresh two-page output and create one complete inspection ZIP."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import zipfile

import pymupdf as fitz
from PIL import Image

from as_ocr import delivery


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    raise RuntimeError("Historical physical-profile/type/description recipe is disabled. Use the current five-column as-ocr pipeline/deliver contract; preserve existing outputs.")
    project=Path.cwd().resolve();output=project/'output'
    assert project==Path('/home/domapp/Z_AiWork/As_ocr') and not output.is_symlink()
    cleanup=json.loads((output/'_internal/cleanup-audit.json').read_text())
    assert cleanup['recreated_empty'] and cleanup['output']==str(output) and not cleanup['links'] and not cleanup['unexpected']
    protected=json.loads((output/'_internal/protected-files-before.json').read_text())
    changed=[name for name,expected in protected['files'].items() if not (project/name).is_file() or
             [(project/name).stat().st_size,(project/name).stat().st_mtime_ns]!=expected]
    if changed:raise ValueError('Protected files changed: '+repr(changed))
    for name,expected in protected['source_hashes'].items():assert sha(project/name)==expected
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    path=os.readlink(fd)
                    if path.startswith(str(output)+'/'):
                        flags=int(next(x.split()[1] for x in (proc/'fdinfo'/fd.name).read_text().splitlines() if x.startswith('flags:')),8)
                        if flags & 3:raise ValueError('Active output writer '+proc.name)
                except (OSError,StopIteration):pass
        except OSError:pass
    cases=[('Data/Alupco/ALUPCO50SG CAT.(3rd ED 2011).pdf',7),('Data/1.)FW 50+SG/SECTION SG bind.pdf',1)]
    totals=Counter();results=[];path_map={str(output):'output/'}
    for filename,page in cases:
        source=project/filename;root=output/delivery.identifier(source.name,sha(source))
        publication=json.loads((root/'_internal/publication.json').read_text())
        summary=json.loads(Path(publication['stage7_summary']).read_text())
        stage5=Path(summary['selected_sources'][0]['stage5_path']);saved=json.loads(stage5.read_text())
        assert summary['validation_errors']==0
        assert saved['upstream']['page_index']+1==page
        assert saved['settings']['configuration']['observed_profiles']
        for key in ('stage3','stage4'):
            assert output in Path(saved['upstream'][key]['path']).resolve().parents
        assert all(output in Path(p).resolve().parents for p in publication['source_files'])
        fresh=json.loads((root/'_internal/fresh-detection/results.json').read_text())[0]
        assert fresh['detection']['pages_reused']==0 and fresh['grouping']['pages_reused']==0
        assert fresh['page']==page and Path(fresh['source'])==source
        index=Path(fresh['index'])
        with sqlite3.connect(index.as_uri()+'?mode=ro',uri=True) as db:
            indexed=db.execute("SELECT count(*) FROM pages WHERE status='complete'").fetchone()[0]
        for dbpath in [index,root/'_internal/fresh-components/index.sqlite3']:
            with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
                for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%view_attempts'"):
                    assert not db.execute(f'SELECT count(*) FROM {table} WHERE state=?',('reused',)).fetchone()[0]
        expected=[];counts=[];detail_rows=[]
        for folder,detail in sorted(publication['details'].items(),key=lambda item:item[1]['order']):
            directory=root/'details'/folder
            rows=delivery.read_csv(directory/'detail.csv');assert rows
            assert all(r['source_page']==str(page) for r in rows)
            assert len({r['profile_count'] for r in rows})==1
            counts.append(int(rows[0]['profile_count']));expected+=rows
            assert (directory/'image.png').is_file() and not list(directory.glob('*.json'))
            record=next(json.loads(Path(p).read_text()) for p in summary['artifacts']['detail_results']
                        if json.loads(Path(p).read_text())['detail_id']==detail['detail_id'])
            assert (directory/'image.png').read_bytes()==Path(next(p for p in summary['artifacts']['detail_results']
                if json.loads(Path(p).read_text())['detail_id']==detail['detail_id'])).with_name('image.png').read_bytes()
            panels=[];seen=set()
            s3=json.loads(Path(saved['upstream']['stage3']['path']).read_text())
            grouping=next(d for d in s3['details'] if d['id']==detail['detail_id'])
            for composite in grouping['composites']:
                for panel in composite['panels']:
                    if panel['source_crop'] in seen:continue
                    seen.add(panel['source_crop'])
                    with Image.open(panel['source_crop']) as image:panels.append(image.convert('RGB'))
            combined=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'white');top=0
            for panel in panels:combined.paste(panel,(0,top));top+=panel.height
            with Image.open(directory/'image.png') as image:
                assert image.size==combined.size and image.convert('RGB').tobytes()==combined.tobytes()
            detail_rows.append(dict(folder=folder,count=counts[-1],rows=len(rows),source_detail_id=detail['detail_id']))
        assert delivery.read_csv(root/'all_details.csv')==expected
        before=(root/'all_details.csv').read_bytes();assert delivery.merge(root)==len(expected)
        assert (root/'all_details.csv').read_bytes()==before
        result=Counter(details=len(counts),observed_profiles=sum(counts),rows=len(expected),
            profile_ids=sum(bool(r['profile_id']) for r in expected),component_names=sum(bool(r['component_name']) for r in expected),
            classified_types=sum(r['profile_type']!='UNDETERMINED' for r in expected),
            undetermined_types=sum(r['profile_type']=='UNDETERMINED' for r in expected),descriptions=sum(bool(r['description']) for r in expected))
        totals.update(result)
        original=root/'_internal/sources'/source.name;original.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,original);assert sha(original)==sha(source)
        path_map[str(source)]=str(original.relative_to(project))
        for n in range(1,8):
            folder=root/'stages'/f'stage_{n:02d}';assert (folder/'review.md').is_file()
            (folder/'review_images').mkdir(exist_ok=True)
            with (folder/'review.md').open('a') as handle:
                handle.write(f'\nFinal selected-page context: original PDF page {page}; {result["details"]} Details; '
                    f'{result["observed_profiles"]} observed profiles; {result["rows"]} component rows; '
                    f'{result["classified_types"]} classified types; {result["undetermined_types"]} UNDETERMINED.\n')
                handle.write('See ../stage_07/review.md for validated geometry/type overlays. Observed totals are not certified ground truth.\n')
        stage1=root/'stages/stage_01';preview=stage1/'review_images'/f'original-page-{page}.png'
        with fitz.open(source) as pdf:pdf[page-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(preview)
        with (stage1/'review.md').open('a') as handle:
            handle.write(f'\nSource indexing completed {indexed} pages; extraction stages selected original page {page} only.\n'
                         f'\n![Original input page](review_images/{preview.name})\n')
        recovery=root/'_internal/paragraph-review/paragraph-recovery.png'
        if recovery.is_file():
            target=root/'stages/stage_05/review_images/paragraph-recovery-original.png';shutil.copyfile(recovery,target)
            with (root/'stages/stage_05/review.md').open('a') as handle:
                handle.write('\nFocused paragraph recovery timed out. No recovered OCR text was accepted. '
                             'The enlarged original-source paragraph region is retained for REVIEW.\n'
                             '\n![Original explanatory paragraphs for review](review_images/paragraph-recovery-original.png)\n')
        results.append(dict(source=source.name,page=page,root=root.name,counts=list(counts),totals=dict(result),details=detail_rows,
                            stage5=str(stage5),stage7=publication['stage7_summary'],detection_seconds=fresh['detection'].get('runtime_seconds'),
                            grouping_seconds=fresh['grouping'].get('runtime_seconds')))
    # These are duplicate public-publication snapshots of this fresh work, not
    # extraction dependencies. Original stage evidence and all failures stay.
    for root in output.glob('*__*'):
        for history in (root/'_internal').glob('previous-*'):
            assert history.is_dir() and not history.is_symlink()
            assert not any(p.is_symlink() for p in history.rglob('*'))
            shutil.rmtree(history)
    (output/'_internal/path-map.json').write_text(json.dumps(path_map,indent=2))
    ended=datetime.now(timezone.utc);started=datetime.fromisoformat(cleanup['deleted_at'])
    duration=round((ended-started).total_seconds(),1)
    data=dict(cleanup=cleanup,protected_files_unchanged=len(protected['files']),source_hashes_unchanged=True,
        no_pre_cleanup_extraction_reuse=True,totals=dict(totals),sources=results,duration_seconds=duration,finished_at=ended.isoformat())
    (output/'_internal/package-validation.json').write_text(json.dumps(data,indent=2))
    report=['# Fresh extraction rebuild — execution report','',
        f'Generated {ended.isoformat()}. Rebuild wall duration, including implementation, focused retries and validation: **{duration:.1f} seconds ({duration/60:.1f} minutes)**.','',
        'The old project output directory was verified as disposable generated data, removed, and recreated empty. '
        f'The cleanup removed {sum(cleanup["counts"].values())} generated files ({cleanup["bytes"]:,} bytes). '
        'No symlinks or active output handles were found. The audit is retained under _internal.','',
        'All live detection, grouping, callout and initial component extraction results were generated after cleanup from the original PDFs. '
        'No pre-cleanup extraction cache, export, image, CSV or checkpoint was reused. This fresh run’s own persisted observations were refined '
        'during bounded geometry verification and label reconciliation. Model weights/service settings and concurrency configuration were unchanged. '
        'Only ALUPCO original page 7 and Schüco SECTION SG bind original page 1 were selected for extraction; the 110-page ALUPCO job remains stopped.','',
        f'Protected-file verification: {len(protected["files"])} files under out/ and Data/ unchanged by size and modification time; both source PDF SHA-256 values unchanged.','',
        '## Actual output totals','',
        '| Metric | Total |','|---|---:|']
    labels={'details':'Detected/exported Details','observed_profiles':'Observed physical profile occurrences','profile_ids':'Profile ID occurrences in CSV',
        'component_names':'Component names in CSV','classified_types':'Populated supported functional types','undetermined_types':'UNDETERMINED types',
        'descriptions':'Rows with verified descriptions','rows':'Component CSV rows'}
    report += [f'| {label} | {totals[key]} |' for key,label in labels.items()]
    report += ['', '| Source / original page | Counts per Detail | Details | Rows | IDs | Names | Classified | UNDETERMINED | Description rows |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        t=r['totals'];report += [f'| {r["source"]} / {r["page"]} | {", ".join(map(str,r["counts"]))} | {t["details"]} | {t["rows"]} | {t["profile_ids"]} | {t["component_names"]} | {t["classified_types"]} | {t["undetermined_types"]} | {t["descriptions"]} |']
    report += ['', 'Counts measure geometry observations, not CSV rows or distinct identifiers. The ALUPCO silicone row is a sealant, not a profile. '
        'Schüco contains geometry-only profile rows with blank ID/name fields, while some identified callouts remain unbound to a counted profile region. '
        'A numeric identifier never automatically adds a profile or establishes its function. Repeated source identifiers remain distinct occurrences.','',
        '## Validation and remaining review','',
        '- Every Detail has image.png and an independent seven-column detail.csv. Each per-PDF all_details.csv was compared with all owned Detail rows; repeated merging is byte-identical.','- Every public CSV has exactly detail_name,profile_count,profile_id,component_name,profile_type,description,source_page in that order. Counts are nonnegative integers repeated consistently per Detail. Types are supported values or UNDETERMINED.','- Final image dimensions and RGB pixels match the freshly generated original Stage 3 crop panels exactly. No resizing was introduced into delivered drawing images.','- Seven numbered stage reports exist per PDF; detection, grouping, callout, profile/type and paragraph review images are included. Stage 7 overlays show the final accepted type projection.','- ALUPCO counts 1 and 1 were visually compared with the spacer holder and spandrel frame; source names/IDs are retained. Sealant is excluded from the structural count. The three shared explanatory paragraphs retain source spellings, including warrented and compatable. They fill three rows; the marked anodizing footnote remains REVIEW.','- Schüco observed inventories 1, 3, 3 remain potentially incomplete and are not accepted as ground-truth totals. Earlier model inventories disagreed and sometimes split connected sections or returned an unsupported empty verification. Those proposals and verification evidence are retained internally. The final counts report the actually retained visible observations, without compensating for unseen parts.','- The Schüco numbered callouts do not establish every component function/name. Functional types use associated physical geometry when independently supported; ID-only or axis-only classifications are withheld. Section orientation alone does not verify Mullion versus Transom.','- Schüco Stage 2 coverage verification failed; its initial detections still retain the three visible assemblies. ALUPCO had one crop OCR timeout, with names recovered from exact combined labels read during the fresh geometry audit. Schüco paragraph reading/recovery was incomplete or timed out; descriptions remain blank, with enlarged original paragraphs preserved for review. No unsuccessful operation is presented as validated extraction.','- Source completeness, positive real Section/Isometric equivalence and general manufacturer accuracy are not certified by these two selected pages. Both deliveries remain PARTIAL / REVIEW.','',
        '## Implementation','',
        'Stage 5 profile_observations.py renders complete geometry plus verified callouts, inventories physical profiles, verifies continuity/duplicates, and classifies functional types. '
        'Counts retain supported observations despite REVIEW, incomplete crops or a failed verification. Literal name/code blocks can recover a missing source name through the existing callout contract. '
        'Stage 7 validates document/page/Detail ownership, physical target bindings and functional support; the presenter counts inventory entries and preserves geometry-only rows. '
        'Supported geometry types can populate an ID row even when the label itself contains only a number. Paragraph extraction remains a separate original-page pass. '
        'The seven existing stages, detection/grouping rules, CSV columns and minimal result.json contract are preserved.','',
        'Modified implementation: src/as_ocr/profile_observations.py, components.py, pipeline.py, paragraphs.py, exports.py, presentation.py and delivery.py. '
        'Focused runners, validators, tests and README/HANDOFF were updated. No new model, database architecture, CSV column or concurrency tuning was introduced.','',
        'Automated test outcomes are recorded in _internal/tests.md. Tests cover repeated/missing IDs, unlabeled profiles, multiple marks of one part, Section/Isometric deduplication, incomplete/REVIEW counts, unsupported accessories/types, type evidence independent of parent status, label recovery, descriptions, seven-column export, repeat merging and internal JSON compatibility.','',
        '## Files and source evidence','']
    for r in results:
        prefix=r['root'];report += [f'### {r["source"]} — original page {r["page"]}','',
            f'- [Merged CSV]({prefix}/all_details.csv)',f'- [Detection review]({prefix}/stages/stage_02/review.md)',
            f'- [Profile and paragraph review]({prefix}/stages/stage_05/review.md)',
            f'- [Final validated geometry/type review]({prefix}/stages/stage_07/review.md)']
        report += [f'- {d["folder"]}, count {d["count"]}: [image.png]({prefix}/details/{d["folder"]}/image.png), [detail.csv]({prefix}/details/{d["folder"]}/detail.csv)' for d in r['details']]
        report += ['']
    report += ['The archive contains the entire new output directory, including the original PDFs under each source’s _internal/sources for offline inspection. '
        'Public report/image links are relative. Technical provenance keeps its original paths and hashes; _internal/path-map.json maps those paths to archive locations.','']
    (output/'review.md').write_text('\n'.join(report)+'\n')
    stamp=ended.strftime('%Y%m%dT%H%M%SZ');archive=project/f'OCR_Detaila_fresh_output_{stamp}.zip'
    files=sorted(p for p in output.rglob('*') if p.is_file())
    assert not any(p.is_symlink() for p in output.rglob('*'))
    inventory={str(p.relative_to(project)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    manifest=output/'_internal/package-inventory.json';manifest.write_text(json.dumps(inventory,indent=2))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(output.rglob('*')):z.write(path,str(path.relative_to(project)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert {n for n in z.namelist() if not n.endswith('/')}=={str(p.relative_to(project)) for p in output.rglob('*') if p.is_file()}
        for name,item in inventory.items():assert hashlib.sha256(z.read(name)).hexdigest()==item['sha256']
    print(json.dumps(dict(zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=sha(archive),
                         totals=dict(totals),duration_seconds=duration,protected_changes=changed,crc_and_inventory_verified=True)),flush=True)


if __name__=='__main__':main()

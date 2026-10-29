"""Stage 1 ingestion. All PDF source geometry uses unrotated MuPDF points."""
from __future__ import annotations

from . import progress

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import uuid

import pymupdf as fitz
from PIL import Image, ImageOps

EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
VERSION = '4'


def flag_native_control_spans(spans):
    """Retain exact native text while exposing suspect font decoding for review."""
    count = 0
    for span in spans:
        if any(0x80 <= ord(char) <= 0x9f for char in span['text']):
            span.setdefault('review_reasons', []).append(
                'Native text contains C1 control codepoints; verify glyph visually')
            count += 1
    return count


def discover(source: Path, output: Path, maximum: int | None = None):
    """Lazy, deterministic per-directory traversal; never enumerate file siblings."""
    source, output = source.resolve(), output.resolve()
    if source.is_file():
        yield source
        return
    count = 0
    def walk(folder):
        nonlocal count
        if maximum is not None and count >= maximum:
            return
        with os.scandir(folder) as entries:
            ordered = sorted(entries, key=lambda e: e.name)
        for entry in ordered:
            if maximum is not None and count >= maximum:
                return
            path = Path(entry.path)
            if path.resolve() == output or entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                yield from walk(path)
            elif entry.is_file() and path.suffix.lower() in EXTENSIONS:
                count += 1
                yield path
    yield from walk(source)


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def encode(value):
    if isinstance(value, bytes):
        return {'hex': value.hex()}
    if isinstance(value, (fitz.Rect, fitz.Point, fitz.Matrix, fitz.Quad)):
        return list(value)
    raise TypeError(type(value).__name__)


def dumps(value):
    return json.dumps(value, default=encode, ensure_ascii=False)


def database(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, format TEXT, page_count INTEGER, metadata TEXT);
    CREATE TABLE IF NOT EXISTS sources(path TEXT, document_id TEXT REFERENCES documents(id), PRIMARY KEY(path,document_id));
    CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, started TEXT DEFAULT CURRENT_TIMESTAMP, finished TEXT, config TEXT, selection TEXT, summary TEXT);
    CREATE TABLE IF NOT EXISTS pages(id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id), page_index INTEGER, config TEXT, status TEXT, width REAL, height REAL, rotation INTEGER, geometry TEXT, signals TEXT, preview TEXT, completed_run TEXT REFERENCES runs(id), UNIQUE(document_id,page_index,config));
    CREATE TABLE IF NOT EXISTS spans(page_id TEXT REFERENCES pages(id), ordinal INTEGER, data TEXT, PRIMARY KEY(page_id,ordinal));
    CREATE TABLE IF NOT EXISTS vectors(page_id TEXT REFERENCES pages(id), ordinal INTEGER, data TEXT, PRIMARY KEY(page_id,ordinal));
    CREATE TABLE IF NOT EXISTS images(page_id TEXT REFERENCES pages(id), ordinal INTEGER, data TEXT, PRIMARY KEY(page_id,ordinal));
    CREATE TABLE IF NOT EXISTS attempts(run_id TEXT REFERENCES runs(id), document_id TEXT, page_index INTEGER, source TEXT, status TEXT, error TEXT);
    ''')
    return db


def pdf_page(page, target, maximum):
    spans = []
    for block in page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                spans.append(dict(span, line_direction=line['dir'], writing_mode=line['wmode']))
    suspect_spans = flag_native_control_spans(spans)
    vectors = page.get_drawings(extended=True)
    images = page.get_image_info(hashes=False, xrefs=True)
    rect = page.rect
    scale = min(1, maximum / max(rect.width, rect.height))
    page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(target)
    source_rect = page.rect * page.derotation_matrix
    area = max(1, source_rect.get_area())
    coverage = min(1, sum((fitz.Rect(i['bbox']) & source_rect).get_area() for i in images) / area)
    label = ('mixed_candidate' if spans else 'scanned_candidate') if images and coverage > .1 else ('native' if spans or vectors else ('unknown' if images else 'blank_candidate'))
    geometry = dict(source_to_display=list(page.rotation_matrix), display_to_source=list(page.derotation_matrix), display_to_preview=[scale, 0, 0, scale, 0, 0], pdf_to_source=list(page.transformation_matrix), source_rect=list(source_rect), cropbox=list(page.cropbox), mediabox=list(page.mediabox))
    signals = dict(native_text_count=len(spans), native_character_count=sum(len(s['text']) for s in spans), native_control_spans=suspect_spans, vector_path_count=len(vectors), embedded_image_count=len(images), approximate_image_coverage=coverage, label=label, needs_visual_text_analysis=bool(images) or not spans or bool(suspect_spans))
    return source_rect.width, source_rect.height, page.rotation, geometry, signals, spans, vectors, images


def image_page(image, target, maximum):
    orientation = image.getexif().get(274, 1)
    # Pillow exposes oriented TIFF size even before load(); tags retain raw size.
    w, h = (int(image.tag_v2[256]), int(image.tag_v2[257])) if image.format == 'TIFF' else image.size
    transforms = {1:[1,0,0,1,0,0],2:[-1,0,0,1,w,0],3:[-1,0,0,-1,w,h],4:[1,0,0,-1,0,h],5:[0,1,1,0,0,0],6:[0,1,-1,0,h,0],7:[0,-1,-1,0,h,w],8:[0,-1,1,0,0,w]}
    displayed = ImageOps.exif_transpose(image)
    dw, dh = displayed.size
    preview = displayed.convert('RGB')
    preview.thumbnail((maximum, maximum))
    preview.save(target, 'PNG')
    matrix = fitz.Matrix(*transforms.get(orientation, transforms[1]))
    geometry = dict(exif_orientation=orientation, source_to_display=list(matrix), display_to_source=list(~matrix), display_to_preview=[preview.width/dw,0,0,preview.height/dh,0,0], source_rect=[0,0,w,h], units='pixels')
    signals = dict(native_text_count=0, native_character_count=0, vector_path_count=0, embedded_image_count=0, approximate_image_coverage=1, label='scanned_candidate', needs_visual_text_analysis=True)
    return w,h,0,geometry,signals,[],[],[]


def run(source, output, max_documents=None, max_pages=None, preview_max=1200, force=False, *, pages=None):
    source, output = Path(source), Path(output)
    if preview_max < 1 or any(n is not None and n < 1 for n in (max_documents, max_pages)):
        raise ValueError('Limits and preview dimension must be positive')
    if not source.exists() or not (source.is_file() or source.is_dir()):
        raise ValueError(f'Input does not exist: {source}')
    if pages is not None and (not source.is_file() or max_pages is not None or not pages or
            any(type(p) is not int or p < 1 for p in pages) or len(set(pages))!=len(pages)):
        raise ValueError('Explicit pages require one file, unique positive original page numbers, and no max_pages')
    if source.is_dir() and source.resolve() == output.resolve():
        raise ValueError('Output directory must differ from input directory')
    output.mkdir(parents=True, exist_ok=True)
    (output/'previews').mkdir(exist_ok=True)
    config = dumps(dict(version=VERSION, preview_max=preview_max, pymupdf=fitz.VersionBind, pillow=Image.__version__))
    config_id = hashlib.sha256(config.encode()).hexdigest()[:16]
    db = database(output/'index.sqlite3')
    run_id = uuid.uuid4().hex
    summary = dict(documents=0, completed=0, reused=0, errors=0, limited=False, incomplete=False)
    db.execute('INSERT INTO runs(id,config,selection) VALUES(?,?,?)', (run_id,config,dumps(dict(source=str(source.resolve()),max_documents=max_documents,max_pages=max_pages,pages=pages,force=force))))
    db.commit()
    try:
        for path in discover(source, output, max_documents):
            summary['documents'] += 1
            doc_id = None
            try:
                if path.suffix.lower() not in EXTENSIONS:
                    raise ValueError('Unsupported input extension')
                doc_id = fingerprint(path)
                is_pdf = path.suffix.lower() == '.pdf'
                with (fitz.open(path) if is_pdf else Image.open(path)) as doc:
                    if is_pdf and doc.needs_pass:
                        raise ValueError('Password-protected PDF requires a password')
                    count = len(doc) if is_pdf else getattr(doc, 'n_frames', 1)
                    metadata = doc.metadata if is_pdf else {k:str(v) for k,v in doc.info.items()}
                    with db:
                        db.execute('INSERT OR REPLACE INTO documents VALUES(?,?,?,?)', (doc_id,path.suffix.lower()[1:],count,dumps(metadata)))
                        db.execute('INSERT OR IGNORE INTO sources VALUES(?,?)',(str(path.resolve()),doc_id))
                    if pages is not None and max(pages)>count:
                        raise ValueError('Selected original page exceeds source page count')
                    selected = sorted(p-1 for p in pages) if pages is not None else list(range(min(count,max_pages) if max_pages is not None else count))
                    if len(selected) < count:
                        summary['limited'] = True
                    for index in selected:
                        progress.start(1,index+1,'Loading and preprocessing PDF page')
                        page_id = f'{doc_id}:{index}:{config_id}'
                        prior = db.execute('SELECT status,preview FROM pages WHERE id=?',(page_id,)).fetchone()
                        if not force and prior and prior[0]=='complete' and (output/prior[1]).is_file():
                            summary['reused'] += 1
                            progress.done(1,index+1,reused=True)
                            continue
                        with db:
                            db.execute("INSERT OR IGNORE INTO pages(id,document_id,page_index,config,status) VALUES(?,?,?,?,'processing')", (page_id,doc_id,index,config))
                            db.execute("UPDATE pages SET status='processing' WHERE id=? AND status!='complete'", (page_id,))
                            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?)', (run_id,doc_id,index,str(path),'processing',None))
                        relative = f'previews/{page_id.replace(":", "_")}_{uuid.uuid4().hex}.png'
                        temporary = output/'previews'/f'{uuid.uuid4().hex}.png'
                        try:
                            if is_pdf:
                                data = pdf_page(doc[index], temporary, preview_max)
                            else:
                                doc.seek(index)
                                data = image_page(doc, temporary, preview_max)
                            w,h,rotation,geometry,signals,spans,vectors,images = data
                            temporary.replace(output/relative)
                            with db:
                                db.execute('INSERT OR REPLACE INTO pages VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(page_id,doc_id,index,config,'complete',w,h,rotation,dumps(geometry),dumps(signals),relative,run_id))
                                for table, rows in [('spans',spans),('vectors',vectors),('images',images)]:
                                    db.execute(f'DELETE FROM {table} WHERE page_id=?',(page_id,))
                                    db.executemany(f'INSERT INTO {table} VALUES(?,?,?)',[(page_id,n,dumps(row)) for n,row in enumerate(rows)])
                                db.execute("UPDATE attempts SET status='complete' WHERE run_id=? AND source=? AND page_index=? AND status='processing'", (run_id,str(path),index))
                            summary['completed'] += 1
                            progress.done(1,index+1)
                        except Exception as exc:
                            temporary.unlink(missing_ok=True)
                            summary['errors'] += 1
                            with db:
                                db.execute("UPDATE pages SET status='error' WHERE id=? AND status!='complete'", (page_id,))
                                db.execute("UPDATE attempts SET status='error',error=? WHERE run_id=? AND source=? AND page_index=? AND status='processing'", (str(exc),run_id,str(path),index))
                            progress.done(1,index+1,ok=False,reason=type(exc).__name__+': '+str(exc))
                            print(f'{path} page/frame {index}: {exc}',file=sys.stderr)
            except Exception as exc:
                summary['errors'] += 1
                with db:
                    db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?)',(run_id,doc_id,None,str(path),'error',str(exc)))
                print(f'{path}: {exc}',file=sys.stderr)
        # Reaching a discovery cap is explicitly limited, without probing another file.
        summary['limited'] |= max_documents is not None and summary['documents'] >= max_documents
        summary['incomplete'] = summary['limited'] or bool(summary['errors'])
        with db:
            db.execute('UPDATE runs SET finished=CURRENT_TIMESTAMP,summary=? WHERE id=?',(dumps(summary),run_id))
    except KeyboardInterrupt:
        summary.update(incomplete=True, interrupted=True)
        with db:
            db.execute('UPDATE runs SET summary=? WHERE id=?', (dumps(summary),run_id))
        raise
    finally:
        db.close()
    return summary


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description='Stage 1 document ingestion and page indexing (no OCR).')
    parser.add_argument('input',type=Path,help='Single file or recursively searched directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-documents',type=positive)
    parser.add_argument('--max-pages',type=positive,help='Maximum pages/frames per document')
    parser.add_argument('--preview-max',type=positive,default=1200)
    parser.add_argument('--force','--reindex',action='store_true',help='Reprocess selected pages; default resumes')
    args = parser.parse_args()
    try:
        summary = run(args.input,args.output,args.max_documents,args.max_pages,args.preview_max,args.force)
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; committed pages will be reused on resume.\n')
    except Exception as exc:
        parser.exit(2,f'Error: {exc}\n')
    print(dumps(summary))
    return 1 if summary['errors'] or not summary['documents'] else 0

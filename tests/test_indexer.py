import json
import sqlite3
from pathlib import Path
import pymupdf as fitz
from PIL import Image
from as_ocr import indexer as ix


def make_pdf(path, count=3):
    doc=fitz.open()
    for n in range(count):
        p=doc.new_page(width=300,height=200)
        if n != 1:
            p.insert_text((20,30),'Exact native text')
            p.draw_line((10,40),(200,40))
        if n:
            import io
            buf=io.BytesIO()
            Image.new('RGB',(200,150),'gray').save(buf,format='PNG')
            p.insert_image(p.rect,stream=buf.getvalue())
        if n == 2:
            p.set_rotation(90)
    doc.save(path)
    doc.close()


def rows(out):
    with sqlite3.connect(out/'index.sqlite3') as db:
        return db.execute("SELECT page_index,geometry,signals,completed_run FROM pages WHERE status='complete' ORDER BY page_index").fetchall()


def test_pdf_resume_config_and_limits(tmp_path):
    source=tmp_path/'sample.PDF'; out=tmp_path/'out'
    make_pdf(source)
    assert ix.run(source,out,max_pages=1)['completed']==1
    assert ix.run(source,out,max_pages=3)['completed']==2
    old=rows(out)
    assert ix.run(source,out)['reused']==3
    assert rows(out)==old
    signals=[json.loads(r[2]) for r in old]
    assert [s['label'] for s in signals]==['native','scanned_candidate','mixed_candidate']
    geom=json.loads(old[2][1])
    point=fitz.Point(22,37)
    result=point*fitz.Matrix(*geom['source_to_display'])*fitz.Matrix(*geom['display_to_source'])
    assert abs(result.x-point.x)<1e-5 and abs(result.y-point.y)<1e-5
    assert ix.run(source,out,preview_max=200)['completed']==3
    assert ix.run(source,out,preview_max=200,force=True)['completed']==3
    make_pdf(tmp_path/'new.pdf',1)
    source.write_bytes((tmp_path/'new.pdf').read_bytes())
    assert ix.run(source,out)['completed']==1
    with sqlite3.connect(out/'index.sqlite3') as db:
        assert db.execute('SELECT COUNT(*) FROM vectors').fetchone()[0]>0
        span=json.loads(db.execute('SELECT data FROM spans LIMIT 1').fetchone()[0])
        assert span['text']=='Exact native text' and 'bbox' in span


def test_images_corruption_and_duplicates(tmp_path):
    inputs=tmp_path/'inputs'; inputs.mkdir(); out=inputs/'out'
    (inputs/'a.pdf').write_bytes(b'broken')
    image=Image.new('RGB',(40,70),'white')
    image.save(inputs/'b.PNG')
    image.save(inputs/'c.TIFF',save_all=True,append_images=[Image.new('RGB',(70,40),'black')])
    (inputs/'d.png').write_bytes((inputs/'b.PNG').read_bytes())
    summary=ix.run(inputs,out)
    assert summary['errors']==1 and summary['completed']==3 and summary['reused']==1
    with sqlite3.connect(out/'index.sqlite3') as db:
        assert db.execute('SELECT COUNT(*) FROM sources').fetchone()[0]==3
    assert len(rows(out))==3


def test_selection_never_hashes_unselected(tmp_path,monkeypatch):
    inputs=tmp_path/'inputs'; inputs.mkdir(); out=tmp_path/'out'
    Image.new('RGB',(10,10)).save(inputs/'a.png')
    (inputs/'b.pdf').write_bytes(b'unrelated')
    original=ix.fingerprint; seen=[]
    def guarded(path):
        seen.append(path.name)
        assert path.name=='a.png'
        return original(path)
    monkeypatch.setattr(ix,'fingerprint',guarded)
    assert ix.run(inputs,out,max_documents=1)['completed']==1
    def forbidden(*args,**kwargs):
        raise AssertionError('single-file discovery enumerated directory')
    monkeypatch.setattr(ix.os,'scandir',forbidden)
    assert ix.run(inputs/'a.png',out)['reused']==1
    assert seen==['a.png','a.png']


def test_failed_page_retried_preserves_success(tmp_path,monkeypatch):
    source=tmp_path/'x.pdf'; out=tmp_path/'out'; make_pdf(source)
    original=ix.pdf_page
    def fail(page,*args):
        if page.number==1: raise RuntimeError('synthetic failure')
        return original(page,*args)
    monkeypatch.setattr(ix,'pdf_page',fail)
    assert ix.run(source,out)['errors']==1
    assert len(rows(out))==2
    monkeypatch.setattr(ix,'pdf_page',original)
    result=ix.run(source,out)
    assert result['completed']==1 and result['reused']==2


def test_image_orientation_and_preview_bounds(tmp_path):
    for extension in ['jpg', 'tiff']:
        source = tmp_path / f'oriented.{extension}'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (40, 70), 'red').save(source, exif=exif)
        out = tmp_path / extension
        assert ix.run(source, out, preview_max=50)['completed'] == 1
        with sqlite3.connect(out/'index.sqlite3') as db:
            width, height, encoded, preview = db.execute('SELECT width,height,geometry,preview FROM pages').fetchone()
        assert (width, height) == (40, 70)
        geometry = json.loads(encoded)
        transformed = fitz.Rect(0, 0, width, height) * fitz.Matrix(*geometry['source_to_display'])
        assert transformed == fitz.Rect(0, 0, 70, 40)
        point = fitz.Point(11, 17)
        restored = point * fitz.Matrix(*geometry['source_to_display']) * fitz.Matrix(*geometry['display_to_source'])
        assert restored == point
        with Image.open(out/preview) as image:
            assert image.size == (50, 29)


def test_interruption_and_force_failure_preserve_success(tmp_path, monkeypatch):
    import pytest
    source = tmp_path/'x.pdf'
    out = tmp_path/'out'
    make_pdf(source)
    original = ix.pdf_page
    def interrupt(page, *args):
        if page.number == 1:
            raise KeyboardInterrupt()
        return original(page, *args)
    monkeypatch.setattr(ix, 'pdf_page', interrupt)
    with pytest.raises(KeyboardInterrupt):
        ix.run(source, out)
    assert len(rows(out)) == 1
    with sqlite3.connect(out/'index.sqlite3') as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE status='processing'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM pages WHERE status='processing'").fetchone()[0] == 1
    monkeypatch.setattr(ix, 'pdf_page', original)
    result = ix.run(source, out)
    assert result['completed'] == 2 and result['reused'] == 1
    old = rows(out)
    def fail(*args):
        raise RuntimeError('synthetic force failure')
    monkeypatch.setattr(ix, 'pdf_page', fail)
    assert ix.run(source, out, force=True)['errors'] == 3
    assert rows(out) == old


def test_cli_errors_and_password(tmp_path):
    import subprocess
    source = tmp_path/'encrypted.pdf'
    doc = fitz.open()
    doc.new_page()
    doc.save(source, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw='owner', user_pw='secret')
    doc.close()
    result = subprocess.run(['as-ocr', str(source), '--output', str(tmp_path/'out')], capture_output=True, text=True)
    assert result.returncode == 1 and 'Password-protected' in result.stderr
    unsupported = tmp_path/'explicit.xyz'
    unsupported.write_text('unsupported')
    result = subprocess.run(['as-ocr', str(unsupported), '--output', str(tmp_path/'out')], capture_output=True, text=True)
    assert result.returncode == 1 and 'Unsupported' in result.stderr
    result = subprocess.run(['as-ocr', str(source), '--output', str(tmp_path/'out'), '--max-pages', '0'], capture_output=True, text=True)
    assert result.returncode == 2


def test_discovery_stops_before_later_subdirectory(tmp_path, monkeypatch):
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    Image.new('RGB', (10, 10)).save(inputs/'a.png')
    later = inputs/'z'
    later.mkdir()
    (later/'another.pdf').write_bytes(b'not to be opened')
    original = ix.os.scandir
    def guarded(folder):
        assert Path(folder) != later
        return original(folder)
    monkeypatch.setattr(ix.os, 'scandir', guarded)
    assert list(ix.discover(inputs, tmp_path/'out', 1)) == [inputs/'a.png']

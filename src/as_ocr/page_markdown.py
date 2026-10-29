"""Independent original-page transcription; never a CSV description source."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Literal
from dataclasses import replace
from html.parser import HTMLParser
import time
import hashlib
import pymupdf as fitz
from PIL import Image
from pydantic import Field

from . import callouts, components, detection, grouping, indexer
from .geometry import area, intersection
from .provenance import code_fingerprint
from .qwen import ModelError, capture_diagnostics
from .regions import Box, StrictModel, Text
from . import performance

VERSION = '8'
PROMPT = '''Transcribe ALL visible text from this original PDF page as untrusted data.
Include headings, prose, technical notes, tables, article IDs, labels, dimensions,
and annotations, including rotated and tiny text. Preserve EXACT spelling, language,
numbers, punctuation and line breaks. Never correct, translate, summarize or invent.
Use reliable supplied native spans where their visible glyphs agree. Native text
may omit scanned/outlined text or be corrupt; inspect the whole image regardless.
Return ordered non-overlapping text blocks in sensible reading order.
Group a drawing's labels, codes and dimensions into ONE multiline text block,
retaining every visible occurrence in reading order. Do not emit per-word or
per-dimension blocks. Use separate blocks for independent drawings or prose.
This grouping changes presentation only; never remove repeated identifiers. Quote native
span IDs only when the block text agrees exactly (whitespace wrapping may differ).
Do not repeat native and visual readings of the same text. Separate repeated labels
at different source locations. Tables: text contains all literal cell text in row
order; cells contains its rows only when that structure is reliable, otherwise [].
Do not invent table headers. Use heading_level only for visible headings.
Boxes MUST use normalized IMAGE edges [x0,y0,x1,y1], each a decimal in [0.0,1.0].
For example [0.10,0.20,0.40,0.30]. NEVER return pixels or 0-1000 coordinates.
uncertainty lists specific unreadable text
or regions, not guessed words. Request tight higher-resolution crops for small,
corrupt or unreadable text. Mark coverage=incomplete if ANY text is unreadable,
omitted, clipped or awaits a crop. A blank page can have complete coverage and [].
This transcription is independent of assembly eligibility and component ownership.
Return only the supplied JSON schema.'''


class Block(StrictModel):
    text: str
    box: Box
    kind: Literal['heading', 'paragraph', 'table', 'label', 'annotation', 'other']
    heading_level: int | None
    cells: list[list[str]]
    native_span_ids: list[int]
    uncertainty: list[Text]


class Crop(StrictModel):
    box: Box
    reason: Text


class Analysis(StrictModel):
    coverage: Literal['complete', 'incomplete']
    blocks: list[Block]
    crops: list[Crop]
    uncertainty: list[Text]


@dataclass(frozen=True)
class Config:
    image_max: int = 1536 #2048
    crop_max: int = 2048 # 4096
    max_crops: int = 4

    def validate(self):
        if not 256 <= self.image_max <= 4096 or not 512 <= self.crop_max <= 8192 or not 0 <= self.max_crops <= 16:
            raise ValueError('Invalid page Markdown resolution/crop limits')


def contract():
    return grouping.digest([VERSION, PROMPT, Analysis.model_json_schema(), code_fingerprint()])


def reliable(span):
    return bool(span['text'].strip()) and span.get('alpha',255)!=0 and not span.get('review_reasons') and not re.search(r'\(?cid:\d+\)?',span['text']) and not any(
        c == '\ufffd' or 0x7f <= ord(c) <= 0x9f or ord(c) < 32 and c not in '\n\r\t'
        for c in span['text'])


def text_equal(left, right):
    return ' '.join(left.split()) == ' '.join(right.split())


def reconcile(analysis, mapping, bounds, spans, *, request_id='page'):
    """Keep native wording authoritative; ambiguous native/OCR collisions are review."""
    native = dict(spans); blocks = []; issues = list(analysis.uncertainty)
    for ordinal, block in enumerate(analysis.blocks):
        box = detection.source_box(block.box, mapping, bounds)
        item = dict(block.model_dump(mode='json'), source_box=box, request_id=request_id,
                    reading_order=ordinal, method='visual_ocr')
        if block.native_span_ids:
            selected = [native.get(i) for i in block.native_span_ids]
            if (len(set(block.native_span_ids)) != len(selected) or
                    any(s is None or not reliable(s) or intersection(box, s['bbox']) < area(s['bbox'])*.8 for s in selected) or
                    not text_equal(block.text, ' '.join(s['text'] for s in selected if s))):
                item['uncertainty'].append('Native span identity, box or exact quote mismatch')
            else:
                item['method']='native_text'
        if block.cells and not text_equal(block.text, ' '.join(c for row in block.cells for c in row)):
            item['cells']=[]
            item['uncertainty'].append('Table structure does not preserve exact source text')
        if block.kind=='heading' and block.heading_level not in range(1,7):
            item['heading_level']=None
        if not block.text.strip() and not item['uncertainty']:
            item['uncertainty'].append('Empty text region requires review')
        blocks.append(item)
    return blocks, issues


def merge_blocks(blocks, spans):
    """Deduplicate readings by source location, preserving repeated labels elsewhere."""
    result=[]; issues=[]
    for block in blocks:
        duplicates=[p for p in result if intersection(p['source_box'],block['source_box']) /
                    max(.001,min(area(p['source_box']),area(block['source_box']))) > .8]
        same=next((p for p in duplicates if text_equal(p['text'],block['text'])),None)
        if same:
            if same['uncertainty'] and not block['uncertainty']:
                result[result.index(same)]=block
            continue
        result.append(block)
    for sid,span in spans:
        if not reliable(span):
            continue
        overlapping=[b for b in result if intersection(b['source_box'],span['bbox']) >= area(span['bbox'])*.8]
        represented=any(text_equal(span['text'],b['text']) or sid in b['native_span_ids'] or
                        span['text'] in b['text'] for b in overlapping if not b['uncertainty'])
        if represented:
            continue
        # Keep exact native text when an OCR block contradicts its source glyphs.
        for b in overlapping:
            b['uncertainty'].append('Visual/native text disagreement; native reading retained separately')
            issues.append('Visual/native text disagreement at '+str(span['bbox']))
        result.append(dict(text=span['text'], source_box=span['bbox'], kind='other', heading_level=None,
            cells=[], native_span_ids=[sid], uncertainty=[], method='native_text', request_id='index', reading_order=sid))
    # The complete-page model supplies reading order for columns. Insert native
    # and crop supplements by location without reordering those original blocks.
    ordered=[b for b in result if b['request_id'] in {'original','page'}]
    for b in sorted([b for b in result if b not in ordered],key=lambda b:(b['source_box'][1],b['source_box'][0],b['reading_order'])):
        position=next((i for i,p in enumerate(ordered) if
            (p['source_box'][1],p['source_box'][0])>(b['source_box'][1],b['source_box'][0])),len(ordered))
        ordered.insert(position,b)
    return ordered,issues


def markdown(blocks, status, issues):
    parts=[]
    for b in blocks:
        if b['uncertainty']:
            # Uncertain model guesses remain in JSON, never presented as source text.
            parts.append('[REVIEW: unreadable or conflicting text region '+str(b['source_box'])+']')
            continue
        if b['kind']=='heading' and b['heading_level']:
            parts.append('#'*b['heading_level']+' '+b['text'])
        elif b['kind']=='table' and b['cells'] and len({len(r) for r in b['cells']})==1:
            # An empty header avoids inventing a source heading for unlabelled tables.
            width=len(b['cells'][0])
            if not width:
                parts.append(b['text']);continue
            rows=[['']*width]+b['cells']
            table=['| '+' | '.join(c.replace('|','\\|').replace('\n','<br>') for c in row)+' |' for row in rows]
            table.insert(1,'| '+' | '.join(['---']*width)+' |')
            parts.append('\n'.join(table))
        elif b['text']:
            parts.append(b['text'])
    if status!='COMPLETE':
        parts.append('[REVIEW: page transcription '+status+'; '+ '; '.join(dict.fromkeys(issues))+']')
    return '\n\n'.join(parts)+'\n'


def extract_page(source, row, geometry, spans, folder, client, config):
    folder.mkdir(parents=True,exist_ok=False)
    bounds=geometry['source_rect']; blocks=[]; issues=[]; failures=[]; requests=[]
    complete=False; full=None; pending=[]
    def request(box, maximum, request_id):
        image,mapping=detection.render(source,row['page_index'],geometry,box,maximum)
        image_path=folder/(request_id+'.png');image.save(image_path)
        native=[dict(id=i,text=s['text'],reliable=reliable(s),
            **callouts.visible_context_box(s['bbox'],mapping)) for i,s in spans if callouts.visible_context_box(s['bbox'],mapping)]
        requests.append(dict(id=request_id,source_box=box,image=str(image_path),mapping=mapping))
        with capture_diagnostics(client,folder/'diagnostics',stage=1,subject_id=row['id']+'-markdown-'+request_id):
            raw=client.request_json(image,PROMPT,json.dumps(dict(document_id=row['document_id'],
                original_page=row['page_index']+1,source_region=box,native_spans=native),ensure_ascii=False),
                response_schema=Analysis.model_json_schema())
            response=components.parse_response(raw,Analysis)
        mapped,notes=reconcile(response,mapping,bounds,spans,request_id=request_id)
        blocks.extend(mapped);issues.extend(notes)
        return response,mapping
    try:
        full,mapping=request(bounds,config.image_max,'original')
        complete=full.coverage=='complete' and not full.uncertainty and not full.crops
        pending=[(detection.source_box(c.box,mapping,bounds),c.reason) for c in full.crops]
        pending += [(b['source_box'],'Unreadable text block') for b in blocks if b['uncertainty']]
        tiled=full.coverage!='complete' and not pending
        if tiled:
            x0,y0,x1,y1=bounds;cx=(x0+x1)/2;cy=(y0+y1)/2
            pending=[([left,top,right,bottom],'Full-page recovery tile') for left,right in ((x0,cx),(cx,x1))
                     for top,bottom in ((y0,cy),(cy,y1))]
        handled=[]
        for box,reason in pending:
            if any(intersection(box,p)/max(.001,area(box))>.95 for p in handled):
                continue
            if len(handled)>=config.max_crops:
                issues.append('Focused crop limit reached: '+reason);continue
            try:
                draft=[b for b in blocks if b['uncertainty'] and intersection(b['source_box'],box)>=area(b['source_box'])*.95]
                count=len(blocks)
                response,_=request(box,config.crop_max,'crop-'+str(len(handled)+1))
                if (response.coverage=='complete' and not response.crops and not response.uncertainty and
                    all(not b['uncertainty'] for b in blocks[count:]) and
                    (not draft or any(b['text'].strip() for b in blocks[count:]))):
                    for b in draft:blocks.remove(b)
                    handled.append(box)
                else:
                    issues.append('Focused crop remains incomplete: '+reason)
                    handled.append(box)
            except (ValueError,OSError,ModelError) as exc:
                failures.append(str(exc));issues.append('Focused crop failed: '+reason)
                handled.append(box)
        if full and full.coverage=='complete' and pending and not issues and not failures:
            complete=True
        # An incomplete overview cannot certify full coverage merely through local crops.
    except (ValueError,OSError,ModelError) as exc:
        failures.append(str(exc))
    blocks,notes=merge_blocks(blocks,spans);issues.extend(notes)
    issues += [r for b in blocks for r in b['uncertainty']]
    issues += failures
    if full and full.coverage!='complete':issues.append('Full-page text coverage remains incomplete')
    status='COMPLETE' if complete and not issues else 'PARTIAL' if any(b['text'] and not b['uncertainty'] for b in blocks) else 'FAILED'
    result=dict(version=VERSION,contract=contract(),document_id=row['document_id'],page_id=row['id'],
        source_page=row['page_index']+1,status=status,blocks=blocks,issues=issues,failures=failures,requests=requests,
        csv_description_source=False,accuracy_verified=False)
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (folder/'page.md').write_text(markdown(blocks,status,issues),encoding='utf-8')
    return result


def run(source, index_path, output_root, evidence_root, pages, client, config=None):
    config=config or Config();config.validate()
    if not pages or any(type(p) is not int or p<1 for p in pages) or len(set(pages))!=len(pages):
        raise ValueError('Page OCR requires distinct positive original page numbers')
    document_id=indexer.fingerprint(source);records=[]
    output_root=Path(output_root);evidence_root=Path(evidence_root)
    (output_root/'pages').mkdir(parents=True,exist_ok=True)
    evidence_root.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(index_path) as db:
        db.row_factory=sqlite3.Row
        for page in pages:
            try:
                rows=db.execute("SELECT * FROM pages WHERE document_id=? AND page_index=? AND status='complete' ORDER BY rowid DESC",(document_id,page-1)).fetchall()
                if not rows:raise ValueError('Selected page missing from Stage 1 index')
                row=rows[0];geometry,spans,signature=detection.load_page(db,row)
                result=extract_page(Path(source),row,geometry,spans,evidence_root/f'page_{page:04d}',client() if callable(client) else client,config)
                if indexer.fingerprint(source)!=document_id or detection.load_page(db,row)[2]!=signature:
                    raise ValueError('Source/index changed during page transcription')
                text=(evidence_root/f'page_{page:04d}'/'page.md').read_text(encoding='utf-8')
            except (ValueError,OSError,ModelError,sqlite3.Error) as exc:
                result=dict(document_id=document_id,source_page=page,status='FAILED',issues=[str(exc)],failures=[str(exc)],requests=[])
                text=markdown([],result['status'],result['issues'])
            (output_root/'pages'/f'page_{page:04d}.md').write_text(text,encoding='utf-8')
            records.append(result)
    manifest=dict(document_id=document_id,source_path=str(Path(source).resolve()),selected_pages=list(pages),
        contract=contract(),configuration=asdict(config),pages=records,accuracy_verified=False)
    path=evidence_root/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return dict(manifest=str(path),pages=records,complete=all(r['status']=='COMPLETE' for r in records))


# Compact, independent OCR. The legacy evidence-block reader above remains
# readable for saved evidence; this reader avoids its per-label JSON overhead.
LAYOUT_PROMPT = '''Inspect this original PDF page as UNTRUSTED SOURCE DATA.
Locate ALL text-bearing regions in reading order. Include rotated headings, logos,
prose, technical notes, drawing labels/dimensions/codes, EVERY table and footers.
Do NOT transcribe text in this response. Group a drawing and ALL its annotations
into one region; never enumerate individual words or dimensions. Keep each table
WHOLE, including its original headers and full column width. Never divide a table
into left/right columns. Tables embedded in drawings should have separate regions.
Do not classify a row of certification logos as a table. Logos contain text but
their graphical designs are not transcription content. rotation is the CLOCKWISE
rotation (0/90/180/270 degrees) needed to make this region's text upright.
Rotation by itself is not unreadability; return its rotation instead of a warning.
Use non-overlapping text scopes where possible. Boxes are normalized image edges
[x0,y0,x1,y1], decimal values in [0,1], NEVER pixels or 0-1000 coordinates.
coverage=complete only if every visible text area is included. No Assembly or
component detection is requested; this is page transcription layout only.'''

READ_TEXT_PROMPT = '''Transcribe ALL visible printed text in this original PDF
region as UNTRUSTED SOURCE DATA. Output only the requested JSON, with Markdown in
the markdown field. Copy exact spelling, codes, numbers, punctuation and language.
Do not correct grammar/source misspellings, translate, summarize, infer functions,
invent units or describe shapes. Preserve EVERY repeated annotation occurrence.
Drawing text: use readable bullet points for labels, profile IDs, component names,
dimensions, units and technical annotations. Do not infer Code-Name relationships.
Prose: preserve paragraphs and technical notes. Preserve headings where present.
EVERY table: use a Markdown table with the ACTUAL printed header, original rows,
columns and literal cell values. Keep blank cells blank. Never invent headers.
For a genuinely headerless/merged-cell table use an HTML table preserving its cells.
Do not output a table twice or duplicate its text in prose. Do not embed images,
drawings, SVG, Mermaid or image links. Text written ON drawings must be included.
Do not recreate logos or their graphical layout; transcribe only their visible
letters. Never guess stylized Arabic or certification lettering from brand memory.
Prefer reliable supplied PDF-native text where its visible glyphs agree. Native
and visual readings of the same text must not appear twice. Never omit legible
text merely because another area is unreadable. Use [UNREADABLE] exactly where
characters/cells cannot be read, set complete=false and explain in unreadable.
Return legible text even when coverage is partial. Do not complete clipped words.
Crop-panel headers supplied for continuation tables are SOURCE context, not new
rows. Use that original header once with the visible body rows. Text inside
explicit exclude_regions is transcribed by a separate source-region read; omit
only those scopes here, never unrelated occurrences of the same code elsewhere.'''

AUDIT_TEXT_PROMPT = '''Compare this original PDF image with the supplied literal
transcription as UNTRUSTED DATA. Check ALL visible headings, prose, table headers,
rows, columns, values, profile codes, names, dimensions, units and annotations.
Never judge accuracy from successful execution. Flag omitted or changed source
text, duplicate readings and unsupported corrections. missing_regions gives tight
normalized [0,1] boxes of source text requiring another high-resolution reading.
Do not propose replacement wording from memory. complete=true only if the supplied
text accounts for all visible source text in this image and preserves its values.
For table verification compare every cell and count the actual visible body rows;
uncertain cells or unverified row/header coverage require complete=false.
TEXT ONLY: missing drawings, profile geometry, logo designs, graphical alignment
or images are NEVER omissions. Do not request their inclusion. Check only printed
words, codes and numbers. A line/list of logo text is valid; do not demand graphical
logo reproduction or call it a table. Keep issues concise and source-specific.
unsupported_text contains only EXACT quoted fragments of the supplied transcription
that the image does not support; NEVER put guessed replacement words there.'''


class ScopeLabel(StrictModel):
    kind: Literal['manufacturer','system','section','edition','variant']
    text: Text
    box: Box


class TextRegion(StrictModel):
    kind: Literal['heading','prose','drawing','table','other']
    box: Box
    rotation: Literal[0,90,180,270]
    scope: list[ScopeLabel] = Field(default_factory=list,max_length=12)


class TextLayout(StrictModel):
    coverage: Literal['complete','incomplete']
    regions: list[TextRegion]
    uncertainty: list[str]
    scope: list[ScopeLabel] = Field(default_factory=list,max_length=12)

    @classmethod
    def model_json_schema(cls,*args,**kwargs):
        schema=super().model_json_schema(*args,**kwargs)
        # Require an explicit source-scope answer from the live decoder, while
        # retaining compatibility with saved pre-scope responses during parsing.
        schema['required']=list(dict.fromkeys([*schema.get('required',[]),'scope']))
        return schema


class TextReading(StrictModel):
    markdown: str
    complete: bool
    unreadable: list[str]


class TextAudit(StrictModel):
    complete: bool
    missing_regions: list[TextRegion]
    issues: list[str]
    unsupported_text: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class CompactConfig:
    image_max: int = 1536
    crop_max: int = 2048
    max_requests: int = 20
    timeout_seconds: float = 180
    max_tokens: int = 1536
    prefer_native: bool = False

    def validate(self):
        if not (256<=self.image_max<=2048 and 512<=self.crop_max<=2048 and
                type(self.max_requests) is int and self.max_requests>0 and 0<self.timeout_seconds<=600 and
                512<=self.max_tokens<=2048):
            raise ValueError('Compact OCR needs bounded requests and images within the 2048-pixel transport limit')


def compact_contract():
    return grouping.digest([VERSION,LAYOUT_PROMPT,READ_TEXT_PROMPT,AUDIT_TEXT_PROMPT,
        TextLayout.model_json_schema(),TextReading.model_json_schema(),TextAudit.model_json_schema(),code_fingerprint()])


def native_image_spans(spans,mapping,exclude=()):
    """Native hints use THIS image's normalized coordinates, never PDF points.

    A clipped span's full quote is not reliable text for the visible crop. The
    original indexed span remains available for source review/supplements.
    """
    result=[]
    for ordinal,span in spans:
        if not reliable(span):continue
        visible=callouts.visible_context_box(span['bbox'],mapping)
        if not visible or visible['clipped'] or any(intersection(span['bbox'],b)>0 for b in exclude):continue
        result.append(dict(id=ordinal,text=span['text'],box=visible['box']))
    return result


def render_text_region(source,page_index,geometry,box,maximum):
    """Up to 8x source resolution for small text, within the same 2048px cap."""
    source=Path(source)
    if source.suffix.lower()!='.pdf':return detection.render(source,page_index,geometry,box,maximum)
    rotation=fitz.Matrix(*geometry['source_to_display']);display=fitz.Rect(box)*rotation
    scale=min(maximum/max(display.width,display.height),8)
    with fitz.open(source) as pdf:
        pix=pdf[page_index].get_pixmap(matrix=fitz.Matrix(scale,scale),clip=display,alpha=False)
    image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    matrix=rotation*fitz.Matrix(scale,scale)*fitz.Matrix(1,0,0,1,-pix.x,-pix.y)
    if max(image.size)>maximum:
        w,h=image.size;image.thumbnail((maximum,maximum));matrix*=fitz.Matrix(image.width/w,image.height/h)
    return image,dict(source_to_pixels=list(matrix),pixels_to_source=list(~matrix),pixel_size=list(image.size),source_context_box=box)


def upright(image,mapping,rotation):
    if not rotation:return image,mapping
    w,h=image.size
    transform={90:fitz.Matrix(0,1,-1,0,h,0),180:fitz.Matrix(-1,0,0,-1,w,h),270:fitz.Matrix(0,-1,1,0,0,w)}[rotation]
    image=image.transpose({90:Image.Transpose.ROTATE_270,180:Image.Transpose.ROTATE_180,270:Image.Transpose.ROTATE_90}[rotation])
    matrix=fitz.Matrix(*mapping['source_to_pixels'])*transform
    return image,dict(mapping,source_to_pixels=list(matrix),pixels_to_source=list(~matrix),pixel_size=list(image.size))


def markdown_tables(value):
    """Parse real GFM headers/rows without normalizing literal cell values."""
    lines=value.splitlines();result=[];i=0
    def cells(line):
        line=line.strip()
        if line.startswith('|'):line=line[1:]
        if line.endswith('|') and not line.endswith('\\|'):line=line[:-1]
        return [s.strip() for s in re.split(r'(?<!\\)\|',line)]
    while i+1<len(lines):
        if '|' in lines[i] and re.fullmatch(r'\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*',lines[i+1]):
            header=cells(lines[i]);start=i;i+=2;rows=[]
            while i<len(lines) and '|' in lines[i] and lines[i].strip():
                rows.append(cells(lines[i]));i+=1
            result.append(dict(kind='gfm',start=start,end=i,headers=header,rows=rows,columns=len(header),
                valid=bool(header) and all(len(row)==len(header) for row in rows)))
        else:i+=1
    class Tables(HTMLParser):
        def __init__(self):super().__init__();self.table=None;self.row=[];self.cell=None;self.is_header=False
        def handle_starttag(self,tag,attrs):
            if tag=='table':self.table=dict(kind='html',start=None,end=None,headers=[],rows=[],columns=0,valid=True)
            elif self.table is not None and tag=='tr':self.row=[];self.is_header=False
            elif self.table is not None and tag in {'td','th'}:
                self.cell='';self.is_header|=tag=='th'
            elif tag=='br' and self.cell is not None:self.cell+='\n'
        def handle_data(self,data):
            if self.cell is not None:self.cell+=data
        def handle_endtag(self,tag):
            if self.table is None:return
            if tag in {'td','th'} and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
            elif tag=='tr' and self.row:
                if self.is_header:self.table['headers']+=self.row
                else:self.table['rows'].append(self.row)
                self.table['columns']=max(self.table['columns'],len(self.row))
            elif tag=='table':result.append(self.table);self.table=None
    if '<table' in value.lower():Tables().feed(value)
    return result


def clean_transcription(value, kind):
    issues=[]
    if re.search(r'!\[|<\s*(?:img|svg)\b|```\s*(?:mermaid|plantuml|dot)',value,re.I):
        value=re.sub(r'!\[[^\]]*\]\([^)]*\)','',value)
        value=re.sub(r'<svg\b.*?</svg>|<img\b[^>]*>','',value,flags=re.I|re.S)
        value=re.sub(r'```\s*(?:mermaid|plantuml|dot).*?```','',value,flags=re.I|re.S)
        issues.append('Non-text image/drawing markup was withheld')
    tables=markdown_tables(value)
    if any(not t['valid'] for t in tables):issues.append('Table row/column structure is inconsistent')
    if kind=='table' and not tables and '<table' not in value.lower():
        issues.append('Located table was not returned with its original table structure')
    if kind=='drawing':
        lines=value.splitlines();rendered=[]
        for line in lines:
            if line.strip() and not re.match(r'\s*(?:[-*+]\s|\d+[.)]\s|#|\||<)',line):
                rendered.append('- '+line)
            else:rendered.append(line)
        value='\n'.join(rendered)
    return value.strip(),issues


def _merge_table_parts(top,bottom):
    a=markdown_tables(top);b=markdown_tables(bottom)
    if len(a)==len(b)==1 and a[0]['kind']==b[0]['kind']=='gfm' and a[0]['valid'] and b[0]['valid'] and a[0]['headers']==b[0]['headers']:
        left=top.splitlines();right=bottom.splitlines()
        # The header is the same original source panel. Never deduplicate data rows.
        return '\n'.join(left[:a[0]['end']]+right[b[0]['start']+2:b[0]['end']]+left[a[0]['end']:]+right[b[0]['end']:]),True
    return top+'\n\n'+bottom,False


def native_supplements(spans,text):
    seen=set();required={};ordered=[]
    for _,span in spans:
        identity=(tuple(round(v,3) for v in span['bbox']),span['text'])
        if not reliable(span) or identity in seen:continue
        seen.add(identity);required[span['text']]=required.get(span['text'],0)+1;ordered.append(span['text'])
    missing={}
    plain=re.sub(r'(?m)^\s*(?:#{1,6}\s+|[-*+]\s+)','',text).replace('`','')
    for value,count in required.items():
        boundary=r'[\w.,]' if re.fullmatch(r'[\d.,]+',value) else r'\w'
        found=len(re.findall(r'(?<!'+boundary+r')'+re.escape(value)+r'(?!'+boundary+r')',plain))
        missing[value]=max(0,count-found)
    result=[]
    for value in ordered:
        if missing[value]:result.append(value);missing[value]-=1
    return result


def extract_page_compact(source,row,geometry,spans,folder,client,config=None,vectors=()):
    """Read bounded text scopes; failed scopes cannot erase successful scopes."""
    config=config or CompactConfig();config.validate()
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    bounds=geometry['source_rect'];requests=[];failures=[];units=[];unresolved=[]
    started=time.monotonic();initial_requests=client.requests
    cache={}
    def rendered(box, maximum, rotation=0):
        key=(tuple(box),maximum,rotation)
        if key not in cache:
            # One region retained: read and audit share source pixels without
            # caching every full-resolution drawing on a dense page.
            cache.clear()
            image,mapping=render_text_region(Path(source),row['page_index'],geometry,box,maximum)
            cache[key]=upright(image,mapping,rotation)
        return cache[key]
    def request(image,prompt,context,schema,label,cap):
        if client.requests-initial_requests>=config.max_requests:raise ModelError('OCR page request budget exhausted')
        if max(image.size)>2048:raise ValueError('OCR region exceeds transport pixel limit')
        path=folder/(label+'.png');image.save(path)
        saved=client.config
        client.config=replace(saved,max_tokens=min(config.max_tokens,cap),timeout=min(saved.timeout,config.timeout_seconds),retries=0)
        event=dict(id=label,image=str(path),image_size=list(image.size),original_page=row['page_index']+1,context=context)
        requests.append(event);before=time.monotonic()
        from . import progress
        progress.request_start();request_error=None;diagnostic_start=len(getattr(client,'request_diagnostics',[]))
        try:
            with capture_diagnostics(client,folder/'diagnostics',stage=1,subject_id=row['id']+'-'+label):
                raw=client.request_json(image,prompt,json.dumps(context,ensure_ascii=False),response_schema=schema.model_json_schema())
                performance.atomic_write(folder/(label+'-response.json'),raw)
                result=components.parse_response(raw,schema)
            event['state']='COMPLETE';return result
        except (ValueError,OSError,ModelError) as exc:
            request_error=callouts.failure_reason(exc)
            event.update(state='FAILED',error=str(exc));failures.append(dict(request=label,error=str(exc)))
            raise
        finally:
            event['duration_seconds']=round(time.monotonic()-before,3);client.config=saved
            diagnostics=getattr(client,'request_diagnostics',[])[diagnostic_start:]
            progress.request_end(event['duration_seconds'],error=request_error,
                timeouts=sum(bool(d.get('timeout_classification')) for d in diagnostics))
    overview,mapping=rendered(bounds,config.image_max)
    native=native_image_spans(spans,mapping)
    try:
        layout=request(overview,LAYOUT_PROMPT,dict(original_page=row['page_index']+1,native_spans=native,
            native_coordinate_space='normalized_current_image'),TextLayout,'layout',config.max_tokens)
        regions=[dict(kind=r.kind,box=detection.source_box(r.box,mapping,bounds),rotation=r.rotation,
                      scope=[s.model_dump(mode='json') for s in r.scope]) for r in layout.regions]
        layout_ok=layout.coverage=='complete' and not layout.uncertainty
        unresolved+=layout.uncertainty
    except (ValueError,OSError,ModelError):
        # Full-width bands preserve columns, unlike the old quadrant fallback.
        x0,y0,x1,y1=bounds;h=(y1-y0)/3
        regions=[dict(kind='other',box=[x0,y0+i*h,x1,y0+(i+1)*h]) for i in range(3)]
        layout_ok=False;unresolved.append('Layout failed; full-width recovery bands need boundary review')
        layout=None
    def read(box,kind,label,depth=0,header=None,exclude=(),rotation=0):
        image,transform=rendered(box,config.crop_max,rotation if header is None else 0)
        panels=[dict(source_box=box,role='body')]
        if header is not None:
            head,_=rendered(header,config.crop_max)
            width=max(image.width,head.width);combined=Image.new('RGB',(width,image.height+head.height+8),'white')
            combined.paste(head,(0,0));combined.paste(image,(0,head.height+8));combined.thumbnail((2048,2048));image=combined
            panels.insert(0,dict(source_box=header,role='original_table_header_context'))
        facts=native_image_spans(spans,transform,exclude) if header is None else []
        literal=''
        try:
            all_spans=[s for _,s in spans if intersection(s['bbox'],box)>0]
            if (config.prefer_native and kind in {'heading','prose'} and not exclude and not rotation
                    and all_spans and all(reliable(s) and intersection(s['bbox'],box)>=area(s['bbox'])*.99 for s in all_spans)):
                literal='\n'.join(s['text'] for s in sorted(all_spans,key=lambda s:(s['bbox'][1],s['bbox'][0])))
                try:
                    audit=request(image,AUDIT_TEXT_PROMPT,dict(original_page=row['page_index']+1,transcription=literal),
                                  TextAudit,label+'-native-check',512)
                except (ValueError,OSError,ModelError):
                    audit=None
                if audit and audit.complete and not (audit.issues or audit.missing_regions or audit.unsupported_text):
                    return dict(source_box=box,kind=kind,rotation=rotation,markdown=literal,complete=True,
                        issues=[],request=label,source_audit=audit.model_dump(mode='json'),native_verified=True)
            result=request(image,READ_TEXT_PROMPT,dict(original_page=row['page_index']+1,kind=kind,source_panels=panels,
                reliable_native_spans=facts,clockwise_deskew=rotation,exclude_regions=[callouts.visible_context_box(b,transform) for b in exclude
                    if callouts.visible_context_box(b,transform)]),TextReading,label,config.max_tokens)
            text,notes=clean_transcription(result.markdown,kind)
            return dict(source_box=box,kind=kind,rotation=rotation,markdown=text,complete=result.complete and not result.unreadable and not notes,
                        issues=result.unreadable+notes,request=label)
        except (ValueError,OSError,ModelError) as exc:
            if depth==0 and not rotation and client.requests-initial_requests<=config.max_requests-3:
                x0,y0,x1,y1=box;middle=(y0+y1)/2;cuts=[]
                for path in vectors:
                    for item in path.get('items',[]):
                        if item[0]=='l':
                            a,b=item[1:3]
                            if abs(a[1]-b[1])<.5 and abs(a[0]-b[0])>=(x1-x0)*.65 and y0+10<a[1]<y1-10:cuts.append(float(a[1]))
                if cuts:middle=min(cuts,key=lambda y:abs(y-middle))
                header_box=[x0,y0,x1,min(middle,y0+min(45,(y1-y0)*.16))] if kind=='table' else None
                top=read([x0,y0,x1,middle],kind,label+'-top',1,exclude=exclude)
                bottom=read([x0,middle,x1,y1],kind,label+'-bottom',1,header_box,exclude)
                text,merged=_merge_table_parts(top['markdown'],bottom['markdown']) if kind=='table' else (top['markdown']+'\n\n'+bottom['markdown'],True)
                issues=top['issues']+bottom['issues']
                if kind=='table' and not merged:issues.append('Continuation-table header could not be reconciled exactly')
                return dict(source_box=box,kind=kind,markdown=text,complete=top['complete'] and bottom['complete'] and merged,
                    issues=issues,request=label,recovered_after=str(exc),parts=[top,bottom])
            return dict(source_box=box,kind=kind,markdown=literal,complete=False,
                        issues=[str(exc)]+(['Native draft retained; visual verification incomplete'] if literal else []),request=label)
    for ordinal,region in enumerate(regions):
        # A table has its own full-column read; drawing readers receive exclusions.
        exclude=[r['box'] for r in regions if r is not region and r['kind']=='table' and
                 region['kind']!='table' and intersection(r['box'],region['box'])>=area(r['box'])*.98]
        unit=read(region['box'],region['kind'],'region-'+str(ordinal+1),exclude=exclude,rotation=region.get('rotation',0))
        tables=markdown_tables(unit['markdown'])
        if unit['markdown'] and not unit.get('native_verified'):
            try:
                image,transform=rendered(region['box'],config.crop_max,region.get('rotation',0))
                audit=request(image,AUDIT_TEXT_PROMPT,dict(original_page=row['page_index']+1,transcription=unit['markdown'],
                    table_structure=[dict(headers=t['headers'],body_rows=len(t['rows']),columns=t['columns']) for t in tables]),
                    TextAudit,('table-check-' if tables else 'text-check-')+str(ordinal+1),512)
                unit['markdown'],audit=audit_formatted_text(unit['markdown'],audit,
                    lambda plain: request(image,AUDIT_TEXT_PROMPT,
                        dict(original_page=row['page_index']+1,transcription=plain,
                             table_structure=[dict(headers=t['headers'],body_rows=len(t['rows']),columns=t['columns']) for t in tables]),
                        TextAudit,'literal-check-'+str(ordinal+1),512))
                unit['source_audit']=audit.model_dump(mode='json')
                if tables:unit['table_audit']=audit.model_dump(mode='json')
                for fragment in audit.unsupported_text:
                    if fragment and fragment in unit['markdown']:
                        boundary=r'[\w.,]' if re.fullmatch(r'[\d.,]+',fragment) else r'\w'
                        unit['markdown']=re.sub(r'(?<!'+boundary+r')'+re.escape(fragment)+r'(?!'+boundary+r')','[UNREADABLE]',unit['markdown'])
                if not audit.complete or audit.issues or audit.missing_regions:
                    unit['complete']=False;unit['issues']+=audit.issues or ['Source text/table values could not be verified']
                if audit.unsupported_text:
                    unit['complete']=False;unit['issues'].append('Unsupported source fragments were withheld')
                    unit['source_audit']['complete']=False
                    if tables:unit['table_audit']['complete']=False
            except (ValueError,OSError,ModelError) as exc:
                unit['complete']=False;unit['issues'].append('Independent source verification failed: '+str(exc))
        unit['scope_claims']=region.get('scope',[])
        source_image=folder/('region-'+str(ordinal+1)+'.png')
        if not source_image.exists():source_image=folder/('region-'+str(ordinal+1)+'-native-check.png')
        if source_image.exists():
            unit['source_image']=str(source_image.resolve())
            unit['source_image_sha256']=hashlib.sha256(source_image.read_bytes()).hexdigest()
        units.append(unit)
        performance.atomic_json(folder/'checkpoint.json',dict(source_page=row['page_index']+1,units=units,failures=failures,requests=requests))
    # Native spans absent from the visual reading remain exact source evidence.
    text='\n\n'.join(u['markdown'] for u in units if u['markdown'])
    supplements=native_supplements(spans,text)
    if supplements:
        text+='\n\n'+'\n'.join('- '+s for s in supplements)
        unresolved.append('Reliable native text supplements require reading-order/source comparison')
    audit_ok=False
    try:
        audit=request(overview,AUDIT_TEXT_PROMPT+'\nThis overview pass checks TEXT COVERAGE and duplicates only. Fine character accuracy was checked on high-resolution regions. Do not criticize absent graphics.',dict(original_page=row['page_index']+1,transcription=text),TextAudit,'page-check',768)
        audit_ok=audit.complete and not audit.issues and not audit.missing_regions and not audit.unsupported_text
        unresolved+=audit.issues
        for n,missing in enumerate(audit.missing_regions):
            if client.requests-initial_requests>=config.max_requests:unresolved.append('Page recovery budget exhausted');break
            box=detection.source_box(missing.box,mapping,bounds)
            matched=next((u for u in units if intersection(box,u['source_box'])>=area(u['source_box'])*.85 and
                          intersection(box,u['source_box'])/max(area(box),.001)>.65),None)
            if matched:
                union=[min(box[0],matched['source_box'][0]),min(box[1],matched['source_box'][1]),
                       max(box[2],matched['source_box'][2]),max(box[3],matched['source_box'][3])]
                extra=read(union,matched['kind'],'missing-'+str(n+1),rotation=missing.rotation or matched.get('rotation',0))
                if extra['complete'] or len(extra['markdown'])>len(matched['markdown']):units[units.index(matched)]=extra
            else:
                extra=read(box,missing.kind,'missing-'+str(n+1),rotation=missing.rotation);units.append(extra)
        text='\n\n'.join(u['markdown'] for u in units if u['markdown'])
        remaining_native=native_supplements(spans,text)
        if remaining_native:text+='\n\n'+'\n'.join('- '+s for s in remaining_native)
        if audit.missing_regions:unresolved.append('Additional source regions recovered; full-page comparison remains REVIEW')
    except (ValueError,OSError,ModelError) as exc:unresolved.append('Page coverage verification failed: '+str(exc))
    unresolved+=[issue for u in units for issue in u['issues']]
    status='COMPLETE' if layout_ok and audit_ok and all(u['complete'] for u in units) and not unresolved else 'PARTIAL' if text.strip() else 'FAILED'
    result=dict(version=VERSION,engine='compact_region_ocr',contract=compact_contract(),document_id=row['document_id'],page_id=row['id'],
        source_page=row['page_index']+1,status=status,units=units,issues=unresolved,failures=failures,requests=requests,
        model_requests=client.requests-initial_requests,duration_seconds=round(time.monotonic()-started,3),accuracy_verified=False,
        native_span_count=len(spans),csv_description_source=False)
    from . import catalog_lookup
    result['scope_evidence']=catalog_lookup.verify_scopes(
        [s.model_dump(mode='json') for s in layout.scope] if layout else [],mapping,bounds,spans,units,source,row['page_index'])
    for unit in units:
        unit['scope_evidence']=catalog_lookup.verify_scopes(unit.get('scope_claims',[]),mapping,bounds,spans,units,source,row['page_index'])
    result['source_path']=str(Path(source).resolve())
    result['configuration']=asdict(config)
    output=f'# Original PDF page {row["page_index"]+1}\n\n**OCR status: {status}**\n\n'+text+'\n'
    if unresolved:output+='\n[REVIEW: '+'; '.join(dict.fromkeys(unresolved))+']\n'
    result['output_sha256']=hashlib.sha256(output.encode()).hexdigest()
    performance.atomic_write(folder/'page.md',output)
    performance.atomic_json(folder/'result.json',result)
    return result


def literal_markdown(text):
    """Remove presentation markers only; preserve digits, punctuation and words."""
    text=re.sub(r'(?m)^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|>\s+)', '', text)
    return re.sub(r'(\*\*|__|`)(.+?)\1',r'\2',text)


def audit_formatted_text(text,audit,reaudit):
    """Recheck visible wording before a formatting complaint can erase it.

    Reuses the existing image, prompt, timeout and request budget. A failed or
    inconclusive recheck does not authorize the text.
    """
    if any(literal_markdown(fragment)!=fragment for fragment in audit.unsupported_text):
        plain=literal_markdown(text)
        if plain!=text:
            try:return plain,reaudit(plain)
            except (ValueError,OSError,ModelError):pass
    return text,audit


def source_page_context(source,page_number):
    """Fresh native text/geometry for exactly one original page; no stage/index run."""
    with fitz.open(source) as pdf:
        page=pdf[page_number-1]
        spans=[]
        for block in page.get_text('dict',flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks']:
            for line in block.get('lines',[]):
                spans.extend(line['spans'])
        indexer.flag_native_control_spans(spans)
        geometry=dict(source_rect=list(page.rect*page.derotation_matrix),source_to_display=list(page.rotation_matrix))
        vectors=page.get_drawings()
    return geometry,list(enumerate(spans)),vectors


def _compact_job(task):
    from . import progress
    page=task[2]
    progress.start(1,page,'Independent compact Markdown OCR')
    try:
        result=_compact_source_job(task)
    except Exception:
        progress.done(1,page,ok=False,failed=True,reason='Markdown OCR execution failed')
        raise
    progress.done(1,page,ok=result['status']=='COMPLETE',failed=result['status']=='FAILED',
                  reason='Markdown OCR '+result['status'])
    return result


def _compact_source_job(task):
    source,index_path,page,folder,model,config = task
    from . import progress
    from .qwen import QwenClient
    document_id=indexer.fingerprint(source)
    with sqlite3.connect(Path(index_path).resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        row=db.execute("SELECT * FROM pages WHERE document_id=? AND page_index=? AND status='complete' ORDER BY rowid DESC LIMIT 1",
                       (document_id,page-1)).fetchone()
        if row is None:raise ValueError('Selected original page missing from source index')
        row=dict(row)
        geometry,spans,signature=detection.load_page(db,row)
        vectors=[json.loads(x[0]) for x in db.execute('SELECT data FROM vectors WHERE page_id=? ORDER BY ordinal',(row['id'],))]
        result=extract_page_compact(source,row,geometry,spans,folder,QwenClient(model),config,vectors)
        if detection.load_page(db,row)[2]!=signature or indexer.fingerprint(source)!=document_id:
            raise ValueError('Source/index changed during compact OCR')
    performance.validate_markdown(folder,document_id,page,compact_contract())
    return result


def run_compact(source,index_path,output_root,evidence_root,pages,model,config=None,workers=1):
    """Independent original-page jobs; dependencies within each page remain ordered."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from . import async_inference,progress
    config=config or CompactConfig();config.validate()
    if (not pages or any(type(p) is not int or p<1 for p in pages) or len(set(pages))!=len(pages)
            or type(workers) is not int or not 1<=workers<=(32 if async_inference.enabled() else 4)):
        raise ValueError('Require explicit unique original pages and 1-32 workers with async inference (otherwise 1-4)')
    document_id=indexer.fingerprint(source)
    with sqlite3.connect(Path(index_path).resolve().as_uri()+'?mode=ro',uri=True) as db:
        row=db.execute('SELECT page_count FROM documents WHERE id=?',(document_id,)).fetchone()
        if row is None or max(pages)>row[0]:raise ValueError('Source/page selection does not match index')
    root=Path(evidence_root);root.mkdir(parents=True,exist_ok=False)
    public=Path(output_root)/'pages';public.mkdir(parents=True,exist_ok=True)
    tasks=[(Path(source),Path(index_path),p,root/f'page_{p:04d}',model,config) for p in sorted(pages)]
    records={}
    def save(task,future):
        page=task[2];folder=task[3]
        try:
            if future is None:_compact_job(task)
            else:future.result()
            result,text=performance.validate_markdown(folder,document_id,page,compact_contract())
        except (ValueError,OSError,ModelError,sqlite3.Error) as exc:
            result=dict(document_id=document_id,source_page=page,status='FAILED',issues=[str(exc)],model_requests=0)
            text=f'# Original PDF page {page}\n\n**OCR status: FAILED**\n\n[REVIEW: {exc}]\n'
            performance.atomic_json(root/f'page_{page:04d}-failure.json',result)
        performance.atomic_write(public/f'page_{page:04d}.md',text)
        records[page]=result
        performance.atomic_json(root/'manifest.json',dict(document_id=document_id,source_path=str(Path(source).resolve()),
            selected_pages=sorted(pages),contract=compact_contract(),configuration=asdict(config),
            pages=[records[p] for p in sorted(records)],pending_pages=sorted(set(pages)-records.keys()),accuracy_verified=False))
    if workers==1:
        for task in tasks:save(task,None)
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),
                **async_inference.worker_options(progress.worker_options())) as pool:
            for task,future in performance.bounded_results(pool,_compact_job,tasks,workers,async_inference.cancel):
                save(task,future)
    return dict(manifest=str(root/'manifest.json'),pages=[records[p] for p in sorted(records)],
                complete=all(r['status']=='COMPLETE' for r in records.values()),
                requests=sum(r.get('model_requests',0) for r in records.values()))


from .fabrication_prompts import OCR_LAYOUT_ADDENDUM, OCR_READ_ADDENDUM
LAYOUT_PROMPT += OCR_LAYOUT_ADDENDUM
READ_TEXT_PROMPT += OCR_READ_ADDENDUM
PROMPT += OCR_READ_ADDENDUM
AUDIT_TEXT_PROMPT += "\nCheck outlined article codes and local numbered keys even when native prose exists. Verify exact printed scope headings and every code/name table cell; never infer scope or identities."

LAYOUT_PROMPT += '''
Native hint boxes are already NORMALIZED to THIS supplied image. Never divide
them by pixel width/height again or interpret them as PDF points. Verify visible
glyphs before using them. Return top-level scope explicitly, including [] if no
printed scope is established. Group contiguous headings/prose of the same
orientation into a useful region rather than many tiny strips; retain every
word and keep independent tables/drawings separate. All returned boxes remain
normalized image edges, not a rescaling of the supplied normalized hints.'''
LAYOUT_PROMPT += '''
When the image prints BOTH a system designation and a product-family banner,
include BOTH system and section in the top-level scope. Quote a complete visible
product-family phrase from a long repeated banner; do not quote an incomplete
continuation beyond the image edge. A chapter heading alone is not that family.
Do not duplicate page-global scope on every region: region scope is for a local
option/namespace that is explicitly different from or refines the page scope.'''
READ_TEXT_PROMPT += '''
Native hint boxes are normalized to THIS source crop and omit clipped spans.
Do not extend a source quote beyond the characters visible in this crop. A missing
native hint is not missing visible text: inspect all original pixels normally.'''

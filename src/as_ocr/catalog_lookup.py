"""Literal, scoped catalog names from independently audited original-page OCR.

No new occurrences or counts. Missing scope and conflicting names fail closed.
"""
from pathlib import Path
import re
import unicodedata

from .geometry import area,intersection

VERSION='1'


def key(value):
    return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())


def audited(unit):
    audit=unit.get('source_audit',{})
    return bool(unit.get('complete') and not unit.get('issues') and audit.get('complete')
        and not any(audit.get(k) for k in ('issues','missing_regions','unsupported_text')))


def verify_scopes(claims,mapping,bounds,spans,units,source=None,page_index=None):
    from .detection import source_box
    from .page_markdown import reliable
    result=[]
    for claim in claims:
        box=source_box(claim['box'],mapping,bounds);text=claim['text']
        native=[n for n,s in spans if s.get('text','').strip()==text.strip()
            and intersection(s['bbox'],box)>=.8*area(s['bbox'])
            and reliable(s)]
        reads=[u for u in units if audited(u) and text in u.get('markdown','')
            and intersection(u['source_box'],box)>=.99*area(box)]
        search=[]
        if not native and not reads and source is not None and page_index is not None and Path(source).is_file() and Path(source).suffix.lower()=='.pdf':
            # A product-family phrase may be only part of a long repeated native
            # banner. Search real glyph locations, not inferred font widths or a
            # whole-span quote whose other end lies outside the page/crop.
            import pymupdf as fitz
            with fitz.open(source) as pdf:
                page=pdf[page_index]
                for rect in page.search_for(text):
                    hit=list(rect)
                    if area(hit)>0 and intersection(hit,bounds)>=area(hit)-.1 and intersection(hit,box)>=.8*area(hit):search.append(hit)
        if native or reads or search:
            result.append(dict(kind=claim['kind'],text=text,box=box,
                method='native_quote' if native else 'native_search_quote' if search else 'independently_audited_visual_quote',
                source_quote_boxes=search,
                native_span_ids=native,verified=True))
    return result


def scope_values(labels):
    result={}
    for kind in ('manufacturer','system','section','edition','variant'):
        values={key(s['text']) for s in labels if s.get('verified') and s.get('kind')==kind}
        if len(values)>1:return None
        result[kind]=next(iter(values),'')
    return result if result['system'] and result['section'] else None


def scoped(page,unit=None):
    labels=list(page.get('scope_evidence',[]))
    local=(unit or {}).get('scope_evidence',[])
    overridden={s['kind'] for s in local if s.get('verified')}
    return scope_values([s for s in labels if s.get('kind') not in overridden]+local)


CODE_HEADERS={'reference','ref','référence','referencia','referenz','code','article','article number',
    'art.-nr.','art. no.','profile id','product code','رقم الصنف','الكود'}
NAME_HEADERS={'description','designation','désignation','name','component name','bezeichnung',
    'descripción','الوصف','اسم المكون'}
CONDITION_HEADERS={'condition','conditions','variant','option','hand','handedness','opening','mounting',
    'system','section','bedingung','variante','condition d’emploi','الشرط','النظام','البديل'}


def rows(unit):
    from .page_markdown import markdown_tables
    from .callouts import non_code_category
    for number,table in enumerate(markdown_tables(unit.get('markdown',''))):
        if not table.get('valid'):continue
        if table.get('kind')=='html' and re.search(r'\b(?:rowspan|colspan)\s*=',unit['markdown'],re.I):
            continue  # Merged-cell scope is retained in OCR, not guessed for lookup.
        headers=[key(h) for h in table['headers']]
        codes=[i for i,h in enumerate(headers) if h in CODE_HEADERS]
        names=[i for i,h in enumerate(headers) if h in NAME_HEADERS]
        if len(codes)!=1 or len(names)>1:continue
        for row_number,row in enumerate(table['rows'],1):
            if len(row)!=len(headers):continue
            literal=row[codes[0]].strip()
            if names:
                code=literal;name=row[names[0]].strip();name_column=names[0]
            else:
                # One printed Reference cell can contain both code and designation.
                match=re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9._/-]{1,79})\s+(.+)',literal)
                if not match:continue
                code,name=match.groups();name_column=codes[0]
            if (not code or len(code)>80 or '\n' in code or not re.search(r'\d',code)
                or non_code_category(code) or not name or name==code
                or re.search(r'\[UNREADABLE\]|\b(?:as per|see table|see page)\b',name,re.I)):
                continue
            conditions=[value for h,value in zip(headers,row) if h in CONDITION_HEADERS and value.strip()]
            yield dict(code=code,name=name,conditions=conditions,table_number=number+1,row_number=row_number,
                code_column=codes[0]+1,name_column=name_column+1,headers=table['headers'],cells=row)


def build(manifest,snapshot,document_id):
    from . import indexer
    saved=snapshot.json(manifest)
    if saved.get('document_id')!=document_id:raise ValueError('Foreign catalog OCR manifest')
    source=Path(saved['source_path'])
    if indexer.fingerprint(source)!=document_id:raise ValueError('Catalog source PDF changed')
    pages={};entries=[]
    for page in saved.get('pages',[]):
        if page.get('document_id')!=document_id:raise ValueError('Foreign catalog OCR page')
        number=page['source_page']
        if number in pages:raise ValueError('Duplicate catalog OCR page')
        pages[number]=page
        for unit in page.get('units',[]):
            scope=scoped(page,unit)
            if not audited(unit) or not scope:continue
            image=unit.get('source_image');sha=unit.get('source_image_sha256')
            if not image or not sha:continue
            snapshot.bytes(image,sha)
            for row in rows(unit):
                if any(key(c) not in scope.values() for c in row['conditions']):continue
                entries.append(dict(row,document_id=document_id,source_page=number,
                    source_box=unit['source_box'],scope=scope,
                    scope_evidence=page.get('scope_evidence',[])+unit.get('scope_evidence',[]),
                    source_image=image,source_image_sha256=sha,ocr_manifest=str(Path(manifest).resolve()),
                    source_audit=unit['source_audit']))
    by_code={}
    for entry in entries:by_code.setdefault(entry['code'],[]).append(entry)
    return dict(document_id=document_id,pages=pages,entries=entries,by_code=by_code,manifest=str(manifest),source_path=str(source))


def lookup(index,code,page_number,view_box):
    if not view_box:return '',[],['CATALOG_TARGET_GEOMETRY_UNAVAILABLE']
    page=index.get('pages',{}).get(page_number,{})
    units=[u for u in page.get('units',[]) if u.get('kind')=='drawing' and
        intersection(u['source_box'],view_box)>=.8*area(view_box)]
    scopes=[scoped(page,u) for u in units] if units else [scoped(page)]
    if not scopes or any(s is None for s in scopes) or any(s!=scopes[0] for s in scopes):
        return '',[],['CATALOG_SCOPE_UNRESOLVED']
    # Codes are literal matches: normalization is only for printed scope wording.
    hits=[e for e in index.get('by_code',{}).get(code,[]) if e['scope']==scopes[0]]
    choices={h['name'] for h in hits}
    if len(choices)!=1:return '',hits,['CATALOG_NAME_CONFLICT' if choices else 'CATALOG_NAME_NOT_FOUND']
    return next(iter(choices)),hits,[]


def apply(rows_,traces,index):
    for row,trace in zip(rows_,traces):
        if not row['profile_id']:continue
        name,evidence,reasons=lookup(index,row['profile_id'],int(row['source_page']),trace['source_structural_box'])
        trace['catalog_lookup']=dict(status='SUPPORTED' if name else 'REVIEW',evidence=evidence,reasons=reasons)
        if name and not row['component_name']:
            row['component_name']=name;trace['name_source']='catalog_table'
            trace['reasons']=[r for r in trace['reasons'] if r!='NO_VERIFIED_NAME_IN_ASSOCIATED_SOURCE_CALLOUT']
        elif name and row['component_name']!=name:
            trace['catalog_lookup'].update(status='LOCAL_NAME_PRESERVED',reasons=['Literal local callout takes precedence'])

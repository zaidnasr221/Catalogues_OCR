"""Stage 6: explicit same-document retrieval and scoped property assignments.

References remain independent candidates. Matching code text never proves scope
or changes a physical membership. Scanned coverage is explicit, never absence.
"""
from __future__ import annotations

from .provenance import code_fingerprint

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import uuid

from PIL import Image
from pydantic import Field, model_validator
import pymupdf as fitz

from . import callouts, components, detection, grouping, indexer
from .geometry import expand, intersection
from .qwen import ModelConfig, ModelError, QwenClient
from .regions import StrictModel, Text

VERSION = '2'
PROMPT_VERSION = '1'
SCHEMA_VERSION = '1'
PROMPT = components.PROMPT + '''
STAGE 6 REFERENCE LOOKUP: inspect ONLY the supplied target codes and this crop.
Do not create callout occurrences. Components here are independent reference
items: occurrence_ids MUST be empty. New reference items may use table_row OR a
code_label whose bounds enclose an explicit code caption and its properties.
Only EXACT raw codes from target_codes are allowed. A normalized spelling is only
a retrieval clue. Do not complete unreadable codes, match by shape, or assume a
missing native-text match means absence. Keep manufacturer/system/edition unknown
unless explicitly labeled in this crop; view titles and filenames are not scope.
Table values require the exact code row, units/header evidence, explicit axes and
conditions. General notes cannot assign properties without demonstrated scope.
Use surface_finish for finishes, never material. If a field has a footnote marker,
return a footnotes link: component local ID, original field index, marker text,
marker evidence, note evidence, and scope evidence. The note must visibly carry
that marker. Scope evidence must name this exact code OR explicitly apply to all
marked items of the item's explicitly labeled category. Uncertain links stay
unresolved; do not apply a general material/finish note to an adjacent drawing.
Return footnotes=[] when no explicit supported link is available. No cross-document
identity decisions or component-to-Detail membership may be proposed.'''


class Window(StrictModel):
    page_index: int = Field(ge=0)
    box: tuple[float,float,float,float]
    reason: Text

    @model_validator(mode='after')
    def valid_box(self):
        grouping.checked_box(list(self.box))
        return self


class LookupPlan(StrictModel):
    document_id: str
    selection_reason: Text
    index_pages: list[int] = Field(max_length=10)
    windows: list[Window] = Field(max_length=20)

    @model_validator(mode='after')
    def bounded_selection(self):
        if len(set(self.index_pages)) != len(self.index_pages) or any(p<0 for p in self.index_pages):
            raise ValueError('Selected page indices must be distinct and nonnegative')
        return self


class Footnote(StrictModel):
    component_local_id: Text
    field_index: int = Field(ge=0)
    marker: Text
    marker_evidence: components.Evidence
    note_evidence: components.Evidence
    scope_evidence: list[components.Evidence] = Field(min_length=1)
    explanation: Text
    uncertainty: list[Text]


class Analysis(components.Analysis):
    footnotes: list[Footnote]


@dataclass(frozen=True)
class Config:
    verify_hits: bool = False
    max_hit_windows: int = 8
    image_max: int = 2048
    hit_margin_fraction: float = .06

    def validate(self):
        if not (0<=self.max_hit_windows<=20 and 256<=self.image_max<=2048 and
                math.isfinite(self.hit_margin_fraction) and 0<=self.hit_margin_fraction<=.25):
            raise ValueError('Invalid bounded enrichment settings')


def load_bundle(path):
    path,s5 = components.read_manifest(path,5)
    s3path = s5['upstream']['stage3']['path']; s4path = s5['upstream']['stage4']['path']
    for key,p in [('stage3',s3path),('stage4',s4path)]:
        if grouping.file_digest(p) != s5['upstream'][key]['sha256']:
            raise ValueError('Stage 5 upstream content changed')
    s2,views,details,s4,upstream = components.load_inputs(s3path,s4path)
    if s5['upstream']['document_id'] != s2['document_id']:
        raise ValueError('Foreign Stage 5 source document')
    if not set(s5['selected_view_ids']) <= {v['id'] for v in views}:
        raise ValueError('Unknown Stage 5 view')
    reference = {'stage5_path':str(path),'stage5_sha256':grouping.file_digest(path),
                 'stage3_path':str(Path(s3path).resolve()),'stage3_sha256':grouping.file_digest(s3path),
                 'stage4_path':str(Path(s4path).resolve()),'stage4_sha256':grouping.file_digest(s4path),
                 'document_id':s2['document_id'],'page_id':s2['page_id'],'page_index':s2['page_index'],
                 'source_path':s2['source_path']}
    return {'reference':reference,'stage5':s5,'stage4':s4,'stage2':s2,'views':views,'details':details,'upstream':upstream}


def indexed_pages(db,document_id):
    # Do not resolve ambiguous Stage 1 configurations silently.
    rows = db.execute("SELECT * FROM pages WHERE document_id=? AND status='complete' ORDER BY page_index,id",(document_id,)).fetchall()
    result = {}
    for row in rows:
        if row['page_index'] in result:
            raise ValueError('Multiple complete indexed configurations for one source page')
        result[row['page_index']] = row
    return result


def index_signature(db,document_ids):
    records = []
    for document_id in sorted(set(document_ids)):
        doc = db.execute('SELECT id,page_count FROM documents WHERE id=?',(document_id,)).fetchone()
        pages = db.execute('SELECT id,page_index,status,config,geometry,signals FROM pages WHERE document_id=? ORDER BY id',(document_id,)).fetchall()
        spans = db.execute('SELECT s.page_id,s.ordinal,s.data FROM spans s JOIN pages p ON p.id=s.page_id WHERE p.document_id=? ORDER BY s.page_id,s.ordinal',(document_id,)).fetchall()
        records.append([list(doc) if doc else None,[list(r) for r in pages],[list(r) for r in spans]])
    return grouping.digest(records)


def index_selected(db,source,document_id,pages,output):
    """Append at most ten explicitly selected pages using the actual Stage 1 extractor."""
    if len(pages)>10 or len(set(pages))!=len(pages):
        raise ValueError('At most ten distinct additional pages may be selected')
    if indexer.fingerprint(source)!=document_id:
        raise ValueError('Selected original source identity changed')
    existing = indexed_pages(db,document_id)
    missing = [p for p in pages if p not in existing]
    if not missing:
        return []
    docrow = db.execute('SELECT page_count,format FROM documents WHERE id=?',(document_id,)).fetchone()
    if docrow is None or docrow['format']!='pdf' or any(p<0 or p>=docrow['page_count'] for p in missing):
        raise ValueError('Additional selection must be pages of the existing PDF')
    config = indexer.dumps({'version':indexer.VERSION,'preview_max':1200,'pymupdf':fitz.VersionBind,
                            'pillow':Image.__version__,'selected_reference_pages':True})
    config_id = hashlib.sha256(config.encode()).hexdigest()[:16]
    run_id = uuid.uuid4().hex
    with db:
        db.execute('INSERT INTO runs(id,config,selection) VALUES(?,?,?)',(run_id,config,indexer.dumps({'source':str(source),'explicit_page_indices':pages,'stage':6})))
    folder = Path(output)/'indexed-previews';folder.mkdir(parents=True,exist_ok=True)
    added=[]
    with fitz.open(source) as doc:
        for p in missing:
            pid=f'{document_id}:{p}:{config_id}';path=folder/(uuid.uuid4().hex+'.png')
            with db:
                db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?)',(run_id,document_id,p,str(source),'processing',None))
            try:
                w,h,rotation,geometry,signals,spans,vectors,images = indexer.pdf_page(doc[p],path,1200)
                with db:
                    # No deletion/replacement of historical page content.
                    if db.execute('SELECT 1 FROM pages WHERE id=?',(pid,)).fetchone():
                        raise ValueError('Existing selected-reference page cannot be overwritten')
                    db.execute('INSERT INTO pages VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                               (pid,document_id,p,config,'complete',w,h,rotation,indexer.dumps(geometry),indexer.dumps(signals),str(path),run_id))
                    for table,values in [('spans',spans),('vectors',vectors),('images',images)]:
                        db.executemany(f'INSERT INTO {table} VALUES(?,?,?)',[(pid,n,indexer.dumps(v)) for n,v in enumerate(values)])
                    db.execute("UPDATE attempts SET status='complete' WHERE run_id=? AND page_index=?",(run_id,p))
                added.append(p)
            except BaseException as exc:
                with db:
                    db.execute("UPDATE attempts SET status='error',error=? WHERE run_id=? AND page_index=?",(type(exc).__name__,run_id,p))
                raise
    with db:
        db.execute('UPDATE runs SET finished=CURRENT_TIMESTAMP,summary=? WHERE id=?',(indexer.dumps({'completed':len(added),'selected_page_indices':pages,'stage':6}),run_id))
    return added


def retrieve(db,document_id,codes):
    pages=indexed_pages(db,document_id);hits=[];coverage=[]
    normalized = {code:callouts.normalize(code) for code in codes}
    for p,row in pages.items():
        spans=[(ordinal,json.loads(data)) for ordinal,data in db.execute('SELECT ordinal,data FROM spans WHERE page_id=? ORDER BY ordinal',(row['id'],))]
        for ordinal,span in spans:
            for code in codes:
                pattern=r'(?<![A-Za-z0-9])'+re.escape(code)+r'(?![A-Za-z0-9])'
                exact = bool(re.search(pattern,span['text']))
                # Whole-span normalization is a collision/retrieval clue only.
                near = not exact and callouts.normalize(span['text'])==normalized[code]
                if exact or near:
                    item={'document_id':document_id,'page_id':row['id'],'page_index':p,'code_raw':code,
                          'source_text':span['text'],'span_ordinal':ordinal,'box':span['bbox'],
                          'match':'EXACT_TEXT_CANDIDATE' if exact else 'NORMALIZED_COLLISION',
                          'status':'UNVERIFIED','method':'indexed_native_text'}
                    item['id']='hit-'+grouping.digest(item)[:20];hits.append(item)
        signals=json.loads(row['signals'])
        coverage.append({'page_id':row['id'],'page_index':p,'native_search':'SEARCHED','native_span_count':len(spans),
                         'visual_search':'UNSEARCHED','visual_required':True,'native_text_risk':not spans or signals.get('needs_visual_text_analysis',False),
                         'absence_established':False})
    return hits,coverage


def scope_signature(identity,db=None):
    values=identity.get('scope',{})
    if identity.get('scope_status')!='SOURCE_VERIFIED' or not all(values.get(k) for k in ('manufacturer','system','edition')):
        return None
    evidence=copy.deepcopy(identity.get('scope_evidence',[]));components.validate_scope(evidence)
    for item in evidence:
        for e in item.get('evidence',[]):
            try:grouping.checked_box(e['box'])
            except (KeyError,ValueError,TypeError):return None
            if e.get('document_id')!=identity.get('document_id') or e.get('page_id')!=identity.get('page_id'):return None
            if db is not None:
                row=db.execute('SELECT geometry,document_id,page_index,status FROM pages WHERE id=?',(e['page_id'],)).fetchone()
                if row is None or row['document_id']!=identity['document_id'] or row['page_index']!=e['page_index'] or row['status']!='complete':return None
                if not components.contained(json.loads(row['geometry'])['source_rect'],e['box']):return None
                if e.get('method')=='native_text':
                    spans=[db.execute('SELECT data FROM spans WHERE page_id=? AND ordinal=?',(e['page_id'],ordinal)).fetchone() for ordinal in e.get('native_span_ids',[])]
                    if not spans or any(s is None for s in spans) or not any(e['quote'] in json.loads(s[0])['text'] for s in spans):return None
    if not all(any(s['field']==key and s.get('accepted',False) and s['value_raw']==values[key] and s['evidence'] for s in evidence)
               for key in ('manufacturer','system','edition')):
        return None
    return tuple(values[k] for k in ('manufacturer','system','edition'))


def match_sources(targets,target_identities,references,reference_identities,observations,db=None):
    identities={i['id']:i for i in target_identities+reference_identities};matches=[];assignments=[]
    for target in targets:
        if target.get('identifier_kind') == 'component_name':
            continue
        for reference in references:
            if reference.get('identifier_kind') == 'component_name':
                continue
            if target['id']==reference['id'] or target['document_id']!=reference['document_id'] or target['page_index']==reference['page_index']:
                continue
            exact=target['code_raw'] is not None and target['code_raw']==reference['code_raw']
            near=not exact and target['code_raw'] and reference['code_raw'] and callouts.normalize(target['code_raw'])==callouts.normalize(reference['code_raw'])
            if not exact and not near:
                continue
            a=scope_signature(identities.get(target['identity_id'],{}),db);b=scope_signature(identities.get(reference['identity_id'],{}),db)
            status='SUPPORTED' if exact and a and a==b else 'REVIEW'
            reason=('Exact code and verified manufacturer/system/edition scope agree' if status=='SUPPORTED' else
                    'Normalized spelling is not exact identity' if near else
                    'Manufacturer/system/edition scope is missing' if not a or not b else 'Scope collision; observations remain separate')
            match={'source_component_id':reference['id'],'target_component_id':target['id'],'code_raw':target['code_raw'],
                   'status':status,'reason':reason,'source_identity_id':reference['identity_id'],'target_identity_id':target['identity_id']}
            match['id']='match-'+grouping.digest(match)[:20];matches.append(match)
            if status=='SUPPORTED':
                for o in observations:
                    if (o['candidate_id']==reference['id'] and o['accepted'] and o['field']!='code'
                            and o.get('relation',{}).get('kind')!='page_paragraph'):
                        assignments.append({'id':'assignment-'+grouping.digest([match['id'],o['id']])[:20],
                                            'match_id':match['id'],'candidate_id':target['id'],'observation_id':o['id'],
                                            'source_candidate_id':reference['id'],'status':'SUPPORTED',
                                            'conditions_preserved':True,'membership_effect':'NONE'})
    return matches,assignments


def apply_footnotes(analysis):
    """Qualify only an explicitly linked marker/note; unresolved scope stays rejected."""
    by_id={c.local_id:c for c in analysis.components};valid=set();links=[]
    for link in analysis.footnotes:
        if link.component_local_id not in by_id or link.field_index>=len(by_id[link.component_local_id].fields):
            raise ValueError('Unknown footnote field reference')
        component=by_id[link.component_local_id];field=component.fields[link.field_index]
        quotes=[field.value_raw]+[e.quote for e in field.evidence]
        marker=(len(link.marker)<=8 and any(link.marker in q for q in quotes) and link.marker in link.marker_evidence.quote)
        note=link.note_evidence.quote.lstrip().startswith(link.marker)
        explicit_scope=any(component.code_raw in e.quote for e in link.scope_evidence)
        categories={f.category_value for f in component.fields if f.field=='category' and not f.review_reasons}
        general_scope=any('marked' in e.quote.casefold() and link.marker in e.quote and any(cat and cat in e.quote.casefold() for cat in categories) for e in link.scope_evidence)
        supported=bool(marker and note and explicit_scope and not link.uncertainty and
                       not any(e.review_reasons for e in [link.marker_evidence,link.note_evidence]+link.scope_evidence))
        record=link.model_dump(mode='json');record['supported']=supported
        record['reason']='Explicit marker, note and scope demonstrated' if supported else 'Footnote marker/note scope unresolved'
        links.append(record)
        if supported:
            field.conditions.append(link.note_evidence);valid.add((link.component_local_id,link.field_index))
        else:
            field.review_reasons.append(record['reason'])
    for component in analysis.components:
        for n,field in enumerate(component.fields):
            if any(re.search(r'[*†‡]',q) for q in [field.value_raw]+[e.quote for e in field.evidence]) and (component.local_id,n) not in valid:
                field.review_reasons.append('Explicit footnote marker-to-note scope has not been verified')
    return links


def migrate(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS enrichment_runs(id TEXT PRIMARY KEY,started TEXT DEFAULT CURRENT_TIMESTAMP,
      finished TEXT,settings TEXT,manifest TEXT,summary TEXT);
    CREATE TABLE IF NOT EXISTS enrichment_lookups(id TEXT PRIMARY KEY,run_id TEXT REFERENCES enrichment_runs(id),
      cache_key TEXT,page_id TEXT,state TEXT,result_path TEXT,artifacts TEXT,error TEXT);
    CREATE INDEX IF NOT EXISTS enrichment_cache ON enrichment_lookups(cache_key,state);
    ''')


def run(inputs,output,*,index_path,plan=None,config=None,model=None,client=None,force=False,candidate_ids=()):
    config=config or Config();config.validate()
    model=model or ModelConfig(max_tokens=4096,retries=0,max_requests=10,timeout=180)
    QwenClient(model);client=client or QwenClient(model);before=client.requests
    bundles=[load_bundle(p) for p in inputs]
    if not bundles:
        raise ValueError('Explicit saved Stage 5 inputs are required')
    documents={b['reference']['document_id'] for b in bundles}
    if len(documents)!=1:
        raise ValueError('Cross-document enrichment is not automatic; select one document')
    if len({b['reference']['page_id'] for b in bundles})!=len(bundles):
        raise ValueError('Select only one saved Stage 5 attempt per source page')
    document_id=next(iter(documents));source=Path(bundles[0]['reference']['source_path'])
    if plan is not None and not isinstance(plan,LookupPlan):
        plan=LookupPlan.model_validate_json(Path(plan).read_text())
    if plan and plan.document_id!=document_id:
        raise ValueError('Lookup plan refers to a different source document')
    targets=[copy.deepcopy(c) for b in bundles for c in b['stage5']['candidates'] if not candidate_ids or c['id'] in candidate_ids]
    if candidate_ids and set(candidate_ids)!={c['id'] for c in targets}:
        raise ValueError('Unknown selected component candidate')
    codes=sorted({c['code_raw'] for c in targets if c['code_raw'] is not None and c.get('identifier_kind') != 'component_name'})
    identities=[i for b in bundles for i in b['stage5']['identities']]
    run_id=uuid.uuid4().hex;folder=Path(output).resolve()/'enrichments'/run_id;folder.mkdir(parents=True)
    settings={'logic_sha256':code_fingerprint(),'version':VERSION,'prompt_version':PROMPT_VERSION,'schema_version':SCHEMA_VERSION,
              'prompt_hash':grouping.digest(PROMPT),'schema_hash':grouping.digest(Analysis.model_json_schema()),
              'configuration':asdict(config),'model':model.public(),'identity_scope':model.identity or run_id,
              'pymupdf':fitz.VersionBind,'pillow':Image.__version__}
    sources=[b['reference'] for b in bundles];failures=[];results=[];hits=[];coverage=[];added=[];count=None;signature=None
    summary={'run_id':run_id,'manifest':str(folder/'manifest.json'),'lookups_completed':0,'lookups_reused':0,'errors':0,'model_requests':0}
    db=None
    try:
        if not Path(index_path).is_file():
            failures.append({'code':'INDEX_UNAVAILABLE','reason':'Existing source index is unavailable; retrieval NOT RUN'})
        elif not source.is_file():
            failures.append({'code':'SOURCE_UNAVAILABLE','reason':'Original source unavailable; retrieval NOT RUN'})
        elif indexer.fingerprint(source)!=document_id:
            failures.append({'code':'SOURCE_IDENTITY_MISMATCH','reason':'Original source identity changed; retrieval NOT RUN'})
        else:
            db=sqlite3.connect(index_path);db.row_factory=sqlite3.Row;db.execute('PRAGMA foreign_keys=ON');migrate(db)
            with db:
                db.execute('INSERT INTO enrichment_runs(id,settings) VALUES(?,?)',(run_id,indexer.dumps(settings)))
            doc=db.execute('SELECT page_count FROM documents WHERE id=?',(document_id,)).fetchone()
            if doc is None:
                raise ValueError('Document identity is absent from explicit index')
            count=doc['page_count']
            if plan:
                added=index_selected(db,source,document_id,plan.index_pages,folder)
            pages=indexed_pages(db,document_id);hits,coverage=retrieve(db,document_id,codes)
            signature=index_signature(db,[document_id])
            windows=list(plan.windows) if plan else []
            if config.verify_hits:
                for hit in [h for h in hits if h['match']=='EXACT_TEXT_CANDIDATE'][:config.max_hit_windows]:
                    row=pages[hit['page_index']];bounds=json.loads(row['geometry'])['source_rect']
                    box=expand(hit['box'],max(bounds[2]-bounds[0],bounds[3]-bounds[1])*config.hit_margin_fraction,bounds)
                    windows.append(Window(page_index=hit['page_index'],box=tuple(box),reason='Verify exact native-text candidate '+hit['id']))
            unique={grouping.digest(w.model_dump(mode='json')):w for w in windows}
            for window in unique.values():
                lookup=uuid.uuid4().hex;row=pages.get(window.page_index)
                if row is None:
                    failures.append({'code':'UNINDEXED_WINDOW','page_index':window.page_index,'reason':'Selected lookup page is not indexed'});continue
                geometry,spans,page_signature=detection.load_page(db,row);bounds=geometry['source_rect']
                if not components.contained(bounds,window.box):
                    failures.append({'code':'INVALID_WINDOW','page_index':window.page_index,'reason':'Lookup box exceeds canonical page bounds'});continue
                window_id='reference-view-'+grouping.digest([document_id,row['id'],window.box])[:20]
                view={'id':window_id,'document_id':document_id,'page_id':row['id'],'page_index':window.page_index,
                      'box':list(window.box),'kind':'reference_only','source_review_status':'PROPOSED',
                      'source_review_reasons':[],'primary_detail_ids':[],'alternative_detail_ids':[],
                      'selection_reason':window.reason,'membership_effect':'NONE'}
                cache_settings={**settings,'model':{k:v for k,v in settings['model'].items() if k!='max_requests'}}
                key=grouping.digest([cache_settings,sources,page_signature,codes,window.model_dump(mode='json')])
                prior=db.execute("SELECT * FROM enrichment_lookups WHERE cache_key=? AND state='complete' ORDER BY rowid DESC LIMIT 1",(key,)).fetchone()
                with db:
                    db.execute('INSERT INTO enrichment_lookups(id,run_id,cache_key,page_id,state) VALUES(?,?,?,?,?)',(lookup,run_id,key,row['id'],'processing'))
                try:
                    if not force and prior and callouts.reusable(prior):
                        result=json.loads(Path(prior['result_path']).read_text());summary['lookups_reused']+=1
                        with db:
                            db.execute("UPDATE enrichment_lookups SET state='reused',result_path=?,artifacts=? WHERE id=?",(prior['result_path'],prior['artifacts'],lookup))
                    else:
                        if client.requests-before>=model.max_requests:
                            raise ModelError('Model request budget exhausted; analysis unfinished')
                        image,mapping=detection.render(source,window.page_index,geometry,list(window.box),config.image_max)
                        native=[{'id':ordinal,'text':span['text'],'box':callouts.normalized_box(span['bbox'],mapping)}
                                for ordinal,span in spans if intersection(span['bbox'],window.box)>0]
                        context={'target_codes':codes,'reference_view_id':window_id,'native_spans':native,'occurrences':[],
                                 'scope_rule':'Never assume filename, unknown scope, or code repetition establishes equivalence'}
                        raw=client.request_json(image,PROMPT,json.dumps(context,ensure_ascii=False),response_schema=Analysis.model_json_schema())
                        analysis=components.parse_response(raw,schema=Analysis)
                        if any(c.code_raw not in codes or c.occurrence_ids for c in analysis.components):
                            raise ValueError('Reference response contains an unselected code or invented occurrence')
                        footnotes=apply_footnotes(analysis)
                        # Reuse field validation with an empty occurrence set; no Stage 4
                        # success or new callout is created for a reference-only crop.
                        empty_context={'state':'COMPLETE','complete':True,'occurrences':[]}
                        candidates,observations,scopes=components.process_analysis(analysis,view,[],empty_context,mapping,bounds,spans,allow_reference_labels=True)
                        for candidate in candidates:
                            candidate['inherited_review_reasons'].append({'stage':6,'reason':'Reference-only crop; no Detail or drawing membership established'})
                        candidates,observations,source_identities=components.summarize(candidates,observations,scopes)
                        source_components={c.local_id:c for c in analysis.components}
                        for link in footnotes:
                            raw_component=source_components[link['component_local_id']]
                            raw_field=raw_component.fields[link['field_index']]
                            code_box=detection.source_box(raw_component.relation.code_evidence.box,mapping,bounds)
                            link['source_component_id']='component-'+grouping.digest([components.VERSION,row['id'],raw_component.code_raw,code_box])[:20]
                            link['observation_ids']=[o['id'] for o in observations if o['candidate_id']==link['source_component_id'] and
                                                     o['field']==raw_field.field and o['value_raw']==raw_field.value_raw and o['name_raw']==raw_field.name_raw]
                            for field in ('marker_evidence','note_evidence'):
                                evidence=components.Evidence.model_validate(link[field],strict=False)
                                link[field]=components.evidence_record(evidence,view,mapping,bounds,spans)
                            link['scope_evidence']=[components.evidence_record(components.Evidence.model_validate(e,strict=False),view,mapping,bounds,spans) for e in link['scope_evidence']]
                            link['id']='footnote-'+grouping.digest([window_id,link])[:20]
                        source_image=folder/(lookup+'-source.png');image.save(source_image)
                        annotated=folder/(lookup+'-evidence.png');components.annotate(image,mapping,observations).save(annotated)
                        result={'view':view,'mapping':mapping,'coverage':analysis.coverage,'coverage_evidence':analysis.evidence,
                                'uncertainty':analysis.uncertainty,'candidates':candidates,'observations':observations,
                                'identities':source_identities,'scope_observations':scopes,'footnotes':footnotes,
                                'source_image':str(source_image),'annotated_image':str(annotated),
                                'actual_response_models':sorted(client.response_models)}
                        path=folder/(lookup+'-result.json');path.write_text(indexer.dumps(result))
                        artifacts={str(p):grouping.file_digest(p) for p in [path,source_image,annotated]}
                        with db:
                            db.execute("UPDATE enrichment_lookups SET state='complete',result_path=?,artifacts=? WHERE id=?",(str(path),indexer.dumps(artifacts),lookup))
                        summary['lookups_completed']+=1
                    results.append(result)
                except (ModelError,ValueError,OSError) as exc:
                    reason=callouts.failure_reason(exc)
                    issues=[{'field':list(e['loc']),'type':e['type']} for e in exc.errors(include_input=False,include_context=False,include_url=False)] if hasattr(exc,'errors') else []
                    failures.append({'code':'LOOKUP_FAILED','page_index':window.page_index,'view_id':window_id,'reason':reason,'validation_issues':issues})
                    with db:
                        db.execute("UPDATE enrichment_lookups SET state='failed',error=? WHERE id=?",(reason,lookup))
            if index_signature(db,[document_id])!=signature or indexer.fingerprint(source)!=document_id:
                raise ValueError('Indexed/source content changed during enrichment')
        references=[c for result in results for c in result['candidates']]
        source_observations=[o for result in results for o in result['observations']]
        source_identities=[i for result in results for i in result['identities']]
        # Explicit saved Stage 5 references are also usable; only cross-page,
        # same-document, verified exact-scope matches can transfer properties.
        all_saved=[c for b in bundles for c in b['stage5']['candidates']]
        saved_obs=[o for b in bundles for o in b['stage5']['observations']]
        matches,assignments=match_sources(targets,identities,references+all_saved,source_identities+identities,source_observations+saved_obs,db=db)
        if db is None:
            assignments=[]
            for match in matches:match.update(status='REVIEW',reason='Source/index unavailable; scope was not reverified')
        for c in coverage:
            inspected=[r for r in results if r['view']['page_id']==c['page_id']]
            if inspected:
                c['visual_search']='WINDOWS_INSPECTED'
                c['reference_view_ids']=[r['view']['id'] for r in inspected]
        indexed={c['page_index'] for c in coverage}
        visual_unsearched=[c['page_index'] for c in coverage]
        # Native text is candidate retrieval, not proof that outlined/image labels
        # were fully searched. Reuse complete original-source Stage 5 page crops.
        if db:
            for bundle in bundles:
                saved=bundle['stage5']
                if saved['state']!='COMPLETE' or not saved['complete']:continue
                completed={c['view_id'] for c in saved.get('coverage',[]) if c['coverage']=='complete' and not c['uncertainty']}
                for image in saved.get('annotated_views',[]):
                    if image['view_id'] not in completed:continue
                    page=db.execute('SELECT geometry FROM pages WHERE id=?',(bundle['reference']['page_id'],)).fetchone()
                    if page and components.contained(image['geometry']['source_context_box'],json.loads(page['geometry'])['source_rect']):
                        p=bundle['reference']['page_index']
                        if p in visual_unsearched:visual_unsearched.remove(p)
                        for c in coverage:
                            if c['page_index']==p:c['visual_search']='SAVED_STAGE5_PAGE_CROP_INSPECTED'

        # A window never establishes full-page absence/coverage unless it truly
        # covers the page and the model completed it without uncertainty.
        for r in results:
            if db:
                row=db.execute('SELECT geometry FROM pages WHERE id=?',(r['view']['page_id'],)).fetchone()
                bounds=json.loads(row['geometry'])['source_rect']
                if components.contained(r['view']['box'],bounds) and r['coverage']=='complete' and not r['uncertainty']:
                    if r['view']['page_index'] in visual_unsearched:visual_unsearched.remove(r['view']['page_index'])
        unindexed=sorted(set(range(count))-indexed) if count is not None else None
        coverage_complete=count is not None and not unindexed and not visual_unsearched
        execution='NOT_RUN' if db is None else 'PARTIAL' if failures or any(r['coverage']!='complete' or r['uncertainty'] for r in results) else 'COMPLETE'
        result={'stage':6,'enrichment_version':VERSION,'run_id':run_id,'execution_status':execution,
                'source_coverage_status':'COMPLETE' if coverage_complete else 'PARTIAL' if count is not None else 'UNAVAILABLE',
                'acceptance_status':'REVIEW','sources':sources,'document_id':document_id,'source_path':str(source),
                'settings':settings,'index_signature':signature,'selected_candidate_ids':[c['id'] for c in targets],
                'selection_plan':plan.model_dump(mode='json') if plan else None,'additional_pages_indexed':added,'selected_reference_pages_in_index':[p for p,row in indexed_pages(db,document_id).items() if json.loads(row['config']).get('selected_reference_pages')] if db else [],
                'coverage':{'document_page_count':count,'indexed_pages_searched':sorted(indexed),'unindexed_pages':unindexed,'native_index_status':'COMPLETE' if count is not None and not unindexed else 'PARTIAL',
                            'visual_unsearched_pages':visual_unsearched,'pages':coverage},
                'retrieval_hits':hits,'reference_views':[r['view'] for r in results],
                'source_candidates':references,'source_identities':source_identities,'source_observations':source_observations,
                'scope_observations':[s for r in results for s in r['scope_observations']],
                'matches':matches,'assignments':assignments,'footnotes':[f for r in results for f in r['footnotes']],
                'lookups':[{k:r[k] for k in ('view','mapping','coverage','coverage_evidence','uncertainty','source_image','annotated_image')} for r in results],
                'failures':failures,'model_requests':client.requests-before,
                'actual_response_models':sorted(set(client.response_models)|{a for r in results for a in r['actual_response_models']})}
        path=folder/'manifest.json';path.write_text(indexer.dumps(result))
        lines=['# Stage 6 retrieval/enrichment review','',f'Execution: {execution}; source coverage: {result["source_coverage_status"]}; acceptance: REVIEW.',
               f'Indexed pages searched (one-based): {[p+1 for p in sorted(indexed)]}.',
               f'Additional pages indexed (one-based): {[p+1 for p in added]}.',
               f'Native hits: {len(hits)}; verified assignments: {len(assignments)}.',
               'Missing text matches do not establish code absence on scanned/outline pages.',
               'Reference properties do not change physical membership in a Detail.','']
        lines += ['- '+f['code']+': '+f['reason'] for f in failures]
        lines += ['- '+m['target_component_id']+' / '+m['source_component_id']+': '+m['reason'] for m in matches]
        lines += ['', 'Unindexed pages and partially/unsearched visual coverage remain explicit in manifest.json.',
                  'Deferred: detection boundaries, leader origins, unknown scope, broad accuracy evaluation and profiling.']
        (folder/'review.md').write_text('\n'.join(lines)+'\n')
        summary.update(errors=len(failures),model_requests=client.requests-before,execution_status=execution,
                       source_coverage_status=result['source_coverage_status'],native_hits=len(hits),supported_assignments=len(assignments))
        if db:
            with db:
                db.execute('UPDATE enrichment_runs SET finished=CURRENT_TIMESTAMP,manifest=?,summary=? WHERE id=?',(str(path),indexer.dumps(summary),run_id))
        return summary
    finally:
        if db:db.close()


def main(argv=None):
    parser=argparse.ArgumentParser(prog='as-ocr enrich',description=__doc__)
    parser.add_argument('stage5',nargs='+',type=Path)
    parser.add_argument('--index',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--lookup-plan',type=Path,help='Explicit zero-based page indices and canonical source boxes; at most ten additional pages')
    parser.add_argument('--candidate',action='append',default=[])
    parser.add_argument('--verify-hits',action='store_true');parser.add_argument('--max-hit-windows',type=int,default=8)
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--qwen-endpoint',default=os.getenv('AS_OCR_QWEN_ENDPOINT'))
    parser.add_argument('--qwen-model',default=os.getenv('AS_OCR_QWEN_MODEL'));parser.add_argument('--qwen-identity',default=os.getenv('AS_OCR_QWEN_IDENTITY'))
    parser.add_argument('--max-tokens',type=int,default=4096);parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--retries',type=int,default=0);parser.add_argument('--max-model-requests',type=indexer.positive,default=10)
    args=parser.parse_args(argv)
    try:
        model=ModelConfig(args.qwen_endpoint,args.qwen_model,args.qwen_identity,args.timeout,args.max_tokens,args.retries,args.max_model_requests,os.getenv('AS_OCR_QWEN_API_KEY'))
        result=run(args.stage5,args.output,index_path=args.index,plan=args.lookup_plan,
                   config=Config(args.verify_hits,args.max_hit_windows),model=model,force=args.force,candidate_ids=args.candidate)
    except KeyboardInterrupt:parser.exit(130,'Interrupted; previous enrichment work preserved.\n')
    except Exception as exc:parser.exit(2,'Stage 6 failed: '+type(exc).__name__+'; previous work preserved.\n')
    print(json.dumps(result));return 1 if result['errors'] or result['execution_status']!='COMPLETE' else 0

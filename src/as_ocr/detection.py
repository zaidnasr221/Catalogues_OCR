"""Stage 2: Qwen regions, source-space validation and append-only exports."""
from __future__ import annotations
from contextlib import nullcontext

from .provenance import code_fingerprint

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

from PIL import Image, ImageDraw, ImageOps
import pymupdf as fitz

from . import indexer, coverage, crop_evidence
from .geometry import area, expand, intersection, union
from .qwen import ModelConfig, ModelError, QwenClient, capture_diagnostics, request_metadata
from .regions import Analysis, PROMPT, PROMPT_VERSION, SCHEMA_VERSION, parse_response

VERSION = '16'


@dataclass(frozen=True)
class Config:
    analysis_max: int = 1536
    crop_max: int = 2400
    overview_max: int = 1600
    context_margin: float = .01
    max_followups: int = 2
    followup_max: int = 2048
    verify_coverage: bool = True

    def validate(self):
        if not (128 <= self.analysis_max <= 2048 and 128 <= self.followup_max <= 2048 and
                128 <= self.crop_max <= 4096 and 128 <= self.overview_max <= 4096 and
                0 <= self.context_margin <= .2 and 0 <= self.max_followups <= 4):
            raise ValueError('Invalid image, context or follow-up limits')


def digest(value):
    return hashlib.sha256(indexer.dumps(value).encode()).hexdigest()


def migrate(db):
    # Historical tables/records remain valid; richer state is held in JSON.
    db.executescript('''
    CREATE TABLE IF NOT EXISTS detection_migrations(version INTEGER PRIMARY KEY);
    INSERT OR IGNORE INTO detection_migrations VALUES(1);
    CREATE TABLE IF NOT EXISTS detection_runs(
      id TEXT PRIMARY KEY, started TEXT DEFAULT CURRENT_TIMESTAMP, finished TEXT,
      source TEXT, config TEXT, summary TEXT);
    CREATE TABLE IF NOT EXISTS detection_pages(
      id TEXT PRIMARY KEY, run_id TEXT REFERENCES detection_runs(id), page_id TEXT,
      cache_key TEXT, state TEXT, model_state TEXT, manifest TEXT, error TEXT);
    CREATE INDEX IF NOT EXISTS detection_cache ON detection_pages(page_id,cache_key,state);
    CREATE TABLE IF NOT EXISTS detection_candidates(
      attempt_id TEXT REFERENCES detection_pages(id), id TEXT, page_id TEXT,
      box TEXT, kind TEXT, view_type TEXT, method TEXT, evidence TEXT, status TEXT,
      review_reasons TEXT, crop TEXT, image_geometry TEXT,
      PRIMARY KEY(attempt_id,id));
    ''')


def render(source, page_index, geometry, box, maximum):
    """Render ORIGINAL source, returning source→export affine and its exact inverse."""
    rotation = fitz.Matrix(*geometry['source_to_display'])
    display_box = fitz.Rect(box)*rotation
    if source.suffix.lower() == '.pdf':
        with fitz.open(source) as document:
            page = document[page_index]
            scale = min(maximum/max(display_box.width,display_box.height),4)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=display_box,alpha=False)
            image = Image.frombytes('RGB',(pixmap.width,pixmap.height),pixmap.samples)
            matrix = rotation * fitz.Matrix(scale,scale) * fitz.Matrix(1,0,0,1,-pixmap.x,-pixmap.y)
    else:
        with Image.open(source) as original:
            original.seek(page_index)
            displayed = ImageOps.exif_transpose(original)
            left,top = max(0,math.floor(display_box.x0)),max(0,math.floor(display_box.y0))
            right,bottom = min(displayed.width,math.ceil(display_box.x1)),min(displayed.height,math.ceil(display_box.y1))
            image = displayed.crop((left,top,right,bottom)).convert('RGB')
            width,height = image.size
            image.thumbnail((maximum,maximum))
            matrix = rotation * fitz.Matrix(1,0,0,1,-left,-top) * fitz.Matrix(image.width/width,image.height/height)
    if max(image.size)>maximum:
        width,height=image.size
        image.thumbnail((maximum,maximum))
        matrix *= fitz.Matrix(image.width/width,image.height/height)
    return image, {'source_to_pixels':list(matrix),'pixels_to_source':list(~matrix),
                   'pixel_size':list(image.size),'source_context_box':list(box)}


def load_page(db, row):
    # Exact indexed content is part of reuse identity, even though vectors no
    # longer discover regions. Native spans are used only for ownership auditing.
    records = {}
    for table in ('vectors','spans','images'):
        records[table] = [(ordinal,json.loads(data)) for ordinal,data in db.execute(
            f'SELECT ordinal,data FROM {table} WHERE page_id=? ORDER BY ordinal',(row['id'],))]
    signature=digest({'page':{key:row[key] for key in ('id','config','width','height','rotation','geometry','signals')},
                      'records':records})
    return json.loads(row['geometry']), records['spans'], signature


def source_box(box, mapping, bounds):
    """Normalized displayed-image edges → canonical unrotated source bounds."""
    width,height = mapping['pixel_size']
    pixels = fitz.Rect(box[0]*width,box[1]*height,box[2]*width,box[3]*height)
    result = list(pixels*fitz.Matrix(*mapping['pixels_to_source']))
    # Raster edge rounding may extend a fraction of a source unit outside page.
    result = [max(bounds[0],result[0]),max(bounds[1],result[1]),
              min(bounds[2],result[2]),min(bounds[3],result[3])]
    if area(result)<=0:
        raise ModelError('Region maps to degenerate source bounds')
    return result


def mapped_regions(analysis, mapping, bounds, request_id):
    result=[]
    for region in analysis.regions:
        required=source_box(region.box,mapping,bounds)
        structural=source_box(region.structural_box,mapping,bounds) if region.structural_box else list(required)
        reviews=list(region.uncertainty)
        kind=region.kind
        if kind=='assembly_candidate' and region.assembly_basis in {'single_component','overview_only','metadata_only'}:
            kind={'single_component':'profile_candidate','overview_only':'legend_notes',
                  'metadata_only':'document_metadata'}[region.assembly_basis]
            reviews.append('Assembly classification contradicted by explicit visible-content basis; retained as non-assembly')
        elif kind=='assembly_candidate' and region.assembly_basis=='uncertain':
            reviews.append('Multi-component connection not established; eligibility unresolved')
        elif kind=='assembly_candidate' and region.assembly_basis is None:
            reviews.append('Explicit multi-component connection basis missing; eligibility requires review')
        elif kind=='assembly_candidate' and region.assembly_basis=='exploded_assembly':
            reviews.append('Exploded assembly retained with source assembly evidence; physical membership requires review')
        reference_marks=[]
        for mark in region.reference_marks:
            item=mark.model_dump(mode='json')
            item['box']=source_box(mark.box,mapping,bounds)
            item['leader_points']=[list(fitz.Point(p[0]*mapping['pixel_size'][0],p[1]*mapping['pixel_size'][1])*fitz.Matrix(*mapping['pixels_to_source'])) for p in mark.leader_points]
            item['source']='model_reference_hint'
            reference_marks.append(item)
            if intersection(required,item['box'])<area(item['box'])-1e-5:
                reviews.append('Reference mark retained beyond initial context; source relation still requires verification')
            required=union(required,item['box'])
        context_evidence=[]
        for annotation in region.context_annotations:
            extent=source_box(annotation.box,mapping,bounds)
            target=source_box(annotation.target_box,mapping,bounds)
            supported=(region.structural_box is not None and
                       intersection(structural,target)>=area(target)-1e-5 and
                       intersection(extent,target)>0)
            context_evidence.append({'box':extent,'target_box':target,
                'association':annotation.association,'evidence':annotation.evidence,
                'label_box':source_box(annotation.label_box,mapping,bounds) if annotation.label_box else None,
                'leader_points':[list(fitz.Point(p[0]*mapping['pixel_size'][0],p[1]*mapping['pixel_size'][1])*fitz.Matrix(*mapping['pixels_to_source'])) for p in annotation.leader_points],
                'status':'model_supported' if supported else 'REVIEW'})
            # Keep uncertain context visible too, without assigning ownership.
            required=union(required,extent)
            if not supported:
                reviews.append('Annotation target/leader does not establish ownership; complete context retained for review')
        if region.incomplete:
            reviews.append('Model reports incomplete region')
        if region.kind=='unknown':
            reviews.append('Region eligibility remains unknown')
        if region.box[0]==0 or region.box[1]==0 or region.box[2]==1 or region.box[3]==1:
            reviews.append('Region touches request boundary; required extents may be clipped')
        title=None
        if region.title:
            title_box=source_box(region.title.box,mapping,bounds)
            if intersection(required,title_box)<area(title_box)-1e-5:
                required=union(required,title_box)
                reviews.append('Required bounds expanded to retain model-supported title; scope needs review')
            title={'text':region.title.text,'box':title_box,'evidence':region.title.evidence,
                   'provenance':{'method':'qwen','request_id':request_id,'region_id':region.id},
                   'scope':'candidate_only','verified':False}
        result.append({'model_id':request_id+':'+region.id,'box':required,'initial_box':source_box(region.box,mapping,bounds),
            'structural_box':structural,'kind':kind,'view_type':region.view_type,
            'title':title,'parent_ref':request_id+':'+region.parent_id if region.parent_id else None,
            'evidence':[{'observations':region.evidence,'request_id':request_id,
                         'normalized_box':list(region.box),
                         'assembly_basis':region.assembly_basis,
                         'context_annotations':context_evidence,
                         'structural_bounds_basis':'model_geometry' if region.structural_box else 'required_region_fallback'}],
            'review_reasons':reviews,'annotations':[],'reference_marks':reference_marks})
    return result


def reconcile_duplicates(regions):
    """Remove exact repeated proposals only; overlap is never identity evidence."""
    kept=[]; aliases={}; seen={}
    for region in regions:
        title=region['title']
        title_key=(title['text'],tuple(title['box'])) if title else None
        key=(tuple(region['box']),tuple(region['structural_box']),region['kind'],region['view_type'],
             title_key,region['parent_ref'])
        prior=seen.get(key)
        if prior is None:
            seen[key]=region;kept.append(region)
        else:
            prior['evidence'].extend(region['evidence'])
            prior.setdefault('reference_marks',[]).extend(region.get('reference_marks',[]))
            prior['review_reasons'].extend(region['review_reasons'])
            prior['review_reasons'].append('Exact repeated region proposal consolidated; source evidence retained')
            aliases[region['model_id']]=prior['model_id']
    for region in kept:
        region['parent_ref']=aliases.get(region['parent_ref'],region['parent_ref'])
    return kept


def validate_relationships(regions):
    by_id={r['model_id']:r for r in regions}
    for region in regions:
        parent=by_id.get(region['parent_ref'])
        if region['parent_ref'] and (not parent or parent is region or parent['kind']!='mixed_content' or
                intersection(parent['box'],region['box'])<area(region['box'])-1e-5):
            region['review_reasons'].append('Invalid/unsupported mixed-region parent reference; containment unresolved')
            region['parent_ref']=None
    for region in regions:
        chain=set()
        current=region
        while current:
            if current['model_id'] in chain:
                region['parent_ref']=None
                region['review_reasons'].append('Cyclic mixed-region provenance rejected')
                break
            chain.add(current['model_id'])
            current=by_id.get(current['parent_ref'])
    for region in regions:
        if region['kind']=='mixed_content':
            children=[r for r in regions if r['parent_ref']==region['model_id']]
            region['review_reasons'].append('Mixed-region containment is provisional; internal-view coverage needs review')
            if not children:
                region['review_reasons'].append('Unresolved internal-view coverage: no separable child views proposed')


def annotate(regions, spans):
    """Audit immutable model bounds; never grow boxes or assign by proximity."""
    audit=[]
    by_id={r['model_id']:r for r in regions}
    for ordinal,span in sorted(spans,key=lambda item:(tuple(item[1]['bbox']),item[1]['text'],item[0])):
        box=list(span['bbox'])
        if area(box)<=0:
            continue
        hits=[r for r in regions if intersection(r['box'],box)>0]
        ancestors=set()
        for hit in hits:
            parent=by_id.get(hit['parent_ref'])
            while parent:
                ancestors.add(parent['model_id'])
                parent=by_id.get(parent['parent_ref'])
        hits=[r for r in hits if r['model_id'] not in ancestors]
        record={'table':'spans','ordinal':ordinal,'text':span['text'],'box':box,
                'status':'unassigned','owner_model_id':None,
                'competing_model_ids':[r['model_id'] for r in hits]}
        if len(hits)==1 and intersection(hits[0]['box'],box)>=area(box)-1e-5:
            record.update(status='attached',owner_model_id=hits[0]['model_id'],
                          reason='Native span fully contained in one independent model region')
            hits[0]['annotations'].append(record)
        else:
            reason=('Competing model regions; annotation ownership unresolved' if len(hits)>1 else
                    'Indexed annotation crosses required bounds; extent/ownership unresolved' if hits else
                    'Unassigned annotation retained in full-page context')
            record.update(status='ambiguous' if len(hits)>1 else 'unassigned',reason=reason)
            for hit in hits:
                hit['review_reasons'].append(reason)
        audit.append(record)
    return audit


def request_analysis(client, image, prompt, request, context_text='', schema=Analysis, response_schema=None):
    before=client.requests
    started=time.monotonic()
    transport=getattr(client,'_client',client)
    diagnostic_start=len(getattr(transport,'request_diagnostics',[]))
    try:
        request['response_format']='json_schema'
        from . import async_inference
        directory=getattr(client,'directory',None) or async_inference.diagnostics_directory()
        scope=capture_diagnostics(client,directory,stage=2,subject_id=request['id']) if async_inference.enabled() else nullcontext()
        with request_metadata(purpose=request.get('purpose',request['id'])),scope:
            raw=client.request_json(image,prompt,context_text,response_schema=response_schema if response_schema is not None else schema.model_json_schema())
            try:
                result,envelope=parse_response(raw,model=schema)
                request['response_envelope']=envelope
            except ValueError:
                # Pydantic error strings may contain response values; never persist them.
                raise ModelError('Invalid region JSON/schema; page analysis failed') from None
        request['state']='succeeded'
        return result
    except ModelError:
        request['state']='failed'
        raise
    finally:
        request['http_attempts']=client.requests-before
        request['duration_seconds']=round(time.monotonic()-started,3)
        request['diagnostic_request_ids']=[item['request_id'] for item in
            getattr(transport,'request_diagnostics',[])[diagnostic_start:]]


def verify_page_coverage(source,row,geometry,config,client,regions,page,requests,full_render=None):
    """Reconcile against the original page; never infer completeness from crops."""
    bounds=geometry['source_rect']
    image,mapping=full_render if full_render is not None else render(source,row['page_index'],geometry,bounds,config.analysis_max)
    def normalized(box):
        pixels=list(fitz.Rect(box)*fitz.Matrix(*mapping['source_to_pixels']))
        return [max(0,min(1,v/mapping['pixel_size'][i%2])) for i,v in enumerate(pixels)]
    proposed=json.loads(json.dumps(regions))
    context=[{'candidate_id':r['model_id'],'kind':r['kind'],'view_type':r['view_type'],
              'box':normalized(r['box']),'structural_box':normalized(r['structural_box'])}
             for r in regions]
    request={'id':'coverage','purpose':'full_page_coverage_verification','image_geometry':mapping}
    requests.append(request)
    audit=request_analysis(client,image,coverage.PROMPT,request,json.dumps(context),schema=coverage.Audit,
                           response_schema=coverage.response_schema([r['model_id'] for r in regions]))
    ids=[a.candidate_id for a in audit.assessments]
    if len(ids)!=len(set(ids)) or set(ids)!={r['model_id'] for r in regions}:
        raise ModelError('Coverage audit did not account for every candidate exactly once')
    generated=[r for a in audit.assessments for r in a.replacements]+list(audit.additions)
    if len({r.id for r in generated})!=len(generated):
        raise ModelError('Coverage audit contains duplicate replacement/addition IDs')
    decisions={a.candidate_id:a for a in audit.assessments}
    revised=[];aliases={};superseded=[]
    def map_audited(items):
        mapped=mapped_regions(Analysis(decision='REVIEW',page_kind='engineering',
            evidence=['Coverage reconciliation'],coverage=audit.coverage,uncertainty=[],
            regions=items,followups=[]),mapping,bounds,'coverage')
        for original,item in zip(items,mapped):
            if original.kind=='assembly_candidate' and original.assembly_basis is None:
                item['review_reasons'].append('Audit supplied no physical assembly basis; unresolved candidate retained for review')
                page['review_reasons'].append('Coverage audit contains an unsupported assembly classification')
            if original.kind=='assembly_candidate' and original.assembly_basis=='uncertain':
                item['kind']='assembly_candidate'
                item['review_reasons'].append('Possible assembly from coverage audit retained for REVIEW')
        return mapped
    for region in regions:
        assessment=decisions[region['model_id']]
        if assessment.action=='replace':
            replacement=map_audited(assessment.replacements)
            revised.extend(replacement);superseded.append(region['model_id'])
            aliases[region['model_id']]=replacement[0]['model_id'] if len(replacement)==1 else None
        else:
            if assessment.action=='review':
                region['review_reasons'].append('Coverage audit unresolved: '+assessment.evidence)
                if assessment.possible_assembly:
                    region['kind']='assembly_candidate'
                    region['review_reasons'].append('Possible assembly retained for REVIEW; eligibility not verified')
            revised.append(region)
    revised.extend(map_audited(audit.additions))
    for region in revised:
        if region['parent_ref'] in aliases:
            region['parent_ref']=aliases[region['parent_ref']]
            region['review_reasons'].append('Parent candidate revised during coverage audit; containment requires review')
    effective_coverage = ('incomplete' if any(r.kind=='assembly_candidate' and
                          r.assembly_basis is None for r in generated) else audit.coverage)
    page['coverage_validation']={'version':coverage.VERSION,'state':'complete',
        'execution_status':'COMPLETE','coverage_status':effective_coverage.upper(),
        'proposed_candidates':proposed,'superseded_model_ids':superseded,
        'audit':audit.model_dump(mode='json'),'request_id':'coverage'}
    page['coverage']=effective_coverage
    page['review_reasons'].extend(audit.uncertainty)
    if audit.coverage=='incomplete' or any(a.action=='review' for a in audit.assessments):
        page['review_reasons'].append('Full-page coverage verification retains unresolved regions')
    if any(r['kind']=='assembly_candidate' for r in revised):
        page['decision']='CANDIDATES'
    return revised


def confirmed_no_detail(page, regions, failures=()):
    """Independent whole-page eligibility proof, separate from non-assembly review."""
    check=page.get('coverage_validation',{});audit=check.get('audit',{})
    assessments=audit.get('assessments',[])
    return bool(not failures and page.get('initial_decision',page.get('decision'))=='NO_DETAIL'
        and page.get('evidence') and page.get('coverage')=='complete' and regions
        and all(r['kind'] in {'table','document_metadata','legend_notes','profile_candidate'} for r in regions)
        and check.get('state')=='complete' and audit.get('coverage')=='complete'
        and audit.get('evidence') and not audit.get('uncertainty') and not audit.get('additions')
        and len(assessments)==len(regions)
        and all(a['action']=='keep' and not a['possible_assembly'] for a in assessments))


def analyze_page(source, row, geometry, config, client):
    bounds=geometry['source_rect']
    regions,requests,failures=[],[],[]
    page={'decision':'FAILED','page_kind':'unknown','evidence':[],'coverage':'incomplete',
          'review_reasons':[]}
    prompt=PROMPT+'\nJSON schema:\n'+json.dumps(Analysis.model_json_schema(),sort_keys=True)
    try:
        image,mapping=render(source,row['page_index'],geometry,bounds,config.analysis_max)
        full_render=(image,mapping)
        request={'id':'full','purpose':'full_page','image_geometry':mapping}
        requests.append(request)
        analysis=request_analysis(client,image,prompt,request)
        regions=mapped_regions(analysis,mapping,bounds,'full')
        eligibility_uncertain=any(r.kind in {'unknown','mixed_content'} or r.uncertainty or r.incomplete
                                  for r in analysis.regions)
        page.update(decision=analysis.decision,page_kind=analysis.page_kind,evidence=analysis.evidence,
                    coverage=analysis.coverage,review_reasons=list(analysis.uncertainty))
        page['initial_decision']=analysis.decision
        if analysis.coverage=='incomplete':
            page['review_reasons'].append('Full-page coverage incomplete; crop checks cannot prove global coverage')
        if analysis.followups:
            page['review_reasons'].append('Targeted source crops requested; global coverage remains provisional')
        for number,window in enumerate(analysis.followups):
            if number>=config.max_followups:
                page['review_reasons'].append('Follow-up window limit reached; coverage unresolved')
                break
            context=source_box(window.box,mapping,bounds)
            image,crop_mapping=render(source,row['page_index'],geometry,context,config.followup_max)
            request={'id':f'crop{number+1}','purpose':window.reason,'image_geometry':crop_mapping}
            requests.append(request)
            # Existing regions are untrusted observations, not grouping evidence.
            known=[{'id':r['model_id'],'box':r['box'],'kind':r['kind']} for r in regions]
            context_text=('This is a targeted source crop. Propose missing independent regions only. '
                          'Do not repeat existing full-page detections. Existing canonical source bounds '
                          '(not normalized crop coordinates): '+json.dumps(known))
            try:
                extra=request_analysis(client,image,prompt,request,context_text)
                regions.extend(mapped_regions(extra,crop_mapping,bounds,request['id']))
                page['review_reasons'].extend(extra.uncertainty)
                if extra.coverage=='incomplete' or extra.followups:
                    page['review_reasons'].append('Targeted crop coverage unresolved; nested follow-ups are not executed')
            except ModelError as exc:
                request['state']='failed'
                failures.append(str(exc))
                page['review_reasons'].append('Targeted crop analysis failed; full-page proposals retained for review')
                break
    except ModelError as exc:
        if requests:
            requests[-1]['state']='failed'
        failures.append(str(exc))
        page['review_reasons'].append('Qwen analysis could not complete; no deterministic fallback')
    if page['decision']!='FAILED' and config.verify_coverage:
        try:
            regions=verify_page_coverage(source,row,geometry,config,client,regions,page,requests,full_render=full_render)
        except ModelError as exc:
            failures.append(str(exc))
            page['coverage']='incomplete'
            page['coverage_validation']={'version':coverage.VERSION,'state':'failed','reason':str(exc)}
            page['review_reasons'].append('Full-page coverage verification failed; all original proposals retained for REVIEW')
    regions=reconcile_duplicates(regions)
    validate_relationships(regions)
    if page['decision']!='FAILED':
        assembly=any(r['kind']=='assembly_candidate' for r in regions)
        # Generated boundary/annotation flags on non-detail metadata do not
        # contradict a confident page eligibility decision. Model doubts do.
        uncertain=(eligibility_uncertain if page['decision']=='NO_DETAIL' else
                   any(r['kind'] in {'unknown','mixed_content'} or r['review_reasons'] for r in regions))
        contradiction=(page['decision']=='NO_DETAIL' and assembly) or (page['decision']=='CANDIDATES' and not assembly)
        if contradiction:
            page['review_reasons'].append('Model page decision conflicts with proposed region eligibility')
        if confirmed_no_detail(page,regions,failures):
            page['decision']='NO_DETAIL'
            page['eligibility_evidence']=page['coverage_validation']['audit']['evidence']
        elif (page['decision']=='REVIEW' or page['coverage']!='complete' or page['review_reasons'] or
                uncertain or contradiction or failures):
            page['decision']='REVIEW'
        else:
            page['decision']='CANDIDATES' if assembly else 'NO_DETAIL'
    if page['decision'] in {'REVIEW','FAILED'}:
        for region in regions:
            region['review_reasons'].append('Page eligibility/coverage remains unresolved; inspect full-page context')
    return regions,page,requests,failures


def crop_context(region, regions, bounds, margin, audit):
    required=region['box']
    requested=expand(required,margin,bounds)
    result=list(requested)
    obstacles=[r['box'] for r in regions if r is not region and
               r['parent_ref']!=region['model_id'] and region['parent_ref']!=r['model_id']]
    owned={a['ordinal'] for a in region['annotations']}
    obstacles.extend(a['box'] for a in audit if a['ordinal'] not in owned)
    reasons=[]
    for box in obstacles:
        if intersection(required,box)>0:
            region['review_reasons'].append('Required rectangular extents overlap neighboring/shared content; separation unresolved')
            continue
        if min(requested[3],box[3])>max(requested[1],box[1]):
            if box[2]<=required[0]:
                result[0]=max(result[0],min(required[0],box[2]))
            elif box[0]>=required[2]:
                result[2]=min(result[2],max(required[2],box[0]))
        if min(requested[2],box[2])>max(requested[0],box[0]):
            if box[3]<=required[1]:
                result[1]=max(result[1],min(required[1],box[3]))
            elif box[1]>=required[3]:
                result[3]=min(result[3],max(required[3],box[1]))
    if result!=requested:
        reasons.append('Optional padding constrained by neighboring regions/native text; required content retained')
    if region['kind']=='assembly_candidate':
        kinds={r['model_id']:r['kind'] for r in regions}
        for annotation in audit:
            box=annotation['box']
            ids=annotation['competing_model_ids']
            if (region['model_id'] not in ids and
                    annotation['owner_model_id']!=region['model_id']):
                continue
            if (area(box)<=0 or intersection(required,box)<.25*area(box) or
                    any(kinds.get(model_id)=='document_metadata' for model_id in ids)):
                continue
            complete=expand(box,2,bounds)
            if intersection(result,box)<area(box)-1e-5:
                result=union(result,complete)
                note='Crop extended to retain intersecting native text; annotation ownership remains unresolved'
                reasons.append(note)
                region['review_reasons'].append(note)
    return result,reasons


def retain_native_context(regions, audit, bounds):
    """Retain clipped native spans for review without changing ownership evidence.

    Audit precedes expansion, so a newly enclosed label is never silently assigned
    to a component/region. The published context bounds match the retained content.
    """
    kinds={r['model_id']:r['kind'] for r in regions}
    for region in regions:
        if region['kind']!='assembly_candidate':continue
        original=list(region['box'])
        for annotation in audit:
            box=annotation['box'];ids=annotation['competing_model_ids']
            if (region['model_id'] not in ids or area(box)<=0 or
                    intersection(original,box)<.25*area(box) or
                    intersection(original,box)>=area(box)-1e-5 or
                    any(kinds.get(i)=='document_metadata' for i in ids)):
                continue
            region['box']=union(region['box'],expand(box,2,bounds))
            region['review_reasons'].append('Context bounds expanded to retain clipped native text; ownership remains unresolved')


def export_page(source,row,geometry,regions,config,output,attempt_id,provenance,page,spans=(),vectors=()):
    parent=output/'detections'; parent.mkdir(parents=True,exist_ok=True)
    staging=parent/f'.{attempt_id}.tmp'; staging.mkdir()
    destination=parent/attempt_id
    bounds=geometry['source_rect']
    overview,overview_geometry=render(source,row['page_index'],geometry,bounds,config.overview_max)
    overview.save(staging/'context.png')
    original_overview=overview.copy()
    for region in regions:
        if region['kind']!='assembly_candidate':continue
        verification=crop_evidence.verify(region,regions,bounds,spans,vectors,original_overview,overview_geometry)
        region['crop_completeness']=verification
        region['box']=verification['corrected_context_box']
        if verification['status']!='COMPLETE':region['review_reasons']+=verification['issues']
    canvas=ImageDraw.Draw(overview)
    matrix=fitz.Matrix(*overview_geometry['source_to_pixels'])
    ids={r['model_id']:digest([row['document_id'],row['page_index'],r['model_id'],r['box'],r['kind']])[:20]
         for r in regions}
    candidates=[]
    skipped=page['decision']=='NO_DETAIL'
    for number,region in enumerate(regions,1):
        cid=ids[region['model_id']]
        crop_name,crop_geometry=None,None
        if not skipped:
            crop_box,padding=crop_context(region,regions,bounds,max(bounds[2]-bounds[0],bounds[3]-bounds[1])*config.context_margin,provenance['annotation_audit'])
            crop,crop_geometry=render(source,row['page_index'],geometry,crop_box,config.crop_max)
            crop_name=cid+'.png'; crop.save(staging/crop_name)
            if region.get('crop_completeness'):
                crop_evidence.write(staging/cid,original_overview,overview_geometry,region,region['crop_completeness'],crop)
                # REVIEW context is the corrected source extent plus existing
                # constrained optional whitespace, never a smaller geometry crop.
                region['crop_completeness']['final_crop_box']=crop_box
        annotation_box=region['title']['box'] if region['title'] else None
        for annotation in region['annotations']:
            annotation_box=union(annotation_box,annotation['box']) if annotation_box else annotation['box']
        if crop_geometry is not None:
            crop_geometry.update(structural_box=region['structural_box'],annotation_box=annotation_box,padding_reasons=padding)
        reviews=list(dict.fromkeys(region['review_reasons']))
        label=f'C{number:03d}-{cid[:6]}'
        record={'id':cid,'label':label,'document_id':row['document_id'],'page_id':row['id'],
                'page_index':row['page_index'],'source_path':str(source),'box':region['box'],
                'structural_box':region['structural_box'],'annotation_box':annotation_box,
                'initial_box':region.get('initial_box',region['box']),
                'crop_completeness':region.get('crop_completeness'),
                'crop_review_directory':str(destination/cid) if region.get('crop_completeness') else None,
                'crop_review_paths':{name:str(destination/cid/name) for name in ['structural_bbox_overlay.png','context_bbox_overlay.png','detail_crop.png','crop_completeness_review.md','crop_completeness.json']} if region.get('crop_completeness') else {},
                'annotations':region['annotations'],'title':region['title'],
                'reference_marks':region.get('reference_marks',[]),
                'kind':region['kind'],'view_type':region['view_type'],'method':'qwen',
                'evidence':region['evidence'],'model_region_id':region['model_id'],
                'parent_id':ids.get(region['parent_ref']),
                'child_ids':[ids[r['model_id']] for r in regions if r['parent_ref']==region['model_id']],
                'status':'REVIEW' if reviews else 'PROPOSED','review_reasons':reviews,
                'crop_path':str(destination/crop_name) if crop_name else None,'crop_geometry':crop_geometry,
                'crop_sha256':indexer.fingerprint(staging/crop_name) if crop_name else None}
        candidates.append(record)
        rectangle=list(fitz.Rect(region['box'])*matrix)
        color='#e07000' if reviews else '#0062d0'
        canvas.rectangle(rectangle,outline=color,width=3)
        canvas.rectangle(list(fitz.Rect(region['structural_box'])*matrix),outline='#008050',width=2)
        xy=(max(0,rectangle[0]),max(0,rectangle[1]-14))
        canvas.rectangle(canvas.textbbox(xy,label),fill='white'); canvas.text(xy,label,fill=color)
    if page['decision'] not in {'FAILED','NO_DETAIL'} and any(c['status']=='REVIEW' for c in candidates):
        page['decision']='REVIEW'
        page['review_reasons'].append('Candidate boundaries, classification or annotation ownership require review')
    canvas.rectangle((0,0,min(overview.width,260),18),fill='white')
    canvas.text((4,2),'Stage 2 | '+page['decision']+' | green: geometry',fill='black')
    overview.save(staging/'overview.png')
    overview.save(staging/'detection_full_page_overlay.png')
    review_paths={}
    audit=page.get('coverage_validation',{})
    if audit:
        proposed=audit.get('proposed_candidates',regions)
        superseded=set(audit.get('superseded_model_ids',[]))
        layers={'proposed':proposed,
                'accepted':[r for r in regions if r['kind']=='assembly_candidate'],
                'rejected':[r for r in proposed if r['model_id'] in superseded]+[
                    r for r in regions if r['kind'] not in {'assembly_candidate','unknown'}],
                'final':regions}
        for name,items in layers.items():
            with Image.open(staging/'context.png') as original:
                layer=original.convert('RGB')
            painter=ImageDraw.Draw(layer)
            for item in items:
                rect=list(fitz.Rect(item['box'])*matrix)
                color='#b00020' if name=='rejected' else '#e07000' if item['review_reasons'] else '#0062d0'
                painter.rectangle(rect,outline=color,width=3)
                painter.rectangle(list(fitz.Rect(item['structural_box'])*matrix),outline='#008050',width=2)
                label=item['model_id']+' '+item['kind']
                xy=(max(0,rect[0]),max(20,rect[1]-14))
                painter.rectangle(painter.textbbox(xy,label),fill='white');painter.text(xy,label,fill=color)
            painter.rectangle((0,0,layer.width,18),fill='white')
            heading={'accepted':'ELIGIBLE / REVIEW candidates, not verified Details',
                     'rejected':'Non-assembly / superseded proposals; evidence retained'}.get(name,name)
            painter.text((4,2),heading+' | green geometry / outer context',fill='black')
            filename='coverage-'+name+'.png';layer.save(staging/filename)
            review_paths[name]=str(destination/filename)
    manifest={'stage':2,'detector_version':VERSION,'attempt_id':attempt_id,
        'page_id':row['id'],'document_id':row['document_id'],'page_index':row['page_index'],
        'source_path':str(source),'coordinate_system':'Stage 1 canonical unrotated source coordinates; PDF points or original image pixels',
        'page_decision':page['decision'],'page_analysis':page,'configuration':asdict(config),'provenance':provenance,
        'overview_path':str(destination/'overview.png'),'context_path':str(destination/'context.png'),
        'overview_geometry':overview_geometry,'annotation_audit':provenance['annotation_audit'],
        'coverage_review_paths':review_paths,
        'downstream_skipped':skipped,'skip_reason':list(page['evidence']) if skipped else None,
        'candidates':candidates,'empty_success':page['decision']=='NO_DETAIL' and not candidates}
    (staging/'manifest.json').write_text(indexer.dumps(manifest),encoding='utf-8')
    staging.rename(destination)
    return destination/'manifest.json',manifest


def reusable(path):
    try:
        manifest=json.loads(Path(path).read_text())
        return (manifest['detector_version']==VERSION and manifest['page_decision']!='FAILED' and
                Path(manifest['overview_path']).is_file() and Path(manifest['context_path']).is_file() and
                (manifest['page_decision']=='NO_DETAIL' or
                 all(indexer.fingerprint(Path(c['crop_path']))==c['crop_sha256'] and
                     all(Path(p).is_file() for p in c.get('crop_review_paths',{}).values()) for c in manifest['candidates'])))
    except (OSError,ValueError,KeyError):
        return False


def run(source,output,config=None,model=None,index_path=None,max_pages=None,force=False,client=None,pages=None):
    started=time.monotonic()
    config=config or Config(); config.validate()
    model=model or (client.config if client else ModelConfig())
    client=client or QwenClient(model)
    if not model.endpoint or not model.model:
        raise ValueError('Qwen endpoint and served alias required; inference NOT RUN. Configure AS_OCR_QWEN_ENDPOINT and AS_OCR_QWEN_MODEL')
    if index_path is None or not Path(index_path).is_file():
        raise ValueError('Existing Stage 1 --index is required; index the selected source first')
    if (max_pages is not None and max_pages<1) or (pages is not None and
            (not pages or any(not isinstance(p,int) or p<1 for p in pages))):
        raise ValueError('Page selections must be positive (one-based --pages)')
    if pages is not None and max_pages is not None:
        raise ValueError('Use --pages or --max-pages, not both')
    source,output=Path(source).resolve(),Path(output).resolve()
    if not source.is_file():
        raise ValueError('Detection requires an explicit single input file')
    source_id=indexer.fingerprint(source)
    db=sqlite3.connect(Path(index_path).resolve()); db.row_factory=sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    try:
        document=db.execute('SELECT page_count FROM documents WHERE id=?',(source_id,)).fetchone()
        if not document:
            raise ValueError('Current source content is not indexed; run Stage 1 first')
        count=document['page_count']
        selected=sorted(set(p-1 for p in pages)) if pages is not None else list(range(min(count,max_pages or count)))
        if any(p>=count for p in selected):
            raise ValueError('Selected page exceeds indexed document page count')
        output.mkdir(parents=True,exist_ok=True)
        migrate(db)
        run_id=uuid.uuid4().hex
        settings={'logic_sha256':code_fingerprint(),'detector_version':VERSION,'configuration':asdict(config),
            'libraries':{'pymupdf':fitz.VersionBind,'pillow':Image.__version__},
            'prompt_version':PROMPT_VERSION,'prompt_hash':digest(PROMPT),
            'schema_version':SCHEMA_VERSION,'schema_hash':digest(Analysis.model_json_schema()),
            'coverage_version':coverage.VERSION,'coverage_prompt_hash':digest(coverage.PROMPT),
            'coverage_schema_hash':digest(coverage.Audit.model_json_schema()),
            'response_format':'json_schema',
            'model':model.public(),'identity_scope':model.identity or 'unverified:'+run_id,
            'cross_run_reuse':bool(model.identity)}
        summary={'run_id':run_id,'pages_completed':0,'pages_reused':0,'candidates':0,'review':0,
            'errors':0,'model_failures':0,'model_requests':0,'limited':len(selected)<count,
            'page_decisions':{},'manifests':[]}
        initial_requests=client.requests
        with db:
            db.execute('INSERT INTO detection_runs(id,source,config) VALUES(?,?,?)',(run_id,str(source),indexer.dumps(settings)))
        for page_index in selected:
            row=db.execute("SELECT p.* FROM pages p LEFT JOIN runs r ON p.completed_run=r.id WHERE p.document_id=? AND p.page_index=? AND p.status='complete' ORDER BY r.started DESC,p.rowid DESC LIMIT 1",(source_id,page_index)).fetchone()
            if row is None:
                summary['errors']+=1
                with db:
                    db.execute('INSERT INTO detection_pages VALUES(?,?,?,?,?,?,?,?)',(uuid.uuid4().hex,run_id,f'{source_id}:{page_index}',None,'error','not_run',None,'No complete Stage 1 page; index selected pages first'))
                continue
            geometry,spans,signature=load_page(db,row)
            cache_key=digest([source_id,signature,settings,str(output)])
            prior=db.execute("SELECT * FROM detection_pages WHERE page_id=? AND cache_key=? AND state='complete' AND model_state='succeeded' ORDER BY rowid DESC LIMIT 1",(row['id'],cache_key)).fetchone()
            if not force and prior and reusable(prior['manifest']):
                manifest=json.loads(Path(prior['manifest']).read_text())
                summary['pages_reused']+=1
                with db:
                    db.execute('INSERT INTO detection_pages VALUES(?,?,?,?,?,?,?,?)',(uuid.uuid4().hex,run_id,row['id'],cache_key,'reused','succeeded',prior['manifest'],None))
                manifest_path=Path(prior['manifest'])
            else:
                if client.requests>=model.max_requests:
                    summary.update(limited=True,stop_reason='Model request budget exhausted; remaining pages not analyzed')
                    summary['errors']+=1
                    break
                attempt=uuid.uuid4().hex
                with db:
                    db.execute('INSERT INTO detection_pages VALUES(?,?,?,?,?,?,?,?)',(attempt,run_id,row['id'],cache_key,'processing','not_run',None,None))
                try:
                    regions,page,requests,failures=analyze_page(source,row,geometry,config,client)
                    audit=annotate(regions,spans)
                    retain_native_context(regions,audit,geometry['source_rect'])
                    provenance={'index_signature':signature,'cache_key':cache_key,'detector_settings':settings,
                        'configured_checkpoint_revision':model.identity,'response_models':sorted(client.response_models),
                        'model_failures':failures,'requests':requests,'annotation_audit':audit}
                    vectors=[(o,json.loads(data)) for o,data in db.execute('SELECT ordinal,data FROM vectors WHERE page_id=? ORDER BY ordinal',(row['id'],))]
                    manifest_path,manifest=export_page(source,row,geometry,regions,config,output,attempt,provenance,page,spans,vectors)
                    failed=manifest['page_decision']=='FAILED'
                    state='error' if failed else 'complete'
                    model_state='failed' if failed else 'partial' if failures or page['coverage']!='complete' else 'succeeded'
                    summary['model_failures']+=bool(failures)
                    summary['errors']+=failed
                    summary['pages_completed']+=not failed
                    with db:
                        for c in manifest['candidates']:
                            db.execute('INSERT INTO detection_candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(
                                attempt,c['id'],row['id'],indexer.dumps(c['box']),c['kind'],c['view_type'],c['method'],
                                indexer.dumps(c['evidence']),c['status'],indexer.dumps(c['review_reasons']),
                                c['crop_path'],indexer.dumps(c['crop_geometry'])))
                        db.execute('UPDATE detection_pages SET state=?,model_state=?,manifest=?,error=? WHERE id=?',
                            (state,model_state,str(manifest_path),'; '.join(failures) or None,attempt))
                except Exception as exc:
                    summary['errors']+=1
                    error=f'Page {page_index+1} export/processing failed ({type(exc).__name__}); prior attempts preserved'
                    with db:
                        db.execute("UPDATE detection_pages SET state='error',error=? WHERE id=?",(error,attempt))
                    print(error,file=sys.stderr)
                    continue
            summary['manifests'].append(str(manifest_path))
            summary['candidates']+=len(manifest['candidates'])
            summary['review']+=sum(c['status']=='REVIEW' for c in manifest['candidates'])
            decision=manifest['page_decision']
            summary['page_decisions'][decision]=summary['page_decisions'].get(decision,0)+1
        summary.update(model_requests=client.requests-initial_requests,runtime_seconds=round(time.monotonic()-started,3),
            incomplete=bool(summary['errors'] or summary['model_failures'] or summary['limited'] or summary['page_decisions'].get('REVIEW')))
        with db:
            db.execute('UPDATE detection_runs SET finished=CURRENT_TIMESTAMP,summary=? WHERE id=?',(indexer.dumps(summary),run_id))
        return summary
    finally:
        db.close()


def main(argv=None):
    parser=argparse.ArgumentParser(prog='as-ocr detect',description='Stage 2 Qwen vision region detection; requires Stage 1 index')
    parser.add_argument('input',type=Path,help='Explicit single source file; no discovery')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--index',required=True,type=Path)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--max-pages',type=indexer.positive)
    selection.add_argument('--pages',type=indexer.positive,nargs='+',help='Explicit one-based page numbers; processed sequentially')
    parser.add_argument('--force',action='store_true',help='Create new attempts instead of compatible resume')
    parser.add_argument('--analysis-max',type=indexer.positive,default=1536)
    parser.add_argument('--crop-max',type=indexer.positive,default=2400)
    parser.add_argument('--overview-max',type=indexer.positive,default=1600)
    parser.add_argument('--context-margin',type=float,default=.01)
    parser.add_argument('--max-followups',type=int,default=2,help='At most four targeted source crops per page, only when requested')
    parser.add_argument('--followup-max',type=indexer.positive,default=2048)
    parser.add_argument('--qwen-endpoint',default=os.getenv('AS_OCR_QWEN_ENDPOINT'))
    parser.add_argument('--qwen-model',default=os.getenv('AS_OCR_QWEN_MODEL'))
    parser.add_argument('--qwen-identity',default=os.getenv('AS_OCR_QWEN_IDENTITY'),help='Verified checkpoint@immutable-revision; absent disables cross-run reuse')
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--max-tokens',type=indexer.positive,default=8192)
    parser.add_argument('--retries',type=int,default=1)
    parser.add_argument('--max-model-requests',type=indexer.positive,default=10,help='Total HTTP attempts including retries and follow-ups')
    args=parser.parse_args(argv)
    cfg=Config(args.analysis_max,args.crop_max,args.overview_max,args.context_margin,args.max_followups,args.followup_max)
    model=ModelConfig(args.qwen_endpoint,args.qwen_model,args.qwen_identity,args.timeout,args.max_tokens,
                      args.retries,args.max_model_requests,os.getenv('AS_OCR_QWEN_API_KEY'))
    try:
        result=run(args.input,args.output,cfg,model,args.index,args.max_pages,args.force,pages=args.pages)
    except KeyboardInterrupt:
        parser.exit(130,'Interrupted; completed attempts preserved.\n')
    except (ValueError,OSError,sqlite3.Error) as exc:
        parser.exit(2,f'Error: {exc}\n')
    print(indexer.dumps(result))
    return 1 if result['errors'] or result['model_failures'] else 0

"""Stage 7: validate the evidence graph and publish linked, reversible exports."""
from __future__ import annotations

from .provenance import code_fingerprint
from . import traceability

import argparse
import copy
import csv
from decimal import Decimal,InvalidOperation
import json
import math
from pathlib import Path
import sqlite3
import uuid

from PIL import Image

from . import components,enrichment,grouping,progress,leaders

VERSION='6'
TABLES={
 'details':['id','document_id','page_id','page_index','name','status','grouping_status','completeness','view_ids','composites','review_ids'],
 'detail_components':['id','detail_id','component_id','occurrence_id','view_id','association_status','target_point','observed_view_ids','context_detail_ids','review_ids'],
 'components':['id','identity_id','catalog_identity_id','document_id','page_id','page_index','code_raw','code_normalized','scope','scope_evidence_ids','category','status','occurrence_ids','observed_view_ids','property_ids','review_ids'],
 'profiles':['component_id','code_raw','dimension_property_ids','linear_weight_property_ids','inertia_property_ids','modulus_property_ids','review_ids'],
 'properties':['id','component_id','field','name_raw','value_raw','unit_raw','normalized','meaning','accepted','source_accepted','validation_status','conditional','evidence_ids','condition_evidence_ids','relation_id','applies_to_component_ids','conflict_ids','review_ids'],
 'relationships':['id','kind','source_id','target_id','status','evidence_ids','data','review_ids'],
 'evidence':['id','document_id','page_id','page_index','view_ids','quote','box','method','native_span_ids','provenance_ids','review_ids'],
 'review':['id','severity','code','message','subject_ids','evidence_ids','provenance_ids'],
}


def detail_image(composites, destination):
    """Use original Detail crops, not the downsampled Stage 3 review thumbnails."""
    panels = []
    seen = set()
    for composite in composites:
        originals = composite.get('panels', [])
        sources = [(p['source_crop'], p.get('source_crop_sha256')) for p in originals] if originals else [(composite['path'], None)]
        for path, expected in sources:
            if path in seen:
                continue
            seen.add(path)
            if expected and grouping.file_digest(path) != expected:
                raise ValueError('Original Detail crop changed after grouping')
            with Image.open(path) as source:
                source.load()
                panels.append(source.convert('RGB'))
    if not panels:
        raise ValueError('Detail has no source-backed composite image')
    image = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), 'white')
    top = 0
    for panel in panels:
        image.paste(panel, (0, top))
        top += panel.height
    image.save(destination)


def dumps(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'))


class Reviews:
    def __init__(self):self.rows={}
    def add(self,code,message,subject=None,*,severity='REVIEW',provenance=None,evidence=()):
        key='review-'+grouping.digest([severity,code,message])[:20]
        row=self.rows.setdefault(key,{'id':key,'severity':severity,'code':code,'message':message,
                                      'subject_ids':[],'evidence_ids':[],'provenance_ids':[]})
        for field,values in [('subject_ids',[subject] if subject else []),('evidence_ids',evidence),('provenance_ids',[provenance] if provenance else [])]:
            row[field]=sorted(set(row[field])|set(values))
        return key
    def errors(self):return [r for r in self.rows.values() if r['severity']=='ERROR']


def issue_locations(issue,graph):
    result=[]
    for subject in issue['subject_ids']:
        record=next((graph[k][subject] for k in ('details','views','occurrences','candidates','observations')
                     if subject in graph.get(k,{})),{})
        location=dict(subject_id=subject)
        if isinstance(record.get('page_index'),int):location['source_page']=record['page_index']+1
        if record.get('raw_code') or record.get('code_raw'):location['label']=record.get('raw_code') or record['code_raw']
        if record.get('detail_ids'):location['detail_ids']=record['detail_ids']
        result.append(location)
    return result


def valid_box(box,bounds=None):
    if not isinstance(box,(list,tuple)) or len(box)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in box):return False
    if box[0]>=box[2] or box[1]>=box[3]:return False
    return bounds is None or bounds[0]-.01<=box[0]<box[2]<=bounds[2]+.01 and bounds[1]-.01<=box[1]<box[3]<=bounds[3]+.01


def valid_point(point,bounds):
    return isinstance(point,(list,tuple)) and len(point)==2 and all(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) for v in point) and (bounds is None or bounds[0]<=point[0]<=bounds[2] and bounds[1]<=point[1]<=bounds[3])


def read_saved(path,stage,review,expected=None):
    try:
        path=Path(path)
        if path.stat().st_size>64*1024*1024:raise ValueError('size')
        if expected and grouping.file_digest(path)!=expected:
            review.add('UPSTREAM_CHANGED','Selected upstream content no longer matches its recorded identity',str(path),severity='ERROR');return None
        def invalid(value):raise ValueError('nonfinite')
        def unique(pairs):
            result={}
            for key,value in pairs:
                if key in result:raise ValueError('duplicate key')
                result[key]=value
            return result
        saved=json.loads(path.read_text(),parse_constant=invalid,object_pairs_hook=unique)
        if saved.get('stage')!=stage:raise ValueError('stage')
        return saved
    except (OSError,ValueError,TypeError):
        review.add('MISSING_UPSTREAM','Selected upstream manifest is missing or invalid',str(path),severity='ERROR');return None


def load_graph(s6,review):
    graph={'details':{},'views':{},'occurrences':{},'candidates':{},'identities':{},'observations':{},'paragraphs':{},'scope':[],
           'provenance':{},'page_decisions':{},'upstream_incomplete':False,'profile_observations':[],'leader_observations':[]}
    selected=set(s6['selected_candidate_ids'])|{m['source_component_id'] for m in s6['matches']}
    def insert(table,record,provenance):
        key=record['id'];prior=graph[table].get(key)
        if prior is not None and dumps(prior)!=dumps(record):
            review.add('IDENTIFIER_COLLISION','Conflicting records use the same identifier; no record is accepted silently',key,severity='ERROR',provenance=provenance)
            # Retain the additional raw record under an explicit variant identity.
            variant=copy.deepcopy(record);variant['id']=key+'-variant-'+grouping.digest(record)[:8]
            if table=='observations':variant['accepted']=False;prior['accepted']=False
            graph[table][variant['id']]=variant
        else:graph[table][key]=copy.deepcopy(record)
        graph['provenance'].setdefault(key,[])
        if provenance not in graph['provenance'][key]:graph['provenance'][key].append(provenance)
    for source in s6['sources']:
        path=source['stage5_path'];s5=read_saved(path,5,review,source['stage5_sha256'])
        s3=read_saved(source['stage3_path'],3,review,source['stage3_sha256'])
        s4=read_saved(source['stage4_path'],4,review,source['stage4_sha256'])
        if s3:
            for table,key in [('details','details'),('views','views')]:
                for item in s3[key]:insert(table,item,source['stage3_path'])
        if s4:
            graph['leader_observations']+=s4.get('leader_observations',[])
            for o in s4['occurrences']:insert('occurrences',o,source['stage4_path'])
            decision=s4['upstream'].get('page_decision')
            graph['page_decisions'][source['page_id']]=decision
            if s4['state'] in {'FAILED','PARTIAL'} or not s4['complete']:
                graph['upstream_incomplete']=True
                review.add('UPSTREAM_INCOMPLETE','Stage 4 execution/coverage remains incomplete',source['page_id'],provenance=source['stage4_path'])
        if s5:
            graph['profile_observations']+=s5.get('profile_observations',[])
            for paragraph in (s5.get('paragraph_analysis') or {}).get('paragraphs', []):
                insert('paragraphs',paragraph,path)
            if s5['state']!='COMPLETE' or not s5['complete']:
                graph['upstream_incomplete']=True
                review.add('UPSTREAM_INCOMPLETE','Stage 5 execution/coverage remains incomplete',source['page_id'],provenance=path)
            candidates=[c for c in s5['candidates'] if c['id'] in selected]
            ids={c['id'] for c in candidates}
            for c in candidates:insert('candidates',c,path)
            for i in s5['identities']:
                if ids&set(i['candidate_ids']):insert('identities',i,path)
            for o in s5['observations']:
                if o['candidate_id'] in ids:insert('observations',o,path)
            graph['scope']+=s5['scope_observations']
    for table,key in [('views','reference_views'),('candidates','source_candidates'),('identities','source_identities'),('observations','source_observations')]:
        for record in s6[key]:insert(table,record,'stage6:'+s6['run_id'])
    graph['scope']+=s6.get('scope_observations',[])
    return graph


class EvidenceRegistry:
    def __init__(self,db,views,review):
        self.db=db;self.views=views;self.review=review;self.rows={};self.valid={};self.pages={}
    def page(self,page_id):
        if page_id not in self.pages:
            row=self.db.execute('SELECT * FROM pages WHERE id=?',(page_id,)).fetchone() if self.db else None
            self.pages[page_id]=dict(row) if row else None
        return self.pages[page_id]
    def add(self,e,subject,provenance):
        # Identical source evidence is stored once even if several crops observed it.
        identity=[e.get(k) for k in ('document_id','page_id','quote','box','method')]
        key='evidence-'+grouping.digest(identity)[:20]
        row=self.rows.setdefault(key,{'id':key,'document_id':e.get('document_id'),'page_id':e.get('page_id'),
            'page_index':e.get('page_index'),'view_ids':[],'quote':e.get('quote'),'box':e.get('box'),
            'method':e.get('method'),'native_span_ids':[],'provenance_ids':[],'review_ids':[]})
        row['view_ids']=sorted(set(row['view_ids'])|({e['view_id']} if e.get('view_id') else set()))
        row['native_span_ids']=sorted(set(row['native_span_ids'])|set(e.get('native_span_ids',[])))
        row['provenance_ids']=sorted(set(row['provenance_ids'])|set(provenance))
        valid=True;page=self.page(e.get('page_id'));bounds=json.loads(page['geometry'])['source_rect'] if page else None
        def error(code,message):
            nonlocal valid
            valid=False;row['review_ids'].append(self.review.add(code,message,subject,severity='ERROR',evidence=[key]))
        if not page:error('MISSING_PAGE','Evidence source page is absent from the selected index')
        elif page['status']!='complete' or page['document_id']!=e.get('document_id') or page['page_index']!=e.get('page_index'):
            error('FOREIGN_EVIDENCE','Evidence document/page identity is inconsistent')
        if not valid_box(e.get('box'),bounds):error('INVALID_COORDINATE','Evidence bounds are invalid or outside its canonical source page')
        view=self.views.get(e.get('view_id'))
        if view is None:error('MISSING_VIEW','Evidence references an unknown source view')
        elif view['page_id']!=e.get('page_id') or view['document_id']!=e.get('document_id'):
            error('FOREIGN_VIEW','Evidence and view document/page differ')
        if not isinstance(e.get('quote'),str) or not e['quote'].strip() or not e.get('method'):
            error('MISSING_EVIDENCE','An asserted field needs a source quote and extraction method')
        if e.get('method')=='native_text':
            native=[]
            for ordinal in e.get('native_span_ids',[]):
                found=self.db.execute('SELECT data FROM spans WHERE page_id=? AND ordinal=?',(e.get('page_id'),ordinal)).fetchone() if self.db else None
                if found:native.append(json.loads(found[0]))
            if not native or not components.paragraphs.quote_supported(e['quote'], [s['text'] for s in native]):
                error('NATIVE_QUOTE_MISMATCH','Native quote is not supported by indexed spans')
            elif any(not valid_box(s['bbox'],bounds) or enrichment.intersection(s['bbox'],e['box'])<=0 for s in native):
                error('NATIVE_COORDINATE_MISMATCH','Native quote bounds do not overlap the indexed text')
        for reason in e.get('review_reasons',[]):
            row['review_ids'].append(self.review.add('EVIDENCE_REVIEW',reason,subject,evidence=[key]))
        self.valid[key]=self.valid.get(key,True) and valid
        row['review_ids']=sorted(set(row['review_ids']))
        return key,valid


def build_tables(s6,graph,db,review):
    tables={name:[] for name in TABLES};registry=EvidenceRegistry(db,graph['views'],review)
    provenance=graph['provenance'];views=graph['views'];candidates=graph['candidates'];occurrences=graph['occurrences']
    def evidence_list(items,subject,prov):
        ids=[];valid=True
        for e in items:
            key,good=registry.add(e,subject,prov);ids.append(key);valid &= good
        return sorted(set(ids)),valid
    def relationship(kind,source,target,status,*,evidence=(),data=None,issues=()):
        row={'kind':kind,'source_id':source,'target_id':target,'status':status,'evidence_ids':sorted(set(evidence)),
             'data':data or {},'review_ids':sorted(set(issues))}
        row['id']='relationship-'+grouping.digest(row)[:20];tables['relationships'].append(row);return row['id']
    for view in views.values():
        page=registry.page(view['page_id']);bounds=json.loads(page['geometry'])['source_rect'] if page else None
        if page is None:review.add('MISSING_VIEW_PAGE','View source page is absent from index',view['id'],severity='ERROR')
        if not valid_box(view['box'],bounds):review.add('INVALID_VIEW_BOUNDS','View bounds are invalid',view['id'],severity='ERROR')
        if page and (page['document_id']!=view['document_id'] or page['page_index']!=view['page_index']):
            review.add('FOREIGN_VIEW','View and indexed page differ',view['id'],severity='ERROR')
        relationship('view_source',view['id'],view['page_id'],'REFERENCE_ONLY' if view.get('kind')=='reference_only' else 'PROVISIONAL',
                     data={k:view[k] for k in ('box','structural_box','crop_path','crop_geometry','source_path') if k in view})
    scope_signatures={};scope_evidence_ids={};effective_scopes={}
    for identity in graph['identities'].values():
        accepted_values={};scope_ids=[]
        for item in identity.get('scope_evidence',[]):
            checked=copy.deepcopy(item);components.validate_scope([checked])
            ids,good=evidence_list(item['evidence'],identity['id'],provenance.get(identity['id'],[]));scope_ids+=ids
            if any(registry.rows[e]['document_id']!=identity['document_id'] for e in ids):
                good=False;review.add('FOREIGN_SCOPE','Scope evidence belongs to another source document',identity['id'],severity='ERROR')
            if checked.get('accepted',False) and good and any(item['value_raw'] in registry.rows[e]['quote'] for e in ids):
                accepted_values.setdefault(item['field'],set()).add(item['value_raw'])
        effective={name:next(iter(accepted_values[name])) if len(accepted_values.get(name,set()))==1 and identity.get('scope',{}).get(name) in accepted_values[name] else None for name in ('manufacturer','system','edition')}
        scope_signatures[identity['id']]=tuple(effective[name] for name in ('manufacturer','system','edition')) if all(effective.values()) else None
        scope_evidence_ids[identity['id']]=sorted(set(scope_ids));effective_scopes[identity['id']]=effective
    valid_details=set()
    for d in graph['details'].values():
        issues=[]
        if d.get('membership_role')=='alternative':
            review.add('ALTERNATIVE_GROUP_WITHHELD','Uncertain merged proposal remains internal; primary independent Details are exported',d['id'])
            continue
        if graph['page_decisions'].get(d['page_id'])=='NO_DETAIL':
            review.add('NO_DETAIL_VIOLATION','NO_DETAIL page cannot contain a Detail',d['id'],severity='ERROR')
            continue
        page=registry.page(d['page_id'])
        if not page or page['document_id']!=d['document_id'] or page['page_index']!=d['page_index']:
            issues.append(review.add('DETAIL_PAGE_IDENTITY','Detail original page does not match the source index',d['id'],severity='ERROR'))
        if not d['view_ids'] or any(v not in views or views[v]['page_id']!=d['page_id'] or views[v]['document_id']!=d['document_id'] for v in d['view_ids']):
            issues.append(review.add('DETAIL_VIEW_INTEGRITY','Detail references missing or foreign views',d['id'],severity='ERROR'))
        reverse_key='primary_detail_ids' if d.get('membership_role')=='primary' else 'alternative_detail_ids'
        if any(d['id'] not in views[v].get(reverse_key,[]) for v in d['view_ids'] if v in views):
            issues.append(review.add('DETAIL_MEMBERSHIP_INCONSISTENT','Detail/view membership references are inconsistent',d['id'],severity='ERROR'))
        if not any(review.rows[r]['severity']=='ERROR' for r in issues):valid_details.add(d['id'])
        for reason in d.get('grouping_review_reasons',[]):issues.append(review.add('GROUPING_REVIEW',reason,d['id'],provenance=provenance.get(d['id'],[None])[0]))
        issues.append(review.add('PROVISIONAL_DETAIL','Detail/composites remain provisional; completeness is not established',d['id']))
        composites=[{'path':c['path'],'pixel_size':c.get('pixel_size'),'status':'PROVISIONAL'} for c in d.get('composites',[])]
        for c in composites:
            if not Path(c['path']).is_file():issues.append(review.add('MISSING_COMPOSITE','Original provisional composite artifact is missing',d['id']))
        tables['details'].append({'id':d['id'],'document_id':d['document_id'],'page_id':d['page_id'],'page_index':d['page_index'],
            'name':d['name'],'status':'PROVISIONAL','grouping_status':d['grouping_status'],'completeness':d['completeness'],
            'view_ids':d['view_ids'],'composites':composites,'review_ids':sorted(set(issues))})
        for v in d['view_ids']:
            relationship('view_in_provisional_detail',v,d['id'],'PROVISIONAL',data={'source_membership_role':d['membership_role']},issues=issues)
    for occ in occurrences.values():
        page=registry.page(occ['page_id']);bounds=json.loads(page['geometry'])['source_rect'] if page else None
        if not valid_box(occ['text_box'],bounds):review.add('INVALID_OCCURRENCE_BOUNDS','Occurrence text bounds are invalid',occ['id'],severity='ERROR')
        if not set(occ['observed_in_view_ids'])<=set(views):review.add('OCCURRENCE_VIEW_INTEGRITY','Occurrence references an unknown observed view',occ['id'],severity='ERROR')
        if (occ['source_view_id'] is None)!=(occ['target_point'] is None) or (occ['source_view_id'] is not None and
            (occ['source_view_id'] not in views or not valid_point(occ['target_point'],bounds) or occ['association_method']=='unresolved' or not occ['association_evidence'])):
            review.add('OCCURRENCE_TARGET_INTEGRITY','Occurrence target/view/evidence references are inconsistent',occ['id'],severity='ERROR')
        for v in occ['observed_in_view_ids']:
            relationship('occurrence_observed_in_view',occ['id'],v,'CONTEXT_ONLY',data={'code_raw':occ['raw_code'],'text_box':occ['text_box']})
    properties={}
    for o in graph['observations'].values():
        oid=o['id'];issues=[];owner=candidates.get(o['candidate_id']);accepted=bool(o['accepted']);prov=provenance.get(oid,[])
        def reject(code,message):
            nonlocal accepted
            accepted=False;issues.append(review.add(code,message,oid,severity='ERROR',provenance=prov[0] if prov else None))
        if owner is None:reject('PROPERTY_OWNER_MISSING','Property references an unknown component')
        eids,good=evidence_list(o.get('evidence',[]),oid,prov)
        conditions,cgood=evidence_list(o.get('conditions',[]),oid,prov)
        if accepted and owner and (any(registry.rows[e]['document_id']!=owner['document_id'] or registry.rows[e]['page_id']!=owner['page_id'] for e in eids+conditions)):
            reject('FOREIGN_PROPERTY_EVIDENCE','Property evidence does not establish its source component/page ownership')
        if accepted and (not eids or not good or not cgood):reject('PROPERTY_EVIDENCE_INVALID','Asserted property lacks valid field/condition evidence')
        quoted = (components.paragraphs.quote_supported(o['value_raw'], [registry.rows[e]['quote'] for e in eids])
                  if o.get('relation',{}).get('kind')=='page_paragraph' else
                  any(o['value_raw'] in registry.rows[e]['quote'] for e in eids))
        if accepted and not quoted:reject('FIELD_QUOTE_MISMATCH','Asserted raw value does not occur in its source evidence')
        if accepted and o.get('name_raw') and not any(o['name_raw'] in registry.rows[e]['quote'] for e in eids):reject('PROPERTY_NAME_MISSING','Asserted property name/axis is absent from evidence')
        if accepted and o['field']=='code' and owner and o['value_raw']!=owner['code_raw']:reject('CODE_MISMATCH','Code observation differs from its component identifier')
        relation=o.get('relation',{});relation_evidence=[]
        if relation.get('code_evidence'):
            ids,ok=evidence_list([relation['code_evidence']],oid,prov);relation_evidence+=ids
            if accepted and not ok:reject('RELATION_EVIDENCE_INVALID','Property/code relationship evidence is invalid')
        ids,ok=evidence_list(relation.get('evidence',[]),oid,prov);relation_evidence+=ids
        if accepted and not ok:reject('RELATION_EVIDENCE_INVALID','Property relationship evidence is invalid')
        if accepted and o['field']!='code':
            kind=relation.get('kind')
            if kind in {'code_label','table_row'}:
                bounds=relation.get('bounds');code=relation.get('code_evidence')
                if not valid_box(bounds) or not code or not components.contained(bounds,code['box']) or code['quote']!=owner['code_raw']:
                    reject('RELATION_CODE_INVALID','Label/row does not establish its exact code relationship')
                elif not any(components.contained(bounds,registry.rows[e]['box']) and o['value_raw'] in registry.rows[e]['quote'] for e in eids):
                    reject('PROPERTY_OUTSIDE_RELATION','Property value is outside its demonstrated label/row')
                if not any(registry.rows[e]['method']!='native_text' for e in relation_evidence):reject('OWNERSHIP_UNVERIFIED','Native text/containment alone is not property ownership')
            elif kind=='stage4_target':
                if not owner or not any(m['target_point'] is not None and m['view_id'] for m in owner['memberships']):reject('NULL_TARGET_UPGRADE','Null physical target cannot establish property ownership')
            elif kind=='stage4_name_reading':
                occurrence=occurrences.get(relation.get('occurrence_id'),{})
                if (o['field']!='component_name' or not owner or occurrence.get('category')!='component_name' or
                    occurrence.get('reading_state')!='exact' or occurrence.get('raw_code')!=o['value_raw'] or
                    occurrence.get('page_id')!=owner['page_id'] or
                    relation.get('occurrence_id') not in owner['occurrence_ids'] or
                    components.provenance.reading_review_reasons(occurrence) or
                    components.provenance.metadata_source(occurrence,[views[v]['kind'] for v in occurrence.get('observed_in_view_ids',[]) if v in views])):
                    reject('NAME_READING_UNVERIFIED','Name requires an exact independent Stage 4 reading; no ownership is inferred')
            elif kind=='page_paragraph':
                records=[graph.get('paragraphs',{}).get(pid) for pid in relation.get('paragraph_ids',[])]
                if (o['field']!='description' or not records or any(not r or not r['accepted'] or
                    o['candidate_id'] not in r['selected_candidate_ids'] for r in records) or
                    o['value_raw']!='\n\n'.join(r['text'] for r in records) or
                    set(relation.get('detail_ids',[]))!=components.paragraphs.memberships(owner,list(graph['details'].values()))):
                    reject('PARAGRAPH_OWNERSHIP_UNVERIFIED','Description lacks its selected original-page paragraph association')
                else:
                    for r in records:
                        if (not r.get('evidence') or r['evidence'].get('page_id')!=owner['page_id'] or
                            r['evidence'].get('document_id')!=owner['document_id'] or
                            not r.get('association_evidence') or r['association_evidence'].get('page_id')!=owner['page_id']):
                            reject('PARAGRAPH_PAGE_INVALID','Paragraph and anchor must belong to the component original page')
                        ids,ok=evidence_list([r['association_evidence']],oid,prov);relation_evidence+=ids
                        if not ok:reject('PARAGRAPH_ANCHOR_INVALID','Paragraph association anchor is invalid')
            else:reject('OWNERSHIP_UNVERIFIED','Asserted property relationship is unresolved')
        normalized=copy.deepcopy(o.get('normalized'))
        if accepted and o['field'] in components.UNITS:
            expected,reason=components.normalize_value(o['field'],o['value_raw'],o['unit_raw'],o.get('decimal_convention','unknown'))
            if reason or expected!=normalized or not any(components.unit_in_quote(o['unit_raw'],registry.rows[e]['quote']) for e in eids):
                reject('INVALID_UNIT_VALUE','Normalized numeric property is incompatible, non-finite, ambiguous or lacks source units');normalized=None
            if o['field']=='dimension' and (o['name_raw'] is None or o['meaning'] not in {'component_dimension','glass_thickness'}):
                reject('DIMENSION_MEANING','Assembly/stock/cut/joint dimensions are not component dimensions')
            if o['field'] in {'inertia','section_modulus'} and not o['name_raw']:reject('STRUCTURAL_AXIS_MISSING','Structural property needs its explicit axis/meaning')
        elif normalized is not None:
            try:
                if not Decimal(str(normalized['value'])).is_finite():raise ValueError('finite')
            except (InvalidOperation,ValueError,KeyError,TypeError):
                reject('NONFINITE_NORMALIZATION','Normalized value is not a finite supported number');normalized=None
        if accepted and o['field']=='material' and re_finish(o['value_raw']):reject('FINISH_NOT_MATERIAL','A surface finish is not an explicit base material')
        if accepted and any(x in o['value_raw'] for x in '*†‡') and not conditions:
            reject('UNRESOLVED_FOOTNOTE','Footnote-qualified value cannot be unconditional')
        for reason in o.get('review_reasons',[]):issues.append(review.add('FIELD_REVIEW',reason,oid,provenance=prov[0] if prov else None))
        rid=relationship('property_source_relation',oid,o['candidate_id'],'SUPPORTED' if accepted else 'REVIEW',evidence=relation_evidence,data={'kind':relation.get('kind'),'bounds':relation.get('bounds'),'explanation':relation.get('explanation')})
        properties[oid]={'id':oid,'component_id':o['candidate_id'],'field':o['field'],'name_raw':o['name_raw'],'value_raw':o['value_raw'],
            'unit_raw':o['unit_raw'],'normalized':normalized if accepted else None,'meaning':o['meaning'],
            'accepted':accepted,'source_accepted':bool(o['accepted']),'validation_status':'SOURCE_SUPPORTED' if accepted else 'UNASSIGNED_OR_INVALID',
            'conditional':bool(conditions),'evidence_ids':eids,'condition_evidence_ids':conditions,'relation_id':rid,
            'applies_to_component_ids':[o['candidate_id']] if owner else [],'conflict_ids':list(o.get('conflicts_with',[])),
            'review_ids':sorted(set(issues))}
    for prop in properties.values():
        if prop['accepted'] and prop['field']=='dimension' and prop['meaning']=='glass_thickness':
            categories={graph['observations'][p['id']].get('category_value') for p in properties.values() if p['component_id']==prop['component_id'] and p['field']=='category' and p['accepted'] and not p['conditional'] and not p['conflict_ids']}
            if categories!={'glass'}:
                prop['accepted']=False;prop['normalized']=None;prop['validation_status']='UNASSIGNED_OR_INVALID'
                prop['review_ids'].append(review.add('GLASS_DIMENSION_SCOPE','Glass thickness requires explicit glass category',prop['id'],severity='ERROR'))
    matches={m['id']:m for m in s6['matches']}
    for match in matches.values():
        a=candidates.get(match['target_component_id']);b=candidates.get(match['source_component_id']);valid=False
        if a and b:
            ia=graph['identities'].get(a['identity_id'],{});ib=graph['identities'].get(b['identity_id'],{})
            sa=scope_signatures.get(ia.get('id'));sb=scope_signatures.get(ib.get('id'))
            valid=bool(a['document_id']==b['document_id'] and a['code_raw']==b['code_raw'] and sa and sa==sb)
        issues=[]
        if match['status']=='SUPPORTED' and not valid:
            issues.append(review.add('INVALID_SCOPE_TRANSFER','Claimed enrichment lacks exact code and verified source scope',match['id'],severity='ERROR'))
        elif not valid:issues.append(review.add('SCOPE_REVIEW',match['reason'],match['id']))
        relationship('same_scoped_article' if valid else 'ambiguous_code_match',match['target_component_id'],match['source_component_id'],
                     'SUPPORTED' if valid else 'REVIEW',data={'source_match_id':match['id']},issues=issues)
        match['_valid']=valid
    for assignment in s6['assignments']:
        prop=properties.get(assignment['observation_id']);match=matches.get(assignment['match_id']);target=candidates.get(assignment['candidate_id'])
        if graph['observations'].get(assignment['observation_id'],{}).get('relation',{}).get('kind')=='page_paragraph':
            review.add('PARAGRAPH_TRANSFER_FORBIDDEN','Original-page descriptions cannot transfer to another occurrence',assignment['id'],severity='ERROR');continue
        if not prop or not match or not match.get('_valid') or not prop['accepted'] or not target or prop['component_id']!=assignment['source_candidate_id'] or assignment.get('membership_effect')!='NONE':
            review.add('INVALID_ASSIGNMENT','Enrichment assignment lacks valid source property/scope or attempts a membership change',assignment['id'],severity='ERROR');continue
        prop['applies_to_component_ids']=sorted(set(prop['applies_to_component_ids'])|{assignment['candidate_id']})
        relationship('property_enrichment',assignment['observation_id'],assignment['candidate_id'],'SUPPORTED',
                     evidence=prop['evidence_ids']+prop['condition_evidence_ids'],data={'source_match_id':match['id'],'membership_effect':'NONE'})
    # Conflict detection runs over original and enriched observations. Conditions
    # remain separate; no component/property projection chooses a winning value.
    groups={}
    for prop in properties.values():
        if not prop['accepted']:continue
        conditions=tuple(sorted(registry.rows[e]['quote'] for e in prop['condition_evidence_ids']))
        for cid in prop['applies_to_component_ids']:
            groups.setdefault((cid,prop['field'],prop['name_raw'],prop['meaning'],conditions),[]).append(prop)
    for group in groups.values():
        for a in group:
            for b in group:
                av=a['normalized'] or [a['value_raw'],a['unit_raw']];bv=b['normalized'] or [b['value_raw'],b['unit_raw']]
                if a['id']!=b['id'] and av!=bv:
                    a['conflict_ids']=sorted(set(a['conflict_ids'])|{b['id']})
                    a['validation_status']='CONFLICT'
                    a['review_ids']=sorted(set(a['review_ids'])|{review.add('PROPERTY_CONFLICT','Supported source observations conflict; no winner selected',a['id'],evidence=a['evidence_ids'])})
    for prop in properties.values():
        if not set(prop['conflict_ids'])<=set(properties):
            prop['review_ids'].append(review.add('CONFLICT_REFERENCE_MISSING','Property conflict references a missing observation',prop['id'],severity='ERROR'))
    assigned_occurrences=set()
    for candidate in candidates.values():
        cid=candidate['id'];issues=[];identity=graph['identities'].get(candidate['identity_id'])
        if identity is None or cid not in identity.get('candidate_ids',[]):issues.append(review.add('IDENTITY_MISSING','Component scoped identity or reverse reference is missing',cid,severity='ERROR'))
        if identity and (identity['document_id']!=candidate['document_id'] or identity['code_raw']!=candidate['code_raw']):issues.append(review.add('IDENTITY_MISMATCH','Scoped identity/code/document is inconsistent',cid,severity='ERROR'))
        if not set(candidate['observed_in_view_ids'])<=set(views) or any(views[v]['document_id']!=candidate['document_id'] or views[v]['page_id']!=candidate['page_id'] for v in candidate['observed_in_view_ids'] if v in views):
            issues.append(review.add('COMPONENT_VIEW_INTEGRITY','Component observed-view references are missing or foreign',cid,severity='ERROR'))
        expected_normalized=enrichment.callouts.normalize(candidate['code_raw']) if candidate['code_raw'] else None
        normalized_code=candidate['code_normalized']
        if expected_normalized!=normalized_code:
            normalized_code=None;issues.append(review.add('NORMALIZED_CODE_MISMATCH','Normalized matching form differs from the exact raw code',cid,severity='ERROR'))
        signature=scope_signatures.get(candidate['identity_id'])
        if identity and identity.get('scope_status')=='SOURCE_VERIFIED' and not signature:issues.append(review.add('SCOPE_EVIDENCE_INVALID','Verified scope lacks required labeled source evidence',cid,severity='ERROR'))
        if not signature:issues.append(review.add('UNKNOWN_SCOPE','Manufacturer/system/catalog edition remain unknown or incomplete',cid))
        for reason in candidate.get('review_reasons',[]):issues.append(review.add('COMPONENT_REVIEW',reason,cid))
        for reason in candidate.get('inherited_review_reasons',[]):issues.append(review.add('INHERITED_STAGE_'+str(reason.get('stage','UNKNOWN')),reason['reason'],cid,provenance=provenance.get(cid,[None])[0]))
        owned=[p for p in properties.values() if cid in p['applies_to_component_ids']]
        categories={graph['observations'][p['id']].get('category_value') for p in owned if p['field']=='category' and p['accepted'] and not p['conflict_ids'] and not p['conditional']}
        category=next(iter(categories)) if len(categories)==1 else 'unknown'
        for p in owned:
            if p['field']=='dimension' and p['meaning']=='glass_thickness' and p['accepted'] and category!='glass':
                p['accepted']=False;p['normalized']=None;p['review_ids'].append(review.add('GLASS_DIMENSION_SCOPE','Glass thickness requires explicit glass category',p['id'],severity='ERROR'))
        catalog='catalog-'+grouping.digest([candidate['document_id'],candidate['code_raw'],signature])[:20] if signature else None
        tables['components'].append({'id':cid,'identity_id':candidate['identity_id'],'catalog_identity_id':catalog,
            'document_id':candidate['document_id'],'page_id':candidate['page_id'],'page_index':candidate['page_index'],
            'code_raw':candidate['code_raw'],'code_normalized':normalized_code,'scope':effective_scopes.get(candidate['identity_id']),'scope_evidence_ids':scope_evidence_ids.get(candidate['identity_id'],[]),
            'category':category,'status':'REVIEW' if issues or candidate.get('status')=='REVIEW' else 'SOURCE_SUPPORTED',
            'occurrence_ids':candidate['occurrence_ids'],'observed_view_ids':candidate['observed_in_view_ids'],
            'property_ids':[p['id'] for p in owned],'review_ids':sorted(set(issues))})
        for oid in candidate['occurrence_ids']:
            assigned_occurrences.add(oid);occ=occurrences.get(oid);member=next((x for x in candidate['memberships'] if x['occurrence_id']==oid),None)
            m_issues=[];view_id=None;point=None;detail_ids=[]
            if not occ or not member:
                m_issues.append(review.add('MEMBERSHIP_REFERENCE_MISSING','Component occurrence/membership reference is missing',cid,severity='ERROR'))
            elif candidate['code_raw']!=occ['raw_code']:
                m_issues.append(review.add('OCCURRENCE_CODE_MISMATCH','Component and occurrence codes differ',cid,severity='ERROR'))
            elif member['target_point'] is not None or member['view_id'] is not None:
                page=registry.page(occ['page_id']);bounds=json.loads(page['geometry'])['source_rect'] if page else None
                if occ['target_point'] is None or occ['source_view_id'] is None or member['target_point']!=occ['target_point'] or member['view_id']!=occ['source_view_id'] or occ['association_method']=='unresolved' or not occ['association_evidence'] or not valid_point(member['target_point'],bounds):
                    m_issues.append(review.add('NULL_MEMBERSHIP_UPGRADE','Physical membership is inconsistent with Stage 4; null targets stay unresolved',cid,severity='ERROR'))
                elif member['view_id'] not in views:
                    m_issues.append(review.add('MEMBERSHIP_VIEW_MISSING','Physical membership references an unknown view',cid,severity='ERROR'))
                elif not valid_point(member['target_point'],views[member['view_id']].get('structural_box',views[member['view_id']]['box'])) or views[member['view_id']]['page_id']!=occ['page_id']:
                    m_issues.append(review.add('MEMBERSHIP_COORDINATE','Physical target is outside its source view or on another page',cid,severity='ERROR'))
                else:
                    view_id=member['view_id'];point=member['target_point']
                    detail_ids=[d['id'] for d in graph['details'].values() if d['id'] in valid_details and view_id in d['view_ids'] and d.get('membership_role')=='primary' and graph['page_decisions'].get(d['page_id'])!='NO_DETAIL']
            if len(detail_ids)>1:
                m_issues.append(review.add('COMPETING_DETAILS','Competing provisional Detail ownership remains unresolved',cid));detail_ids=[]
            if member and not set(member.get('detail_context_ids',[]))<=set(graph['details']):
                m_issues.append(review.add('DETAIL_CONTEXT_REFERENCE','Membership context references an unknown Detail',cid,severity='ERROR'))
            status='SUPPORTED_VIEW_IN_PROVISIONAL_DETAIL' if detail_ids and not m_issues else 'SUPPORTED_VIEW_ONLY' if view_id and not m_issues else 'UNRESOLVED'
            if status=='UNRESOLVED':m_issues.append(review.add('UNRESOLVED_MEMBERSHIP','Component occurrence has no confirmed drawing/Detail target',cid))
            relationship('component_occurrence',cid,oid,status,data={'target_point':point,'view_id':view_id},issues=m_issues)
            for did in detail_ids or [None]:
                row={'detail_id':did,'component_id':cid,'occurrence_id':oid,'view_id':view_id,'association_status':status,
                     'target_point':point,'observed_view_ids':occ['observed_in_view_ids'] if occ else [],
                     'context_detail_ids':[d for d in member.get('detail_context_ids',[]) if d in graph['details'] and graph['page_decisions'].get(graph['details'][d]['page_id'])!='NO_DETAIL'] if member else [],'review_ids':sorted(set(m_issues))}
                row['id']='membership-'+grouping.digest([cid,oid,did])[:20];tables['detail_components'].append(row)
        if category=='profile':
            valid=[p for p in owned if p['accepted'] and not p['conflict_ids']]
            tables['profiles'].append({'component_id':cid,'code_raw':candidate['code_raw'],
                'dimension_property_ids':[p['id'] for p in valid if p['field']=='dimension'],
                'linear_weight_property_ids':[p['id'] for p in valid if p['field']=='linear_weight'],
                'inertia_property_ids':[p['id'] for p in valid if p['field']=='inertia'],
                'modulus_property_ids':[p['id'] for p in valid if p['field']=='section_modulus'],'review_ids':sorted(set(issues))})
    for oid,occ in occurrences.items():
        if oid not in assigned_occurrences:
            issues=[review.add('UNLINKED_OCCURRENCE','Readable/unreadable occurrence is preserved without a selected component',oid)]
            tables['detail_components'].append({'id':'membership-'+grouping.digest([oid,None])[:20],'detail_id':None,'component_id':None,
                'occurrence_id':oid,'view_id':None,'association_status':'UNRESOLVED','target_point':None,
                'observed_view_ids':occ['observed_in_view_ids'],'context_detail_ids':occ['detail_ids'],'review_ids':issues})
    for footnote in s6.get('footnotes',[]):
        eid,good=evidence_list([footnote['marker_evidence'],footnote['note_evidence']]+footnote['scope_evidence'],footnote['id'],['stage6:'+s6['run_id']])
        relationship('footnote_scope',footnote['id'],footnote.get('source_component_id'),
                     'SUPPORTED_CONDITION' if good and footnote['supported'] else 'REVIEW',evidence=eid,
                     data={'marker':footnote['marker'],'property_ids':footnote.get('observation_ids',[]),'explanation':footnote['explanation']})
    validated_paragraphs={}
    for paragraph in graph.get('paragraphs',{}).values():
        if not paragraph.get('accepted') or paragraph.get('scope')!='detail':continue
        for did in paragraph.get('detail_ids',[]):
            detail=graph['details'].get(did)
            if did not in valid_details:continue
            items=[paragraph.get('evidence'),paragraph.get('association_evidence')]
            if (not paragraph.get('applies_to_all') or paragraph.get('text_kind')!='explanation' or
                paragraph.get('basis') not in {'heading','annotation','structure','explicit_name','explicit_identifier'} or
                paragraph.get('review_reasons') or any(not e or e['page_id']!=detail['page_id'] or
                    e['document_id']!=detail['document_id'] for e in items)):
                review.add('DETAIL_PARAGRAPH_WITHHELD','Detail paragraph lacks validated same-page ownership',did)
                continue
            eids,good=evidence_list(items,paragraph['id'],graph['provenance'].get(paragraph['id'],[]))
            if good and components.paragraphs.quote_supported(paragraph['text'],[items[0]['quote']]):
                validated_paragraphs.setdefault(did,[]).append(dict(id=paragraph['id'],kind='page_paragraph',
                    document_id=detail['document_id'],page_id=detail['page_id'],page_index=detail['page_index'],
                    text=paragraph['text'],source_box=paragraph['source_box'],evidence_ids=eids))
    graph['validated_detail_paragraphs']=validated_paragraphs
    metadata={}
    for did in valid_details:
        detail=graph['details'][did];values={}
        for item in graph['scope']:
            if item.get('field') not in {'manufacturer','system'}:continue
            checked=copy.deepcopy(item);components.validate_scope([checked])
            if not checked.get('accepted'):continue
            items=item.get('evidence',[])
            if not items or any(e['page_id']!=detail['page_id'] or e['document_id']!=detail['document_id'] or
                    e['view_id'] not in detail['view_ids'] for e in items):continue
            eids,good=evidence_list(items,did,[])
            if good:values.setdefault(item['field'],[]).append(dict(value=item['value_raw'],evidence_ids=eids))
        metadata[did]=values
    graph['validated_detail_metadata']=metadata
    tables['properties']=list(properties.values());tables['evidence']=list(registry.rows.values())
    return tables


def re_finish(value):
    import re
    return bool(re.fullmatch(r'(?:(?:black|white|clear|silver)\s+)?(?:anodi[sz]ed|powder[ -]?coated|painted|polished)(?:[*!])?',value,re.I))


def page_ranges(values):
    if values is None:return None
    result=[]
    for value in sorted(set(values)):
        if result and value==result[-1][1]+1:result[-1][1]=value
        else:result.append([value,value])
    return result


def coverage_summary(source):
    return {'page_index_base':0,'document_page_count':source['document_page_count'],
            'indexed_pages_searched_count':len(source['indexed_pages_searched']),
            'indexed_page_ranges':page_ranges(source['indexed_pages_searched']),
            'unindexed_page_count':len(source['unindexed_pages']) if source['unindexed_pages'] is not None else None,
            'unindexed_page_ranges':page_ranges(source['unindexed_pages']),
            'visual_unsearched_page_count':len(source['visual_unsearched_pages']),
            'visual_unsearched_page_ranges':page_ranges(source['visual_unsearched_pages']),
            'detail_reference':'Full per-page/window coverage is in the enrichment manifest'}


def cell(value):
    if isinstance(value,(list,dict,bool)):return dumps(value)
    if value is None:return ''
    return str(value)


def write_csv(path,fields,rows):
    expected=[{name:cell(row.get(name)) for name in fields} for row in rows]
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,quoting=csv.QUOTE_ALL);writer.writeheader();writer.writerows(expected)
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);actual=list(reader)
        if reader.fieldnames!=fields or actual!=expected:raise ValueError('CSV round-trip failed')
    for raw,row in zip(actual,rows):
        for key,value in row.items():
            if key in fields and isinstance(value,(list,dict,bool)) and json.loads(raw[key])!=value:
                raise ValueError('Structured CSV round-trip failed')


def reusable(row):
    try:return all(grouping.file_digest(path)==sha for path,sha in json.loads(row['artifacts']).items())
    except (OSError,ValueError,TypeError):return False


def run(stage6,output,*,index_path,force=False):
    review=Reviews();s6=read_saved(stage6,6,review)
    if s6 is None:raise ValueError('Explicit saved Stage 6 manifest is required')
    graph=load_graph(s6,review);db=None;signature=None
    try:
        if Path(index_path).is_file():
            db=sqlite3.connect(index_path);db.row_factory=sqlite3.Row;db.execute('PRAGMA foreign_keys=ON')
            try:
                signature=enrichment.index_signature(db,[s['document_id'] for s in s6['sources']])
                for violation in db.execute('PRAGMA foreign_key_check'):
                    review.add('SQLITE_REFERENTIAL_INTEGRITY','Existing SQLite foreign key check failed',str(violation[0]),severity='ERROR')
            except sqlite3.Error:
                review.add('INDEX_INVALID','Existing database is not a usable source index','index',severity='ERROR');db.close();db=None
        else:review.add('INDEX_UNAVAILABLE','Source index is missing; evidence validation is partial','index',severity='ERROR')
        output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
        settings={'logic_sha256':code_fingerprint(),'version':VERSION,'summary_version':'2','csv_schema_hash':grouping.digest(TABLES),'encoding':'utf-8-sig','quoting':'QUOTE_ALL','structured_cells':'compact JSON'}
        hashes=[]
        for source in s6['sources']:
            for key in ('stage3_path','stage4_path','stage5_path'):
                path=Path(source[key]);hashes.append([str(path),grouping.file_digest(path) if path.is_file() else None])
        key=grouping.digest([settings,grouping.file_digest(stage6),hashes,signature])
        if db:
            db.execute('CREATE TABLE IF NOT EXISTS export_runs(id TEXT PRIMARY KEY,started TEXT DEFAULT CURRENT_TIMESTAMP,finished TEXT,cache_key TEXT,state TEXT,result_path TEXT,artifacts TEXT,summary TEXT)');db.commit()
            prior=db.execute("SELECT * FROM export_runs WHERE cache_key=? AND state IN ('complete','partial') ORDER BY rowid DESC LIMIT 1",(key,)).fetchone()
            if not force and prior and reusable(prior):
                result=json.loads(Path(prior['result_path']).read_text());return {**result,'reused':True}
        run_id=uuid.uuid4().hex
        if db:
            with db:db.execute("INSERT INTO export_runs(id,cache_key,state) VALUES(?,?,'processing')",(run_id,key))
        if s6['execution_status']!='COMPLETE':review.add('ENRICHMENT_INCOMPLETE','Stage 6 processing is failed, unfinished or partial',s6['run_id'])
        if s6['source_coverage_status']!='COMPLETE':review.add('UNSEARCHED_COVERAGE','Unindexed or visually unsearched source coverage remains',s6['run_id'])
        for failure in s6['failures']:review.add('ENRICHMENT_'+failure['code'],failure['reason'],failure.get('view_id',s6['run_id']))
        for hit in s6['retrieval_hits']:review.add('RETRIEVAL_CANDIDATE','Native code match is a retrieval candidate, not property or identity proof',hit['id'])
        tables=build_tables(s6,graph,db,review)
        leader_inventory=leaders.validate_saved(graph['leader_observations'],graph['details'],graph['views'],review)
        for did in graph['details']:
            if did not in leader_inventory:review.add('MISSING_LEADER_INVENTORY','No verified component leader inventory; leader diagnostic withheld, CSV component count is independent',did)
        tables['review']=list(review.rows.values())
        partial=bool(review.errors()) or graph['upstream_incomplete'] or s6['execution_status']!='COMPLETE' or s6['source_coverage_status']!='COMPLETE'
        state='PARTIAL' if partial else 'COMPLETE'
        destination=output/('partial' if partial else 'complete')/run_id
        staging=output/('.'+run_id+'.tmp');staging.mkdir()
        artifacts={name:str(destination/(name+'.csv')) for name in TABLES}
        artifacts['leader_observations']=str(destination/'leader-observations.json')
        artifacts['detail_paragraphs']=str(destination/'detail-paragraphs.json')
        (staging/'detail-paragraphs.json').write_text(dumps(graph.get('validated_detail_paragraphs',{}))+'\n')
        artifacts['detail_metadata']=str(destination/'detail-metadata.json')
        (staging/'detail-metadata.json').write_text(dumps(graph.get('validated_detail_metadata',{}))+'\n')
        artifacts['occurrence_audit']=str(destination/'occurrence-audit.json')
        traceability.write(staging/'occurrence-audit.json',traceability.audit(
            list(graph['occurrences'].values()),list(graph['candidates'].values()),tables['detail_components']))
        (staging/'leader-observations.json').write_text(dumps(leader_inventory)+'\n')
        for name,fields in TABLES.items():write_csv(staging/(name+'.csv'),fields,tables[name])
        detail_paths=[]
        export_pages=sorted({d['page_index']+1 for d in tables['details']})
        progress.plan(7,export_pages)
        for d in tables['details']:
            progress.operation(7,d['page_index']+1,'Saving '+d['id']+' image.png and result.json')
            members=[x for x in tables['detail_components'] if x['detail_id']==d['id']]
            component_ids=sorted({x['component_id'] for x in members if x['component_id']})
            property_ids=sorted({p['id'] for p in tables['properties'] if set(p['applies_to_component_ids'])&set(component_ids)})
            relevant=[p for p in tables['properties'] if p['id'] in property_ids]
            evidence_ids=sorted({e for p in relevant for e in p['evidence_ids']+p['condition_evidence_ids']})
            context_occurrences=sorted({o['id'] for o in graph['occurrences'].values() if set(o['observed_in_view_ids'])&set(d['view_ids'])})
            result={'stage':7,'detail_id':d['id'],'status':'PROVISIONAL','completeness':d['completeness'],
                    'source_views':[{'id':v,'crop_path':graph['views'][v].get('crop_path'),'source_box':graph['views'][v]['box'],
                                     'crop_geometry':graph['views'][v].get('crop_geometry')} for v in d['view_ids'] if v in graph['views']],
                    'provisional_composites':d['composites'],'component_ids':component_ids,'property_ids':property_ids,'evidence_ids':evidence_ids,
                    'context_occurrence_ids':context_occurrences,'context_does_not_establish_membership':True,
                    'review_ids':sorted(set(d['review_ids'])|{r for x in members for r in x['review_ids']}),
                    'tables':{name:'../../'+name+'.csv' for name in ('components','properties','evidence','review','relationships')}}
            folder=staging/'details'/d['id'];folder.mkdir(parents=True)
            detail_image(graph['details'][d['id']]['composites'], folder/'image.png')
            inventory=leader_inventory.get(d['id']) or dict(contract=leaders.CONTRACT,detail_id=d['id'],document_id=d['document_id'],page_id=d['page_id'],page_index=d['page_index'],complete=False,coverage='incomplete',observations=[],issues=['No Stage 4 leader inventory available'])
            leaders.write(folder,dict(inventory),graph['details'][d['id']],list(graph['views'].values()),published_folder=destination/'details'/d['id'])
            path=folder/'result.json';path.write_text(dumps(result)+'\n');assert json.loads(path.read_text())==result
            detail_paths.append(str(destination/'details'/d['id']/'result.json'))
        issue_lines=['# Validation and deferred issues','',f'Export: {state}. Execution, coverage and acceptance are distinct.','']
        issue_lines += ['- '+r['severity']+' '+r['code']+': '+r['message']+' ['+r['id']+'] '+
                        json.dumps(issue_locations(r,graph),ensure_ascii=False) for r in tables['review']]
        issue_lines += ['', 'Full-document extraction accuracy is not established by this run; consult the per-stage execution and source-review evidence.',
                        'CSV is UTF-8 with BOM; import code/text columns as Text in Excel to retain leading zeros. Structured cells are compact JSON.']
        (staging/'issues.md').write_text('\n'.join(issue_lines)+'\n')
        summary={'stage':7,'export_version':VERSION,'run_id':run_id,'export_status':state,
                 'verified_component_leaders':sum(r.get('observed_accepted_leaders',0) for r in leader_inventory.values()),
                 'leader_count_contract':leaders.CONTRACT,
                  'profile_count_contract':leaders.COMPONENT_COUNT_CONTRACT,
                 'execution_status':'PARTIAL' if review.errors() or graph['upstream_incomplete'] or s6['execution_status']!='COMPLETE' else 'COMPLETE',
                 'source_coverage_status':s6['source_coverage_status'],'acceptance_status':'REVIEW' if tables['review'] else 'SOURCE_SUPPORTED',
                 'complete':not partial,'selected_sources':[{'document_id':s['document_id'],'source_path':s['source_path'],'stage5_path':s['stage5_path']} for s in s6['sources']],
                 'coverage':coverage_summary(s6['coverage']),'additional_pages_indexed':s6['additional_pages_indexed'],
                 'reference_page_index_selection':s6.get('selection_plan',{}).get('index_pages',[]) if s6.get('selection_plan') else [],
                 'counts':{name:len(rows) for name,rows in tables.items()},'validation_errors':len(review.errors()),
                 'failures':s6['failures'],'model_requests':s6['model_requests'],'supported_assignments':len(s6['assignments']),
                 'artifacts':{**artifacts,'detail_results':detail_paths,'issues':str(destination/'issues.md'),
                              'summary':str(destination/'run-summary.json'),'enrichment':str(Path(stage6).resolve())},
                 'settings':settings,'reused':False}
        (staging/'run-summary.json').write_text(dumps(summary)+'\n');assert json.loads((staging/'run-summary.json').read_text())==summary
        destination.parent.mkdir(parents=True,exist_ok=True);staging.rename(destination)
        for page in export_pages:
            progress.done(7,page,ok=not review.errors(),reason='Published Detail files; review status remains in evidence')
        files={str(p):grouping.file_digest(p) for p in destination.rglob('*') if p.is_file()}
        if db:
            with db:db.execute('UPDATE export_runs SET finished=CURRENT_TIMESTAMP,state=?,result_path=?,artifacts=?,summary=? WHERE id=?',
                               ('partial' if partial else 'complete',str(destination/'run-summary.json'),dumps(files),dumps(summary),run_id))
        if not partial:
            pointer=output/('.latest-complete-'+run_id+'.tmp');pointer.write_text(dumps({'run_id':run_id,'summary':str(destination/'run-summary.json')}));pointer.replace(output/'latest-complete.json')
        return summary
    except Exception as exc:
        # A write/round-trip failure never replaces an earlier complete export.
        failed_id=locals().get('run_id',uuid.uuid4().hex)
        failed=Path(output).resolve()/'partial'/(failed_id+'-failed')
        failed.parent.mkdir(parents=True,exist_ok=True)
        pending=locals().get('staging')
        if pending and pending.is_dir():pending.rename(failed)
        else:failed.mkdir(exist_ok=True)
        summary={'stage':7,'export_version':VERSION,'run_id':failed_id,'export_status':'FAILED','execution_status':'FAILED',
                 'source_coverage_status':s6.get('source_coverage_status','UNAVAILABLE'),'acceptance_status':'REVIEW','complete':False,
                 'failures':[{'code':'EXPORT_FAILED','reason':type(exc).__name__}],
                 'artifacts':{'summary':str(failed/'run-summary.json')},'reused':False}
        (failed/'run-summary.json').write_text(dumps(summary)+'\n')
        if db:
            try:
                with db:db.execute("UPDATE export_runs SET finished=CURRENT_TIMESTAMP,state='failed',result_path=?,summary=? WHERE id=?",(str(failed/'run-summary.json'),dumps(summary),failed_id))
            except sqlite3.Error:pass
        return summary
    finally:
        if db:db.close()


def main(argv=None):
    parser=argparse.ArgumentParser(prog='as-ocr export',description=__doc__)
    parser.add_argument('stage6',type=Path);parser.add_argument('--index',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--force',action='store_true');args=parser.parse_args(argv)
    try:result=run(args.stage6,args.output,index_path=args.index,force=args.force)
    except KeyboardInterrupt:parser.exit(130,'Interrupted; previous complete exports preserved.\n')
    except Exception as exc:parser.exit(2,'Stage 7 failed: '+type(exc).__name__+'; previous complete exports preserved.\n')
    print(json.dumps({k:result[k] for k in ('run_id','export_status','execution_status','source_coverage_status','acceptance_status','complete','reused')} | {'counts':result.get('counts',{}),'validation_errors':result.get('validation_errors',0),'summary':result['artifacts']['summary']}));return 0 if result['complete'] else 1

"""Five-column source-callout projection; no inferred names or legacy counts."""
from __future__ import annotations

from pathlib import Path
import re
import pymupdf as fitz

from . import leaders, provenance
from .geometry import area, intersection

COLUMNS=('detail_name','profile_count','profile_id','component_name','source_page')


def detail_name(pdf, ordinal):
    """Filename stem + one-based Detail ordinal across selected PDF pages."""
    if type(ordinal) is not int or ordinal < 1:
        raise ValueError('Detail ordinal must be a positive integer')
    return f'{Path(pdf).stem}_{ordinal:02d}'


def literal_name(occurrence,candidates,observations,view):
    """Exact source name in THIS associated callout; never transfer enrichment."""
    from . import local_legend
    legend_name=local_legend.name(occurrence)
    if legend_name and provenance.supported_product_reading(occurrence,[view['kind']]):
        return legend_name,[]
    if (occurrence.get('category')=='component_name' and occurrence.get('reading_state')=='exact' and not occurrence.get('reading_clipped')
        and not provenance.reading_review_reasons(occurrence)):
        return occurrence.get('raw_code') or '',[]
    ids={c['id'] for c in candidates if occurrence['id'] in c.get('occurrence_ids',[])
        and c.get('page_id')==occurrence['page_id'] and c.get('document_id')==occurrence['document_id']}
    choices=[];eids=[]
    for o in observations:
        if o.get('candidate_id') not in ids or o.get('field')!='component_name' or not o.get('accepted'):
            continue
        if o.get('conditions') or o.get('review_reasons') or o.get('conflicts_with'):continue
        relation=o.get('relation',{});code=relation.get('code_evidence') or {};bounds=relation.get('bounds')
        if relation.get('kind')!='code_label' or not bounds or code.get('quote')!=occurrence.get('raw_code'):continue
        b=code.get('box');text=occurrence.get('text_box')
        if not b or not text or intersection(b,text)<.5*min(area(b),area(text)):continue
        if not relation.get('evidence') or not any(e.get('method')!='native_text' for e in relation['evidence']):continue
        value=o.get('value_raw')
        if not value or value==occurrence.get('raw_code'):continue
        good=[]
        for e in [code,*o.get('evidence',[]),*relation['evidence']]:
            if (e.get('page_id')!=occurrence['page_id'] or e.get('document_id')!=occurrence['document_id']
                or e.get('view_id')!=view['id'] or e.get('review_reasons')):break
            box=e.get('box')
            if not box or intersection(bounds,box)<area(box)-1e-5:break
            good.append(e)
        else:
            # Require the name itself as a literal quotation, not a paraphrase
            # that happens to contain a familiar engineering type.
            if any(e.get('quote')==value for e in o.get('evidence',[])):
                choices.append(value);eids.append(o['id'])
    return (choices[0],eids) if choices and len(set(choices))==1 else ('',[])


def project(summary_path,Snapshot,index,structured,unsafe,*,details_only=False):
    snapshot=Snapshot();summary=snapshot.json(Path(summary_path).resolve())
    if summary.get('stage')!=7 or summary.get('export_status') not in {'COMPLETE','PARTIAL'}:
        raise ValueError('Select a published COMPLETE or PARTIAL Stage 7 snapshot')
    sources=summary['selected_sources'];catalogs={s['document_id'] for s in sources}
    names={Path(s['source_path']).name for s in sources}
    if len(catalogs)!=1 or len(names)!=1:raise ValueError('Select exactly one source catalog')
    catalog=next(iter(catalogs));filename=next(iter(names))
    if not re.fullmatch('[0-9a-f]{64}',catalog):raise ValueError('Invalid source SHA-256')
    artifacts=summary['artifacts'];details=index(snapshot.table(artifacts['details']))
    # Retain linked audit tables in the snapshot even though public rows are
    # rebuilt exclusively from original same-page callouts.
    members=[];reviews=[]
    for name in ('detail_components','components','properties','evidence','review','relationships'):
        if artifacts.get(name):snapshot.table(artifacts[name])
        if name=='detail_components':members=snapshot.table(artifacts[name])
        if name=='review':reviews=snapshot.table(artifacts[name])
    if any(m.get('detail_id') and m['detail_id'] not in details for m in members):
        raise ValueError('Dangling Detail membership')
    if any(r.get('severity')=='ERROR' and set(structured(r.get('subject_ids','[]')))&set(details) for r in reviews):
        raise ValueError('Invalid Detail source page, geometry or evidence ownership; publication blocked')
    s6=snapshot.json(artifacts['enrichment']) if artifacts.get('enrichment') else {}
    occurrences={};views={};source_details={};candidates=[];observations=[];missing_source=False
    for source in s6.get('sources',[]):
        if source['document_id']!=catalog:raise ValueError('Foreign catalog provenance')
        for n in (3,4,5):
            if not source.get(f'stage{n}_path'):
                missing_source=True;continue
            if not Path(source[f'stage{n}_path']).is_file():
                missing_source=True;continue
            saved=snapshot.json(source[f'stage{n}_path'],source.get(f'stage{n}_sha256'))
            if saved.get('stage')!=n:raise ValueError('Invalid upstream stage')
            if n==3:
                for v in saved['views']:
                    if v['document_id']!=catalog:raise ValueError('Foreign source view')
                    views[v['id']]=v
                for d in saved['details']:source_details[d['id']]=d
            elif n==4:
                for o in saved['occurrences']:
                    if o['id'] in occurrences and occurrences[o['id']]!=o:raise ValueError('Conflicting source occurrence')
                    occurrences[o['id']]=o
            else:
                candidates+=saved.get('candidates',[]);observations+=saved.get('observations',[])
    inventories=snapshot.json(artifacts['leader_observations']) if artifacts.get('leader_observations') else {}
    rows=[];traces=[]
    source_order={did:n for n,did in enumerate(source_details)}
    partial=(missing_source or not summary.get('complete') or summary['export_status']!='COMPLETE'
        or summary.get('execution_status')!='COMPLETE' or summary.get('source_coverage_status')!='COMPLETE'
        or bool(summary.get('validation_errors')))
    for ordinal,d in enumerate(sorted(details.values(),key=lambda d:(int(d['page_index']),source_order.get(d['id'],0),d['id'])),1):
        if d['document_id']!=catalog or not re.fullmatch(r'0|[1-9][0-9]*',str(d['page_index'])):
            raise ValueError('Invalid Detail source page or identity')
        did=d['id'];page=int(d['page_index'])+1
        vids=set(structured(d['view_ids']));record=inventories.get(did)
        original_detail=source_details.get(did)
        if original_detail and (original_detail['document_id']!=catalog or original_detail['page_id']!=d['page_id']
            or original_detail['page_index']!=page-1 or set(original_detail['view_ids'])!=vids):
            raise ValueError('Detail source page or membership differs from original grouping evidence')
        for vid in vids:
            view=views.get(vid)
            if view and (view['document_id']!=catalog or view['page_id']!=d['page_id'] or view.get('page_index',page-1)!=page-1):
                raise ValueError('View belongs to another original source page')
            if view and view.get('crop_path'):snapshot.bytes(view['crop_path'],view.get('crop_sha256'))
        if record and (record.get('document_id')!=catalog or record.get('page_id')!=d['page_id']
            or record.get('page_index')!=page-1):raise ValueError('Foreign leader inventory')
        # Re-verify immutable source pixels and ownership at the final boundary.
        if record:
            from .exports import Reviews
            review=Reviews();validated=leaders.validate_saved([record],source_details,views,review)
            if review.errors():raise ValueError('Invalid leader source evidence at publication')
            record=validated.get(did)
            for c in (record or {}).get('view_coverage',[]):snapshot.bytes(c['source_image'],c['source_image_sha256'])
            for o in (record or {}).get('observations',[]):
                if o.get('accepted'):snapshot.bytes(o['source_image'],o['source_image_sha256'])
        leader_count,leader_status=leaders.count_state(record)
        if leader_status not in {'COMPLETE','VERIFIED_ZERO'}:partial=True
        name=detail_name(filename,ordinal)
        detail_rows=[]
        for o in occurrences.values():
            vid=o.get('source_view_id');view=views.get(vid)
            if (vid not in vids or not view or o.get('document_id')!=catalog or o.get('page_id')!=d['page_id']
                or o.get('page_index',page-1)!=page-1
                or view.get('page_id')!=d['page_id'] or not o.get('target_point')
                or o.get('association_method')=='unresolved' or not o.get('association_evidence')):continue
            metadata_boxes=[v['box'] for v in views.values() if v['kind']=='document_metadata'
                and v['page_id']==d['page_id'] and v.get('box')]
            if provenance.metadata_source(o,[view['kind']],metadata_boxes):continue
            target=o['target_point']
            owners=[v['id'] for v in views.values() if v['page_id']==d['page_id'] and v['kind']=='assembly_candidate'
                and fitz.Rect(v['structural_box']).contains(fitz.Point(*target))]
            if owners!=[vid]:continue
            if o.get('category') not in {'product_code','uncertain_code','component_name'}:continue
            pid=o.get('raw_code') if provenance.supported_product_reading(o,[view['kind']],metadata_boxes) else ''
            cname,pids=literal_name(o,candidates,observations,view)
            detail_rows.append((o['id'],pid,cname,pids))
        count,component_identity=leaders.extracted_component_count(detail_rows,occurrences)
        status='EXTRACTED_COMPONENTS' if component_identity else 'NO_SUPPORTED_COMPONENTS'
        if not component_identity:count=''
        if not detail_rows:detail_rows=[('', '', '', [])]
        for oid,pid,cname,pids in sorted(detail_rows):
            row=dict(detail_name=name,profile_count=count,profile_id=pid,component_name=cname,source_page=str(page))
            reasons=[] if component_identity else ['PROFILE_COUNT_'+status]
            if leader_status not in {'COMPLETE','VERIFIED_ZERO'}:reasons.append('LEADER_REVIEW_'+leader_status)
            if not cname:reasons.append('NO_VERIFIED_NAME_IN_ASSOCIATED_SOURCE_CALLOUT')
            if not pid:reasons.append('NO_EXACT_SUPPORTED_PROFILE_CODE')
            if not pid and not cname:reasons.append('SOURCE_READING_WITHHELD');partial=True
            trace=dict(row_key=[catalog,did,oid],csv_row_number=len(rows)+2,component_ids=[],membership_ids=[],
                property_ids=pids,evidence_ids=[],review_ids=[],reasons=reasons,withheld={},
                detail_status=d['status'],completeness=d['completeness'],profile_count_status=status,
                profile_count_verified=bool(component_identity),profile_count_contract=leaders.COMPONENT_COUNT_CONTRACT,
                component_occurrence_id=component_identity.get(oid),
                leader_count=leader_count,leader_count_status=leader_status)
            trace['membership_ids']=[m['id'] for m in members if m.get('detail_id')==did and m.get('occurrence_id')==oid]
            trace.update(catalog_name=filename,page=str(page),system_name='',review_status='REVIEW',
                association_status='SUPPORTED' if oid else 'NO_CONFIRMED_COMPONENTS')
            trace['source_structural_box']=views.get(occurrences.get(oid,{}).get('source_view_id'),{}).get('structural_box')
            if occurrences.get(oid,{}).get('legend_resolution'):
                trace['legend_resolution']=occurrences[oid]['legend_resolution']
            for key,value in list(row.items()):
                if unsafe(value):
                    row[key]='';trace['withheld'][key]=value;trace['reasons'].append('UNSAFE_CSV_'+key);partial=True
            rows.append(row);traces.append(trace)
    snapshot.verify()
    unassigned=[m for m in members if not m.get('detail_id') or m.get('association_status')!='SUPPORTED_VIEW_IN_PROVISIONAL_DETAIL']
    return summary,catalog,filename,rows,traces,snapshot,unassigned,partial

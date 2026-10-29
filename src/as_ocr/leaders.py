"""Independent leader diagnostics and extracted associated component counting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw

from . import line_evidence as lines
from .geometry import area, distance, intersection

VERSION='4'
CONTRACT='distinct_verified_component_leaders_v1'
COMPONENT_COUNT_CONTRACT='distinct_extracted_associated_components_v1'
EXCLUDED={'dimension','quantity','page_number','figure_reference','item_marker'}


def extracted_component_count(rows,occurrences):
    """Count supported associated occurrences, independently of line validation.

    Upstream already fuses repeated reads of one label. A code and a separate
    name on the same proposed physical leader represent one component. Identical
    codes on distinct leaders remain distinct; proximity alone never merges them.
    """
    groups=[];identity={}
    for oid,code,name,_ in rows:
        if not oid or not (code or name):continue
        occurrence=occurrences[oid];path=occurrence.get('leader_path')
        match=None
        for group in groups:
            prior=occurrences[group[0]]
            codes={r[1] for r in group[1] if r[1]}
            if code and codes and code not in codes:continue
            if (occurrence.get('source_view_id')==prior.get('source_view_id') and
                lines.same_line(path,prior.get('leader_path'))):
                match=group;break
        if match is None:
            match=[oid,[]];groups.append(match)
        match[1].append((oid,code,name))
        identity[oid]=match[0]
    return str(len(groups)),identity


def identifier(page,path):
    forward=[[round(x,4) for x in p] for p in path]
    shape=min(forward,list(reversed(forward)))
    return 'leader-'+hashlib.sha256(json.dumps([page,shape]).encode()).hexdigest()[:20]


def context_census_issues(view,observations):
    """Every already source-verified Detection leader must be accounted for.

    This is a coverage cross-check, never a count taken from Detection totals.
    Additional independently verified Callout leaders are retained normally.
    """
    crop=view.get('crop_completeness') or view.get('source_candidate',{}).get('crop_completeness') or {}
    if crop.get('status')!='COMPLETE':return ['Source annotation context is not complete']
    proofs=[p for p in crop.get('annotation_proofs',[]) if p.get('ownership_verified')]
    if not proofs:return ['No source-verified Detection annotation census is available']
    accepted=[o for o in observations if o.get('accepted') and o.get('view_id')==view['id']]
    issues=[]
    for proof in proofs:
        path=proof.get('annotation',{}).get('leader_points',[])
        matches=[o for o in accepted if lines.same_line(path,o['path'],.8)]
        if len(matches)!=1:
            issues.append('A source-verified Detection component leader is missing or ambiguous in the Callout census')
    return list(dict.fromkeys(issues))


def inventory(detail,views,view_results,vectors=()):
    """Build from RAW per-view observations, before text-occurrence fusion."""
    selected={v['id']:v for v in views if v['id'] in detail['view_ids']}
    outcomes={r['view_id']:r for r in view_results};observations=[];issues=[];coverage=[]
    complete=bool(selected) and detail.get('membership_role','primary')=='primary'
    for vid,view in selected.items():
        result=outcomes.get(vid)
        if not result:
            complete=False;issues.append('Source view not examined: '+vid);continue
        if (result.get('leader_coverage')!='complete' or not result.get('leader_coverage_evidence')
            or result.get('leader_audit',{}).get('status')!='COMPLETE'):
            complete=False;issues.append('Leader coverage is incomplete or unavailable: '+vid)
        crop=(view.get('crop_completeness') or view.get('source_candidate',{}).get('crop_completeness') or {})
        if crop.get('status')!='COMPLETE':
            complete=False;issues.append('Full annotation context is not deterministically complete: '+vid)
        source=Path(result['source_image'])
        with Image.open(source) as original:image=original.convert('RGB')
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        coverage.append(dict(view_id=vid,leader_coverage=result.get('leader_coverage'),
            independent_audit_status=result.get('leader_audit',{}).get('status'),
            evidence=result.get('leader_coverage_evidence',[]),source_image=str(source),
            source_image_sha256=source_hash,source_mapping=result['mapping']))
        for record in result.get('records',[]):
            proposal=record.get('proposed_geometry',{})
            path=record.get('leader_path') or (record.get('source_trace') or {}).get('path') or proposal.get('leader_path') or []
            kind=record.get('leader_kind','uncertain')
            if not path:
                if record.get('category') in EXCLUDED or record.get('association_method')=='qwen_explicit_location':continue
                if proposal.get('leader_state')=='missing':continue
                complete=False;issues.append('Candidate component leader has no traceable path');continue
            item=dict(id=identifier(detail['page_index'],path),view_id=vid,path=path,
                vector_evidence=record.get('vector_evidence') or (record.get('source_trace') or {}).get('vectors',[]),
                label_id=record.get('label_id') or 'label-'+hashlib.sha256(json.dumps(
                    [detail['page_id'],record.get('text_box'),record.get('raw_code')]).encode()).hexdigest()[:20],
                source_page=detail['page_index']+1,raw_text=record.get('raw_code'),
                label_box=record.get('text_box'),target_point=record.get('target_point'),
                category=record.get('category'),leader_kind=kind,accepted=False,status='REVIEW',reasons=[],
                detection_passes=[record.get('leader_audit_pass','unspecified')],
                source_image=str(source),source_image_sha256=source_hash,source_mapping=result['mapping'],
                model_evidence=record.get('association_evidence',[]))
            contradiction=(record.get('category') in EXCLUDED and kind=='component')
            if contradiction:item['reasons'].append('Component-leader and non-component annotation classifications conflict')
            if not contradiction and (record.get('category') in EXCLUDED or kind in {'dimension','construction','hatching','irrelevant'}):
                item.update(status='EXCLUDED',reasons=['Non-component line']);observations.append(item);continue
            if kind!='component':item['reasons'].append('Component-leader classification is uncertain')
            if record.get('category') not in {'product_code','component_name'} or not record.get('raw_code'):
                item['reasons'].append('No readable component name or identifier on this leader')
            if proposal.get('leader_state')!='clear':item['reasons'].append('Leader is crossed, branched, clipped, missing or uncertain')
            if record.get('source_view_id')!=vid or record.get('target_point') is None or not item['model_evidence']:
                item['reasons'].append('Visual annotation-to-component ownership is unresolved')
            if not lines.valid_path(path):item['reasons'].append('Invalid source polyline')
            else:
                if not item['label_box'] or distance(item['label_box'],[*path[0],*path[0]])>lines.label_start_tolerance(item['label_box']):
                    item['reasons'].append('Polyline does not begin at its source label')
                owners=[v['id'] for v in views if v['kind']=='assembly_candidate' and
                    fitz.Rect(v['structural_box']).contains(fitz.Point(*path[-1]))]
                if owners!=[vid]:item['reasons'].append('Target has ambiguous or unsupported assembly ownership')
                item['source_verification']=lines.verify(path,vectors,image,result['mapping'])
                if not item['source_verification']['verified']:item['reasons'].append(item['source_verification']['reason'])
                arrowheads=lines.arrowhead_segments(vectors,path[-1])
                for ordinal,number,a,b in lines.vector_segments(vectors):
                    if (ordinal,number) in arrowheads:continue
                    if any(lines.same_line([c,z],[a,b],.8) for c,z in zip(path,path[1:])):continue
                    if any(lines.crossing_outside_target(c,z,a,b,view['structural_box']) for c,z in zip(path,path[1:])):
                        item['reasons'].append('Indexed source line crosses proposed leader; ownership requires REVIEW');break
            observations.append(item)
    distinct=[]
    for item in observations:
        if item['status']=='EXCLUDED':distinct.append(item);continue
        duplicate=next((old for old in distinct if old['status']!='EXCLUDED' and lines.same_line(old['path'],item['path'])),None)
        if duplicate:
            duplicate.setdefault('duplicate_detections',[]).append(item)
            duplicate['detection_passes']=sorted(set(duplicate['detection_passes']+item['detection_passes']))
            # Repeated observations cannot resolve conflicting ownership.
            if (duplicate['target_point'] is None or item['target_point'] is None or
                lines.point_distance(duplicate['target_point'],item['path'][-2],item['path'][-1])>.15 or
                duplicate['leader_kind']!=item['leader_kind'] or duplicate['view_id']!=item['view_id']):
                duplicate['reasons'].append('Duplicate physical-line observations disagree')
            duplicate['reasons']=list(dict.fromkeys(duplicate['reasons']+item['reasons']))
        else:distinct.append(item)
    for i,left in enumerate(distinct):
        if left['status']=='EXCLUDED' or not lines.valid_path(left['path']):continue
        for right in distinct[i+1:]:
            shared_label=(left.get('label_box') if left.get('label_box') and right.get('label_box')
                          and left.get('raw_text')==right.get('raw_text')
                          and intersection(left['label_box'],right['label_box'])/max(area(left['label_box']),area(right['label_box']))>=.8 else None)
            if (shared_label and left.get('vector_evidence') and right.get('vector_evidence')
                and lines.shared_label_branch(left['path'],right['path'],shared_label)):continue
            if right['status']!='EXCLUDED' and lines.valid_path(right['path']) and lines.conflict(left['path'],right['path'],shared_label):
                for item in (left,right):item['reasons'].append('Ambiguous crossing or shared branch; inspect original page')
            elif (right['status']!='EXCLUDED' and lines.same_line(left['path'],right['path'],.8)
                and not lines.same_line(left['path'],right['path'])):
                for item in (left,right):item['reasons'].append('Near-coincident traces may describe one physical line; deduplication requires REVIEW')
    accepted=0
    for item in distinct:
        item['reasons']=list(dict.fromkeys(item['reasons']))
        if item['status']!='EXCLUDED':
            if not {'primary','independent_audit'}<=set(item['detection_passes']):
                complete=False
                issues.append('Independent leader census does not confirm this physical line')
            item['accepted']=not item['reasons'];item['status']='ACCEPTED' if item['accepted'] else 'REVIEW'
            if item['accepted']:accepted+=1;item['number']=accepted
            else:complete=False
    for view in selected.values():
        missing=context_census_issues(view,distinct)
        if missing:complete=False;issues+=missing
    return dict(contract=CONTRACT,document_id=detail['document_id'],page_id=detail['page_id'],
        page_index=detail['page_index'],detail_id=detail['id'],coverage='complete' if complete else 'incomplete',
        required_view_ids=list(detail['view_ids']),view_coverage=coverage,
        complete=complete,observations=distinct,issues=list(dict.fromkeys(issues)),
        no_leader_evidence=[r.get('leader_coverage_evidence',[]) for r in view_results if r['view_id'] in selected],
        verified_count=accepted if complete else None,observed_accepted_leaders=accepted)


def count_state(record):
    """Return the extracted source-supported count, even for incomplete coverage."""
    if not record or record.get('contract')!=CONTRACT:return '', 'MISSING_LEADER_EVIDENCE'
    if not record.get('view_coverage'):return '', 'MISSING_LEADER_EVIDENCE'
    accepted=[o for o in record.get('observations',[]) if o.get('accepted')]
    if any(o.get('leader_kind')!='component' or o.get('category') not in {'product_code','component_name'}
           or not o.get('raw_text') or not lines.valid_path(o.get('path'))
           or not o.get('source_verification',{}).get('verified') for o in accepted):
        return '', 'INVALID_LEADER_EVIDENCE'
    if any(lines.same_line(a['path'],b['path']) for i,a in enumerate(accepted) for b in accepted[i+1:]):
        return '', 'DUPLICATE_LEADER_EVIDENCE'
    if [o.get('number') for o in accepted]!=list(range(1,len(accepted)+1)):
        return '', 'INVALID_LEADER_NUMBERING'
    coverage=record.get('view_coverage',[])
    complete=(record.get('complete') and record.get('coverage')=='complete' and not record.get('issues')
        and bool(record.get('required_view_ids'))
        and {r['view_id'] for r in coverage}==set(record['required_view_ids'])
        and all(r.get('leader_coverage')=='complete' and r.get('independent_audit_status')=='COMPLETE'
                and r.get('evidence') for r in coverage)
        and all({'primary','independent_audit'}<=set(o.get('detection_passes',[])) for o in accepted)
        and not any(o.get('status')=='REVIEW' for o in record.get('observations',[])))
    if complete:
        if type(record.get('verified_count')) is not int or record['verified_count']!=len(accepted):
            return '', 'COUNT_EVIDENCE_MISMATCH'
        if not accepted and not any(record.get('no_leader_evidence',[])):
            complete=False
    return str(len(accepted)), ('COMPLETE' if accepted else 'VERIFIED_ZERO') if complete else 'OBSERVED_PARTIAL'


def write(folder,record,detail,views,*,published_folder=None):
    """Number accepted leaders on actual original-source panels; no inference."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    panels=[]
    for vid in detail['view_ids']:
        v=next(v for v in views if v['id']==vid)
        with Image.open(v['crop_path']) as original:image=original.convert('RGB')
        panels.append((vid,image,v['crop_geometry']))
    if not panels:raise ValueError('Leader review needs original source Detail views')
    width=max(im.width for _,im,_ in panels);height=sum(im.height+24 for _,im,_ in panels)
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas);top=0
    for vid,image,mapping in panels:
        canvas.paste(image,(0,top+24));draw.text((4,top+3),'Original source view '+vid,fill='black')
        transform=fitz.Matrix(*mapping['source_to_pixels'])
        for item in record['observations']:
            if item['view_id']!=vid or not lines.valid_path(item['path']):continue
            points=[list(fitz.Point(*p)*transform) for p in item['path']]
            points=[(p[0],p[1]+top+24) for p in points]
            color='#008000' if item['accepted'] else '#a0a0a0' if item['status']=='EXCLUDED' else '#bd0000'
            draw.line(points,fill=color,width=3)
            label=str(item['number']) if item['accepted'] else item['status']
            x,y=points[0];draw.rectangle(draw.textbbox((x,y),label),fill='white');draw.text((x,y),label,fill=color)
        top+=image.height+24
    artifact_root=Path(published_folder or folder)
    overlay=folder/'component_leaders_overlay.png';canvas.save(overlay)
    record['overlay_path']=str(artifact_root/overlay.name)
    path=folder/'leader_observations.json';record['observations_path']=str(artifact_root/path.name)
    review=folder/'profile_count_review.md';record['review_path']=str(artifact_root/review.name)
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    count,status=count_state(record)
    public_count=count if status in {'COMPLETE','VERIFIED_ZERO'} else ''
    rows=['# Component callout leader count review','',f'Original source page: {record["page_index"]+1}',
        f'Detail: {record["detail_id"]}',f'Leader-derived diagnostic count: {public_count or "WITHHELD"}; status: {status}',
        'Final CSV profile_count counts extracted associated components; this file audits lines separately.',
        f'Observed accepted leaders (not necessarily the complete count): {count or "UNKNOWN"}',
        'Count is distinct verified physical component leader lines; text, IDs, rows and structural geometry are not counts.','']
    rows += [f'- {o.get("number",o["status"])} {o["id"]}: '+('; '.join(o['reasons']) or 'Original-source line and ownership verified') for o in record['observations']]
    rows += ['',*record['issues'],'']
    review.write_text('\n'.join(rows))
    return record


def validate_saved(records,details,views,review):
    result={}
    for record in records:
        did=record.get('detail_id');detail=details.get(did)
        try:
            if (not detail or record.get('document_id')!=detail['document_id'] or record.get('page_id')!=detail['page_id']
                or record.get('page_index')!=detail['page_index']):
                raise ValueError('Leader inventory belongs to another Detail/source page')
            if did in result:raise ValueError('Duplicate saved Detail leader inventory')
            if record.get('complete'):
                for vid in detail['view_ids']:
                    missing=context_census_issues(views[vid],record.get('observations',[]))
                    if missing:raise ValueError('; '.join(missing))
            for c in record.get('view_coverage',[]):
                if c['view_id'] not in detail['view_ids']:raise ValueError('Foreign coverage view')
                path=Path(c['source_image'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=c['source_image_sha256']:
                    raise ValueError('Coverage source pixels changed')
                with Image.open(path) as im:
                    if list(im.size)!=c['source_mapping']['pixel_size']:raise ValueError('Coverage source transform has wrong pixel size')
                crop=views[c['view_id']].get('crop_completeness') or views[c['view_id']].get('source_candidate',{}).get('crop_completeness') or {}
                if record.get('complete') and crop.get('status')!='COMPLETE':raise ValueError('Coverage source annotation context is unverified')
            for o in record.get('observations',[]):
                if o.get('view_id') not in detail['view_ids']:raise ValueError('Foreign leader view')
                if o.get('accepted'):
                    view=views[o['view_id']]
                    crop=view.get('crop_completeness') or view.get('source_candidate',{}).get('crop_completeness') or {}
                    if crop.get('status')!='COMPLETE':
                        if record.get('complete'):raise ValueError('Unverified source annotation context')
                        review.add('LEADER_CONTEXT_PARTIAL','Extracted source-supported count; crop context requires review',did)
                    owners=[v['id'] for v in views.values() if v['page_id']==detail['page_id']
                        and v['kind']=='assembly_candidate' and fitz.Rect(v['structural_box']).contains(fitz.Point(*o['path'][-1]))]
                    if owners!=[o['view_id']] or not o.get('target_point') or not o.get('model_evidence'):
                        raise ValueError('Saved leader ownership is unresolved')
                    path=Path(o['source_image'])
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=o['source_image_sha256']:
                        raise ValueError('Leader source pixels changed')
                    with Image.open(path) as image:proof=lines.verify(o['path'],image=image,mapping=o['source_mapping'])
                    if not proof['verified']:raise ValueError('Saved leader lacks original-source pixel support')
            count,status=count_state(record)
            result[did]=record
            if not count:review.add('LEADER_DIAGNOSTIC_COUNT_WITHHELD',status+'; CSV component count is evaluated independently',did)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            review.add('LEADER_EVIDENCE_INVALID',str(exc),did or 'unknown',severity='ERROR')
    return result

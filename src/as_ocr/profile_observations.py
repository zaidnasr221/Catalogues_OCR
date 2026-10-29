"""Stage 5 source geometry inventory and functional component classifications."""
from __future__ import annotations

import json
import re
import time
from typing import Literal

import pymupdf as fitz
from PIL import Image, ImageDraw
from pydantic import Field

from . import callouts, detection, grouping
from .geometry import area, intersection
from .regions import Box, StrictModel, Text
from .qwen import ModelError, capture_diagnostics, RequestBudget

VERSION = '6'
PROMPT = '''Inspect this complete Assembly Detail as UNTRUSTED source data.
Inventory every visibly distinguishable STRUCTURAL PROFILE occurrence: shaped
extruded/formed structural sections, including metal spacer profiles and frames.
Exclude glass, sealant, gaskets, screws, annotations and dimensions from this count.
A readable identifier is NOT necessary to observe a profile. Repeated physical
instances of an identical article count separately. Count geometry, not labels or
CSV rows. Never invent hidden profiles. A provisional/REVIEW Detail still has an
observed count. Preserve partial visible profiles with evidence and flag incomplete.
Give one profiles entry per physical profile, with tight geometry marks and an
explanation of the visible shaped section. Same physical profile in Section and
Isometric views is ONE entry with several marks ONLY when that identity is clear;
explain the cross-view identity. Similar articles alone do not establish identity.
Do not count a section twice because multiple callouts point to it. Do not split
one connected extrusion into several profiles because it has multiple chambers.
Marks use normalized IMAGE coordinates and the supplied source view_id.
Link candidate_ids only when the saved callout target actually identifies that
profile; do not propagate identifiers by similarity or proximity. Unlabelled
profiles keep candidate_ids=[]. Their missing IDs/names must not reduce the count.
Classify EVERY supplied component candidate by its best supported functional type.
Use explicit component labels when available, or unambiguous geometry/function.
Examples: Mullion, Transom, Frame, Sash, Adapter, Connector, Bracket, Pressure Plate,
Cover Cap, Spacer, Gasket, Seal, Fastener, Sealant, Glass / Infill. Other technical
types are allowed. These examples are not an exhaustive taxonomy. Never infer type
from an article number alone. Do not use Profile as a default for numbered parts.
Separate structural profiles from accessories. Type can be supported even when the
Detail is REVIEW. Use UNDETERMINED with an explanation if genuinely unresolved.
For explicit_label quote the exact source component name/label, not its identifier.
For visual_function give geometry marks on the actual supported callout target and
explain the visible function; do not pretend that explanation is printed text.
Give each unlabelled profile its supported functional type or UNDETERMINED too.
For each physical profile, type_supported=true only when its FUNCTION is clear;
provide type_evidence explaining that function. A countable structural section can
still have type_supported=false. Shape/orientation alone cannot distinguish
Mullion from Transom. Never fill a specific function merely to avoid UNDETERMINED.
No new identifiers, names, dimensions or descriptions. Paragraphs are a separate pass.
If no profile is visible return profiles=[] and a concrete no_profile_reason.
coverage=incomplete for clipped geometry, uncertain physical identity or missed
parts. Report issues honestly. Return only the supplied JSON schema.'''


class Mark(StrictModel):
    view_id: Text
    box: Box
    explanation: Text


class PhysicalProfile(StrictModel):
    local_id: Text
    marks: list[Mark] = Field(min_length=1)
    candidate_ids: list[str]
    identity_evidence: Text
    profile_type: Text
    type_supported: bool
    type_evidence: Text
    review_reasons: list[Text]


class Classification(StrictModel):
    candidate_id: Text
    profile_type: Text
    type_supported: bool
    basis: Literal['explicit_label', 'visual_function', 'unresolved']
    source_quote: str | None
    marks: list[Mark]
    explanation: Text
    review_reasons: list[Text]


class Analysis(StrictModel):
    coverage: Literal['complete', 'incomplete']
    profiles: list[PhysicalProfile]
    component_types: list[Classification]
    no_profile_reason: str | None
    review_reasons: list[Text]


AUDIT_PROMPT = PROMPT + '''
VERIFICATION: proposed_inventory is an untrusted FIRST DRAFT, not ground truth.
Inspect the original geometry again and return the corrected complete inventory.
Trace continuity of each section: an upper hollow chamber and its connected lower
flanges/webs are ONE extrusion, not several profiles. Overlapping boxes alone do
not prove either identity or distinctness. Different names do not split one part.
Do not count a screw/bolt, gasket, sealant, or glass as a structural profile.
Conversely preserve genuinely separate repeated structural parts. Ensure every
retained profile has its own visible material boundary, not a mixture of adjacent
parts or merely an empty bounding rectangle. Do not infer Mullion versus Transom
from image orientation alone; source function must be clear. UNDETERMINED is valid.
A normal section not showing hidden length is not an incomplete crop. Coverage
concerns the available Detail, not hypothetical unseen geometry. Reassess each
candidate's functional type against its actual callout target. Correct the first
draft freely; do not force any count, agreement, symmetry or manufacturer pattern.
'''


def contract():
    return grouping.digest([VERSION,PROMPT,AUDIT_PROMPT,Analysis.model_json_schema()])


def owner(candidate, detail):
    return [m for m in candidate['memberships'] if m['view_id'] in detail['view_ids']
            and m['status']=='SUPPORTED_BY_STAGE4' and m['target_point'] is not None]


def facts(candidates, observations, detail):
    result=[]
    for c in candidates:
        memberships=owner(c,detail)
        if not memberships and not set(c.get('observed_in_view_ids', [])) & set(detail['view_ids']):
            continue
        def values(field):
            return sorted({o['value_raw'] for o in observations if o['candidate_id']==c['id'] and
                o['field']==field and o['accepted'] and not o['conditions'] and not o['conflicts_with']})
        result.append(dict(candidate_id=c['id'],profile_ids=values('code'),component_names=values('component_name'),
                           targets=[dict(view_id=m['view_id'],source_point=m['target_point']) for m in memberships]))
    return result


def render(source,page_index,geometry,detail,views,image_max,observations=()):
    panels=[]
    for view in views:
        if view['id'] not in detail['view_ids']:
            continue
        context_box=view.get('crop_geometry',{}).get('source_context_box',view['box'])
        # Count the complete assembly geometry and its own callout blocks. Page
        # paragraphs have a separate full-page pass and cannot name nearby parts.
        region=fitz.Rect(view.get('structural_box') or view['box'])
        for field in observations:
            if field['field'] not in {'code','component_name'} or not field['accepted']:continue
            for evidence in field['evidence']:
                if evidence['view_id']==view['id']:region |= fitz.Rect(evidence['box'])
        region=fitz.Rect(region.x0-3,region.y0-3,region.x1+3,region.y1+3) & fitz.Rect(context_box)
        box=list(region)
        image,mapping=detection.render(source,page_index,geometry,box,image_max)
        panels.append((view,image,mapping))
    if not panels:
        raise ValueError('No original-source panels for profile inventory')
    width=max(im.width for _,im,_ in panels);height=sum(im.height for _,im,_ in panels)
    composite=Image.new('RGB',(width,height),'white');top=0;layout=[]
    scale=min(1,image_max/max(width,height))
    final_size=(max(1,round(width*scale)),max(1,round(height*scale)))
    for view,image,mapping in panels:
        composite.paste(image,(0,top))
        matrix=fitz.Matrix(*mapping['source_to_pixels'])*fitz.Matrix(1,0,0,1,0,top)*fitz.Matrix(final_size[0]/width,final_size[1]/height)
        layout.append(dict(view_id=view['id'],view_type=view.get('view_type','unknown'),source_box=mapping['source_context_box'],structural_box=view.get('structural_box',view['box']),
            image_box=[0,top/height,image.width/width,(top+image.height)/height],
            mapping=dict(source_to_pixels=list(matrix),pixels_to_source=list(~matrix),pixel_size=list(final_size))))
        top+=image.height
    if composite.size!=final_size:
        composite=composite.resize(final_size,Image.Resampling.LANCZOS)
    return composite,layout


def request_context(detail,layout,candidates,observations,image):
    components=facts(candidates,observations,detail);panels={p['view_id']:p for p in layout}
    for component in components:
        for target in component['targets']:
            point=fitz.Point(*target['source_point'])*fitz.Matrix(*panels[target['view_id']]['mapping']['source_to_pixels'])
            target['image_point']=[point.x/image.width,point.y/image.height]
    return dict(detail_id=detail['id'],panels=[{k:v for k,v in p.items() if k!='mapping'} for p in layout],components=components)


def _inside(outer, inner):
    return outer[0]-.01<=inner[0]<inner[2]<=outer[2]+.01 and outer[1]-.01<=inner[1]<inner[3]<=outer[3]+.01


def _target(marks, fact):
    return any(m['view_id']==t['view_id'] and m['box'][0]-.5<=t['source_point'][0]<=m['box'][2]+.5 and
               m['box'][1]-.5<=t['source_point'][1]<=m['box'][3]+.5 for m in marks for t in fact['targets'])


def _type(value):
    if not value or value.strip().casefold() in {'unknown','undetermined','profile','component','n/a','none','not identifiable','unidentifiable'}:
        return 'UNDETERMINED'
    if len(value)>100 or re.fullmatch(r'[\d\W]+',value) or value[0] in '=+-@' or re.search(r'\b(anodized|anodised|dimension)\b',value,re.I):
        return 'UNDETERMINED'
    if re.fullmatch(r'\d+(?:[.,]\d+)?(?:\s*[x×]\s*\d+(?:[.,]\d+)?)*\s*(?:mm|cm|m|in|inch|inches|ft)',value,re.I):
        return 'UNDETERMINED'
    return value.strip()


def nonstructural(value):
    return ' '.join(value.casefold().replace('/',' / ').split()) in {
        'glass','glass / infill','infill','glazing','gasket','gaskets','epdm gasket','rubber gasket',
        'seal','seals','sealant','sealants','silicone sealant','fastener','fasteners',
        'screw','screws','bolt','bolts','nut','nuts','washer','washers','rivet','rivets'}


def overlapping(left,right):
    return any(a['view_id']==b['view_id'] and intersection(a['box'],b['box'])/
               max(area(a['box']),area(b['box']),.001)>.85 for a in left for b in right)


def same_observation(left,right,left_evidence,right_evidence):
    def boxes(marks):return sorted((m['view_id'],tuple(round(v,4) for v in m['box'])) for m in marks)
    return boxes(left)==boxes(right) and left_evidence==right_evidence


def label_name(quote,fact):
    """Split a literal name/code label only at its exact verified trailing ID."""
    if not quote:return None
    for code in fact['profile_ids']:
        if quote.endswith(code) and quote.count(code)==1:
            prefix=quote[:-len(code)]
            if not prefix or not prefix[-1].isspace():continue
            name=prefix.strip()
            if name and any(c.isalpha() for c in name) and _type(name)!='UNDETERMINED':return name
    return None


def label_supported(quote,fact,marks):
    return bool(quote and (any(quote in n for n in fact['component_names']) or
                           label_name(quote,fact) and _target(marks,fact)))


def reconcile(analysis,detail,context,layout,bounds):
    lookup={p['view_id']:p for p in layout};known={c['candidate_id']:c for c in context['components']}
    reasons=list(analysis.review_reasons);rejected=[]
    def marks(items):
        result=[]
        for mark in items:
            panel=lookup.get(mark.view_id)
            if not panel or not _inside(panel['image_box'],mark.box):
                raise ValueError('Profile/type mark is outside its original Detail panel')
            box=detection.source_box(mark.box,panel['mapping'],bounds)
            if intersection(box,panel['structural_box'])<=0:
                raise ValueError('Geometry mark has no intersection with the assembly geometry')
            result.append(dict(view_id=mark.view_id,box=box,explanation=mark.explanation))
        return result
    profiles=[];local=set();ambiguous=False
    for p in analysis.profiles:
        try:
            if p.local_id in local:raise ValueError('Duplicate physical-profile identity')
            local.add(p.local_id);mapped=marks(p.marks)
            if any(same_observation(prior['marks'],mapped,prior['identity_evidence'],p.identity_evidence) for prior in profiles):
                raise ValueError('Duplicate physical geometry was proposed twice')
            if any(overlapping(prior['marks'],mapped) for prior in profiles):
                ambiguous=True
                reasons.append('Overlapping physical observations have unresolved identity; geometry retained and count withheld')
            linked=[]
            for cid in p.candidate_ids:
                if cid not in known or not _target(mapped,known[cid]):
                    reasons.append('Profile retained without unsupported callout binding: '+cid)
                else:linked.append(cid)
            category=_type(p.profile_type)
            if nonstructural(category):
                raise ValueError('Non-profile accessory/glass cannot contribute to structural profile count')
            item=dict(id='physical-'+grouping.digest([detail['id'],mapped,p.identity_evidence])[:20],marks=mapped,
                candidate_ids=sorted(set(linked)),identity_evidence=p.identity_evidence,profile_type=category,
                type_supported=p.type_supported,type_evidence=p.type_evidence,
                review_reasons=list(p.review_reasons))
            profiles.append(item)
        except ValueError as exc:
            rejected.append(dict(local_id=p.local_id,reason=str(exc)))
    classifications=[]
    by_id={}
    for c in analysis.component_types:by_id.setdefault(c.candidate_id,[]).append(c)
    for cid,fact in known.items():
        entries=by_id.get(cid,[]);category='UNDETERMINED';issue=[];mapped=[];basis='unresolved';quote=None;explanation='No classification returned for this component'
        if len(entries)==1:
            c=entries[0];basis=c.basis;quote=c.source_quote;explanation=c.explanation
            if not c.type_supported:issue.append('Functional classification remains unverified')
            try:
                mapped=marks(c.marks)
                if basis=='explicit_label':
                    if not label_supported(quote,fact,mapped):
                        issue.append('Functional terminology is not supported by a verified component label')
                elif basis=='visual_function':
                    if not mapped or not _target(mapped,fact):issue.append('Visual function does not identify the saved component target')
                else:issue.append('Component function remains unresolved')
                if c.profile_type in fact['profile_ids']:
                    issue.append('An article identifier is not a functional component type')
                category=_type(c.profile_type) if not issue else 'UNDETERMINED'
            except ValueError as exc:issue.append(str(exc))
        else:issue.append('Missing or conflicting classification attempts')
        if category=='UNDETERMINED' and not issue:issue.append('No specific technical function is verified')
        classifications.append(dict(candidate_id=cid,profile_type=category,basis=basis,source_quote=quote,
            type_supported=category!='UNDETERMINED',marks=mapped,explanation=explanation,
            review_reasons=list(dict.fromkeys(issue+([*entries[0].review_reasons] if len(entries)==1 else [])))))
    incomplete=analysis.coverage!='complete' or bool(rejected) or bool(reasons) or any(p['review_reasons'] for p in profiles)
    if not profiles:
        incomplete |= not bool(analysis.no_profile_reason)
        reasons.append(analysis.no_profile_reason or 'No supported physical profile occurrence could be identified')
    reasons += ['Rejected profile '+r['local_id']+': '+r['reason'] for r in rejected]
    return dict(document_id=detail['document_id'],page_id=detail['page_id'],page_index=detail['page_index'],detail_id=detail['id'],
        observed_count=len(profiles),profiles=profiles,component_types=classifications,rejected_profiles=rejected,
        no_profile_reason=analysis.no_profile_reason,
        count_ambiguous=ambiguous,
        coverage=analysis.coverage,review_reasons=reasons,
        potentially_incomplete=bool(incomplete))


def annotate(image,layout,record):
    result=image.copy();draw=ImageDraw.Draw(result);lookup={p['view_id']:p for p in layout}
    def box(mark):return list(fitz.Rect(mark['box'])*fitz.Matrix(*lookup[mark['view_id']]['mapping']['source_to_pixels']))
    for n,profile in enumerate(record['profiles'],1):
        for mark in profile['marks']:
            category=profile['profile_type'] if profile.get('type_supported',False) else 'UNDETERMINED'
            if profile['candidate_ids']:
                supported={c['profile_type'] for c in record['component_types'] if c['candidate_id'] in profile['candidate_ids'] and c['profile_type']!='UNDETERMINED'}
                if len(supported)==1:category=next(iter(supported))
            b=box(mark);draw.rectangle(b,outline='#008040',width=3);draw.text((b[0],max(0,b[1]-12)),f'P{n}: '+category,fill='#008040')
    for c in record['component_types']:
        for mark in c['marks']:
            b=box(mark);draw.rectangle(b,outline='#0055bf',width=2);draw.text((b[0],b[3]),c['profile_type'],fill='#0055bf')
    draw.text((4,4),f'Observed structural profiles: {record["observed_count"]}'+(' / REVIEW' if record['potentially_incomplete'] else ''),fill='#b03000')
    return result


def retain_observed(draft,verified):
    """An incomplete verification cannot erase an already observed inventory."""
    prior={t['candidate_id']:t for t in draft['component_types'] if t['profile_type']!='UNDETERMINED' and t.get('type_supported',True)}
    current={t['candidate_id']:t for t in verified['component_types']}
    for cid,supported in prior.items():
        missing=current.get(cid)
        if missing is None or missing['profile_type']=='UNDETERMINED' and missing.get('basis')=='unresolved':
            current[cid]=dict(supported,explanation=supported['explanation']+'; prior source-supported classification retained after missing secondary classification')
    verified=dict(verified,component_types=list(current.values()))
    if draft['observed_count'] and not verified['observed_count'] and verified['coverage']!='complete':
        retained=dict(draft)
        retained.update(coverage='incomplete',potentially_incomplete=True,
            review_reasons=draft['review_reasons']+verified['review_reasons']+
                ['Incomplete verification returned no profiles; retained observed first-pass geometry for REVIEW, not certified absence'],
            verification_observed_count=0,verification_disagreement=True)
        return retained
    return verified


def count_state(record):
    """Zero requires demonstrated absence; incomplete positive counts are observations."""
    if not record:
        return '', 'MISSING'
    count = record.get('observed_count')
    if type(count) is not int or count < 0 or count != len(record.get('profiles', [])):
        raise ValueError('Invalid observed profile inventory ownership/count')
    if record.get('count_ambiguous'):
        return '', 'AMBIGUOUS_IDENTITY'
    if count:
        return str(count), 'PARTIAL' if record.get('potentially_incomplete', True) else 'COMPLETE'
    if (record.get('coverage') == 'complete' and not record.get('potentially_incomplete', True)
            and record.get('no_profile_reason') and not record.get('rejected_profiles')):
        return '0', 'VERIFIED_ZERO'
    return '', 'INCOMPLETE_OR_FAILED'


def extract(source,page_index,geometry,details,views,candidates,observations,folder,client,image_max=2048,*,inventory_max_requests=None,verification_max_requests=None):
    from .components import parse_response
    results=[]
    inventory=RequestBudget(inventory_max_requests if inventory_max_requests is not None else client.config.max_requests,'physical_profile_inventory_page')
    verification=RequestBudget(verification_max_requests if verification_max_requests is not None else client.config.max_requests,'profile_verification_page')
    for detail in details:
        started=time.monotonic();record=None
        try:
            image,layout=render(source,page_index,geometry,detail,views,image_max,observations)
            context=request_context(detail,layout,candidates,observations,image)
            image_path=folder/(detail['id']+'-profiles-source.png');image.save(image_path)
            with inventory.scope(client), capture_diagnostics(client,folder/'diagnostics',stage=5,subject_id=detail['id']+'-profiles'):
                raw=client.request_json(image,PROMPT,json.dumps(context,ensure_ascii=False),response_schema=Analysis.model_json_schema())
                analysis=parse_response(raw,Analysis)
                record=reconcile(analysis,detail,context,layout,geometry['source_rect'])
            draft=record
            proposed=analysis.model_dump(mode='json')
            with verification.scope(client), capture_diagnostics(client,folder/'diagnostics',stage=5,subject_id=detail['id']+'-profiles-audit'):
                raw=client.request_json(image,AUDIT_PROMPT,json.dumps(dict(context,proposed_inventory=proposed),ensure_ascii=False),response_schema=Analysis.model_json_schema())
                analysis=parse_response(raw,Analysis)
                record=reconcile(analysis,detail,context,layout,geometry['source_rect'])
                record=retain_observed(draft,record)
            record.update(initial_proposed_count=len(proposed['profiles']),initial_inventory=proposed)
            path=folder/(detail['id']+'-profiles.png');annotate(image,layout,record).save(path)
            record.update(source_image=str(image_path),review_image=str(path),layout=layout)
        except (ValueError,OSError,ModelError) as exc:
            reason='Profile observation/verification failed: '+callouts.failure_reason(exc)
            if record is None:
                record=dict(document_id=detail['document_id'],page_id=detail['page_id'],page_index=detail['page_index'],detail_id=detail['id'],
                    observed_count=0,profiles=[],component_types=[],review_reasons=[reason])
            else:
                record['review_reasons'].append(reason+'; observed first-pass profiles retained for REVIEW')
                path=folder/(detail['id']+'-profiles.png');annotate(image,layout,record).save(path)
                record.update(source_image=str(image_path),review_image=str(path),layout=layout)
            record.update(coverage='incomplete',potentially_incomplete=True)
        record['duration_seconds']=round(time.monotonic()-started,3);results.append(record)
    for record in results:
        record['page_request_budgets']={'inventory':inventory.public(),'verification':verification.public()}
    return results


def verify_existing(source,page_index,geometry,details,views,candidates,observations,records,folder,client,image_max=2048,*,verification_max_requests=None):
    """Complete the geometry-verification phase for this fresh run's first drafts."""
    from .components import parse_response
    result=[];original={r['detail_id']:r for r in records}
    verification=RequestBudget(verification_max_requests if verification_max_requests is not None else client.config.max_requests,'profile_verification_page')
    for detail in details:
        started=time.monotonic();prior=original[detail['id']]
        image,layout=render(source,page_index,geometry,detail,views,image_max,observations)
        panels={p['view_id']:p for p in layout}
        def image_marks(items):
            result=[]
            for m in items:
                b=list(fitz.Rect(m['box'])*fitz.Matrix(*panels[m['view_id']]['mapping']['source_to_pixels']))
                result.append(dict(view_id=m['view_id'],box=[max(0,min(1,b[0]/image.width)),max(0,min(1,b[1]/image.height)),
                    max(0,min(1,b[2]/image.width)),max(0,min(1,b[3]/image.height))],explanation=m['explanation']))
            return result
        proposed=dict(coverage=prior['coverage'],profiles=[dict(local_id=p['id'],marks=image_marks(p['marks']),
            candidate_ids=p['candidate_ids'],identity_evidence=p['identity_evidence'],profile_type=p['profile_type'],
            type_supported=p.get('type_supported',False),type_evidence=p.get('type_evidence','Functional type not independently established'),
            review_reasons=p['review_reasons']) for p in prior['profiles']],
            component_types=[dict(c,marks=image_marks(c['marks'])) for c in prior['component_types']],
            review_reasons=prior['review_reasons'])
        context=dict(request_context(detail,layout,candidates,observations,image),proposed_inventory=proposed)
        image_path=folder/(detail['id']+'-profiles-source.png');image.save(image_path)
        with verification.scope(client), capture_diagnostics(client,folder/'diagnostics',stage=5,subject_id=detail['id']+'-profiles-audit'):
            raw=client.request_json(image,AUDIT_PROMPT,json.dumps(context,ensure_ascii=False),response_schema=Analysis.model_json_schema())
            analysis=parse_response(raw,Analysis)
            record=reconcile(analysis,detail,context,layout,geometry['source_rect'])
            record=retain_observed(prior,record)
        path=folder/(detail['id']+'-profiles.png');annotate(image,layout,record).save(path)
        record.update(source_image=str(image_path),review_image=str(path),layout=layout,
                      initial_proposed_count=prior['observed_count'],initial_inventory=prior,
                      initial_review_image=prior.get('review_image'),
                      duration_seconds=round(time.monotonic()-started,3))
        result.append(record)
    return result


def recover_label_fields(records,candidates,observations,details):
    """Retain names read by the fresh audit using the existing code-label contract."""
    new=[];by_id={c['id']:c for c in candidates};detail_map={d['id']:d for d in details}
    for record in records:
        known={f['candidate_id']:f for f in facts(candidates,observations,detail_map[record['detail_id']])}
        for item in record['component_types']:
            cid=item['candidate_id'];fact=known[cid]
            if item['basis']!='explicit_label' or not label_supported(item.get('source_quote'),fact,item['marks']):continue
            # A geometry-backed type remains useful if an earlier name read failed.
            if item['profile_type']=='UNDETERMINED' and item['review_reasons']==['Functional terminology is not supported by a verified component label']:
                types={p['profile_type'] for p in record['profiles'] if cid in p['candidate_ids'] and p['profile_type']!='UNDETERMINED'}
                if len(types)==1:item.update(profile_type=next(iter(types)),review_reasons=[])
            if fact['component_names']:continue
            name=label_name(item.get('source_quote'),fact)
            code=next((o for o in observations if o['candidate_id']==cid and o['field']=='code' and o['accepted']),None)
            if not name or not code:continue
            evidence=dict(code['evidence'][0],quote=item['source_quote'],method='visual_label',review_reasons=[])
            relation=dict(kind='code_label',bounds=evidence['box'],code_evidence=code['evidence'][0],evidence=[evidence],
                          explanation='Exact identifier and literal component name read together in the associated callout label block')
            field=dict(candidate_id=cid,field='component_name',name_raw=None,value_raw=name,unit_raw=None,category_value=None,
                normalized=None,meaning='unspecified',decimal_convention='unknown',accepted=True,conditions=[],conflicts_with=[],
                review_reasons=[],evidence=[evidence],relation=relation)
            field['id']='field-'+grouping.digest(field)[:20];new.append(field)
    return new


def validate_saved(records,details,views,candidates,observations,properties,review):
    """CPU-only export check; never derive counts from identifiers or CSV rows."""
    result={};invalid=set();valid_fields={p['id'] for p in properties if p['accepted'] and not p['conditional'] and not p['conflict_ids']}
    fields=[dict(o,accepted=o['accepted'] and o['id'] in valid_fields) for o in observations.values()]
    for record in records:
        did=record.get('detail_id');detail=details.get(did)
        try:
            if did in invalid:raise ValueError('Detail has conflicting or invalid profile inventories')
            if detail and detail.get('membership_role')=='alternative':
                review.add('ALTERNATIVE_PROFILE_INVENTORY_WITHHELD','Uncertain merged inventory retained in Stage 5 source evidence',did)
                continue
            if not detail or any(record.get(k)!=detail[k] for k in ('document_id','page_id','page_index')):
                raise ValueError('Foreign Detail/page in profile inventory')
            known={f['candidate_id']:f for f in facts(list(candidates.values()),fields,detail)}
            profiles=[];seen=set();ambiguous=record.get('count_ambiguous',False)
            for p in record['profiles']:
                if p['id'] in seen:raise ValueError('Duplicate physical profile identifier')
                seen.add(p['id'])
                if not p['marks'] or not p['identity_evidence']:raise ValueError('Physical profile lacks geometry evidence')
                if any(same_observation(prior['marks'],p['marks'],prior['identity_evidence'],p['identity_evidence']) for prior in profiles):
                    raise ValueError('Repeated physical geometry in saved inventory')
                ambiguous |= any(overlapping(prior['marks'],p['marks']) for prior in profiles)
                for mark in p['marks']:
                    view=views.get(mark['view_id'])
                    if (not view or view['id'] not in detail['view_ids'] or
                        view['document_id']!=detail['document_id'] or view['page_id']!=detail['page_id'] or
                        not _inside(view.get('crop_geometry',{}).get('source_context_box',view['box']),mark['box'])):
                        raise ValueError('Profile geometry is outside its source Detail')
                if any(cid not in known or not _target(p['marks'],known[cid]) for cid in p['candidate_ids']):
                    raise ValueError('Profile/callout target ownership is invalid')
                if (not p['candidate_ids'] and p['profile_type'] in {'Mullion','Transom'} and
                    all(views[m['view_id']].get('view_type')=='section' for m in p['marks'])):
                    p=dict(p,type_supported=False,review_reasons=p['review_reasons']+
                        ['Section geometry without an axis reference does not verify Mullion versus Transom function'])
                    review.add('PROFILE_FUNCTION_UNDETERMINED','Unlabelled section cannot establish Mullion/Transom function from image orientation',p['id'])
                profiles.append(p)
            if type(record['observed_count']) is not int or record['observed_count']!=len(profiles):
                raise ValueError('Observed count differs from physical geometry inventory')
            for p in profiles:
                if nonstructural(_type(p['profile_type'])):
                    raise ValueError('Nonstructural item in physical profile inventory')
            types=[]
            for c in record['component_types']:
                fact=known.get(c['candidate_id']);value=_type(c['profile_type'])
                if not fact:raise ValueError('Type refers to another or unresolved component')
                if c['profile_type'] in fact['profile_ids']:value='UNDETERMINED'
                supported=(c['basis']=='explicit_label' and label_supported(c.get('source_quote'),fact,c['marks']) or
                           c['basis']=='visual_function' and c['marks'] and _target(c['marks'],fact))
                if value!='UNDETERMINED' and (not supported or not c.get('type_supported',not c['review_reasons'])):
                    value='UNDETERMINED'
                    review.add('TYPE_UNVERIFIED','Functional type was withheld because its source support is invalid',c['candidate_id'])
                geometry=[p for p in profiles if c['candidate_id'] in p['candidate_ids'] and p.get('type_supported',False)
                          and _type(p['profile_type'])!='UNDETERMINED']
                values={p['profile_type'] for p in geometry}
                if value=='UNDETERMINED' and len(values)==1:
                    value=next(iter(values))
                    types.append(dict(c,profile_type=value,type_supported=True,basis='visual_function',source_quote=None,
                        marks=[m for p in geometry for m in p['marks']],
                        explanation='Function verified from the associated physical profile geometry: '+
                            '; '.join(p.get('type_evidence',p['identity_evidence']) for p in geometry)))
                else:types.append(dict(c,profile_type=value))
            if did in result:raise ValueError('Repeated Detail profile inventory')
            result[did]=dict(record,profiles=profiles,component_types=types,count_ambiguous=ambiguous)
            _, status = count_state(result[did])
            if status in {'INCOMPLETE_OR_FAILED','AMBIGUOUS_IDENTITY'}:
                review.add('PROFILE_COUNT_WITHHELD','Count not verified: '+status,did)
            for reason in record['review_reasons']:
                review.add('PROFILE_OBSERVATION_REVIEW',reason,did)
            for c in types:
                if c['profile_type']=='UNDETERMINED':review.add('TYPE_UNDETERMINED',c['explanation'],c['candidate_id'])
        except (ValueError,KeyError,TypeError) as exc:
            result.pop(did,None);invalid.add(did)
            review.add('PROFILE_INVENTORY_INVALID',str(exc),did,severity='ERROR')
    return result

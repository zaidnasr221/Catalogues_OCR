"""Same-page fabrication references, corroborated by original native text.

Model hints alone never confirm membership. Unknown/raster-only captions remain
review evidence. This module does not assign component IDs, names or counts.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import re
import unicodedata

import pymupdf as fitz
from PIL import Image

from .geometry import area, intersection, distance
from . import line_evidence


def plain(text):
    if not isinstance(text,str):return ''
    text=unicodedata.normalize('NFKC',text).translate(str.maketrans('–—−','---'))
    return re.sub(r'\s+','',text).casefold()


def valid_box(box):
    return (isinstance(box,(list,tuple)) and len(box)==4 and
            all(type(x) in (int,float) and math.isfinite(x) for x in box) and
            box[2]>box[0] and box[3]>box[1])


def source_quote(view,mark):
    """Verify the complete literal quote against attached native source spans."""
    try:
        box=mark['box'];text=mark['text']
        if not valid_box(box) or not isinstance(text,str) or not text.strip():return False
        if intersection(box,view['box'])<area(box)-1e-5:return False
        nearby=[a for a in view.get('source_candidate',{}).get('annotations',[])
                if a.get('status')=='attached' and valid_box(a.get('box'))
                and isinstance(a.get('text'),str)
                and intersection(a['box'],box)>=.5*min(area(a['box']),area(box))]
        if any(plain(a['text'])==plain(text) for a in nearby):return True
        nearby.sort(key=lambda a:(a['box'][1],a['box'][0]))
        return bool(nearby) and plain(' '.join(a['text'] for a in nearby))==plain(text)
    except (KeyError,TypeError,ValueError):
        return False


def variant_scope(view):
    return {plain(m['marker']) for m in view.get('source_candidate',{}).get('reference_marks',[])
            if m.get('kind')=='variant' and m.get('role')=='caption'
            and plain(m.get('marker',''))==plain(m.get('text','')) and source_quote(view,m)}


def same_variant(views):
    scopes=[variant_scope(v) for v in views]
    return not any(scopes) or all(len(s)==1 and s==scopes[0] for s in scopes)


def reference_key(mark):
    marker=plain(mark.get('marker',''))
    if mark.get('kind')=='detail_locator':
        return marker if re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,15}',marker) else None
    if mark.get('kind')=='section':
        pair=re.fullmatch(r'([a-z][a-z0-9]{0,3})-\1',marker)
        if pair:marker=pair[1]
    return marker if re.fullmatch(r'[a-z][a-z0-9]{0,7}',marker) else None


def caption_matches(mark,key):
    text=plain(mark.get('text',''))
    if mark['kind']=='section':
        return bool(re.fullmatch(r'(?:section|schnitt|coupe|corte)?'+re.escape(key)+r'-'+re.escape(key)+r'(?:\(?\d+:\d+\)?)?',text))
    # A bare repeated X/E1 alone is not evidence of an enlargement relation.
    return bool(re.fullmatch(r'(?:detail|ausschnitt|détail|detalle)?'+re.escape(key)+r'\(?\d+:\d+\)?',text)
                or re.fullmatch(r'(?:detail|ausschnitt|détail|detalle)'+re.escape(key),text))


def source_anchor(view,mark,views):
    """A marker is on one assembly, or its source-backed pointer reaches it."""
    box=mark['box'];center=fitz.Point((box[0]+box[2])/2,(box[1]+box[3])/2)
    same_page=[v for v in views if (v['document_id'],v['page_index'])==(view['document_id'],view['page_index'])
               and v['kind']=='assembly_candidate']
    owners=[v['id'] for v in same_page if fitz.Rect(v['structural_box']).contains(center)]
    if owners==[view['id']]:return True
    path=mark.get('leader_points',[])
    if not line_evidence.valid_path(path) or distance(box,[*path[0],*path[0]])>2:return False
    targets=[v['id'] for v in same_page if any(fitz.Rect(v['structural_box']).contains(fitz.Point(*p)) for p in path)]
    if targets!=[view['id']]:return False
    try:
        if view.get('crop_sha256'):
            import hashlib
            if hashlib.sha256(Path(view['crop_path']).read_bytes()).hexdigest()!=view['crop_sha256']:return False
        with Image.open(view['crop_path']) as image:
            return line_evidence.verify(path,image=image,mapping=view['crop_geometry'])['verified']
    except (OSError,ValueError,KeyError,TypeError):return False


def proposals(views,existing):
    """Add uniquely evidenced reference groups; retain all uncertain proposals."""
    by_id={v['id']:v for v in views};buckets=defaultdict(lambda:defaultdict(dict));clues=[]
    locators=defaultdict(lambda:defaultdict(dict))
    for view in views:
        for mark in view.get('source_candidate',{}).get('reference_marks',[]):
            base=dict(view_ids=[view['id']],reference_mark=mark,membership_established=False)
            if mark.get('kind')=='detail_locator':
                key=reference_key(mark)
                if key and mark.get('role') in {'source','caption'} and mark.get('evidence'):
                    locators[(view['document_id'],view['page_index'],key)][mark['role']][view['id']]=mark
                else:clues.append(dict(base,observations=['Invalid detail locator; REVIEW']))
                continue
            if mark.get('kind') not in {'enlargement','section'}:
                clues.append(dict(base,observations=['Context/variant/step/alternative code only; not a component or membership assignment']))
                continue
            key=reference_key(mark)
            if (not key or not mark.get('evidence') or not source_quote(view,mark)
                    or mark.get('role') not in {'source','caption'}):
                clues.append(dict(base,observations=['Reference needs original-source quote/marker verification; REVIEW']))
                continue
            if view['kind'] not in {'assembly_candidate','profile_candidate'}:
                clues.append(dict(base,observations=['Table or metadata reference cannot establish an assembly view']))
                continue
            if mark['role']=='caption' and not caption_matches(mark,key):
                clues.append(dict(base,observations=['Caption does not explicitly identify this enlargement or section']))
                continue
            if mark['role']=='source' and not source_anchor(view,mark,views):
                clues.append(dict(base,observations=['Source reference lacks unique geometry/pointer ownership']))
                continue
            bucket=(view['document_id'],view['page_index'],mark['kind'],key)
            buckets[bucket][mark['role']][view['id']]=mark
    for (document,page,key),roles in locators.items():
        sources=roles['source'];targets=roles['caption'];ids=sorted(set(sources)|set(targets))
        unique=len(sources)==len(targets)==1 and not set(sources)&set(targets)
        verified=unique and all(source_quote(by_id[vid],m) for vid,m in [*sources.items(),*targets.items()])
        if verified:
            target=next(iter(targets.values()))
            verified=bool(re.fullmatch(r'(?:detail|détail|detalle|ausschnitt)'+re.escape(key),plain(target['text'])))
        clues.append(dict(view_ids=ids,relation_type='detail_locator_context',marker=key,
            source_marks=list(sources.values()),target_captions=list(targets.values()),
            source_quote_verified=bool(verified),membership_established=False,
            observations=['Unique source-backed detail locator; context only, never merge joints' if verified else
                          'Detail locator retained; ambiguous or visual-only source needs REVIEW']))
    checked=[]
    for group in existing:
        if group['strength']=='explicit' and not same_variant([by_id[v] for v in group['view_ids']]):
            group={**group,'strength':'scoped','review_reasons':group['review_reasons']+['Conflicting or incomplete Standard/HI/SI variant scope; membership remains REVIEW']}
        checked.append(group)
    target_claims=defaultdict(set)
    for bucket,roles in buckets.items():
        for vid in roles['caption']:target_claims[vid].add((bucket[2],bucket[3]))
    edges=[]
    for (document,page,kind,key),roles in buckets.items():
        sources=roles['source'];targets=roles['caption']
        ids=sorted(set(sources)|set(targets))
        if len(sources)!=1 or len(targets)!=1 or set(sources)&set(targets):
            clues.append(dict(view_ids=ids,observations=['Reference marker has missing or competing source/target views'],membership_established=False))
            continue
        left=next(iter(sources));right=next(iter(targets))
        if len(target_claims[right])>1:
            clues.append(dict(view_ids=ids,observations=['Target crop contains competing reference captions; split/review the source views first'],membership_established=False))
            continue
        if not same_variant([by_id[left],by_id[right]]):
            clues.append(dict(view_ids=ids,observations=['Different or incomplete variant scopes; reference cannot merge Standard/HI/SI'],membership_established=False))
            continue
        edges.append(dict(view_ids=[left,right],evidence=dict(type='source_verified_'+kind+'_reference',
            marker=key,source_view_id=left,target_view_id=right,source_mark=sources[left],target_caption=targets[right],
            document_id=document,page_index=page,source_quote_verified=True)))
    if not edges:return checked,clues
    # Compose explicit reference chains (main -> A-A and main -> B-B) so the
    # justified links do not become competing overlapping pair proposals.
    parent={vid:vid for vid in by_id}
    def find(vid):
        while parent[vid]!=vid:
            parent[vid]=parent[parent[vid]];vid=parent[vid]
        return vid
    def join(ids):
        for vid in ids[1:]:parent[find(vid)]=find(ids[0])
    for edge in edges:join(edge['view_ids'])
    safe=[g for g in checked if g['strength']=='explicit' and not g['review_reasons']
          and not any(other is not g and other['strength']=='explicit'
              and set(g['view_ids'])&set(other['view_ids'])
              and set(g['view_ids'])!=set(other['view_ids']) for other in checked)]
    for group in safe:join(group['view_ids'])
    roots={find(e['view_ids'][0]) for e in edges}
    merged=[];used=set()
    for root in roots:
        ids=sorted(v for v in by_id if find(v)==root)
        if len({(by_id[v]['document_id'],by_id[v]['page_index']) for v in ids})!=1 or not same_variant([by_id[v] for v in ids]):
            clues.append(dict(view_ids=ids,observations=['Reference chain has conflicting page/variant scope; REVIEW'],membership_established=False))
            continue
        evidence=[e['evidence'] for e in edges if find(e['view_ids'][0])==root];titles=[]
        for n,g in enumerate(checked):
            if g in safe and set(g['view_ids'])<=set(ids):
                used.add(n);evidence+=g['evidence'];titles+=g['titles']
        merged.append(dict(view_ids=ids,strength='explicit',titles=titles,evidence=evidence,review_reasons=[]))
    return [g for n,g in enumerate(checked) if n not in used]+merged,clues

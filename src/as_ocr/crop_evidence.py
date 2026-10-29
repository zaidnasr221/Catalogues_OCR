"""CPU verification of source context. Unproven ownership is never ACCEPTED.

Native spans and outlined glyph bounds are evidence of extent, not of wording.
Only traced source lines establish ownership; proximity never does.
"""
from __future__ import annotations

import math
import json
from pathlib import Path

import pymupdf as fitz
from PIL import ImageDraw

from . import line_evidence as lines
from .geometry import area, distance, expand, intersection, union

VERSION='1'


def contains(outer,inner):
    return area(inner)>0 and intersection(outer,inner)>=area(inner)-1e-5


def valid(box,bounds):
    return (isinstance(box,(list,tuple)) and len(box)==4 and
        all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in box)
        and contains(bounds,box))


def text_extents(spans,vectors):
    result=[dict(box=s.get('bbox',s.get('box')),kind='native_text',ordinal=o,text=s.get('text','')) for o,s in spans]
    # PDF outlines cannot be read as native text. Preserve small filled/curved
    # source paths as possible glyphs; their semantics remain explicitly unknown.
    glyphs=[]
    for o,p in vectors:
        b=p.get('rect');items=p.get('items',[])
        if not b or area(b)<=0:continue
        w,h=b[2]-b[0],b[3]-b[1]
        if max(w,h)<=18 and min(w,h)<=10 and (p.get('fill') is not None or any(i[0]=='c' for i in items)):
            glyphs.append(dict(box=list(b),kind='possible_outlined_text',ordinal=o))
    # Join only same-baseline adjacent glyph extents, never different assemblies.
    buckets={}
    for glyph in sorted(glyphs,key=lambda g:(round(g['box'][1]/2),g['box'][0])):
        b=glyph['box'];key=round(b[1]/2)
        nearby=[r for k in (key-1,key,key+1) for r in buckets.get(k,[])]
        prior=next((r for r in reversed(nearby) if abs(r['box'][1]-b[1])<2 and abs(r['box'][3]-b[3])<2
            and 0<=b[0]-r['box'][2]<=max(2,b[3]-b[1])),None)
        if prior:prior['box']=union(prior['box'],b)
        else:result.append(glyph);buckets.setdefault(key,[]).append(glyph)
    return result


def outward_paths(segments,structure):
    """Trace unique native elbow chains; branches/limits cannot prove ownership."""
    grid={};cell=.25
    def key(p):return round(p[0]/cell),round(p[1]/cell)
    for n,s in enumerate(segments):
        for p in s[2:]:grid.setdefault(key(p),[]).append((n,p))
    result=[]
    for n,(_,_,a,b) in enumerate(segments):
        ia=fitz.Rect(structure).contains(fitz.Point(*a));ib=fitz.Rect(structure).contains(fitz.Point(*b))
        if ia==ib:continue
        path=[a,b] if ia else [b,a];used={n};ambiguous=False
        for _ in range(128):
            p=path[-1];kx,ky=key(p);matches=set()
            for x in (kx-1,kx,kx+1):
                for y in (ky-1,ky,ky+1):
                    for i,q in grid.get((x,y),[]):
                        if i not in used and math.dist(p,q)<=.1:matches.add(i)
            if len(matches)!=1:
                ambiguous=bool(matches);break
            i=matches.pop();used.add(i);c,d=segments[i][2:]
            q=d if math.dist(p,c)<=.1 else c
            if fitz.Rect(structure).contains(fitz.Point(*q)):ambiguous=True;break
            path.append(q)
        else:ambiguous=True
        result.append(dict(path=path,ambiguous=ambiguous))
    return result


def boundary_ink(context,bounds,structure,annotations,regions,image,mapping):
    """Separate source-page clipping/metadata from a newly clipped detail crop."""
    transform=fitz.Matrix(*mapping['source_to_pixels']);inverse=fitz.Matrix(*mapping['pixels_to_source'])
    x0,y0,x1,y1=[round(v) for v in list(fitz.Rect(context)*transform)]
    rgb=image.convert('RGB');notes=[];unresolved=False
    metadata=[r['box'] for r in regions if r['kind']=='document_metadata' and
              intersection(r['box'],structure)==0 and
              not any(intersection(r['box'],a['box'])>0 for a in annotations)]
    edges=([(x,y0) for x in range(x0,x1+1)],[(x,y1) for x in range(x0,x1+1)],
           [(x0,y) for y in range(y0,y1+1)],[(x1,y) for y in range(y0,y1+1)])
    for edge in edges:
        unknown=0
        for x,y in edge:
            if not (0<=x<rgb.width and 0<=y<rgb.height) or min(rgb.getpixel((x,y)))>=220:continue
            p=fitz.Point(x,y)*inverse
            if any(abs(p[i%2]-bounds[i])<=.75 for i in range(4)):
                notes.append('Ink already touches original page edge; no additional crop loss established')
            elif any(fitz.Rect(b).contains(p) for b in metadata):
                notes.append('Crop-edge ink belongs to separate document metadata')
            else:unknown+=1
        unresolved |= unknown>1
    return unresolved,list(dict.fromkeys(notes))


def verify(region,regions,bounds,spans=(),vectors=(),image=None,mapping=None):
    initial=list(region.get('initial_box',region['box']));context=list(region['box']);structure=region['structural_box']
    issues=[];corrections=[];proofs=[];ambiguous=[];boundary_notes=[]
    if not valid(context,bounds) or not valid(structure,bounds) or not contains(context,structure):
        issues.append('Invalid structural/context bounds or coordinate transform')
    if mapping:
        try:
            forward=fitz.Matrix(*mapping['source_to_pixels']);inverse=fitz.Matrix(*mapping['pixels_to_source'])
            for p in ((bounds[0],bounds[1]),(bounds[2],bounds[3])):
                if abs(fitz.Point(*p)*forward*inverse-fitz.Point(*p))>.01:raise ValueError('inverse')
        except (ValueError,KeyError,TypeError):issues.append('Source/pixel coordinate transforms do not round-trip')
    annotations=[a for e in region.get('evidence',[]) for a in e.get('context_annotations',[])]
    others=[r for r in regions if r is not region and r['kind']=='assembly_candidate']
    for a in annotations:
        path=a.get('leader_points',[]);label=a.get('label_box');target=a.get('target_box')
        proposed_path=path
        if label and lines.valid_path(path):
            from .callouts import vector_trace
            traced,_,issue=vector_trace(path,vectors,2,structure,label)
            if traced and not issue:path=traced
        proof=lines.verify(path,vectors,image,mapping)
        owners=[r['model_id'] for r in regions if r['kind']=='assembly_candidate' and lines.valid_path(path)
            and fitz.Rect(r['structural_box']).contains(fitz.Point(*path[-1]))]
        supported=(label and valid(label,bounds) and target and contains(structure,target)
            and lines.valid_path(path) and distance(label,[*path[0],*path[0]])<=lines.label_start_tolerance(label)
            and fitz.Rect(target).contains(fitz.Point(*path[-1])) and owners==[region['model_id']]
            and proof['verified'])
        item=dict(annotation=dict(a,leader_points=path),proposed_path=proposed_path,
                  source_verification=proof,ownership_verified=bool(supported));proofs.append(item)
        if not supported:
            issues.append('Annotation leader or label ownership is not source-verified');ambiguous.append(a['box']);continue
        required=union(a['box'],label)
        for p in path:required=union(required,[p[0]-.01,p[1]-.01,p[0]+.01,p[1]+.01])
        if not contains(context,required):corrections.append(dict(reason='Source-verified complete label/leader',box=required));context=union(context,required)
    extents=text_extents(spans,vectors)
    segments=list(lines.vector_segments(vectors))
    outward=outward_paths(segments,structure)
    for text in extents:
        b=text['box']
        if not valid(b,bounds):continue
        owners=[p for p in proofs if p['ownership_verified'] and contains(p['annotation']['label_box'],b)]
        # Entire outside labels connected by a native line are retained even if
        # the model omitted them. This is candidate ownership, not acceptance.
        touching=[]
        if not owners and not contains(context,b):
            for trace in outward:
                endpoint=trace['path'][-1]
                if distance(b,[*endpoint,*endpoint])<=2:touching.append(trace)
        clipped=intersection(context,b)>0 and not contains(context,b)
        if owners or touching or clipped:
            if not contains(context,b):
                context=union(context,b);corrections.append(dict(reason='Retain complete source text extent',**text))
            if not owners:
                issues.append('Text extent retained but annotation ownership remains ambiguous');ambiguous.append(b)
    for trace in outward:
        path=trace['path']
        if any(not fitz.Rect(context).contains(fitz.Point(*p)) for p in path):
            extent=[min(p[0] for p in path)-.01,min(p[1] for p in path)-.01,
                    max(p[0] for p in path)+.01,max(p[1] for p in path)+.01]
            extent=[max(bounds[0],extent[0]),max(bounds[1],extent[1]),min(bounds[2],extent[2]),min(bounds[3],extent[3])]
            context=union(context,extent);corrections.append(dict(reason='Retain complete source elbow/leader chain',box=extent))
            issues.append('Unenumerated source line chain requires annotation ownership REVIEW')
        if trace['ambiguous']:issues.append('Source line chain branches, crosses ownership, or exceeds bounded tracing; REVIEW')
    # A visible line that exits the context cannot be silently clipped. Retain
    # its full extent; do not decide whether it is a dimension or component.
    for o,n,a,b in segments:
        inside=[fitz.Rect(context).contains(fitz.Point(*p)) for p in (a,b)]
        if inside[0]!=inside[1] and (fitz.Rect(structure).contains(fitz.Point(*a)) or fitz.Rect(structure).contains(fitz.Point(*b))):
            extent=[min(a[0],b[0])-.01,min(a[1],b[1])-.01,max(a[0],b[0])+.01,max(a[1],b[1])+.01]
            extent=[max(bounds[0],extent[0]),max(bounds[1],extent[1]),min(bounds[2],extent[2]),min(bounds[3],extent[3])]
            context=union(context,extent);corrections.append(dict(reason='Retain source line crossing crop boundary',box=extent,vector=[o,n]))
            issues.append('Boundary-crossing source line requires component/dimension ownership review')
    for other in others:
        if intersection(context,other['structural_box'])>0:
            issues.append('Context overlaps an adjacent assembly; competing ownership requires REVIEW')
    # Inspect actual page ink immediately on the four final crop edges. Text
    # bounds alone do not protect raster-only pages or an unenumerated leader.
    if image is not None and mapping:
        clipped,boundary_notes=boundary_ink(context,bounds,structure,annotations,regions,image,mapping)
        if clipped:
            issues.append('Original-source ink touches corrected crop boundary; possible clipped label/leader')
    if not spans and not vectors:issues.append('Raster-only context lacks indexed text/vector completeness evidence')
    if not annotations:issues.append('No structured annotation/leader census; completeness cannot be demonstrated')
    if region.get('review_reasons'):issues.append('Detection has unresolved eligibility or annotation reviews')
    if not contains(bounds,context):issues.append('Corrected context leaves source page')
    return dict(version=VERSION,status='REVIEW' if issues else 'COMPLETE',initial_context_box=initial,boundary_notes=boundary_notes,
        structural_box=list(structure),corrected_context_box=context,corrections=corrections,
        annotation_proofs=proofs,ambiguous_context=ambiguous,issues=list(dict.fromkeys(issues)),
        wording_not_inferred=True)


def write(folder,source_image,mapping,region,review,crop):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    matrix=fitz.Matrix(*mapping['source_to_pixels'])
    for name,boxes in [('structural_bbox_overlay.png',[(review['structural_box'],'green')]),
        ('context_bbox_overlay.png',[(review['initial_context_box'],'red'),(review['corrected_context_box'],'blue')])]:
        layer=source_image.copy();draw=ImageDraw.Draw(layer)
        for box,color in boxes:draw.rectangle(list(fitz.Rect(box)*matrix),outline=color,width=3)
        layer.save(folder/name)
    crop.save(folder/'detail_crop.png')
    text=['# Crop completeness review','',f'Status: {review["status"]}',
        'Red = initial model context; blue = source-corrected context; green = structural geometry.',
        f'Initial context: {review["initial_context_box"]}',f'Corrected context: {review["corrected_context_box"]}',
        '',*[f'- {r}' for r in review['issues']],
        '',*[f'- {r["reason"]}: {r["box"]}' for r in review['corrections']],
        '', 'Outlines indicate source extents only; they do not supply guessed text.','']
    (folder/'crop_completeness_review.md').write_text('\n'.join(text))
    (folder/'crop_completeness.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')

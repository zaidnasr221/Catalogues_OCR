"""Deterministic original-source polyline evidence; never counts model totals."""
from __future__ import annotations

import math
import pymupdf as fitz


def valid_path(path):
    return (isinstance(path,list) and 2<=len(path)<=100 and
        all(isinstance(p,(tuple,list)) and len(p)==2 and all(isinstance(x,(int,float)) and
            not isinstance(x,bool) and math.isfinite(x) for x in p) for p in path) and
        all(math.dist(a,b)>1e-6 for a,b in zip(path,path[1:])))


def length(path):
    return sum(math.dist(a,b) for a,b in zip(path,path[1:]))


def point_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    t=max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy or 1)))
    return math.dist(p,(a[0]+t*dx,a[1]+t*dy))


def samples(path,step=.5):
    for a,b in zip(path,path[1:]):
        n=max(1,math.ceil(math.dist(a,b)/step))
        for i in range(n):yield [a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n]
    yield list(path[-1])


def same_line(a,b,tolerance=.15):
    """Whole-path identity, including bends, independent of text and sampling."""
    if not valid_path(a) or not valid_path(b):return False
    if min(max(math.dist(a[0],b[0]),math.dist(a[-1],b[-1])),
           max(math.dist(a[0],b[-1]),math.dist(a[-1],b[0])))>tolerance:return False
    return all(min(point_distance(p,c,d) for c,d in zip(right,right[1:]))<=tolerance
        for left,right in ((a,b),(b,a)) for p in samples(left,max(.05,tolerance)))


def crosses(a,b,c,d):
    def turn(p,q,r):return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    return turn(a,b,c)*turn(a,b,d)<-1e-7 and turn(c,d,a)*turn(c,d,b)<-1e-7


def crossing_outside_target(a,b,c,d,target_box=None):
    """A drawing contour inside the owned assembly is not a competing leader.

    Peer leader crossings are checked separately by conflict(), without this exemption.
    """
    if not crosses(a,b,c,d):return False
    if target_box is None:return True
    dx,dy=b[0]-a[0],b[1]-a[1]
    ex,ey=d[0]-c[0],d[1]-c[1]
    t=((c[0]-a[0])*ey-(c[1]-a[1])*ex)/(dx*ey-dy*ex)
    return not fitz.Rect(target_box).contains(fitz.Point(a[0]+t*dx,a[1]+t*dy))


def conflict(left,right,shared_label_box=None):
    if same_line(left,right):return False
    common_start=(shared_label_box and all(shared_label_box[0]-.15<=p[0]<=shared_label_box[2]+.15
        and shared_label_box[1]-.15<=p[1]<=shared_label_box[3]+.15 for p in (left[0],right[0])))
    return (any(math.dist(p,q)<.15 and not (common_start and i==j==0)
        for i,p in enumerate((left[0],left[-1])) for j,q in enumerate((right[0],right[-1]))) or
        any(same_line([a,b],[c,d]) for a,b in zip(left,left[1:]) for c,d in zip(right,right[1:])) or
        any(crosses(a,b,c,d) or
        (point_distance(c,a,b)<.15 and min(math.dist(c,a),math.dist(c,b))>.3) or
        (point_distance(d,a,b)<.15 and min(math.dist(d,a),math.dist(d,b))>.3)
        for a,b in zip(left,left[1:]) for c,d in zip(right,right[1:])))


def vector_segments(vectors):
    for ordinal,path in vectors:
        for number,item in enumerate(path.get('items',[])):
            if item[0]=='l' and math.dist(item[1],item[2])>1e-6:
                yield ordinal,number,list(item[1]),list(item[2])


def label_start_tolerance(box):
    """Same source-coordinate label allowance in extraction and leader review."""
    return max(2.0, (box[3]-box[1])*.8) if box else 2.0


def arrowhead_segments(vectors,tip):
    """Small filled, closed source triangles sharing the exact leader tip."""
    result=set()
    for ordinal,vector in vectors:
        items=vector.get('items',[])
        if vector.get('fill') is None or len(items)!=3 or any(i[0]!='l' for i in items):continue
        if not all(math.dist(items[n][2],items[(n+1)%3][1])<=.1 for n in range(3)):continue
        points=[i[1] for i in items]
        if min(math.dist(p,tip) for p in points)<=.15 and max(math.dist(p,tip) for p in points)<=8:
            result.update((ordinal,i) for i in range(3))
    return result


def trace_label_target(label, target, vectors, tolerance=2):
    """Trace a unique source route between two anchors, not model-made bends.

    Only source endpoints join (.1 PDF unit). Interior intersections are never
    graph junctions. A bounded target allowance cannot select a nearest route:
    two routes, even with unequal distances, remain unresolved.
    """
    from .geometry import distance
    if not label or not target:return None, [], None
    allowance=max(tolerance,min(4.0,(label[3]-label[1])*.25))
    start_allowance=label_start_tolerance(label)
    reach=math.dist([(label[0]+label[2])/2,(label[1]+label[3])/2],target)
    margin=max(12,reach*.2)
    bounds=[min(label[0],target[0])-margin,min(label[1],target[1])-margin,
            max(label[2],target[0])+margin,max(label[3],target[1])+margin]
    nodes=[];grid={};adj={};edges=[]
    def node(p):
        key=(math.floor(p[0]/.1),math.floor(p[1]/.1))
        near=[n for x in range(key[0]-1,key[0]+2) for y in range(key[1]-1,key[1]+2)
              for n in grid.get((x,y),[]) if math.dist(p,nodes[n])<=.1]
        if len(near)>1:raise ValueError('Ambiguous source endpoint junction')
        if near:return near[0]
        n=len(nodes);nodes.append(p);grid.setdefault(key,[]).append(n);adj[n]=[];return n
    try:
        for ordinal,item,a,b in vector_segments(vectors):
            if not all(bounds[0]<=p[0]<=bounds[2] and bounds[1]<=p[1]<=bounds[3] for p in (a,b)):continue
            if len(edges)>=12000:return None, [], 'Indexed leader search limit; path unresolved'
            u,v=node(a),node(b)
            if u==v:continue
            n=len(edges);edges.append((ordinal,item,a,b));adj[u].append((v,n));adj[v].append((u,n))
    except ValueError as exc:return None, [], str(exc)
    starts=[n for n,p in enumerate(nodes) if len(adj[n])==1 and
            distance(label,[*p,*p])<=start_allowance and math.dist(p,target)>allowance]
    goals={n for n,p in enumerate(nodes) if math.dist(p,target)<=allowance}
    stack=[([n],[],0.) for n in starts];routes=[];steps=0
    while stack:
        route,used,total=stack.pop();steps+=1
        if steps>4096:return None, [], 'Indexed leader search limit; path unresolved'
        n=route[-1]
        if n in goals:
            routes.append((route,used))
            if len(routes)>1:return None, [], 'Indexed leader anchor route is not unique'
            continue
        if len(route)>=100:continue
        for nxt,e in adj[n]:
            dist=total+math.dist(nodes[n],nodes[nxt])
            if nxt not in route and dist<=reach*2.5+2*start_allowance:
                stack.append((route+[nxt],used+[e],dist))
    if not routes:return None, [], None
    route,used=routes[0]
    # Keep original segment coordinates, not graph-cluster approximations.
    path=[];matched=[]
    for n,e in zip(route,used):
        ordinal,item,a,b=edges[e]
        if math.dist(nodes[n],b)<math.dist(nodes[n],a):a,b=b,a
        if not path:path.append(a)
        path.append(b);matched.append((ordinal,item))
    return path,matched,None


def shared_label_branch(left,right,label):
    """Same label and source-verified paths may share a trunk, never rejoin.

    Callers must establish indexed-source support for both paths separately.
    """
    from .geometry import distance
    if not label or not valid_path(left) or not valid_path(right):return False
    if any(distance(label,[*p[0],*p[0]])>label_start_tolerance(label) for p in (left,right)):return False
    n=0
    while n<min(len(left),len(right)) and math.dist(left[n],right[n])<=.15:n+=1
    if n<2 or n>=min(len(left),len(right)):return False
    fork=left[n-1];box=[fork[0]-.15,fork[1]-.15,fork[0]+.15,fork[1]+.15]
    if conflict(left[n-1:],right[n-1:],box):return False
    # Neither outgoing branch may cross or touch the earlier common trunk.
    return not any(crosses(a,b,c,d) or point_distance(d,a,b)<.15
        for path in (left,right) for a,b in zip(left[:n-1],left[1:n])
        for c,d in zip(path[n-1:],path[n:]))


def match_vector_path(path, vectors, tolerance=.8):
    """Return one connected source chain per proposed leg, preserving bends.

    Proposal tolerance never bridges source gaps: joins retain the existing .1
    source-unit limit. Multiple routes are unresolved, never nearest-line wins.
    Search is bounded so dense drawings cannot trigger unbounded traversal.
    """
    if not valid_path(path):return None, [], 'Invalid source polyline'
    source=list(vector_segments(vectors));points=[];used=[]
    for a,b in zip(path,path[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        edges=[]
        for ordinal,item,c,d in source:
            if point_distance(c,a,b)>tolerance or point_distance(d,a,b)>tolerance:continue
            progress=(d[0]-c[0])*dx+(d[1]-c[1])*dy
            if abs(progress)<1e-9:continue
            if progress<0:c,d=d,c
            edges.append((ordinal,item,c,d))
        if len(edges)>2048:return None, [], 'Indexed leader search limit; path unresolved'
        stack=[([edge], {edge[:2]}) for edge in edges if math.dist(a,edge[2])<=tolerance]
        routes=[];steps=0
        while stack:
            chain,seen=stack.pop();steps+=1
            if steps>4096:return None, [], 'Indexed leader search limit; path unresolved'
            end=chain[-1][3]
            if math.dist(end,b)<=tolerance:
                routes.append(chain)
                if len(routes)>1:return None, [], 'Indexed leader match is not unique'
                continue
            if len(chain)>=99:return None, [], 'Indexed leader chain exceeds path limit'
            for edge in edges:
                if edge[:2] not in seen and math.dist(end,edge[2])<=.1:
                    if steps+len(stack)>=4096:
                        return None, [], 'Indexed leader search limit; path unresolved'
                    stack.append((chain+[edge],seen|{edge[:2]}))
        if not routes:return None, [], None
        for ordinal,item,c,d in routes[0]:
            if (ordinal,item) in used:return None, [], 'Indexed leader reuses a source segment'
            if points and math.dist(points[-1],c)>.1:
                return None, [], 'Indexed leader path is discontinuous'
            if not points:points.append(c)
            points.append(d);used.append((ordinal,item))
    if not valid_path(points):return None, [], 'Invalid source chain'
    return points,used,None


def verify(path,vectors=(),image=None,mapping=None,tolerance=.8):
    """Match every proposed segment, or verify continuous ink on original pixels.

    This verifies geometry only; callers independently check component semantics,
    label/target ownership, complete coverage and competing lines.
    """
    if not valid_path(path):return dict(verified=False,reason='Invalid or missing leader polyline')
    traced,used,issue=match_vector_path(path,vectors,tolerance)
    if issue:return dict(verified=False,reason=issue)
    if traced:
        return dict(verified=True,method='indexed_source_vectors',segments=[list(x) for x in used])
    if image is None or mapping is None:
        return dict(verified=False,reason='No unique source-vector trace or original raster verification')
    transform=fitz.Matrix(*mapping['source_to_pixels'])
    pixel_path=[list(fitz.Point(*p)*transform) for p in path]
    rgb=image.convert('RGB');pixels=rgb.load();hits=0;total=0;run=0;longest=0
    for x,y in samples(pixel_path,.75):
        total+=1;x,y=round(x),round(y)
        if not 0<=x<rgb.width or not 0<=y<rgb.height:
            return dict(verified=False,reason='Leader leaves original source viewport')
        # Saturated cyan/yellow engineering lines can be bright in grayscale.
        ink=any(min(pixels[i,j])<200 for i in range(max(0,x-1),min(rgb.width,x+2))
                for j in range(max(0,y-1),min(rgb.height,y+2)))
        hits+=ink;run=0 if ink else run+1;longest=max(longest,run)
    ratio=hits/max(1,total)
    return dict(verified=ratio>=.97 and longest<=3,method='original_source_pixels',
        supported_fraction=ratio,max_missing_run_pixels=longest*.75,
        reason=None if ratio>=.97 and longest<=3 else 'Proposed line lacks continuous original-source ink')

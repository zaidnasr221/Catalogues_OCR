"""Resolve printed local item-marker legends without inventing component lines."""
import re
from .geometry import area,intersection
from . import line_evidence,provenance


def contained(outer,inner):
    return bool(outer and inner and area(inner)>0 and intersection(outer,inner)>=area(inner)-1e-5)


def valid(record,view):
    link=record.get('legend_link') or {}
    context=view.get('crop_geometry',{}).get('source_context_box',view['box'])
    if (record.get('category')!='item_marker' or record.get('reading_state')!='exact'
        or record.get('reading_review_reasons') or record.get('reading_clipped')
        or record.get('association_method')=='unresolved' or record.get('source_view_id')!=view['id']
        or record.get('leader_kind')!='component' or not line_evidence.valid_path(record.get('leader_path'))):return False
    if (not link or link.get('marker')!=record.get('raw_code') or not link.get('scope_text')
        or not link.get('evidence') or not contained(context,link.get('scope_box'))
        or not contained(link['scope_box'],record.get('text_box'))
        or not contained(link['scope_box'],link.get('row_box'))
        or not contained(link['row_box'],link.get('code_box'))):return False
    code=link.get('code','');name=link.get('name')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{1,79}',code) or not re.search(r'\d',code):return False
    from .callouts import non_code_category
    if non_code_category(code) or provenance.standalone_item_number(code):return False
    if code not in link.get('row_text','') or link['marker'] not in link['row_text']:return False
    if name and (name not in link['row_text'] or not contained(link['row_box'],link.get('name_box'))):return False
    return True


def resolve(records,view,views=()):
    proposals=[r for r in records if valid(r,view) and not any(
        other['id']!=view['id'] and other.get('kind')=='assembly_candidate'
        and intersection(r['legend_link']['scope_box'],other['structural_box'])>0 for other in views)]
    # All proposed readings in the same explicit scope must agree on this marker.
    for record in proposals:
        link=record['legend_link']
        same=[r for r in proposals if r['legend_link']['marker']==link['marker']
            and r['legend_link']['scope_text']==link['scope_text']
            and intersection(r['legend_link']['scope_box'],link['scope_box'])>=.9*area(link['scope_box'])]
        readings={(r['legend_link']['code'],r['legend_link'].get('name'),r['legend_link']['row_text']) for r in same}
        confirms=[r for r in same if line_evidence.same_line(record['leader_path'],r['leader_path'],.8)
            and intersection(r['legend_link']['row_box'],link['row_box'])>=.8*min(area(r['legend_link']['row_box']),area(link['row_box']))]
        if len(readings)!=1 or {r.get('leader_audit_pass') for r in confirms}!={'primary','independent_audit'}:
            record.setdefault('review_reasons',[]).append('Local legend ambiguous or missing independent source confirmation')
            continue
        record['legend_resolution']=dict(link,status='SUPPORTED',
            marker_box=record['text_box'],source_page=record['page_index']+1)
        record.update(category='product_code',raw_code=link['code'],normalized_code=link['code'],
            reading_method='independent_local_legend_readings',
            reading_support=[dict(raw_text=link['code'],method='local_legend',evidence=link['evidence'])])


def name(occurrence):
    link=occurrence.get('legend_resolution') or {}
    if link.get('status')=='SUPPORTED' and link.get('code')==occurrence.get('raw_code'):
        return link.get('name') or ''
    return ''

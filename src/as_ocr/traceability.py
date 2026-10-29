"""Occurrence accounting independent of physical-profile counting."""
from __future__ import annotations

import json


def audit(occurrences, candidates, memberships, traces=(), *, target='final CSV'):
    records = []
    for occurrence in occurrences:
        oid = occurrence['id']
        linked = [c for c in candidates if oid in c.get('occurrence_ids', [])]
        members = [m for m in memberships if m.get('occurrence_id') == oid]
        rows = [t for t in traces if t['row_key'][2] == oid]
        reasons = list(occurrence.get('review_reasons', []))
        reasons += [r for t in rows for r in t.get('reasons', [])]
        if rows:
            status = 'EXPORTED'
        elif not occurrence.get('raw_code') or occurrence.get('association_method') == 'unresolved':
            status = 'UNRESOLVED'
            reasons.append('Reading or Detail ownership unresolved; retained in source evidence')
        else:
            status = 'WITHHELD'
            reasons.append('No eligible row reached '+target)
        if not linked:
            reasons.append('No Stage 5 candidate retained for this Stage 4 occurrence')
        if not members:
            reasons.append('No Stage 7 membership row retained')
        records.append(dict(occurrence_id=oid, document_id=occurrence['document_id'],
            page_id=occurrence['page_id'], raw_text=occurrence.get('raw_code'),
            text_box=occurrence.get('text_box'), status=status, export_target=target,
            candidate_ids=[c['id'] for c in linked], membership_ids=[m['id'] for m in members],
            csv_rows=[t['csv_row_number'] for t in rows],
            reasons=list(dict.fromkeys(reasons)) or ['Independent source occurrence preserved']))
    if len({r['occurrence_id'] for r in records}) != len(records):
        raise ValueError('Duplicate occurrence identity in audit')
    return records


def from_snapshot(summary, snapshot, traces):
    """Read every Stage 4 occurrence, including candidates withheld before Stage 7."""
    s6 = snapshot.json(summary['artifacts']['enrichment']) if summary['artifacts'].get('enrichment') else {}
    occurrences = {}; candidates = {};missing_source=False
    for source in s6.get('sources',[]):
        try:
            s4 = snapshot.json(source['stage4_path'], source['stage4_sha256'])
        except FileNotFoundError:
            missing_source=True
            continue
        for o in s4['occurrences']:
            if o['id'] in occurrences and occurrences[o['id']] != o:
                raise ValueError('Conflicting occurrence audit source')
            occurrences[o['id']] = o
        if source.get('stage5_path'):
            s5 = snapshot.json(source['stage5_path'], source.get('stage5_sha256'))
            for c in s5.get('candidates', []):
                candidates[c['id']] = c
    members = snapshot.table(summary['artifacts']['detail_components'])
    if missing_source or not s6:
        components={c['id']:c for c in snapshot.table(summary['artifacts']['components'])}
        for m in members:
            if m['occurrence_id'] in occurrences:continue
            component=components.get(m['component_id'],{})
            occurrences[m['occurrence_id']]=dict(id=m['occurrence_id'],document_id=component.get('document_id'),
                page_id=component.get('page_id'),raw_code=None,association_method='unresolved',
                review_reasons=['Stage 4 source manifest unavailable; full occurrence inventory cannot be verified'])
    return audit(list(occurrences.values()), list(candidates.values()), members, traces)


def write(path, records):
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def markdown(records):
    lines = ['## Stage 4 occurrence accounting', '',
             'Every independent occurrence is accounted for; counts are separate geometry observations.', '']
    for r in records:
        lines.append(f'- {r["occurrence_id"]}: {r["status"]}; rows {r["csv_rows"]}; '+
                     '; '.join(r['reasons']))
    return '\n'.join(lines)+'\n'

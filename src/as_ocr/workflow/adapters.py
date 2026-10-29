"""New serialization/admission adapters around unchanged validated primitives."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import hashlib
import tempfile

import pymupdf as fitz

from as_ocr import components, delivery, exports, grouping, leaders, presentation, provenance, public_contract

WORKSPACE = Path(__file__).resolve().parents[3]
CSV_COLUMNS = tuple(presentation.COLUMNS)


def check_workspace():
    # A source-only NEW_VOCR release has no Git checkout or fixed installation path.
    if not (WORKSPACE/'pyproject.toml').is_file() or not (WORKSPACE/'src/as_ocr/pipeline.py').is_file():
        raise ValueError('Run from the complete NEW_VOCR project')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def admit_paths(output, internal):
    """Admit exclusively fresh public/internal paths; never adopt old results."""
    check_workspace()
    output, internal = Path(output).absolute(), Path(internal).absolute()
    scratch = WORKSPACE / '_internal'
    temporary = Path(tempfile.gettempdir()).resolve()
    for target in (output, internal):
        if target.exists() or target.is_symlink():
            raise FileExistsError('Refusing existing run directory: ' + str(target))
        if target.resolve() != target:
            raise ValueError('Run paths may not traverse symlinks or dot segments')
    if scratch not in internal.parents and temporary not in internal.parents:
        raise ValueError('Internal evidence must stay under _internal/ or a temporary test directory')
    if output.parent != WORKSPACE and scratch not in output.parents and temporary not in output.parents:
        raise ValueError('Use a fresh workspace-level public directory or an internal validation directory')
    if output == internal or output in internal.parents or internal in output.parents:
        raise ValueError('Public results and internal checkpoints must be separate')
    return output, internal


def read_json(path):
    # Existing strict parser rejects duplicate keys, non-finite values and oversized files.
    return presentation.Snapshot().json(path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def asset(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle, source.open('rb') as original:
        shutil.copyfileobj(original, handle)
    if grouping.file_digest(destination) != grouping.file_digest(source):
        raise ValueError('Copied source evidence changed')
    return str(destination)


def group_page(stage2, folder, *, index_path=None):
    """Run approved deterministic same-page grouping, without model assistance."""
    result = grouping.run([stage2], folder, config=grouping.Config(mode='deterministic'), index_path=index_path)
    if result['errors'] or len(result['manifests']) != 1:
        raise ValueError('Same-page grouping did not produce exactly one validated page manifest')
    return Path(result['manifests'][0])


def detail_image(detail, destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Existing helper checks crop hashes and preserves original resolution.
    exports.detail_image(detail['composites'], destination)


def recognize_snapshot(stage3, stage4, names=None):
    """Reuse existing stage handoff validation, names and leader validation."""
    source, views, details, calls, upstream = components.load_inputs(stage3, stage4)
    components.check_upstream(upstream, views, Path(source['source_path']))
    fields = read_json(names) if names else {}
    if names and fields['upstream']['stage4']['sha256'] != grouping.file_digest(stage4):
        raise ValueError('Name evidence belongs to another callout snapshot')
    review = exports.Reviews()
    counts = leaders.validate_saved(calls.get('leader_observations', []),
                                   {d['id']: d for d in details}, {v['id']: v for v in views}, review)
    return dict(source=source, views=views, details=details, callouts=calls,
                candidates=fields.get('candidates', []), observations=fields.get('observations', []),
                scope_observations=fields.get('scope_observations', []),
                inventories=counts, validation=list(review.rows.values()), validation_errors=review.errors(),
                internal_sources=dict(stage3=str(stage3), stage4=str(stage4), names=str(names) if names else None))


def project_detail(snapshot, detail, ordinal):
    """New five-column projection; extraction/acceptance helpers stay unchanged."""
    count, count_status = leaders.count_state(snapshot['inventories'].get(detail['id']))
    if snapshot['validation_errors']:
        count, count_status = '', 'INVALID_SAVED_LEADER_EVIDENCE'
    page = detail['page_index'] + 1
    view_ids = set(detail['view_ids'])
    views = {v['id']: v for v in snapshot['views']}
    rows, traces, exported = [], [], set()
    for call in snapshot['callouts']['occurrences']:
        vid = call.get('source_view_id')
        view = views.get(vid)
        if (vid not in view_ids or not view or call['document_id'] != detail['document_id']
                or call['page_id'] != detail['page_id'] or call.get('page_index') != page - 1
                or call.get('association_method') == 'unresolved' or not call.get('target_point')
                or not call.get('association_evidence')):
            continue
        metadata = [v['box'] for v in views.values() if v['kind'] == 'document_metadata'
                    and v['page_id'] == detail['page_id']]
        if provenance.metadata_source(call, [view['kind']], metadata):
            continue
        owners = [v['id'] for v in views.values() if v['page_id'] == detail['page_id']
                  and v['kind'] == 'assembly_candidate'
                  and fitz.Rect(v['structural_box']).contains(fitz.Point(*call['target_point']))]
        if owners != [vid]:
            continue
        code = call['raw_code'] if provenance.supported_product_reading(call, [view['kind']], metadata) else ''
        name, field_ids = public_contract.literal_name(call, snapshot['candidates'], snapshot['observations'], view)
        row = dict(detail_name=public_contract.detail_name(snapshot['source']['source_path'],ordinal), profile_count=count,
                   profile_id=code, component_name=name, source_page=str(page))
        trace = dict(detail_id=detail['id'], occurrence_id=call['id'], view_id=vid,
                     name_field_ids=field_ids, profile_count_status=count_status, withheld={}, reasons=[])
        if count_status not in {'COMPLETE', 'VERIFIED_ZERO'}:
            trace['reasons'].append('Extracted count retained; leader coverage requires review')
        if not name:
            trace['reasons'].append('No verified name in the associated original-source callout')
        if not code:
            trace['reasons'].append('No exact supported profile code')
        for key, value in list(row.items()):
            if presentation.unsafe(value):
                trace['withheld'][key] = value
                row[key] = ''
                trace['reasons'].append('CSV formula-safety withholding: ' + key)
        rows.append(row)
        traces.append(trace)
        exported.add(call['id'])
    if not rows:
        rows = [dict(detail_name=public_contract.detail_name(snapshot['source']['source_path'],ordinal), profile_count=count,
                     profile_id='', component_name='', source_page=str(page))]
        traces = [dict(detail_id=detail['id'], occurrence_id=None, profile_count_status=count_status,
                       reasons=['No source-supported associated component observation'], withheld={})]
    return rows, traces, exported


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if set(row) != set(CSV_COLUMNS) or any(not isinstance(value, str) for value in row.values()):
            raise ValueError('OutputV3 CSV must contain exactly five string fields')
        if (not re.fullmatch(r'[1-9][0-9]*', row['source_page']) or
                row['profile_count'] and not re.fullmatch(r'0|[1-9][0-9]*', row['profile_count'])):
            raise ValueError('Invalid original page or verified count')
    # Reserve an exclusive new file before using the existing public writer.
    with path.open('x'):
        pass
    delivery.write_csv(path, rows)
    if delivery.read_csv(path) != rows:
        raise ValueError('Five-column CSV round-trip mismatch')


def public_inventory(record):
    # Preserve all original evidence; do not recompute acceptance or totals.
    return deepcopy(record) if record else None

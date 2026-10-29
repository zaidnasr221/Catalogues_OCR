"""CPU-only Detail-centric presentation of explicitly selected Stage 7 snapshots.

Reads saved CSV/JSON and original-source review pixels. No PDF, database or
model inference is used; source leader evidence is revalidated before publication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

from . import traceability, public_contract

VERSION = '12'
COLUMNS = public_contract.COLUMNS
TABLES = ('details', 'detail_components', 'components', 'properties', 'evidence', 'review')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def structured(value):
    return json.loads(value) if value else []


def unsafe(value):
    # Detection only: do not normalize or alter any source string.
    return bool(value) and (value[0] in '\t\r\n' or
                           value.lstrip(' \t\r\n\v\f\0\ufeff\u200b')[0:1]
                           in ('=', '+', '-', '@', '＝', '＋', '－', '＠'))


class Snapshot:
    def __init__(self):
        self.files = {}

    def bytes(self, path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        if len(data) > 64 * 1024 * 1024:
            raise ValueError('Saved artifact exceeds size limit')
        sha = digest(data)
        if expected and sha != expected:
            raise ValueError('Saved provenance hash mismatch: ' + str(path))
        previous = self.files.setdefault(str(path), sha)
        if previous != sha:
            raise ValueError('Input changed during presentation')
        return data

    def json(self, path, expected=None):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON key')
                result[key] = value
            return result
        def invalid(value):
            raise ValueError('Non-finite JSON constant')
        return json.loads(self.bytes(path, expected), object_pairs_hook=unique,
                          parse_constant=invalid)

    def table(self, path):
        reader = csv.DictReader(io.StringIO(self.bytes(path).decode('utf-8-sig'), newline=''))
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError('Malformed saved CSV')
        return rows

    def verify(self):
        for path, sha in self.files.items():
            if digest(Path(path).read_bytes()) != sha:
                raise ValueError('Input changed during presentation: ' + path)


def index(rows):
    result = {}
    for row in rows:
        if row['id'] in result:
            raise ValueError('Duplicate saved identifier: ' + row['id'])
        result[row['id']] = row
    return result


def project(summary_path, *, details_only=False):
    return public_contract.project(summary_path, Snapshot, index, structured, unsafe,
                                   details_only=details_only)


def run(summary_paths, *, output=Path('output'), internal_root=Path('out'), details_only=False):
    output, internal_root = Path(output).resolve(), Path(internal_root).resolve()
    if output == internal_root or output in internal_root.parents or internal_root in output.parents:
        raise ValueError('Public and internal roots must be separate')
    prepared = [project(p, details_only=details_only) for p in summary_paths]
    if len({p[1] for p in prepared}) != len(prepared):
        raise ValueError('Select only one snapshot per catalog')
    publications = []
    for summary, catalog, filename, rows, traces, snapshot, unassigned, partial in prepared:
        occurrence_audit=traceability.from_snapshot(summary,snapshot,[t for r,t in zip(rows,traces) if r['profile_id'] or r['component_name']])
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
        data = buffer.getvalue().encode('utf-8-sig')
        if list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))) != rows:
            raise ValueError('Presentation CSV round-trip failed')
        stem = re.sub('[^A-Za-z0-9_-]+', '_', Path(filename).stem).strip('_')[:80] or 'catalog'
        name = stem + '__' + catalog + ('__PARTIAL' if partial else '') + '.csv'
        destination = output / name
        if destination.is_symlink():
            raise ValueError('Public destination must not be a symbolic link')
        if destination.exists():
            saved_reader=csv.DictReader(io.StringIO(destination.read_text(encoding='utf-8-sig')))
            old = list(saved_reader)
            if saved_reader.fieldnames!=list(COLUMNS):
                raise ValueError('Historical schema is preserved; select a fresh public output root')
            if old and 'catalog_id' in old[0]:
                owned = all(r['catalog_id'] == catalog for r in old)
            else:
                old_sha = digest(destination.read_bytes())
                owned = False
                for path in (internal_root/'presentations'/catalog).glob('*/manifest.json'):
                    previous = json.loads(path.read_text(encoding='utf-8'))
                    if (previous.get('catalog_id') == catalog and previous.get('csv_path') == str(destination)
                            and previous.get('csv_sha256') == old_sha):
                        owned = True
                        break
            if not owned:
                raise ValueError('Public catalog filename collision')
        if str(destination) in snapshot.files:
            raise ValueError('Publication would overwrite a saved artifact')
        manifest = {'version': VERSION, 'catalog_id': catalog, 'catalog_file': filename,
            'public_columns': list(COLUMNS), 'details_only': details_only,
            'stage7_run_id': summary['run_id'], 'publication_status': 'PARTIAL' if partial else 'COMPLETE',
            'source_acceptance_status': summary.get('acceptance_status'), 'csv_path': str(destination),
            'csv_sha256': digest(data), 'row_count': len(rows), 'source_files': snapshot.files,
            'unassigned_memberships': unassigned}
        trace_data = ''.join(encoded(t) + '\n' for t in traces).encode()
        publication_id = digest(encoded(manifest).encode() + trace_data)
        folder = internal_root / 'presentations' / catalog / publication_id
        folder.mkdir(parents=True, exist_ok=True)
        for path, content in ((folder/'manifest.json', (encoded(manifest)+'\n').encode()),
                              (folder/'rows.jsonl', trace_data), (folder/'catalog.csv', data)):
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError('Immutable presentation artifact differs')
            else:
                path.write_bytes(content)
        traceability.write(folder/'occurrence-audit.json',occurrence_audit)
        snapshot.verify()
        output.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() == data:
            publications.append({**manifest, 'trace': str(folder/'rows.jsonl')})
            continue
        # Stage outside output/: that directory contains only public catalog CSVs.
        with tempfile.NamedTemporaryFile(dir=internal_root, prefix='.present-', delete=False) as staging:
            staging.write(data)
            staging.flush()
            os.fsync(staging.fileno())
            temporary = Path(staging.name)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        publications.append({**manifest, 'trace': str(folder/'rows.jsonl')})
    return publications


def main(argv=None):
    parser = argparse.ArgumentParser(prog='as-ocr present', description=__doc__)
    parser.add_argument('summaries', type=Path, nargs='+', help='Explicit saved Stage 7 run-summary.json files')
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--internal-root', type=Path, default=Path('out'))
    parser.add_argument('--details-only', action='store_true', help='Publish only Detail-associated rows; unresolved occurrences remain internal')
    args = parser.parse_args(argv)
    try:
        publications = run(args.summaries, output=args.output, internal_root=args.internal_root, details_only=args.details_only)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Presentation failed: {type(exc).__name__}: {exc}\n')
    for item in publications:
        print(encoded({k: item[k] for k in ('catalog_id', 'publication_status', 'csv_path', 'row_count', 'trace')}))
    return 1 if any(p['publication_status'] == 'PARTIAL' for p in publications) else 0

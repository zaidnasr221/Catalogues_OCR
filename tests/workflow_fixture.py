"""Explicit synthetic fixture; scripted responses, no network/model inference."""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf as fitz

from as_ocr.qwen import ModelError

from dataclasses import replace

PATHS = [[[50, 80], [125, 100]], [[50, 150], [125, 135]]]
LABELS = [[18, 72, 51, 83], [18, 142, 51, 153]]
TABLE = '| ID | Value |\n| --- | --- |\n| 0007 | Ø 12,5 mm |\n| 0008 | 20 mm |'


def make_pdf(path, pages=1, *, assembly=True):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as pdf:
        for _ in range(pages):
            page = pdf.new_page(width=200, height=200)
            page.insert_text((10, 15), 'Assembly Notes', fontsize=8)
            page.insert_text((10, 28), 'Use source data only.', fontsize=5)
            for y, left, right in [(40, 'ID', 'Value'), (50, '0007', 'Ø 12,5 mm'), (60, '0008', '20 mm')]:
                page.insert_text((10, y), left, fontsize=5)
                page.insert_text((50, y), right, fontsize=5)
            if assembly:
                page.draw_rect((110, 90, 180, 150))
                page.draw_rect((120, 95, 175, 145))
                for line in PATHS:
                    page.draw_line(*line)
                    page.insert_text((20, line[0][1]), 'EXACT-001', fontsize=5)
        pdf.save(path)
    return path


class ScriptedClient:
    def __init__(self, backend, role, page, model):
        self.backend, self.role, self.page, self.config = backend, role, page, model
        self.requests = 0
        self.response_models = {'cpu-fixture'}

    def request_json(self, image, prompt, context_text='', *, response_schema):
        if self.requests >= self.config.max_requests:
            raise ModelError('CPU fixture manual request budget exhausted')
        self.requests += 1
        self.backend.requests.append(dict(role=self.role, page=self.page, schema=list(response_schema['properties'])))
        context = json.loads(context_text) if context_text else {}
        if self.backend.fail_role == self.role:
            raise ModelError('Scripted CPU fixture failure')
        if self.backend.cancel_role == self.role and self.requests == 2:
            raise KeyboardInterrupt('Scripted CPU cancellation after first reading')
        properties = response_schema['properties']
        if self.role == 'ocr':
            if 'regions' in properties:
                regions = [
                    dict(kind='heading', box=[.04, .02, .8, .10], rotation=0),
                    dict(kind='prose', box=[.04, .10, .8, .16], rotation=0),
                    dict(kind='table', box=[.04, .17, .6, .32], rotation=0)]
                if not self.backend.no_detail:
                    regions.append(dict(kind='drawing', box=[.04, .35, .95, .85], rotation=0))
                return json.dumps(dict(coverage='complete', uncertainty=[], regions=regions))
            if 'markdown' in properties:
                text = dict(heading='Assembly Notes', prose='Use source data only.', table=TABLE,
                            drawing='- EXACT-001\n- EXACT-001')[context['kind']]
                return json.dumps(dict(markdown=text, complete=True, unreadable=[]))
            return json.dumps(dict(complete=True, missing_regions=[], issues=[], unsupported_text=[]))
        if self.role == 'detection':
            if 'assessments' in properties:
                return json.dumps(dict(coverage='complete', evidence=['Scripted source coverage fixture'], uncertainty=[],
                    assessments=[dict(candidate_id=c['candidate_id'], action='keep', possible_assembly=True,
                        evidence='Known synthetic source assembly and complete label paths', replacements=[]) for c in context], additions=[]))
            annotations = []
            for line, label in zip(PATHS, LABELS):
                target = [124, line[-1][1] - 1, 126, line[-1][1] + 1]
                extent = [min(label[0], target[0]), min(label[1], target[1]), max(label[2], target[2]), max(label[3], target[3])]
                annotations.append(dict(box=[x / 200 for x in extent], label_box=[x / 200 for x in label],
                    target_box=[x / 200 for x in target], leader_points=[[x / 200, y / 200] for x, y in line],
                    association='leader', evidence='Known source vector from code label to component'))
            regions = [] if self.backend.no_detail else [dict(id='assembly', box=[.075, .34, .94, .81],
                structural_box=[.55, .45, .9, .75], kind='assembly_candidate', view_type='section', title=None,
                parent_id=None, evidence=['Two source fixture sections assembled'], uncertainty=[], incomplete=False,
                assembly_basis='connected_components', context_annotations=annotations)]
            return json.dumps(dict(decision='CANDIDATES' if regions else 'NO_DETAIL', page_kind='engineering',
                evidence=['Scripted source eligibility fixture'], coverage='complete', uncertainty=[], regions=regions, followups=[]))
        if self.role == 'callouts':
            # Existing context mapper tells us the exact original source crop;
            # no original PDF coordinates are invented by the test transport.
            mapping = context['image_geometry'] if 'image_geometry' in context else None
            # The known fixture context has the maintained default optional margin:
            # [15,68,188,162] expanded by 2 source units, within source bounds.
            crop = [13, 66, 190, 164]
            def point(p):
                return [(p[0] - crop[0]) / (crop[2] - crop[0]), (p[1] - crop[1]) / (crop[3] - crop[1])]
            observations = []
            for line, label in zip(PATHS, LABELS):
                observations.append(dict(raw_text='EXACT-001', category='product_code', reading_state='exact',
                    leader_kind='component', text_box=[*point(label[:2]), *point(label[2:])], native_candidate_ids=[],
                    reading_evidence=['Exact generated source label'], owner_view_id=context['focus_view_id'],
                    association_method='leader', leader_state='clear', leader_points=[point(p) for p in line],
                    target_point=point(line[-1]), association_evidence=['Known original source vector to component'], uncertainty=[]))
            return json.dumps(dict(coverage='complete', evidence=['Known source crop fully inspected'], uncertainty=[],
                observations=observations, leader_coverage='complete', leader_coverage_evidence=['Both physical source leaders accounted for']))
        if self.role == 'components':
            return json.dumps(dict(coverage='complete', evidence=['No explicit component names in generated callouts'],
                uncertainty=[], scope=[], components=[]))
        raise AssertionError('Unexpected model role: ' + self.role)


class CPUBackend:
    mode = 'CPU_SYNTHETIC'

    def __init__(self, *, fail_role=None, no_detail=False, cancel_role=None):
        self.fail_role, self.no_detail = fail_role, no_detail
        self.cancel_role = cancel_role
        self.requests, self.clients = [], []

    def prepare_model(self, role, model):
        return replace(model, endpoint='http://cpu.invalid/v1', model='cpu-fixture',
                       identity='cpu@scripted-fixture', api_key=None)

    def client(self, role, page, model):
        client = ScriptedClient(self, role, page, model)
        self.clients.append(client)
        return client


def source_regressions(tmp_path):
    """Reproduce the baseline with real PDF pixels/vectors and saved source boxes."""
    from copy import deepcopy
    import hashlib
    from as_ocr import crop_evidence, detection, leaders
    from test_leader_correction import fixture

    saved = json.loads((Path(__file__).parent / 'fixtures/alupco-page1-context.json').read_text())
    source = Path(saved['source'])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == saved['sha256']
    with fitz.open(source) as pdf:
        page = pdf[0]
        geometry = dict(source_to_display=list(page.rotation_matrix),
                        source_rect=list(page.rect * page.derotation_matrix))
        vectors = list(enumerate(page.get_drawings(extended=True)))
    image, mapping = detection.render(source, 0, geometry, geometry['source_rect'], 1600)
    reviews = [crop_evidence.verify(r, saved['regions'], geometry['source_rect'], [], vectors, image, mapping)
               for r in saved['regions'] if r['kind'] == 'assembly_candidate']
    detail, view, result, vec, _ = fixture(tmp_path)
    record = leaders.inventory(detail, [view], [result], vec)
    omitted = deepcopy(result)
    omitted['records'] = [omitted['records'][0], omitted['records'][2]]
    missing = leaders.inventory(detail, [view], [omitted], vec)
    duplicate = deepcopy(result)
    duplicate['records'].append(deepcopy(duplicate['records'][0]))
    repeated = leaders.inventory(detail, [view], [duplicate], vec)
    capture = dict(source_sha256=digest, source_page=1, render_mapping=mapping,
        page_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(), crop_reviews=reviews,
        leaders=dict(complete=leaders.count_state(record),
                     missing_known_detection_leader=leaders.count_state(missing),
                     duplicate=leaders.count_state(repeated),
                     accepted_paths=[o['path'] for o in record['observations'] if o['accepted']],
                     missing_issues=missing['issues']))
    return json.loads(json.dumps(capture))


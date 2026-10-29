"""Offline regressions: python -B -m as_ocr.line_matching_checks (no model)."""
import unittest
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from . import callouts, leaders, line_evidence as lines


def vectors(*pairs):
    return [(0, {'items': [['l', a, b] for a, b in pairs]})]


class LineMatchingChecks(unittest.TestCase):
    def trace(self, path, source, target=None):
        return callouts.vector_trace(path, source, 2.0, target)

    def test_whole_and_split_source_are_equivalent(self):
        path = [[10, 20], [70, 20]]
        for source in (vectors(*[path]), vectors(([10, 20], [35, 20]), ([35, 20], [70, 20]))):
            actual, evidence, issue = self.trace(path, source)
            self.assertIsNone(issue)
            self.assertTrue(lines.same_line(path, actual))
            self.assertEqual(len(evidence), len(source[0][1]['items']))
            self.assertTrue(lines.verify(path, source)['verified'])
            self.assertTrue(lines.verify(actual, source)['verified'])

    def test_split_elbow_preserves_corner_and_reversed_source(self):
        path = [[10, 20], [50, 20], [50, 60]]
        source = vectors(([30, 20], [10, 20]), ([50, 20], [30, 20]),
                         ([50, 40], [50, 20]), ([50, 60], [50, 40]))
        actual, evidence, issue = self.trace(path, source)
        self.assertIsNone(issue)
        self.assertEqual(actual, [[10, 20], [30, 20], [50, 20], [50, 40], [50, 60]])
        self.assertEqual(len(evidence), 4)

    def test_proposal_offset_snaps_to_source(self):
        actual, _, issue = self.trace([[10, 20.5], [70, 20.5]],
            vectors(([10, 20], [35, 20]), ([35, 20], [70, 20])))
        self.assertIsNone(issue)
        self.assertEqual(actual, [[10, 20], [35, 20], [70, 20]])

    def test_source_gap_not_bridged_by_proposal_tolerance(self):
        source = vectors(([10, 20], [35, 20]), ([35.5, 20], [70, 20]))
        actual, _, _ = self.trace([[10, 20], [70, 20]], source)
        self.assertIsNone(actual)
        self.assertFalse(lines.verify([[10, 20], [70, 20]], source)['verified'])

    def test_two_parallel_routes_stay_unresolved(self):
        source = vectors(([10, 19], [70, 19]), ([10, 21], [70, 21]))
        actual, _, issue = self.trace([[10, 20], [70, 20]], source)
        self.assertIsNone(actual)
        self.assertIn('not unique', issue)

    def test_duplicate_source_segments_stay_unresolved(self):
        pair = ([10, 20], [70, 20])
        actual, _, issue = self.trace([*pair], vectors(pair, pair))
        self.assertIsNone(actual)
        self.assertIn('not unique', issue)

    def test_branch_at_new_split_vertex_stays_unresolved(self):
        source = vectors(([10, 20], [35, 20]), ([35, 20], [70, 20]), ([35, 20], [35, 50]))
        actual, _, issue = self.trace([[10, 20], [70, 20]], source)
        self.assertIsNone(actual)
        self.assertIn('branch', issue)

    def test_crossing_outside_target_stays_unresolved(self):
        source = vectors(([10, 20], [35, 20]), ([35, 20], [70, 20]), ([45, 10], [45, 30]))
        actual, _, issue = self.trace([[10, 20], [70, 20]], source, [60, 15, 80, 30])
        self.assertIsNone(actual)
        self.assertIn('cross', issue)

    def test_contour_inside_target_retains_existing_exemption(self):
        source = vectors(([10, 20], [35, 20]), ([35, 20], [70, 20]), ([65, 10], [65, 30]))
        actual, _, issue = self.trace([[10, 20], [70, 20]], source, [60, 15, 80, 30])
        self.assertIsNotNone(actual)
        self.assertIsNone(issue)

    def test_old_elbow_branch_and_crossing_guards(self):
        path = [[10, 10], [40, 40], [80, 50]]
        source = vectors(*zip(path, path[1:]))
        self.assertIsNotNone(self.trace(path, source)[0])
        for extra in (([20, 40], [40, 20]), ([40, 40], [90, 20])):
            actual, _, issue = self.trace(path, source + [(2, {'items': [['l', *extra]]})])
            self.assertIsNone(actual)
            self.assertIsNotNone(issue)

    def test_no_invented_elbow(self):
        source = vectors(([10, 20], [50, 20]), ([50, 20], [50, 60]))
        self.assertIsNone(self.trace([[10, 20], [50, 60]], source)[0])

    def test_corrected_path_still_requires_real_pixels(self):
        path = [[10, 20], [70, 20]]
        actual, _, _ = self.trace(path, vectors(([10, 20], [35, 20]), ([35, 20], [70, 20])))
        mapping = {'source_to_pixels': [1, 0, 0, 1, 0, 0]}
        image = Image.new('RGB', (100, 100), 'white')
        self.assertFalse(lines.verify(actual, image=image, mapping=mapping)['verified'])
        ImageDraw.Draw(image).line([tuple(p) for p in path], fill='black')
        self.assertTrue(lines.verify(actual, image=image, mapping=mapping)['verified'])

    def test_label_allowance_uses_source_text_height(self):
        self.assertEqual(lines.label_start_tolerance([0, 0, 20, 10]), 8)
        self.assertEqual(lines.label_start_tolerance([0, 0, 20, 1]), 2)

    def test_component_count_contract_is_unchanged(self):
        path = [[10, 20], [70, 20]]
        split = [[10, 20], [35, 20], [70, 20]]
        occurrences = {
            'code': {'source_view_id': 'v', 'leader_path': path},
            'name': {'source_view_id': 'v', 'leader_path': split},
            'other': {'source_view_id': 'v', 'leader_path': [[10, 40], [70, 40]]},
        }
        rows = [('code', 'A1', '', []), ('name', '', 'Mullion', []), ('other', 'A1', '', [])]
        count, identities = leaders.extracted_component_count(rows, occurrences)
        self.assertEqual(count, '2')
        self.assertEqual(identities['code'], identities['name'])
        self.assertNotEqual(identities['code'], identities['other'])

    def test_callout_to_inventory_and_final_pixel_validation(self):
        from . import exports
        mapping = dict(source_to_pixels=[1, 0, 0, 1, 0, 0], pixels_to_source=[1, 0, 0, 1, 0, 0],
                       pixel_size=[100, 100], source_context_box=[0, 0, 100, 100])
        path = [[15, 20], [70, 20]]
        source = vectors(([15, 20], [40, 20]), ([40, 20], [70, 20]))
        observation = callouts.Observation.model_validate_json(json.dumps(dict(
            raw_text='A1', category='product_code', reading_state='exact', leader_kind='component',
            text_box=[.05, .15, .10, .25], native_candidate_ids=[], reading_evidence=['Visible code'],
            owner_view_id='v', association_method='leader', leader_state='clear',
            leader_points=[[.15, .20], [.70, .20]], target_point=[.70, .20],
            association_evidence=['Visible line to component'], uncertainty=[])))
        view = dict(id='v', document_id='doc', page_id='page', page_index=0, kind='assembly_candidate',
                    structural_box=[60, 15, 90, 40], primary_detail_ids=['d'], alternative_detail_ids=[],
                    source_review_reasons=[], source_review_status='SUPPORTED',
                    crop_completeness=dict(status='COMPLETE', annotation_proofs=[dict(
                        ownership_verified=True, annotation=dict(leader_points=path))]))
        detail = dict(id='d', document_id='doc', page_id='page', page_index=0, view_ids=['v'],
                      grouping_review_reasons=[], review_status='SUPPORTED')
        record = callouts.observation_record(observation, view, [view], [detail], mapping,
            [0, 0, 100, 100], [], source, callouts.Config())
        self.assertEqual(record['association_method'], 'indexed_vector_leader_with_visual_interpretation')
        self.assertEqual(record['leader_path'], [[15, 20], [40, 20], [70, 20]])
        with tempfile.TemporaryDirectory(prefix='callout-integration-') as temp:
            image = Image.new('RGB', (100, 100), 'white')
            ImageDraw.Draw(image).line([tuple(p) for p in path], fill='black')
            filename = Path(temp)/'source.png'
            image.save(filename)
            result = dict(view_id='v', leader_coverage='complete', leader_coverage_evidence=['Entire line inspected'],
                          leader_audit={'status': 'COMPLETE'}, source_image=str(filename), mapping=mapping,
                          records=[dict(record, leader_audit_pass=p) for p in ('primary', 'independent_audit')])
            inventory = leaders.inventory(detail, [view], [result], source)
            self.assertEqual(leaders.count_state(inventory), ('1', 'COMPLETE'))
            reviews = exports.Reviews()
            self.assertIn('d', leaders.validate_saved([inventory], {'d': detail}, {'v': view}, reviews))
            self.assertFalse(reviews.errors())

    def test_dense_ambiguous_search_is_bounded(self):
        source = vectors(*[([10, 20], [70, 20]) for _ in range(2050)])
        actual, _, issue = self.trace([[10, 20], [70, 20]], source)
        self.assertIsNone(actual)
        self.assertIn('search limit', issue)


if __name__ == '__main__':
    unittest.main()

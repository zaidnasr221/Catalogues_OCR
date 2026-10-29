"""Offline accuracy regressions. Optional --saved-run replays saved model replies."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image, ImageDraw
from . import callouts, components, crop_evidence, delivery, detection, grouping, leaders, line_evidence as lines, page_markdown
from .line_matching_checks import vectors


class AccuracyChecks(unittest.TestCase):
    def test_routes_generalize_across_orientation_position_and_scale(self):
        for turns in range(4):
            for scale in (.5,1,2):
                for offset in ((0,0),(340,510)):
                    def point(p):
                        x,y=p
                        for _ in range(turns):x,y=-y,x
                        return [x*scale+offset[0],y*scale+offset[1]]
                    def box(b):
                        corners=[point([x,y]) for x in (b[0],b[2]) for y in (b[1],b[3])]
                        return [min(p[0] for p in corners),min(p[1] for p in corners),
                                max(p[0] for p in corners),max(p[1] for p in corners)]
                    with self.subTest(turns=turns,scale=scale,offset=offset):
                        expected=[point(p) for p in [[15,20],[40,30],[70,60]]]
                        source=vectors(*zip(expected,expected[1:]))
                        path,_,issue=lines.trace_label_target(box([0,15,10,25]),point([70,60]),source)
                        self.assertIsNone(issue);self.assertEqual(path,expected)
                        left=[point(p) for p in [[15,60],[30,60],[70,20]]]
                        right=[point(p) for p in [[15,60],[30,60],[70,45]]]
                        branch=vectors(*zip(left,left[1:]),(right[1],right[2]))
                        label=box([0,55,10,65])
                        traced,_,issue=callouts.vector_trace([left[1],left[2]],branch,2,None,label,[right])
                        self.assertIsNone(issue);self.assertEqual(traced,left)
                        self.assertIsNone(callouts.vector_trace([left[1],left[2]],branch,2,None,label)[0])

    def test_no_detail_uses_content_evidence_not_page_or_catalog_identity(self):
        for page_number in (2,73,411):
            for kind in ('table','document_metadata','legend_notes','profile_candidate','assembly_candidate','mixed_content','unknown'):
                with self.subTest(page=page_number,kind=kind):
                    page=dict(page_index=page_number-1,initial_decision='NO_DETAIL',coverage='complete',evidence=['Source inspected'],
                        review_reasons=['Formatting review'],coverage_validation=dict(state='complete',audit=dict(
                            coverage='complete',evidence=['Independent inspection'],uncertainty=[],additions=[],
                            assessments=[dict(action='keep',possible_assembly=False)])))
                    expected=kind in {'table','document_metadata','legend_notes','profile_candidate'}
                    self.assertEqual(detection.confirmed_no_detail(page,[dict(kind=kind)]),expected)
                    page['coverage_validation']['audit']['assessments'][0]['possible_assembly']=True
                    self.assertFalse(detection.confirmed_no_detail(page,[dict(kind=kind)]))

    def test_formatting_recheck_does_not_depend_on_part_code_or_language(self):
        for literal in ('000731','ZX-42/B','Gasket for transom','جوان الزجاج'):
            for prefix in ('- ','## ','> '):
                with self.subTest(literal=literal,prefix=prefix):
                    audit=page_markdown.TextAudit(complete=False,issues=['Formatting'],missing_regions=[],unsupported_text=[prefix+literal])
                    good=page_markdown.TextAudit(complete=True,issues=[],missing_regions=[],unsupported_text=[])
                    text,checked=page_markdown.audit_formatted_text(prefix+literal,audit,lambda text:good)
                    self.assertEqual(text,literal);self.assertTrue(checked.complete)

    def test_single_view_does_not_invent_missing_multiview_evidence(self):
        view=dict(id='v',kind='assembly_candidate',structural_box=[0,0,20,20],box=[0,0,20,20],supported_titles=[],
                  document_id='doc',page_id='p',page_index=0,source_review_status='PROPOSED',
                  source_review_reasons=[],primary_detail_ids=[],alternative_detail_ids=[])
        details,_=grouping.build_details([view],[],grouping.Config())
        self.assertEqual(details[0]['grouping_status'],'SINGLETON')
        self.assertEqual(details[0]['grouping_review_reasons'],[])
        self.assertFalse(details[0]['complete'])
        other=copy.deepcopy(view);other.update(id='other',primary_detail_ids=[])
        proposal=dict(view_ids=['v','other'],evidence=[],titles=[],review_reasons=['Unproved shared title'],strength='tentative')
        details,_=grouping.build_details([view,other],[proposal],grouping.Config())
        self.assertTrue(all(d['grouping_review_reasons'] for d in details))

    def test_anchor_trace_uses_source_bends_and_rejects_alternatives(self):
        source=vectors(([15,20],[40,30]),([40,30],[70,60]))
        label=[0,15,10,25];wrong=[[15,20],[15,60],[70,60]]
        path,_,issue=callouts.vector_trace(wrong,source,2,[60,50,80,70],label)
        self.assertIsNone(issue);self.assertEqual(path,[[15,20],[40,30],[70,60]])
        other=source+vectors(([14,20],[70,61]))
        self.assertIsNotNone(lines.trace_label_target(label,[70,60],other)[2])
        self.assertIsNone(lines.trace_label_target(label,[90,60],source)[0])

    def test_shared_trunk_needs_other_source_verified_target(self):
        source=vectors(([15,60],[30,60]),([30,60],[70,20]),([30,60],[70,45]))
        label=[0,55,10,65];a=[[30,60],[70,20]];b=[[30,60],[70,45]]
        left=lines.trace_label_target(label,a[-1],source)[0]
        right=lines.trace_label_target(label,b[-1],source)[0]
        self.assertTrue(lines.shared_label_branch(left,right,label))
        self.assertIsNone(callouts.vector_trace(a,source,2,None,label)[0])
        self.assertIsNotNone(callouts.vector_trace(a,source,2,None,label,[right])[0])
        self.assertFalse(lines.shared_label_branch(left,list(reversed(right)),label))
        self.assertTrue(lines.conflict(left,right,label))  # unverified shared paths remain conflicts

    def test_formatting_recheck_preserves_reading_but_not_wrong_content(self):
        audit=page_markdown.TextAudit(complete=False,issues=['Formatting'],missing_regions=[],unsupported_text=['- VE116'])
        good=page_markdown.TextAudit(complete=True,issues=[],missing_regions=[],unsupported_text=[])
        text,checked=page_markdown.audit_formatted_text('- VE116',audit,lambda text: good)
        self.assertEqual(text,'VE116');self.assertTrue(checked.complete)
        bad=page_markdown.TextAudit(complete=False,issues=['Wrong digit'],missing_regions=[],unsupported_text=['VE116'])
        self.assertEqual(page_markdown.audit_formatted_text('- VE116',audit,lambda text: bad)[1].unsupported_text,['VE116'])
        self.assertEqual(page_markdown.literal_markdown('0012 / 1.5 mm'),'0012 / 1.5 mm')

    def test_no_detail_requires_independent_complete_eligibility(self):
        regions=[dict(kind='table')]
        audit=dict(coverage='complete',evidence=['Tables only'],uncertainty=[],additions=[],
                   assessments=[dict(action='keep',possible_assembly=False)])
        page=dict(initial_decision='NO_DETAIL',coverage='complete',evidence=['Tables only'],
                  review_reasons=['Header clipped'],coverage_validation=dict(state='complete',audit=audit))
        self.assertTrue(detection.confirmed_no_detail(page,regions))
        audit['assessments'][0]['possible_assembly']=True
        self.assertFalse(detection.confirmed_no_detail(page,regions))
        audit['assessments'][0]['possible_assembly']=False
        self.assertFalse(detection.confirmed_no_detail(page,regions,['Timeout']))

    def test_annotation_crop_ambiguity_does_not_become_no_detail(self):
        page=dict(initial_decision='REVIEW',coverage='complete',evidence=['Maybe table'],
                  coverage_validation=dict(state='complete',audit=dict(coverage='complete',evidence=['Table'],
                      uncertainty=[],additions=[],assessments=[dict(action='keep',possible_assembly=False)])))
        self.assertFalse(detection.confirmed_no_detail(page,[dict(kind='table')]))

    def test_ocr_format_recheck_failure_retains_withholding(self):
        audit=page_markdown.TextAudit(complete=False,issues=['Check'],missing_regions=[],unsupported_text=['- VE116'])
        from .qwen import ModelError
        def fail(text):raise ModelError('OCR page request budget exhausted')
        self.assertEqual(page_markdown.audit_formatted_text('- VE116',audit,fail),('- VE116',audit))

    def test_source_arrowhead_is_not_a_crossed_competing_leader(self):
        source=vectors(([15,20],[70,20]))
        triangle=(1,dict(fill=[0,0,0],items=[['l',[70,20],[65,18]],['l',[65,18],[65,22]],['l',[65,22],[70,20]]]))
        source.append(triangle)
        path,_,issue=callouts.vector_trace([[15,20],[70,20]],source,2,None,[0,15,10,25])
        self.assertIsNone(issue);self.assertIsNotNone(path)
        triangle[1]['fill']=None
        self.assertIsNone(callouts.vector_trace([[15,20],[70,20]],source,2,None,[0,15,10,25])[0])

    def test_anchor_search_does_not_bridge_source_gap(self):
        source=vectors(([15,20],[40,20]),([40.5,20],[70,60]))
        self.assertIsNone(lines.trace_label_target([0,15,10,25],[70,60],source)[0])

    def test_progress_distinguishes_failed_and_incomplete(self):
        import io
        from collections import defaultdict
        from . import progress
        reporter=object.__new__(progress.Reporter)
        reporter.source='fixture.pdf';reporter.total=230;reporter.stage=4;reporter.started=0
        reporter.active={};reporter.plans={1:set(range(1,11)),4:set(range(1,11))};reporter.finished=defaultdict(set)
        reporter.completed=set();reporter.failed=set();reporter.incomplete=set()
        reporter.details={};reporter.callouts={};reporter.errors=reporter.timeouts=0
        reporter.stream=io.StringIO();reporter.log=None
        reporter.consume(dict(event='done',pid=1,stage=4,page=1,ok=False,failed=False))
        reporter.consume(dict(event='done',pid=2,stage=4,page=2,ok=False,failed=True))
        reporter.summary(now=1)
        self.assertEqual(reporter.failed,{2});self.assertEqual(reporter.incomplete,{1})
        self.assertIn('/10 selected',reporter.stream.getvalue())

    def test_page_edge_is_distinguished_from_new_crop_loss(self):
        image=Image.new('RGB',(101,101),'white');draw=ImageDraw.Draw(image)
        draw.line([(0,30),(40,30)],fill='black',width=3)
        mapping=dict(source_to_pixels=[1,0,0,1,0,0],pixels_to_source=[1,0,0,1,0,0])
        args=([0,0,100,100],[5,5,90,90],[],[],image,mapping)
        self.assertFalse(crop_evidence.boundary_ink([0,0,100,100],*args)[0])
        self.assertTrue(crop_evidence.boundary_ink([20,0,100,100],*args)[0])


def replay(run):
    """Read-only source index + saved responses. No network calls or model."""
    run=Path(run)
    db=sqlite3.connect((run/'stage_1/index.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    totals={}
    for manifest_path in run.glob('stage_4/callouts/*/manifest.json'):
        saved=json.loads(manifest_path.read_text());up=saved['upstream'];page=up['page_index']+1
        if page not in [49,51,203]:continue
        s2,views,details,_=callouts.load_inputs(up['manifest_path'],up['stage3']['path'])
        row=db.execute('SELECT * FROM pages WHERE id=?',(up['page_id'],)).fetchone()
        geometry,spans,_=detection.load_page(db,row)
        source=[(o,json.loads(data)) for o,data in db.execute('SELECT ordinal,data FROM vectors WHERE page_id=? ORDER BY ordinal',(up['page_id'],))]
        tasks=[]
        for p in manifest_path.parent.glob('*-view.json'):
            v=json.loads(p.read_text());tasks.append((v['view_id'],v))
        for meta_path in manifest_path.parent.glob('diagnostics/*.json'):
            meta=json.loads(meta_path.read_text());vid=meta.get('subject_id')
            response=meta_path.with_suffix('.response.txt')
            if response.exists():tasks.append((vid,callouts.parse_response(response.read_text())))
        for vid,saved_view in tasks:
            view=next((v for v in views if v['id']==vid),None)
            if not view:continue
            image,mapping=detection.render(Path(s2['source_path']),up['page_index'],geometry,view['crop_geometry']['source_context_box'],2048)
            if isinstance(saved_view,dict):
                import pymupdf as fitz
                def normalized(p):
                    q=fitz.Point(*p)*fitz.Matrix(*mapping['source_to_pixels'])
                    return [q.x/mapping['pixel_size'][0],q.y/mapping['pixel_size'][1]]
                obs=[]
                for r in saved_view['records']:
                    if r.get('leader_audit_pass')!='primary':continue
                    g=r['proposed_geometry']
                    obs.append(callouts.Observation.model_validate_json(json.dumps(dict(
                        raw_text=r['raw_code'],category=r['category'],reading_state=r['reading_state'],
                        leader_kind=r['leader_kind'],text_box=callouts.normalized_box(r['text_box'],mapping),
                        native_candidate_ids=r['native_span_ids'],reading_evidence=r['reading_evidence'],
                        owner_view_id=g['owner_view_id'],association_method='leader',leader_state=g['leader_state'],
                        leader_points=[normalized(p) for p in g['leader_path']],target_point=normalized(g['target_point']),
                        association_evidence=r['association_evidence'],uncertainty=[]))))
            else:obs=saved_view.observations
            candidates=callouts.native_candidates(spans,view['crop_geometry']['source_context_box'])
            records,failures=callouts.observation_records(obs,view,views,details,mapping,geometry['source_rect'],candidates,source,callouts.Config())
            reconciled=callouts.reconcile(records,up)
            print('REPLAY',page,'rejected',failures,'records',[(r['raw_code'],r['association_method'],r['leader_path'],r['review_reasons']) for r in reconciled])
            for r in reconciled:
                if r.get('source_trace'):
                    proof=lines.verify(r['source_trace']['path'],image=image,mapping=mapping)
                    print('PIXEL',page,r['raw_code'],proof)
                    assert proof['verified'],(page,r['raw_code'],proof)
            totals.setdefault(page,[]).extend(reconciled)
            if page==203:
                assert len(failures)==1 and len(records)==3
                assert {r['raw_code'] for r in records}=={'2920','HM507','EE167'}
                with tempfile.TemporaryDirectory(prefix='.accuracy-check-',dir=Path(__file__).parent) as temp:
                    temp=Path(temp);copied=sqlite3.connect(temp/'index.sqlite3');db.backup(copied);copied.close()
                    from .qwen import ModelConfig
                    model=ModelConfig(endpoint='http://127.0.0.1:1/v1',model='offline-fixture',retries=0)
                    class SavedClient:
                        requests=0;response_models={'offline-fixture'}
                        def request_json(self,*args,**kwargs):
                            self.requests+=1;return saved_view.model_dump_json()
                    summary=callouts.run(up['manifest_path'],temp/'callouts',index_path=temp/'index.sqlite3',
                        stage3=up['stage3']['path'],view_ids=[vid],config=callouts.Config(details_only=True),
                        model=model,client=SavedClient(),force=True)
                    manifest=json.loads(Path(summary['manifest']).read_text())
                    assert manifest['state']=='PARTIAL' and len(manifest['occurrences'])==3,summary
                    assert any(v.get('observation_failures') for v in
                               (json.loads(p.read_text()) for p in Path(summary['manifest']).parent.glob('*-view.json')))
                    print('INTEGRATION 203: partial view saved; three occurrences retained')
    assert len(totals[49])==2 and all(r['vector_evidence'] for r in totals[49])
    assert len(totals[51])==2 and all(r['source_trace'] for r in totals[51])
    assert sum(r['source_view_id'] is not None for r in totals[51])==1  # upper screw is outside saved structural bounds
    for p in run.glob('stage_2/detections/*/manifest.json'):
        saved=json.loads(p.read_text())
        if saved['page_index']!=54:continue
        page=copy.deepcopy(saved['page_analysis']);regions=copy.deepcopy(page['coverage_validation']['proposed_candidates'])
        page['initial_decision']='NO_DETAIL'
        assert detection.confirmed_no_detail(page,regions)
        primary=SimpleNamespace(decision='NO_DETAIL',page_kind='table_sheet',evidence=page['evidence'],coverage='complete',
            uncertainty=['Header/table formatting requires review'],followups=[],
            regions=[SimpleNamespace(kind=r['kind'],uncertainty=[],incomplete=False) for r in regions])
        def independent(source,row,geometry,config,client,items,state,requests,**kwargs):
            state['coverage_validation']=page['coverage_validation'];return items
        with patch.object(detection,'render',return_value=(Image.new('RGB',(100,100)),{})),\
             patch.object(detection,'request_analysis',return_value=primary),\
             patch.object(detection,'mapped_regions',return_value=regions),\
             patch.object(detection,'verify_page_coverage',side_effect=independent):
            _,decision,_,errors=detection.analyze_page(Path(saved['source_path']),dict(page_index=54),
                dict(source_rect=[0,0,600,842]),detection.Config(),None)
        assert decision['decision']=='NO_DETAIL' and not errors
        assert decision['review_reasons']  # notes preserved, not erased to force success
        print('INTEGRATION 55: independent table-only proof skips extraction while retaining review notes')
    for p in run.glob('stage_4/callouts/*/manifest.json'):
        saved=json.loads(p.read_text());up=saved['upstream']
        if up['page_index'] not in [189,190]:continue
        from .qwen import ModelConfig
        class NoRequests:
            requests=0;response_models=set()
            def request_json(self,*args,**kwargs):raise AssertionError('No field request can recover missing occurrences')
        with tempfile.TemporaryDirectory(prefix='.accuracy-check-',dir=Path(__file__).parent) as temp:
            temp=Path(temp);copied=sqlite3.connect(temp/'index.sqlite3');db.backup(copied);copied.close()
            summary=components.run(up['stage3']['path'],p,temp/'fields',index_path=temp/'index.sqlite3',
                config=components.Config(details_only=True,paragraph_descriptions=False,observed_profiles=False),
                model=ModelConfig(endpoint='http://127.0.0.1:1/v1',model='offline-fixture'),client=NoRequests(),force=True)
            result=json.loads(Path(summary['manifest']).read_text())
            assert summary['model_requests']==0 and result['state']=='PARTIAL'
            print('INTEGRATION',up['page_index']+1,'zero field requests; upstream failure remains partial')
    summary_path=next(run.glob('stage_7/partial/*/run-summary.json'))
    public_root=run.parent.parent.parent
    original_csv=(public_root/'all_details.csv').read_bytes()
    with tempfile.TemporaryDirectory(prefix='.accuracy-check-',dir=Path(__file__).parent) as temp:
        published=delivery.publish(summary_path,output=Path(temp))
        target=Path(published['root']);rows=delivery.read_csv(target/'all_details.csv')
        assert published['rows']==10 and len(rows)==10,published
        assert (target/'all_details.csv').read_bytes()==original_csv
        assert len(list((target/'incomplete_details').glob('*/image.png')))==9
        assert (public_root/'all_details.csv').read_bytes()==original_csv
        print('INTEGRATION delivery: 10 unchanged CSV rows; 9 incomplete detail images retained')
    db.close()
    return totals


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--saved-run',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(AccuracyChecks))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.saved_run:replay(args.saved_run)

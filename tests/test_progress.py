"""Live reports reflect events, never guesses about worker or GPU state."""
import io
import time
from as_ocr import progress


def event(kind, stage=2, page=3, pid=123, **values):
    return dict(event=kind,stage=stage,page=page,pid=pid,at=100.,**values)


def test_waiting_request_does_not_mark_page_complete(tmp_path):
    stream=io.StringIO()
    reporter=progress.Reporter('catalog.pdf',110,4,tmp_path/'progress.log',stream=stream)
    reporter.consume(event('plan',planned_stage=2,pages=list(range(1,111))))
    reporter.consume(event('start',operation='Detecting Assembly Details'))
    reporter.consume(event('request_start'))
    reporter.summary(now=145.)
    assert not reporter.finished[2] and not reporter.completed
    assert '123:S2/P3' in stream.getvalue() and 'request wait 45.0s' in stream.getvalue()
    assert 'pending this stage=109' in stream.getvalue()
    assert 'queue/inference split unavailable' in stream.getvalue()
    reporter.consume(event('request_end',seconds=45,error='timeout',timeouts=1))
    reporter.consume(event('done',ok=False,reason='Detection failed'))
    assert reporter.failed=={3} and not reporter.completed
    assert reporter.timeouts==1 and not reporter.active
    reporter.events.close()


def test_out_of_order_workers_are_reported_without_changing_page_identity(tmp_path):
    stream=io.StringIO()
    reporter=progress.Reporter('catalog.pdf',110,4,tmp_path/'progress.log',stream=stream)
    for page in (4,1,3,2):
        reporter.consume(event('start',page=page,pid=page+100,operation='Detecting'))
    reporter.consume(event('done',page=3,pid=103,ok=True,empty=True))
    reporter.summary(now=130.)
    assert reporter.completed=={3}
    assert {e['page'] for e in reporter.active.values()}=={1,2,4}
    assert 'active=[1, 2, 4]' in stream.getvalue()
    reporter.events.close()


def test_live_reporter_prints_periodic_summary_without_filesystem_polling(tmp_path):
    stream=io.StringIO()
    with progress.Reporter('catalog.pdf',110,4,tmp_path/'progress.log',interval=.03,stream=stream):
        progress.plan(2,range(1,111))
        progress.start(2,1,'Detecting Assembly Details')
        progress.request_start()
        time.sleep(.08)
        progress.request_end(.08)
        progress.done(2,1,empty=True)
    text=stream.getvalue()
    assert text.count('[Status elapsed=')>=2
    assert 'pages completed=1/110' in text
    assert (tmp_path/'progress.log').read_text()==text
    assert progress.worker_options()=={}



def test_late_worker_event_cannot_move_current_stage_backwards(tmp_path):
    reporter=progress.Reporter('catalog.pdf',110,4,tmp_path/'progress.log',stream=io.StringIO())
    reporter.consume(event('plan',planned_stage=3,pages=[1,2]))
    reporter.consume(event('done',stage=2,page=2,ok=True))
    assert reporter.stage==3
    reporter.events.close()

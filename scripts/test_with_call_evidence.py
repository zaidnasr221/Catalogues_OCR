"""Run explicitly named unit tests and retain project function execution evidence."""
import json
from pathlib import Path
import sys

import pytest


def main():
    if len(sys.argv)<3:
        raise SystemExit('Usage: test_with_call_evidence.py EVIDENCE.json TEST [TEST ...]')
    output=Path(sys.argv[1]);root=Path(__file__).resolve().parents[1]/'src'
    observed=set();previous=sys.getprofile()
    class Results:
        def __init__(self):self.outcomes={}
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed:self.outcomes[report.nodeid]=report.outcome
    results=Results();status=None
    def profile(frame,event,arg):
        if event=='call':
            if frame.f_code.co_filename.startswith(str(root)+'/'):
                path=Path(frame.f_code.co_filename)
                module='.'.join(path.relative_to(root).with_suffix('').parts)
                if module.endswith('.__init__'):module=module[:-9]
                observed.add(module+'.'+frame.f_code.co_qualname.replace('.<locals>.','.'))
    sys.setprofile(profile)
    try:
        status=pytest.main(sys.argv[2:],plugins=[results])
    finally:
        sys.setprofile(previous)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(dict(tests=sys.argv[2:],observed_calls=sorted(observed),
            exit_code=int(status) if status is not None else None,test_outcomes=results.outcomes,
            evidence_scope='Focused unit/contract tests only; no real PDF validation'),indent=2)+'\n')
    return status


if __name__=='__main__':raise SystemExit(main())

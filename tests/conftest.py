"""Test adapters only: production Linux checkpoint durability stays unchanged."""
import os
import pytest


@pytest.fixture
def portable_checkpoint_io(monkeypatch):
    if os.name!='nt':return
    from as_ocr import performance
    original=performance.atomic_write
    def write(path,text):
        try:return original(path,text)
        except AttributeError as exc:
            # atomic_write has already fsynced the file and replaced it. Windows
            # lacks the final POSIX directory fsync; this adapter tests OCR only.
            if 'O_DIRECTORY' not in str(exc):raise
    monkeypatch.setattr(performance,'atomic_write',write)

"""Presentation-only OCR formatting. Never repairs source wording or topology."""
from __future__ import annotations

import hashlib
import re

from as_ocr import page_markdown


def format_page(result, original):
    """Preserve the engine's reading order, repeats, supplements and partial text.

    Only recognized one-line headings gain heading syntax, and existing GFM
    tables gain blank-line boundaries. Inconsistent tables remain REVIEW.
    """
    text = original
    changes, issues = [], []
    wrapper = re.search(r'\*\*OCR status: (?:COMPLETE|PARTIAL|FAILED)\*\*', text)
    cursor = wrapper.end() if wrapper else 0
    for unit in result.get('units', []):
        literal = unit.get('markdown', '')
        if not literal:
            continue
        position = text.find(literal, cursor)
        if position < 0:
            issues.append('A layout unit could not be matched exactly to emitted OCR; text retained verbatim')
            continue
        rendered = literal
        if unit.get('kind') == 'heading' and not re.match(r'^\s*(?:#{1,6}\s|<h[1-6]\b)', literal, re.I):
            if '\n' not in literal and literal.strip():
                rendered = '## ' + literal
                changes.append('Added Markdown heading syntax to an existing recognized heading')
            else:
                issues.append('Multiline heading hierarchy cannot safely be inferred; source text retained')
        text = text[:position] + rendered + text[position + len(literal):]
        cursor = position + len(rendered)
    lines = text.splitlines()
    tables = page_markdown.markdown_tables(text)
    before, after = set(), set()
    for table in tables:
        if not table.get('valid'):
            issues.append('Table has inconsistent columns; no cells were invented or discarded')
        if table['kind'] == 'gfm':
            before.add(table['start'])
            after.add(table['end'])
    if re.search(r'<(?:td|th)\b[^>]*(?:rowspan|colspan)\s*=', text, re.I):
        issues.append('Merged HTML cells retained verbatim; table topology needs source review')
    formatted = []
    fenced = False
    for number, line in enumerate(lines):
        if number in before | after and not fenced and formatted and formatted[-1].strip():
            formatted.append('')
            changes.append('Inserted a blank-line table boundary without changing cells')
        formatted.append(line)
        if re.match(r'^\s*(```|~~~)', line):
            fenced = not fenced
    if fenced:
        issues.append('Unbalanced code fence retained for source review')
    output = '\n'.join(formatted).rstrip() + '\n'
    clean, image_issues = page_markdown.clean_transcription(output, 'other')
    if image_issues:
        output = clean.rstrip() + '\n'
        issues += image_issues
    status = result['status']
    if issues and status == 'COMPLETE':
        status = 'PARTIAL'
        output = re.sub(r'\*\*OCR status: COMPLETE\*\*', '**OCR status: PARTIAL**', output, count=1)
    if issues:
        output += '\n[OutputV3 layout REVIEW: ' + '; '.join(dict.fromkeys(issues)) + ']\n'
    return output, dict(status=status, legacy_status=result['status'],
                        formatting_changes=list(dict.fromkeys(changes)),
                        layout_issues=list(dict.fromkeys(issues)), reading_order='UNCHANGED_FROM_EXISTING_OCR',
                        output_sha256=hashlib.sha256(output.encode()).hexdigest())

"""Independent reading acceptance; association reviews do not establish identity."""
import re
import hashlib
from pathlib import Path

from .geometry import area, intersection

METADATA_KINDS = {'document_metadata'}
ASSOCIATION_WORDS = ('association', 'leader', 'target', 'membership', 'endpoint',
                     'pointed to', 'component', 'ownership', 'duplicate occurrence identity',
                     'detail context')
SEMANTIC_DOUBTS = ('classification', 'drawing number', 'document number',
                   'product code or', 'code or reference', 'reading is',
                   'unreadable', 'illegible', 'conflicting reading', 'reading or',
                   'page identifier', 'page number', 'code is ambiguous')


def code_fingerprint():
    """Invalidate extraction caches when any pipeline implementation changes."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reading_review_reasons(occurrence):
    """Typed reviews take precedence; retain conservative legacy interpretation."""
    if 'reading_review_reasons' in occurrence:
        return list(occurrence['reading_review_reasons'])
    result = []
    for reason in occurrence.get('review_reasons', ()):
        lowered = reason.casefold()
        semantic = (any(term in lowered for term in SEMANTIC_DOUBTS) or
                    re.search(r'\b(?:date|revision|glyph|character|spelling)\b', lowered))
        if semantic or not any(term in lowered for term in ASSOCIATION_WORDS):
            result.append(reason)
    return result


def metadata_source(occurrence, source_kinds=(), metadata_boxes=()):
    if (occurrence.get('source_role') == 'document_metadata' or
            METADATA_KINDS.intersection(source_kinds)):
        return True
    box = occurrence.get('text_box')
    return bool(box and area(box) > 0 and any(
        intersection(box, metadata) / area(box) >= .8 for metadata in metadata_boxes))


def standalone_item_number(value):
    """Parenthesized short drawing positions are not standalone profile IDs."""
    return bool(value and re.fullmatch(r'\(\s*\d{1,2}(?:\s*,\s*\d{1,2})*\s*\)', value.strip()))


def supported_product_reading(occurrence, source_kinds=(), metadata_boxes=()):
    """Association doubts do not erase a readable code; identity doubts do.

    Full raw readings remain in evidence. A reconciled selection can reference
    fully observed readings separately from clipped duplicate observations.
    Unknown legacy reading doubts and genuine contradictions remain unresolved.
    """
    code = occurrence.get('raw_code')
    if (not code or standalone_item_number(code) or occurrence.get('category') != 'product_code' or
            occurrence.get('reading_state') != 'exact' or
            metadata_source(occurrence, source_kinds, metadata_boxes)):
        return False
    for reading in occurrence.get('reading_support', occurrence.get('readings', ())):
        if reading.get('raw_text') != code:
            return False
    return not reading_review_reasons(occurrence)

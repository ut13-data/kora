"""
Stage 4: Document Identification

Looks inside a cleaned document's text for structured metadata
(Document ID, Title, Department, etc.) and extracts whatever it can find.

Fields that can't be found in the text are marked "unknown" — never guessed.

This stage only applies to ExtractedDocument (text-based files).
CSV files (ExtractedTable) don't go through this — they're structured
data, not documents with this kind of header metadata.
"""

import re
from dataclasses import dataclass, field

from src.extraction.extractor import ExtractedDocument

UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    document_id: str = UNKNOWN
    title: str = UNKNOWN
    department: str = UNKNOWN
    document_type: str = UNKNOWN
    owner_role: str = UNKNOWN
    version: str = UNKNOWN
    effective_date: str = UNKNOWN
    status: str = UNKNOWN
    related_processes: list[str] = field(default_factory=list)
    related_roles: list[str] = field(default_factory=list)


# Maps each metadata field to the possible label variants a document might
# use, so "Doc ID:", "Document ID:", and "ID:" can all resolve to the same field.
_FIELD_PATTERNS: dict[str, list[str]] = {
    "document_id": [r"Document ID", r"Doc ID", r"ID"],
    "title": [r"Title"],
    "department": [r"Department"],
    "document_type": [r"Document Type", r"Doc Type", r"Type"],
    "owner_role": [r"Owner", r"Owner/Role", r"Owner Role"],
    "version": [r"Version"],
    "effective_date": [r"Effective Date", r"Date"],
    "status": [r"Status"],
}

# These two fields can hold multiple comma-separated values, so they're
# handled separately from the single-value fields above.
_LIST_FIELD_PATTERNS: dict[str, list[str]] = {
    "related_processes": [r"Related Processes"],
    "related_roles": [r"Related Roles"],
}


def _find_single_value(text: str, label_variants: list[str]) -> str:
    """Search text for the first matching 'Label: value' line, return its value."""
    for label in label_variants:
        pattern = rf'^{label}\s*:\s*(.+)$'
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return UNKNOWN


def _find_list_value(text: str, label_variants: list[str]) -> list[str]:
    """Same as _find_single_value, but splits the value on commas into a list."""
    for label in label_variants:
        pattern = rf'^{label}\s*:\s*(.+)$'
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            raw_value = match.group(1).strip()
            return [item.strip() for item in raw_value.split(",") if item.strip()]
    return []


def identify_document_metadata(document: ExtractedDocument) -> DocumentMetadata:
    """Scan a cleaned document's text and extract whatever metadata is present.

    Any field not found in the text is left as "unknown" (or an empty list
    for the list fields) — never inferred or invented.
    """
    text = document.text

    metadata = DocumentMetadata()

    for field_name, label_variants in _FIELD_PATTERNS.items():
        value = _find_single_value(text, label_variants)
        setattr(metadata, field_name, value)

    for field_name, label_variants in _LIST_FIELD_PATTERNS.items():
        values = _find_list_value(text, label_variants)
        setattr(metadata, field_name, values)

    return metadata
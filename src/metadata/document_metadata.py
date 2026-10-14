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
from pathlib import Path

from src.extraction.extractor import ExtractedDocument

UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    source_filename: str = UNKNOWN
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
    # Where document_id / title came from (Phase 4 robustness, mixed real-world files):
    #   "header"             read from the document's own header block
    #   "derived_from_path"  no ID in the header: a stand-in built from the file path
    #   "derived_from_text"  no title in the header: the first heading / first line
    #   ""                   not recorded (output written before this field existed)
    id_source: str = ""
    title_source: str = ""


# Maps each metadata field to the possible label variants a document might
# use, so "Doc ID:", "Document ID:", and "ID:" can all resolve to the same field.
_FIELD_PATTERNS: dict[str, list[str]] = {
    "document_id": [r"Document ID", r"Doc ID", r"ID"],
    "title": [r"Document Title", r"Title"],
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


# A header line: "**Label:** value" (Nexora) or plain "Label: value", label
# short and without sentence punctuation. The header block is the run of such
# lines at the very top of the document.
_HEADER_LINE = re.compile(r"^\*{0,2}\s*[A-Za-z][A-Za-z0-9 /&()-]{0,40}?\s*:\s*\*{0,2}\s*\S")


def header_block(text: str) -> str:
    """Only the metadata block at the top of the document.

    Metadata used to be searched in the WHOLE text, so a body line such as
    "Status: Archived" (describing a ticket, say) could mark a real-world
    document as archived and hide it. Nexora's own headers are always at the
    top, so their values are unchanged."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            if out:            # one blank line may separate header groups
                i += 1
                continue
            break
        if not _HEADER_LINE.match(stripped):
            break
        out.append(stripped)
        i += 1
    return "\n".join(out)


def _find_single_value(text: str, label_variants: list[str]) -> str:
    """Search text for the first matching 'Label: value' line, return its value."""
    for label in label_variants:
        pattern = rf'^\*{{0,2}}\s*{re.escape(label)}\s*:\s*\*{{0,2}}\s*(.+)$'
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return UNKNOWN


def _find_list_value(text: str, label_variants: list[str]) -> list[str]:
    """Same as _find_single_value, but splits the value on commas into a list."""
    for label in label_variants:
        pattern = rf'^\*{{0,2}}\s*{re.escape(label)}\s*:\s*\*{{0,2}}\s*(.+)$'
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            raw_value = match.group(1).strip()
                       # Split on commas, but NOT commas inside (brackets).
            parts = re.split(r",\s*(?![^()]*\))", raw_value)
            return [item.strip() for item in parts if item.strip()]
    return []


def identify_document_metadata(document: ExtractedDocument) -> DocumentMetadata:
    """Scan a cleaned document's text and extract whatever metadata is present.

    Any field not found in the text is left as "unknown" (or an empty list
    for the list fields) — never inferred or invented.
    """
    text = header_block(document.text)

    metadata = DocumentMetadata()
    metadata.source_filename = document.file_info.filename  ##return the document name 

    for field_name, label_variants in _FIELD_PATTERNS.items():
        value = _find_single_value(text, label_variants)
        setattr(metadata, field_name, value)

    for field_name, label_variants in _LIST_FIELD_PATTERNS.items():
        values = _find_list_value(text, label_variants)
        setattr(metadata, field_name, values)

    metadata.id_source = "header" if metadata.document_id != UNKNOWN else ""
    metadata.title_source = "header" if metadata.title != UNKNOWN else ""
    return metadata


def derived_document_id(relative_path: str) -> str:
    """Stable stand-in ID for a document without one in its header, from its
    path inside the raw folder: 'hr/policy.md' -> 'FILE-HR-POLICY'. Two files
    with the same name in different folders get different IDs."""
    stem = str(Path(relative_path).with_suffix("")).replace("\\", "/")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").upper()
    return f"FILE-{slug or 'UNNAMED'}"


def derived_title(text: str) -> str | None:
    """First Markdown heading, else the first non-empty line (<= 120 chars)."""
    first_line = None
    for line in text.splitlines():
        stripped = line.strip().strip("*").strip()
        if not stripped or _HEADER_LINE.match(line.strip()) or stripped == "---":
            continue
        if line.lstrip().startswith("#"):
            return stripped.lstrip("#").strip()[:120] or None
        if first_line is None:
            first_line = stripped[:120]
    return first_line


def apply_fallback_identity(metadata: DocumentMetadata, document: ExtractedDocument,
                            relative_path: str) -> DocumentMetadata:
    """Fill in a missing ID / title with clearly LABELLED stand-ins. The
    *_source fields say where each came from, so nothing derived is ever
    presented as if the document stated it."""
    if metadata.document_id == UNKNOWN:
        metadata.document_id = derived_document_id(relative_path)
        metadata.id_source = "derived_from_path"
    if metadata.title == UNKNOWN:
        title = derived_title(document.text)
        if title:
            metadata.title = title
            metadata.title_source = "derived_from_text"
    return metadata
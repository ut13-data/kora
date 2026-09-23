"""
Stage 3: Cleaning

Takes the raw output of Stage 2 (Extraction) and removes noise —
excess whitespace, repeated blank lines, encoding artifacts —
WITHOUT altering the actual meaning of the content.

Does NOT chunk (Stage 5) or assign metadata (Stage 4).
"""

import re

from src.extraction.extractor import ExtractedDocument, ExtractedTable


def clean_text(text: str) -> str:
    """Clean a block of free text (used for TXT, Markdown, DOCX, PDF output)."""

    # Normalize encoding artifacts: some PDF/DOCX extractions leave behind
    # non-breaking spaces (\xa0) or other odd whitespace-like characters.
    text = text.replace("\xa0", " ")

    # Collapse 3+ consecutive newlines down to exactly 2 (preserves paragraph
    # breaks, removes excessive gaps).
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse repeated spaces/tabs into a single space, but leave newlines alone.
    text = re.sub(r'[ \t]+', ' ', text)

    # Trim trailing whitespace at the end of each line.
    text = re.sub(r'[ \t]+\n', '\n', text)

    # Trim leading/trailing whitespace from the whole document.
    text = text.strip()

    return text


def clean_row(row: dict) -> dict:
    """Clean a single CSV row: trim whitespace on each cell, keep structure intact."""
    return {
        key: (value.strip() if isinstance(value, str) else value)
        for key, value in row.items()
    }


def clean_document(document: ExtractedDocument) -> ExtractedDocument:
    """Return a new ExtractedDocument with cleaned text."""
    cleaned_text = clean_text(document.text)
    return ExtractedDocument(file_info=document.file_info, text=cleaned_text)


def clean_table(table: ExtractedTable) -> ExtractedTable:
    """Return a new ExtractedTable with each row's cells trimmed."""
    cleaned_rows = [clean_row(row) for row in table.rows]
    return ExtractedTable(file_info=table.file_info, rows=cleaned_rows)


def clean_extracted(extracted: ExtractedDocument | ExtractedTable) -> ExtractedDocument | ExtractedTable:
    """Dispatch to the right cleaning function based on the extracted type."""
    if isinstance(extracted, ExtractedDocument):
        return clean_document(extracted)
    elif isinstance(extracted, ExtractedTable):
        return clean_table(extracted)
    else:
        raise TypeError(f"Unexpected type passed to clean_extracted: {type(extracted)}")
"""Unit tests: Stage 5 (structure-aware chunking)."""

from src.chunking.chunker import chunk_document
from src.ingestion.file_intake import FileType
from src.metadata.document_metadata import identify_document_metadata

STEPS = "\n".join(f"- Step {i}: do the thing number {i} carefully." for i in range(1, 51))

DOC = f"""**Document ID:** DOC-T-001
**Document Title:** Chunk Test
**Version:** 1.0
**Status:** Active

---

# Chunk Test

## Purpose
Why this exists: to test chunking.

## Steps
{STEPS}
"""


def _chunk(make_document, text=DOC, file_type=FileType.MARKDOWN, **kwargs):
    doc = make_document(text, file_type=file_type)
    return chunk_document(doc, identify_document_metadata(doc), **kwargs)


def test_metadata_header_is_not_in_chunk_text(make_document):
    for chunk in _chunk(make_document):
        assert "**Document ID:**" not in chunk.chunk_text


def test_heading_stays_with_its_content(make_document):
    purpose = [c for c in _chunk(make_document) if "## Purpose" in c.chunk_text]
    assert len(purpose) == 1
    assert "Why this exists" in purpose[0].chunk_text
    assert purpose[0].section == "Chunk Test > Purpose"


def test_long_section_respects_size_limit(make_document):
    chunks = _chunk(make_document, chunk_size=300)
    assert all(len(c.chunk_text) <= 300 for c in chunks)
    steps = [c for c in chunks if c.section == "Chunk Test > Steps"]
    assert len(steps) > 1                              # it really was split
    assert len({c.parent_id for c in steps}) == 1      # ...but all share one parent


def test_chunk_ids_are_stable_across_runs(make_document):
    first = [c.chunk_id for c in _chunk(make_document)]
    second = [c.chunk_id for c in _chunk(make_document)]
    assert first == second
    assert first[0] == "DOC-T-001_v1.0_000"


def test_non_markdown_files_have_no_section(make_document):
    chunks = _chunk(make_document, text="Plain text.\n\nSecond paragraph.", file_type=FileType.TXT)
    assert chunks and all(c.section == "" for c in chunks)
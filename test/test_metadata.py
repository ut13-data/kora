"""Unit tests: Stage 4 (document metadata)."""

from src.metadata.document_metadata import identify_document_metadata

NEXORA_STYLE = """**Document ID:** PROC-TEST-001
**Document Title:** Test Process
**Department:** HR
**Version:** 2.1
**Status:** Active
**Related Processes:** PROC-A-001 (Alpha, with a comma), PROC-B-002 (Beta)
**Related Roles:** HR Manager, Recruiter

---

# Test Process
Body text."""


def test_bold_header_fields_are_extracted(make_document):
    m = identify_document_metadata(make_document(NEXORA_STYLE))
    assert m.document_id == "PROC-TEST-001"
    assert m.title == "Test Process"
    assert m.department == "HR"
    assert m.version == "2.1"
    assert m.status == "Active"
    assert m.source_filename == "sample.md"


def test_commas_inside_brackets_do_not_split(make_document):
    m = identify_document_metadata(make_document(NEXORA_STYLE))
    assert m.related_processes == ["PROC-A-001 (Alpha, with a comma)", "PROC-B-002 (Beta)"]
    assert m.related_roles == ["HR Manager", "Recruiter"]


def test_missing_fields_are_unknown_not_invented(make_document):
    m = identify_document_metadata(make_document("Just some text with no header."))
    assert m.document_id == "unknown"
    assert m.title == "unknown"
    assert m.owner_role == "unknown"
    assert m.related_roles == []
"""Integration tests: the whole pipeline on the real Nexora data.

If Nexora's source data changes on purpose (e.g. the ID mismatch gets
fixed), update the expected values here, deliberately.
"""

from pathlib import Path

import pytest

from src.metadata.relationships import RelationType
from src.pipeline import process_folder
from src.validation.validator import Severity
from src.ingest import IngestionSettings, run_ingestion

RAW = Path("data/raw")


@pytest.fixture(scope="module")
def result():
    """Run the full pipeline ONCE, shared by every test in this file."""
    if not RAW.exists():
        pytest.skip("Nexora data not found in data/raw")
    return process_folder(RAW)


def test_every_file_is_processed(result):
    assert len(result.documents) == 53
    assert len(result.tables) == 7
    assert result.skipped == []


def test_only_the_known_error_remains(result):
    errors = [i.message for i in result.validation_issues if i.severity == Severity.ERROR]
    assert errors == ["Header ID 'PROC-IMPL-001' doesn't match documents.csv ID 'PROC-IMPL-001-V1'"]


# Answer containment: each key fact must sit WHOLE inside at least one chunk
# of its document. If chunking ever cuts one in half, its test fails.
FACTS = [
    ("PROC-HR-001", "within the first 10 working days of their start date"),
    ("PROC-IMPL-001", "provides final go-live approval"),
    ("PROC-SALES-002", "requires Sales Manager approval"),
    ("PROC-IT-001", "explicit Data & Analytics Manager approval before IT provisions"),
    ("PROC-CS-001", "does not automatically trigger a Level 2 customer escalation"),
    ("DOC-PROD-NEXFLOW", "cannot currently be retroactively enabled without a data migration project"),
    ("DOC-CARE-002", "Registering the address promptly, since it blocks several later steps"),
]


def _flatten(text: str) -> str:
    """Line breaks inside a sentence shouldn't break the check."""
    return " ".join(text.split())


@pytest.mark.parametrize("doc_id, fact", FACTS)
def test_answer_sits_whole_inside_one_chunk(result, doc_id, fact):
    chunks = [c for d in result.documents for c in d.chunks if c.document_id == doc_id]
    assert chunks, f"No chunks found for {doc_id}"
    assert any(fact in _flatten(c.chunk_text) for c in chunks), \
        f"'{fact}' is split across chunks (or missing) in {doc_id}"


def test_munich_new_joiner_chain(result):
    targets = {r.target for r in result.relationships
               if r.source == "LOC-MUC" and r.relation_type in
               (RelationType.LOCATION_HAS_CHECKLIST, RelationType.CULTURE_TONE_GUIDANCE)}
    assert targets == {"DOC-CARE-002", "DOC-CARE-003", "DOC-CARE-004", "DOC-COMM-002"}


def test_onboarding_process_is_linked_to_nexflow(result):
    nexflow = result.registry.id_for_name("products", "NexFlow")
    assert nexflow is not None
    assert any(r.source == "PROC-IMPL-001" and r.target == nexflow
               and r.relation_type == RelationType.PROCESS_RELATED_TO_PRODUCT
               for r in result.relationships)


def test_exclude_archived_setting():
    if not RAW.exists():
        pytest.skip("Nexora data not found in data/raw")
    summary, result = run_ingestion(IngestionSettings(include_archived=False, save_output=False))
    assert all(d.metadata.status.lower() != "archived" for d in result.documents)
    assert sorted(summary.excluded_archived) == [
        "archived_new_hire_security_training_v1_ARCHIVED.md",
        "customer_onboarding_v1_ARCHIVED.md",
    ]
    assert summary.documents == 51
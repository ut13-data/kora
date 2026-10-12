"""Phase 3, Milestone 5: validation, the test questions, and the full output contract."""

import copy
import json

import pytest

from src.build_intelligence import IntelligenceSettings, run_intelligence
from src.intelligence.validation import validate_outputs

SPEC_OUTPUTS = ["entities", "relationships", "processes", "topics", "knowledge_gaps", "conflicts",
                "outdated_knowledge", "employee_context", "germany_checklists", "communication_guidance"]


def _check(validation, name):
    return next(c for c in validation["checks"] if c["check"] == name)


# ---------------- validation ----------------

def test_nothing_fails_and_the_only_warning_is_a_source_data_issue(intel):
    v = intel.validation
    assert v["summary"]["failed"] == 0
    assert [c["check"] for c in v["checks"] if c["status"] == "warn"] == ["document_owners_resolved"]


def test_every_spec_check_is_present(intel):
    names = {c["check"] for c in intel.validation["checks"]}
    assert {"entity_ids_valid", "no_duplicate_ids", "relationships_reference_entities",
            "processes_reference_valid_roles_departments", "employees_reference_valid_entities",
            "document_owners_valid", "checklist_links_valid", "conflicts_reference_documents",
            "no_fabricated_entities", "confidence_values_valid", "cited_chunks_exist", "rules_keep_sources"} <= names


def test_validation_catches_broken_outputs(intel):
    """Break the outputs on purpose: validation must FAIL, not shrug."""
    broken = copy.deepcopy(intel.outputs)
    broken["relationships"]["relationships"][0]["confidence"] = 1.7
    broken["conflicts"]["conflicts"][0]["documents_involved"].append("DOC-DOES-NOT-EXIST")
    broken["roles"]["approval_rules"][0]["chunk_ids"] = ["CHUNK-THAT-DOES-NOT-EXIST"]
    broken["entities"]["entities"].append({**broken["entities"]["entities"][0], "sources": [],
                                           "key": "employee:EMP-9999", "entity_id": "EMP-9999"})
    v = validate_outputs(broken, intel.phase2.chunks)
    for name in ("confidence_values_valid", "conflicts_reference_documents", "cited_chunks_exist",
                 "entities_have_sources"):
        assert _check(v, name)["status"] == "fail", name


def test_every_cited_chunk_is_real(intel):
    assert _check(intel.validation, "cited_chunks_exist")["status"] == "pass"


# ---------------- test questions ----------------

def _answer(intel, qid):
    return next(a for a in intel.answers if a["id"] == qid)


def test_at_least_twenty_questions_all_answered_with_sources(intel):
    assert len(intel.answers) >= 20
    for a in intel.answers:
        assert a["status"] == "answered", (a["id"], a.get("error"))
        assert a["sources"], a["id"]
        assert a["supported_by"], a["id"]
    categories = {a["category"] for a in intel.answers}
    assert {"organization", "processes", "relationships", "knowledge_quality", "germany", "communication"} <= categories


@pytest.mark.parametrize("qid, expected", [
    ("Q01", "11 people"),
    ("Q02", "Lena Hoffmann"),
    ("Q04", "VP Sales"),
    ("Q13", "Housing"),
    ("Q21", "Divya Iyer"),
    ("Q23", "10 working days"),
    ("Q25", "Implementation Manager"),
])
def test_answers_contain_the_right_facts(intel, qid, expected):
    assert expected in _answer(intel, qid)["answer"]


def test_current_answers_leave_archived_knowledge_out(intel):
    assert not any("PROC-IMPL-001-V1" in str(i) for i in _answer(intel, "Q03")["items"])
    assert not any("ADDENDUM" in i for i in _answer(intel, "Q08")["items"])


def test_tone_answer_keeps_the_fact(intel):
    fact = "Please complete the security training by Friday."
    assert all(item.split(": ", 1)[1].startswith(fact) for item in _answer(intel, "Q19")["items"])


# ---------------- the full output contract ----------------

def test_all_outputs_and_the_report_are_written(phase2_dirs, tmp_path):
    out = tmp_path / "intelligence"
    summary, _ = run_intelligence(IntelligenceSettings(
        processed_dir=phase2_dirs / "processed", chunks_dir=phase2_dirs / "chunks", output_dir=out))
    for stem in SPEC_OUTPUTS + ["intelligence_validation", "test_questions"]:
        path = out / f"{stem}.json"
        assert path.exists(), stem
        json.loads(path.read_text(encoding="utf-8"))
    report = (out / "knowledge_report.md").read_text(encoding="utf-8")
    for heading in ("Departments discovered", "Roles discovered", "Employees discovered", "Processes discovered",
                    "Products discovered", "Topics discovered", "Relationships discovered", "Knowledge gaps",
                    "Conflicts", "Outdated documents", "Germany employee-care knowledge",
                    "Communication and tone knowledge", "Validation", "Test questions"):
        assert f"## {heading}" in report, heading
    assert summary.validation["failed"] == 0

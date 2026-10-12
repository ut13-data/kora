"""Phase 3, Milestone 1: entity discovery and normalization on the real Nexora data.

Phase 2 is run into a temporary folder first, so these tests never read or
overwrite your real data/processed output.
"""

# The `intel` fixture (Phase 2 + Phase 3 run once) lives in conftest.py.


def _dups(intel, method):
    return [d for d in intel.duplicates["potential_duplicates"] if d["method"] == method]


def test_entity_counts_match_the_source_tables(intel):
    counts = intel.entities.to_dict()["counts_by_type"]
    assert counts["department"] == 12
    assert counts["role"] == 48
    assert counts["employee"] == 137
    assert counts["location"] == 5
    assert counts["product"] == 7
    assert counts["process"] == 18
    assert counts["document"] == 53
    assert counts["checklist"] == 3
    assert counts["policy"] == 7      # every header Document Type containing "Policy"


def test_no_entity_key_is_added_twice(intel):
    assert intel.entities.collisions == []


def test_customers_come_only_from_the_company_profile_list(intel):
    names = {e.name for e in intel.entities.of_type("customer")}
    assert names == {"Balaji Pharma Distribution", "Rheinland Industrial Supply",
                     "Meridian Retail Group", "Crestline Manufacturing Co."}


def test_project_lead_is_a_documented_alias_of_implementation_manager(intel):
    role = intel.entities.get("role", "ROLE-022")
    assert role.name == "Implementation Manager"
    assert "Project Lead" in role.aliases
    assert any("TN-006" in ev for ev in role.attributes["alias_evidence"]["Project Lead"])


def test_glossary_acronyms_become_role_aliases(intel):
    assert "CSM" in intel.entities.get("role", "ROLE-026").aliases
    assert "AE" in intel.entities.get("role", "ROLE-003").aliases


def test_department_documents_add_acronym_aliases(intel):
    role = intel.entities.get("role", "ROLE-039")       # HR Business Partner
    assert "HRBP" in role.aliases
    assert any("hr.md" in ev for ev in role.attributes["alias_evidence"]["HRBP"])


def test_a_combined_department_value_is_reported_once_as_written(intel):
    unmatched = {(u["expected_type"], u["name"]) for u in intel.duplicates["unmatched_names"]}
    assert ("department", "Executive / Corporate") in unmatched
    assert ("department", "Executive") not in unmatched


def test_terminology_notes_are_classified_from_their_own_wording(intel):
    relation = {e.entity_id: e.attributes["relation"]
                for e in intel.entities.of_type("terminology")
                if e.attributes["kind"] == "terminology_note"}
    assert relation == {"TERM-TN-002": "overloaded", "TERM-TN-004": "not_interchangeable",
                        "TERM-TN-006": "equivalent", "TERM-TN-008": "equivalent"}


def test_archived_file_keeps_index_id_and_header_id_as_alias(intel):
    doc = intel.entities.get("document", "PROC-IMPL-001-V1")
    assert doc.aliases == ["PROC-IMPL-001"]
    assert doc.attributes["status"] == "Archived"


def test_processes_point_at_the_right_version_of_their_document(intel):
    assert intel.entities.get("process", "PROC-IMPL-001").attributes["document_id"] == "PROC-IMPL-001"
    assert intel.entities.get("process", "PROC-IMPL-001-V1").attributes["document_id"] == "PROC-IMPL-001-V1"


def test_a_header_naming_two_departments_resolves_both(intel):
    doc = intel.entities.get("document", "PROC-SUP-002")
    assert doc.attributes["department_ids"] == ["DEPT-SUP", "DEPT-IT"]


def test_every_employee_points_at_known_role_department_and_location(intel):
    for emp in intel.entities.of_type("employee"):
        a = emp.attributes
        assert intel.entities.get("role", a["role_id"]), emp.entity_id
        assert intel.entities.get("department", a["dept_id"]), emp.entity_id
        assert intel.entities.get("location", a["location_id"]), emp.entity_id


def test_onboarding_processes_are_documented_as_distinct(intel):
    pairs = {frozenset((d["entity_a"], d["entity_b"])) for d in _dups(intel, "documented_distinct")}
    assert frozenset(("process:PROC-HR-001", "process:PROC-IMPL-001")) in pairs


def test_shared_header_id_is_reported_not_merged(intel):
    found = _dups(intel, "shared_header_id")
    assert {(d["entity_a"], d["entity_b"]) for d in found} == {
        ("document:PROC-IMPL-001", "document:PROC-IMPL-001-V1")}
    assert all(d["status"] == "unreviewed" for d in found)


def test_unresolved_engineering_suggests_software_engineering(intel):
    found = [d for d in _dups(intel, "unresolved_reference") if d["entity_b"] == "unresolved:Engineering"]
    assert [d["entity_a"] for d in found] == ["department:DEPT-ENG"]
    assert found[0]["confidence"] == 0.5


def test_names_with_no_entity_are_reported_not_guessed(intel):
    unmatched = {(u["expected_type"], u["name"]) for u in intel.duplicates["unmatched_names"]}
    assert ("role", "Chief Executive Officer") in unmatched


def test_support_engineer_levels_are_not_flagged_as_duplicates(intel):
    for d in _dups(intel, "name_similarity"):
        assert not ({d["entity_a"], d["entity_b"]} <= {"role:ROLE-029", "role:ROLE-030", "role:ROLE-031"})


def test_nothing_is_ever_marked_merged(intel):
    assert {d["status"] for d in intel.duplicates["potential_duplicates"]} <= {"documented", "unreviewed"}

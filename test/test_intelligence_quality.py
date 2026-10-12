"""Phase 3, Milestone 3: topics, conflicts, outdated knowledge and gaps,
on the real Nexora data. The `intel` fixture lives in conftest.py."""


def _conflicts(intel, conflict_type):
    return [c for c in intel.conflicts["conflicts"] if c["conflict_type"] == conflict_type]


def _corroborations(intel, kind):
    return [c for c in intel.conflicts["corroborations"] if c["type"] == kind]


def _gaps(intel, gap_type):
    return [g for g in intel.gaps.to_dict()["gaps"] if g["gap_type"] == gap_type]


# ---------------- conflicts ----------------

def test_security_training_conflict_is_found_not_decided(intel):
    [c] = _conflicts(intel, "numeric_fact")
    assert set(c["documents_involved"]) == {"PROC-HR-001", "PROC-HR-001-ADDENDUM"}
    values = {s["document_id"]: s["values"] for s in c["sides"]}
    assert values == {"PROC-HR-001": ["10 working day"], "PROC-HR-001-ADDENDUM": ["30 day"]}
    # Signals point one way and both documents declare the change, but KORA only reports it.
    assert c["status"] == "superseded_per_source"
    assert c["signals"]["active_vs_archived"]["active"] == ["PROC-HR-001"]
    assert {d["document_id"] for d in c["signals"]["declared_by_source"]} == {"PROC-HR-001", "PROC-HR-001-ADDENDUM"}
    assert "does not decide" in c["note"]


def test_office_location_count_contradicts_locations_csv(intel):
    [c] = _conflicts(intel, "count_claim")
    assert c["topic"] == "number of locations"
    assert set(c["documents_involved"]) == {"DOC-CORP-001", "DOC-DEPT-HR", "DOC-DEPT-OPS"}
    assert c["sides"][-1]["values"] == ["5"]


def test_org_chart_contradicts_its_own_head_rule(intel):
    [c] = _conflicts(intel, "org_statement")
    assert "Implementation Manager" in c["topic"]


def test_tiers_and_aliases_are_not_reported_as_approver_conflicts(intel):
    assert _conflicts(intel, "approver") == []
    tiers = _corroborations(intel, "approval_tiers")
    assert any(set(t["documents"]) == {"DOC-DEPT-FIN", "PROC-FIN-002"} for t in tiers)
    same = _corroborations(intel, "approver")
    assert any(set(c["documents"]) == {"PROC-IMPL-001", "PROC-IMPL-001-V1"} and c["approver"] == ["ROLE-022"]
               for c in same)


def test_approximate_headcount_is_a_corroboration_not_a_conflict(intel):
    company = [c for c in _corroborations(intel, "count_claim") if c["document"] == "DOC-CORP-001"]
    assert company and company[0]["claimed"] == 130 and company[0]["actual"] == 137 and company[0]["approximate"]


def test_every_conflict_names_real_documents(intel):
    for c in intel.conflicts["conflicts"]:
        assert len(c["sides"]) >= 2
        for doc_id in c["documents_involved"]:
            assert intel.entities.get("document", doc_id), doc_id


# ---------------- topics ----------------

def test_expected_topics_exist(intel):
    names = {t.name for t in intel.topics.topics}
    assert {"IT / Security", "Customer Onboarding", "Employee Onboarding", "NexView Analytics", "NexFlow",
            "NexCRM", "NexConnect", "Invoice and Payment Process", "New Joiner Life & Settling In",
            "Communication Culture & Tone Guidelines"} <= names


def test_every_document_has_a_topic(intel):
    assert intel.topics.documents_without_topic == []


def test_archived_document_found_by_similarity(intel):
    topic = next(t for t in intel.topics.topics if t.name == "Customer Onboarding")
    entry = next(d for d in topic.documents if d["document_id"] == "PROC-IMPL-001-V1")
    assert entry["method"] == "text_similarity" and entry["similarity"]["shared_terms"]


def test_programs_fold_nested_checklists(intel):
    programs = {t.name: t.home_documents for t in intel.topics.topics if t.level == "program"}
    assert programs["New Joiner Life & Settling In"] == ["DOC-CARE-001", "DOC-CARE-002", "DOC-CARE-003", "DOC-CARE-004"]
    assert len(programs) == 2


def test_topics_are_in_the_graph(intel):
    assert "topic:TOPIC-ACCESS-CONTROL-POLICY" in intel.graph.neighbors("document:DOC-FAQ-001", "discusses")


# ---------------- outdated knowledge ----------------

def test_archived_documents_point_at_their_replacement(intel):
    superseded = {s["document_id"]: s for s in intel.outdated["superseded_documents"]}
    assert set(superseded) == {"PROC-HR-001-ADDENDUM", "PROC-IMPL-001-V1"}
    for doc_id, current in (("PROC-HR-001-ADDENDUM", "PROC-HR-001"), ("PROC-IMPL-001-V1", "PROC-IMPL-001")):
        [target] = superseded[doc_id]["superseded_by"]
        assert target["document_id"] == current
        assert any(e.startswith("its own text") for e in target["evidence"])


def test_version_diff_resolves_the_renamed_approver(intel):
    v1 = next(s for s in intel.outdated["superseded_documents"] if s["document_id"] == "PROC-IMPL-001-V1")
    assert v1["approver_changes"]["unchanged"] is True                    # Project Lead = Implementation Manager
    assert [t["term"] for t in v1["terminology"]] == ["Project Lead"]
    step6 = next(c for c in v1["changes"] if c["step"] == 6)
    assert "Account Management" in step6["before"] and "Customer Success" in step6["after"]


def test_active_documents_are_not_stale(intel):
    assert intel.outdated["legacy_terminology_in_active_documents"] == []
    assert intel.outdated["stale_references_in_active_documents"] == []


# ---------------- gaps ----------------

def test_declared_known_gaps_are_high_and_cited(intel):
    declared = _gaps(intel, "declared_gap")
    assert len(declared) == 3 and all(g["severity"] == "high" for g in declared)
    kg01 = next(g for g in declared if g["evidence"][0]["gap_code"] == "KG-01")
    assert {"PROC-IMPL-001", "DOC-DEPT-SALES", "DOC-SCEN-001"} <= set(kg01["source_documents"])


def test_products_without_implementation_guidance(intel):
    missing = {g["related_entities"][0] for g in _gaps(intel, "product_without_implementation_guidance")}
    assert missing == {"product:PROD-NEXCRM", "product:PROD-NEXAUTO"}


def test_undefined_owner_is_medium(intel):
    ceo = next(g for g in _gaps(intel, "undefined_role") if "Chief Executive Officer" in g["description"])
    assert ceo["severity"] == "medium"


def test_unroutable_escalation_level_is_medium(intel):
    level3 = next(g for g in _gaps(intel, "escalation_target_unclear") if "Level 3" in g["description"])
    assert level3["severity"] == "medium"


def test_clean_checks_are_reported_not_hidden(intel):
    checks = {c["check"]: c["gaps_found"] for c in intel.gaps.to_dict()["checks"]}
    for name in ("process_owner_missing", "procedure_without_steps", "role_without_responsibilities",
                 "role_without_employees"):
        assert checks[name] == 0

"""Phase 3, Milestone 2: rules, processes, roles, relationships and the graph,
on the real Nexora data. The `intel` fixture lives in conftest.py."""

from src.intelligence.graph import IN, OUT
from src.intelligence.relationships import relationship_id


def _rel(intel, source, name, target):
    return next((r for r in intel.relationships
                 if r.source_key == source and r.relationship == name and r.target_key == target), None)


def _rules(intel, rule_type, document_id):
    return [r for r in intel.rules if r.rule_type == rule_type and r.document_id == document_id]


# ---------------- relationships ----------------

def test_every_relationship_has_both_ends_and_evidence(intel):
    assert intel.dangling_relationships == []
    assert all(r.evidence for r in intel.relationships)
    assert all(0 < r.confidence <= 1 for r in intel.relationships)


def test_relationship_ids_are_unique_and_stable(intel):
    ids = [r.relationship_id for r in intel.relationships]
    assert len(ids) == len(set(ids))
    r = intel.relationships[0]
    assert r.relationship_id == relationship_id(r.source_key, r.relationship, r.target_key)


def test_archived_file_keeps_its_own_links(intel):
    rel = _rel(intel, "document:PROC-IMPL-001-V1", "relevant_to", "role:ROLE-022")   # "Project Lead"
    assert rel is not None and rel.historical


def test_archived_process_is_documented_by_its_own_file(intel):
    assert _rel(intel, "process:PROC-IMPL-001-V1", "documented_by", "document:PROC-IMPL-001-V1")
    assert not _rel(intel, "process:PROC-IMPL-001-V1", "documented_by", "document:PROC-IMPL-001")


def test_employee_department_comes_from_employees_csv(intel):
    rel = _rel(intel, "employee:EMP-0077", "works_in", "department:DEPT-IMPL")
    assert rel and rel.confidence == 1.0 and rel.evidence[0].source_file == "employees.csv"


def test_customer_used_by_comes_from_product_documentation(intel):
    rel = _rel(intel, "product:PROD-NEXVIEW", "used_by", "customer:CUST-BALAJI-PHARMA-DISTRIBUTION")
    assert rel and "reference customer" in rel.evidence[0].quote


def test_department_presence_is_read_clause_by_clause(intel):
    assert _rel(intel, "department:DEPT-ENG", "present_in", "location:LOC-BER")
    assert not _rel(intel, "department:DEPT-PROD", "present_in", "location:LOC-BLR")


def test_onboarding_is_governed_by_access_control_via_its_step(intel):
    rel = _rel(intel, "process:PROC-HR-001", "governed_by", "policy:PROC-IT-001")
    assert rel and rel.confidence == 0.85


# ---------------- processes ----------------

def test_processes_are_classified_from_their_document_type(intel):
    kinds = {p.process_id: p.kind for p in intel.processes}
    assert kinds["PROC-SALES-001"] == "procedure"
    assert kinds["PROC-IT-001"] == "policy"
    assert kinds["PROC-SUP-002"] == "policy_and_procedure"
    assert kinds["PROC-HR-003"] == "guideline"


def test_customer_onboarding_process(intel):
    p = intel.process("PROC-IMPL-001")
    assert len(p.steps) == 6
    assert [o["role_id"] for o in p.owner_roles] == ["ROLE-022"]
    assert p.department["dept_id"] == "DEPT-IMPL"
    assert {"ROLE-022", "ROLE-023", "ROLE-024", "ROLE-026"} <= {x["role_id"] for x in p.participants}
    assert {"PROD-NEXFLOW", "PROD-NEXCONNECT"} <= set(p.steps[1]["product_ids"])
    assert [g["gap_codes"] for g in p.known_gaps] == [["KG-01"]]


def test_inputs_and_outputs_only_when_stated(intel):
    onboarding = intel.process("PROC-IMPL-001")
    assert {i["item"] for i in onboarding.inputs} == {"signed contract", '"Deal Closed" handoff record'}
    assert {o["item"] for o in onboarding.outputs} == {"live production usage"}
    assert intel.process("PROC-SALES-001").inputs_outputs_stated is False


def test_proposal_hands_off_to_customer_onboarding(intel):
    rel = _rel(intel, "process:PROC-SALES-002", "triggers", "process:PROC-IMPL-001")
    assert rel and "Deal Closed" in rel.evidence[0].quote


# ---------------- rules: approvals and escalations ----------------

def test_go_live_approval_current_and_archived(intel):
    current = [r for r in _rules(intel, "approval", "PROC-IMPL-001") if "go-live" in r.statement]
    archived = [r for r in _rules(intel, "approval", "PROC-IMPL-001-V1") if "go-live" in r.statement]
    assert [r.role_ids for r in current] == [["ROLE-022"]]
    assert [r.role_ids for r in archived] == [["ROLE-022"]]       # 'Project Lead', a documented alias
    assert archived[0].historical and not current[0].historical


def test_discount_approval_tiers(intel):
    rules = [r for r in _rules(intel, "approval", "PROC-SALES-002") if r.conditions["percents"]]
    by_percent = {tuple(r.conditions["percents"]): r for r in rules}
    assert by_percent[("15%",)].no_approval_required
    assert by_percent[("15–30%",)].role_ids == ["ROLE-002"]      # Sales Manager
    assert by_percent[("30%",)].role_ids == ["ROLE-001"]         # VP Sales


def test_access_approver_table(intel):
    table = [r for r in _rules(intel, "approval", "PROC-IT-001") if r.method == "approver_table"]
    assert len(table) == 5
    warehouse = next(r for r in table if "warehouse" in r.statement)
    assert warehouse.role_ids == ["ROLE-018"] and warehouse.unresolved_targets == ["IT provisioning"]


def test_generic_manager_is_kept_as_written(intel):
    rule = next(r for r in _rules(intel, "approval", "PROC-FIN-002") if "Direct manager" in r.statement)
    assert rule.role_ids == [] and rule.generic_roles == ["Direct manager"]


def test_security_escalation_goes_to_the_ciso(intel):
    rule = next(r for r in _rules(intel, "escalation", "PROC-SUP-002") if "unauthorized data access" in r.statement)
    assert rule.role_ids == ["ROLE-042"] and "PROC-SUP-002" in rule.process_ids


def test_escalation_levels(intel):
    levels = {r.level: r for r in _rules(intel, "escalation", "PROC-CS-001") if r.level}
    assert levels[1].role_ids == ["ROLE-026"]
    assert levels[2].role_ids == ["ROLE-025"]
    assert levels[3].role_ids == [] and levels[3].unresolved_targets == ["Executive (CEO/CTO)"]


# ---------------- roles ----------------

def test_every_role_has_documented_responsibilities(intel):
    missing = [r.role_id for r in intel.roles if not r.responsibilities]
    assert missing == []


def test_org_chart_reporting_lines(intel):
    def parents(role_id):
        return {line["role_id"] for line in intel.role(role_id).reports_to if line["role_id"]}
    assert parents("ROLE-013") == {"ROLE-012"}                    # Engineering Manager -> VP Engineering
    assert parents("ROLE-022") == {"ROLE-025"}                    # Implementation Manager -> VP Customer Success
    for level in ("ROLE-029", "ROLE-030", "ROLE-031"):            # Support Engineer L3/L2/L1
        assert parents(level) == {"ROLE-028"}
    unresolved = {name for e in intel.org_chart["unresolved"] for name in e["unresolved"]}
    assert {"CEO", "CTO", "Marketing Manager"} <= unresolved


# ---------------- graph ----------------

def test_employee_to_product_documentation_walk(intel):
    steps = [("has_role", OUT), ("in_department", OUT), ("owned_by", IN), ("uses", OUT), ("documented_by", OUT)]
    ends = intel.graph.reach("employee:EMP-0077", steps)
    assert "document:DOC-PROD-NEXFLOW" in ends


def test_current_only_leaves_out_archived_knowledge(intel):
    everything = intel.graph.neighbors("role:ROLE-022", "owns", OUT)
    current = intel.graph.neighbors("role:ROLE-022", "owns", OUT, current_only=True)
    assert "process:PROC-IMPL-001-V1" in everything
    assert "process:PROC-IMPL-001-V1" not in current


def test_explanations_name_their_source(intel):
    rel = _rel(intel, "employee:EMP-0077", "has_role", "role:ROLE-022")
    assert "employees.csv" in intel.graph.explain(rel)
    hops = intel.graph.path("employee:EMP-0077", "product:PROD-NEXFLOW")
    assert hops is not None and len(hops) <= 4

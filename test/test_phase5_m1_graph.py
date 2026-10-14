"""
Phase 5 M1: the graph at runtime -- loading, provenance checks, per-user view, quality report.

Runs on the real Nexora data, built into a temporary folder (your data/ is never touched).
"""

import json
from dataclasses import replace

import pytest

from src.graph.schema import DERIVED, DOCUMENTED, HEADER, MENTION, QUOTE, RECORD, VALID_SHAPES
from src.intelligence.graph import BOTH, IN
from src.retrieval.access import User


def view(built, users, config, emp):
    from src.graph.view import GraphView
    return GraphView.for_user(built["graph"], users[emp] if emp else None, config)


def _rel(graph, source, relationship, target):
    return next(r for r in graph.relationships
                if (r.source_key, r.relationship, r.target_key) == (source, relationship, target))


# ---------------- loading ----------------

def test_saved_graph_loads_with_every_entity_and_link(built, intel):
    g = built["graph"]
    assert len(g.index.entities) == len(intel.entities.entities)
    assert len(g.relationships) == len(intel.relationships)
    assert built["loaded"].dangling == []


def test_loading_keeps_ids_confidence_and_history(built, intel):
    saved = {r.relationship_id: r for r in built["graph"].relationships}
    for r in intel.relationships:
        s = saved[r.relationship_id]
        assert (s.source_key, s.relationship, s.target_key) == (r.source_key, r.relationship, r.target_key)
        assert s.confidence == r.confidence and s.historical == r.historical


def test_phase3_traversals_work_on_the_loaded_graph(built):
    g = built["graph"]
    approvers = g.neighbors("process:PROC-IMPL-001", "approves", IN)
    assert approvers == ["role:ROLE-022"]                          # Implementation Manager
    assert g.path("employee:EMP-0077", "process:PROC-IMPL-001") is not None


# ---------------- schema ----------------

def test_every_nexora_link_has_a_known_shape(built):
    unknown = {(r.source_type, r.relationship, r.target_type) for r in built["graph"].relationships} - VALID_SHAPES
    assert unknown == set()


def test_a_new_shape_is_reported_not_accepted(built):
    from src.graph.quality import quality_report
    from src.intelligence.graph import KnowledgeGraph
    g = built["graph"]
    odd = replace(g.relationships[0], relationship_id="REL-TEST", relationship="aproves")
    report = quality_report(KnowledgeGraph(g.index, [odd]), built["tracer"], built["dir"])
    assert len(report["unknown_shapes"]) == 1 and "aproves" in report["unknown_shapes"][0]


# ---------------- provenance: the right tier, and checks that can FAIL ----------------

def test_tiers_of_known_links(built):
    g, t = built["graph"], built["tracer"]
    assert t.tier(_rel(g, "role:ROLE-022", "approves", "process:PROC-IMPL-001")) == QUOTE
    assert t.tier(_rel(g, "role:ROLE-033", "owns", "process:PROC-FIN-002")) == RECORD
    assert t.tier(_rel(g, "role:ROLE-022", "participates_in", "process:PROC-IMPL-001")) == HEADER
    assert t.tier(_rel(g, "employee:EMP-0085", "eligible_for", "checklist:DOC-CARE-002")) == DERIVED


def test_every_nexora_evidence_item_verifies(built):
    g, t = built["graph"], built["tracer"]
    failed = [(r.relationship_id, p.reason) for r in g.relationships for p in t.trace(r) if not p.verified]
    assert failed == []


def test_record_citation_names_file_row_column_and_value(built):
    g, t = built["graph"], built["tracer"]
    [p] = t.trace(_rel(g, "role:ROLE-033", "owns", "process:PROC-FIN-002"))
    assert p.citation["file"] == "processes.csv" and p.citation["key"] == "PROC-FIN-002"
    assert p.citation["column"] == "owner_role" and "Finance Manager" in p.citation["value"]


def test_table_rule_is_cited_as_the_verbatim_row(built):
    g, t = built["graph"], built["tracer"]
    rel = _rel(g, "role:ROLE-013", "approves", "process:PROC-IT-001")       # from a table, paraphrased
    p = next(p for p in t.trace(rel) if p.tier == QUOTE)
    assert p.verified and p.citation["text"] == "Production code deployment | Engineering Manager"


def _tracer_with(built, **changes):
    from src.graph.trace import Tracer
    base = built["tracer"]
    return Tracer(built["graph"], changes.get("chunks", base.chunks), changes.get("tables", base.tables),
                  changes.get("raw_files", base.raw_files))


def test_changed_quote_fails_the_check(built):
    g = built["graph"]
    rel = _rel(g, "role:ROLE-022", "approves", "process:PROC-IMPL-001")
    chunks = {cid: dict(c) for cid, c in built["tracer"].chunks.items()}
    for cid in ("DOC-DEPT-IMPL_v2.0_001", "PROC-IMPL-001_v2.0_002"):
        chunks[cid]["chunk_text"] = "This text no longer says who approves anything."
    t = _tracer_with(built, chunks=chunks)
    assert all(not p.verified for p in t.trace(rel)) and t.tier(rel) is None


def test_changed_table_value_fails_the_check(built):
    g = built["graph"]
    rel = _rel(g, "role:ROLE-033", "owns", "process:PROC-FIN-002")
    tables = dict(built["tracer"].tables)
    tables["processes"] = [dict(r, owner_role="CFO") if r["process_id"] == "PROC-FIN-002" else r
                           for r in tables["processes"]]
    t = _tracer_with(built, tables=tables)
    assert t.tier(rel) is None          # unverified: not quietly downgraded to "derived from the row"


def test_changed_header_line_fails_the_check(built, tmp_path):
    g = built["graph"]
    rel = _rel(g, "role:ROLE-022", "participates_in", "process:PROC-IMPL-001")
    original = built["tracer"].raw_files["customer_onboarding.md"]
    copy = tmp_path / "customer_onboarding.md"
    copy.write_text(original.read_text(encoding="utf-8").replace(
        "**Related Roles:** Implementation Manager, ", "**Related Roles:** "), encoding="utf-8")
    t = _tracer_with(built, raw_files={**built["tracer"].raw_files, "customer_onboarding.md": copy})
    header = [p for p in t.trace(rel) if p.tier == HEADER]
    assert header and not any(p.verified for p in header)


def test_deleted_chunk_fails_a_mention(built):
    g, base = built["graph"], built["tracer"]
    rel, prov = next((r, p) for r in g.relationships for p in base.trace(r)
                     if p.tier == MENTION and p.evidence.chunk_ids)
    chunks = {k: v for k, v in base.chunks.items() if k not in prov.evidence.chunk_ids}
    p = next(p for p in _tracer_with(built, chunks=chunks).trace(rel) if p.evidence is prov.evidence)
    assert not p.verified and "not in the index" in p.reason


# ---------------- the per-user view: access control ----------------

HIDDEN_FOR_ENGINEERING = {"document:PROC-FIN-003", "process:PROC-FIN-003",
                          "topic:TOPIC-INVOICE-AND-PAYMENT-PROCESS", "document:DOC-DEPT-FIN"}


def test_restricted_document_its_process_and_topic_vanish(built, users, config):
    eng = view(built, users, config, "EMP-0040")
    assert {k for k in built["graph"].index.entities if not eng.visible(k)} == HIDDEN_FOR_ENGINEERING
    fin = view(built, users, config, "EMP-0113")
    assert all(fin.visible(k) for k in HIDDEN_FOR_ENGINEERING)


def test_hidden_things_are_never_reached_linked_or_counted(built, users, config):
    eng = view(built, users, config, "EMP-0040")
    full = view(built, users, config, None)
    for r in eng.relationships:
        assert r.source_key not in HIDDEN_FOR_ENGINEERING and r.target_key not in HIDDEN_FOR_ENGINEERING
        for e in r.evidence:
            assert e.source_document not in {"PROC-FIN-003", "DOC-DEPT-FIN"}
            assert not any(c.startswith(("PROC-FIN-003_", "DOC-DEPT-FIN_")) for c in e.chunk_ids)
    assert eng.node("process:PROC-FIN-003") is None
    assert eng.hops("process:PROC-FIN-003", direction=BOTH) == []
    assert eng.path("employee:EMP-0040", "process:PROC-FIN-003") is None
    assert all("FIN-003" not in h.node for h in eng.hops("process:PROC-FIN-002", direction=BOTH))
    s = eng.summary()
    assert s["nodes"] == full.summary()["nodes"] - len(HIDDEN_FOR_ENGINEERING)
    assert s["edges"] == len(eng.relationships) < len(full.relationships)
    assert not any(k in HIDDEN_FOR_ENGINEERING for k in s["isolated_nodes"])
    assert all(m["key"] not in HIDDEN_FOR_ENGINEERING for m in s["most_connected"])


def test_visible_node_loses_only_its_hidden_documents(built, users, config):
    finance_topic = view(built, users, config, "EMP-0040").node("topic:TOPIC-FINANCE")
    assert "PROC-FIN-003" not in finance_topic.source_documents
    assert "PROC-FIN-002" in finance_topic.source_documents
    assert "keywords" not in finance_topic.attributes          # computed from hidden text too
    assert "keywords" in built["graph"].node("topic:TOPIC-FINANCE").attributes


def test_link_with_mixed_evidence_keeps_only_readable_evidence(built, users, config):
    g = built["graph"]
    mixed = next(r for r in g.relationships
                 if {e.source_document for e in r.evidence} >= {"DOC-DEPT-FIN"}
                 and any(e.source_document not in ("DOC-DEPT-FIN", "PROC-FIN-003") for e in r.evidence))
    seen = view(built, users, config, "EMP-0040").visible_relationship(mixed)
    assert seen is not None and seen.relationship_id == mixed.relationship_id
    assert all(e.source_document != "DOC-DEPT-FIN" for e in seen.evidence)
    assert len(seen.evidence) < len(mixed.evidence)


def test_backend_view_without_a_user_sees_everything(built, users, config):
    full = view(built, users, config, None)
    assert len(full.relationships) == len(built["graph"].relationships)
    assert all(full.visible(k) for k in built["graph"].index.entities)


# ---------------- the per-user view: colleagues' private details ----------------

def test_colleague_sees_directory_fields_only(built, users, config):
    eng = view(built, users, config, "EMP-0040")
    julia = eng.node("employee:EMP-0085")
    assert set(julia.attributes) == {"role_id", "role_title", "dept_id", "location_id", "manager_id"}
    assert eng.hops("employee:EMP-0085", "eligible_for") == []
    assert {h.relationship.relationship for h in eng.hops("employee:EMP-0085")} == \
        {"has_role", "works_in", "located_in", "reports_to"}
    assert built["graph"].node("employee:EMP-0085").attributes["international_relocation_germany"] == "Yes"


def test_the_person_and_admins_see_private_details(built, users, config):
    for emp in ("EMP-0085", "EMP-0120"):                         # herself; an admin
        v = view(built, users, config, emp)
        assert "international_relocation_germany" in v.node("employee:EMP-0085").attributes
        assert v.hops("employee:EMP-0085", "eligible_for")


def test_admins_get_no_extra_document_access(built, users, config):
    from src.graph.view import GraphView
    plain_admin = User("EMP-0120", "Admin without manager group", frozenset({"all", "dept:DEPT-HR"}))
    v = GraphView.for_user(built["graph"], plain_admin, config)
    assert v.admin and not v.visible("process:PROC-FIN-003")


def test_visible_fields_come_from_the_config(built, users):
    from src.graph.view import GraphView
    from src.retrieval.config import parse_config
    open_cfg = parse_config({"person_visibility": {"colleague_fields": ["role_title", "international_relocation_germany"]}})
    v = GraphView.for_user(built["graph"], users["EMP-0040"], open_cfg)
    assert v.hops("employee:EMP-0085", "eligible_for")
    assert v.hops("employee:EMP-0085", "works_in") == []         # dept_id no longer listed


# ---------------- config ----------------

def test_new_config_sections_are_validated():
    from src.retrieval.config import ConfigError, KoraConfig, parse_config
    assert parse_config({}).admins == [] and "manager_id" in KoraConfig().colleague_fields
    for bad in ({"admins": ["EMP-1"]}, {"admins": {"employees": [1]}}, {"admins": {"employees": [], "x": 1}},
                {"person_visibility": {"colleague_fields": "role_id"}},
                {"person_visibility": {"colleague_fields": [], "y": True}}):
        with pytest.raises(ConfigError):
            parse_config(bad)


def test_is_admin(config):
    assert config.is_admin(User("EMP-0120")) and not config.is_admin(User("EMP-0040"))
    assert not config.is_admin(None)


def test_shipped_config_names_real_employees(users):
    from src.retrieval.config import load_config
    cfg = load_config()
    assert cfg.admins and all(a in users for a in cfg.admins)


# ---------------- quality report ----------------

def test_quality_report_on_nexora(built):
    from src.graph.quality import quality_report
    r = quality_report(built["graph"], built["tracer"], built["dir"], built["loaded"].dangling)
    assert r["links"] == len(built["graph"].relationships)
    assert sum(r["links_by_tier"].values()) == r["links"]                  # every link counted once
    assert r["links_by_tier"]["unverified"] == 0 and r["unverified_evidence"] == []
    assert r["processes_without_owner"] == []
    assert r["processes_without_participants"] == ["PROC-HR-003"]         # a guideline for all employees
    assert 0.5 < r["documented_share"] < 1
    assert set(r["definitions"]) >= {"links_by_tier", "documented_share", "open_conflicts"}


def test_quality_report_for_a_user_never_names_hidden_things(built, users, config):
    from src.graph.quality import quality_report, render
    from src.retrieval.access import AccessPolicy
    policy = AccessPolicy(config.access_rules)
    eng = users["EMP-0040"]
    r = quality_report(view(built, users, config, "EMP-0040"), built["tracer"], built["dir"],
                       can_read=lambda d: policy.can_read(eng, d))
    text = render(r, "test") + json.dumps(r)
    for hidden in ("PROC-FIN-003", "DOC-DEPT-FIN", "Invoice and Payment"):
        assert hidden not in text

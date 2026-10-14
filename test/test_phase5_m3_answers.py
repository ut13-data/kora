"""
Phase 5 M3: graph answers inside Kora's normal answer flow.

  * relationship questions go through the graph; everything else is Phase 4, unchanged
  * graph evidence (quote, header line, company record) is verified again at
    answer time; a line that fails is dropped, and if no fact survives,
    Phase 4's answer stands
  * conflicts are never hidden behind a graph fact
  * access control and the log work exactly as before
"""

import json
from dataclasses import replace

import pytest

from src.retrieval.access import User
from src.retrieval.answer import Answer, AnswerLine, KoraAnswerer
from src.retrieval.config import load_config, parse_config

ENGINEER = User("EMP-0040", "Engineer", frozenset({"all", "emp:EMP-0040", "dept:DEPT-ENG"}))
FINANCE = User("EMP-0113", "Finance Manager", frozenset({"all", "emp:EMP-0113", "dept:DEPT-FIN", "manager"}))


@pytest.fixture(scope="module")
def kora(intel):
    cfg = load_config()
    assert cfg.graph_enabled, "the shipped Nexora config switches the graph on"
    return KoraAnswerer.from_intelligence(intel, cfg)


@pytest.fixture(scope="module")
def kora_without_graph(intel):
    cfg = load_config()
    cfg.graph_enabled = False
    return KoraAnswerer.from_intelligence(intel, cfg)


# ---------------- routing ----------------

def test_relationship_question_is_answered_from_the_graph(kora):
    a = kora.answer("Which teams are involved in customer onboarding?")
    assert a.status == "answered" and a.decision["rule"] == "graph_involved"
    assert a.decision["graph"]["used"] and a.decision["phase4_status"] == "not_found"
    assert all(l.kind in ("relation", "possible") for l in a.lines)
    assert "graph_ms" in a.timings and "graph" in a.startup


def test_other_questions_are_exactly_phase4(kora, kora_without_graph):
    for q in ("How many days of annual leave do Germany-based employees get?",
              "Who approves purchases above 10000 euros?",
              "What is the company policy on bringing pets to the office?",
              "Are there conflicting instructions about security training?"):
        with_graph, without = kora.answer(q), kora_without_graph.answer(q)
        assert "graph" not in with_graph.decision
        assert (with_graph.status, [l.text for l in with_graph.lines], [c.chunk_id for c in with_graph.citations]) \
            == (without.status, [l.text for l in without.lines], [c.chunk_id for c in without.citations])


def test_graph_is_off_without_config():
    assert parse_config({}).graph_enabled is False


def test_graph_config_is_validated():
    from src.retrieval.config import ConfigError
    for bad in ({"graph": True}, {"graph": {"enabled": "yes"}}, {"graph": {"enabled": True, "depth": 3}}):
        with pytest.raises(ConfigError):
            parse_config(bad)


# ---------------- citations and runtime verification ----------------

def test_company_record_is_cited_as_a_record(kora):
    a = kora.answer("Who owns the procurement process?")
    owner = next(l for l in a.lines if "Finance Manager is the recorded owner" in l.detail)
    assert owner.chunk_id == "record:processes.csv#process_id=PROC-FIN-002#owner_role"
    c = next(c for c in a.citations if c.number == owner.citation)
    assert c.kind == "record" and c.label().startswith(f"[{c.number}] Company record: processes.csv")
    assert kora.verify_line(owner)


def test_header_line_is_cited_as_the_documents_header(kora):
    a = kora.answer("Which policies govern the incident management process?")
    line = next(l for l in a.lines if l.chunk_id.startswith("header:"))
    c = next(c for c in a.citations if c.number == line.citation)
    assert c.kind == "header" and "document header" in c.label() and line.text.startswith("Related Processes: ")
    assert kora.verify_line(line)


def test_changed_text_fails_verification(kora):
    a = kora.answer("Who owns the procurement process?")
    line = a.lines[0]
    assert kora.verify_line(line)
    assert not kora.verify_line(replace(line, text=line.text.replace("Finance Manager", "CFO")))
    assert not kora.verify_line(replace(line, kind="quote"))            # pseudo IDs only for graph lines
    assert not kora.verify_line(AnswerLine("relation", "Owner/Role: CFO", "header:procurement_process.md#Owner/Role"))


def test_record_that_changed_after_loading_is_dropped_and_phase4_stands(intel):
    k = KoraAnswerer.from_intelligence(intel, load_config())
    tables = dict(k.graph.tracer.tables)
    tables["processes"] = [dict(r, owner_role="CFO", owning_dept_id="DEPT-OPS")
                           if r["process_id"] == "PROC-FIN-002" else r for r in tables["processes"]]
    k.graph.tracer.tables = tables                     # the CSV changed; the graph is from before
    a = k.answer("Who owns the procurement process?")
    # caught by the evidence check (the link no longer verifies) before any line is built
    assert "graph" not in a.decision or a.decision["graph"]["used"] is False
    assert not any(l.chunk_id and l.chunk_id.startswith("record:") for l in a.lines)
    assert "Finance Manager is the recorded owner" not in a.render()


def test_every_graph_line_in_the_dev_set_is_faithful(kora):
    from pathlib import Path
    from src.evaluation.run_eval import faithful
    users = kora.retriever.users
    questions = json.loads(Path("data/evaluation/graph_dev_questions.json").read_text(encoding="utf-8"))["questions"]
    graph_lines = 0
    for q in questions:
        a = kora.answer(q["question"], user=users.get(q["as_user"]) if q.get("as_user") else None)
        assert faithful(a, kora), q["id"]
        graph_lines += sum(l.kind in ("relation", "possible") for l in a.lines)
    assert graph_lines > 30


# ---------------- conflicts, access, log, render ----------------

def test_a_phase4_conflict_is_kept_next_to_graph_facts(kora):
    g = kora.graph.queries(None).answer("Who owns the procurement process?")
    side = AnswerLine("conflict_side", "Security training: 10 working days", None, "PROC-HR-001")
    phase4 = Answer("q", "conflict", [side], [], ["Two sources disagree."])
    decision = {"rule": "conflict"}
    a = kora._from_graph(g, phase4, decision)
    assert a.status == "conflict" and a.lines[0] is side
    assert any(l.kind == "relation" for l in a.lines) and "Two sources disagree." in a.notes


def test_access_control_holds_through_the_graph(kora):
    q = "Who takes part in the invoice and payment process?"
    eng, fin = kora.answer(q, user=ENGINEER), kora.answer(q, user=FINANCE)
    assert eng.status == "not_found" and "graph" not in eng.decision
    text = json.dumps(eng.to_log(), default=str)
    for hidden in ("PROC-FIN-003", "Invoice and Payment", "AP/AR"):
        assert hidden not in text.replace(q, "")
    assert fin.status == "answered" and fin.decision["graph"]["used"]


def test_log_record_is_json_and_names_the_route(kora):
    a = kora.answer("Why is the customer onboarding process delayed?")
    rec = json.loads(json.dumps(a.to_log()))
    assert rec["status"] == "partial" and rec["decision"]["rule"] == "graph_delay"
    assert rec["decision"]["graph"]["kind"] == "delay"
    assert any(l["kind"] == "relation" for l in rec["lines"])


def test_render_shows_claim_evidence_and_why(kora):
    a = kora.answer("What comes after the proposal and quotation process?")
    text = a.render(why=True)
    assert "Proposal and Quotation triggers Customer Onboarding" in text
    assert "knowledge graph:" in text and "Phase 4 alone would have said:" in text


def test_named_in_a_step_not_doing_it(kora):
    a = kora.answer("Which teams are involved in customer onboarding?")
    ae = next(l for l in a.lines if "Account Executive" in l.detail)
    assert "is named in a step of" in ae.detail and "has a step" not in ae.detail


def test_citation_precision_ignores_company_records():
    from src.evaluation.run_eval import citation_precision
    assert citation_precision(["PROC-FIN-002"], ["PROC-FIN-002"], []) == 1.0
    assert citation_precision([], ["PROC-FIN-002"], []) is None


# ---------------- the always-current index reloads the graph ----------------

def test_rebuild_reloads_the_graph(tmp_path):
    from test_robustness import _write_company
    from src.index import Kora, Paths
    raw = tmp_path / "raw"
    _write_company(raw)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"graph": {"enabled": True}}), encoding="utf-8")
    kora = Kora.open(Paths(raw_dir=raw, processed_dir=tmp_path / "p", chunks_dir=tmp_path / "c",
                           intelligence_dir=tmp_path / "i", context_dir=tmp_path / "x",
                           manifest=tmp_path / "m.json", config=config, log_dir=None))
    before = kora.answerer.graph
    assert before is not None
    (raw / "bike.md").write_text("# Bike Leasing\n\nEmployees can lease one bike through payroll "
                                 "for 36 months with HR approval.\n", encoding="utf-8")
    kora.ask("What is the bike leasing policy?")
    assert kora.answerer.graph is not None and kora.answerer.graph is not before

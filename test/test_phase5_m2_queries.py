"""
Phase 5 M2: relationship questions answered from the graph.

What must hold:
  * the graph only answers what it recognises (intent + a visible entity) AND
    can back with documented evidence; everything else goes to Phase 4;
  * every statement's verb matches its evidence ("lists as related" for a
    header line, never "governs");
  * every citation can be found again in its source;
  * hidden documents and colleagues' private details never appear.
"""

import json
import re
from pathlib import Path

import pytest

from src.intelligence.text import norm_key


@pytest.fixture(scope="module")
def ask(built, users, config):
    from src.graph.query import GraphQueries
    from src.graph.view import GraphView
    cache = {}

    def _ask(question, emp=None):
        if emp not in cache:
            v = GraphView.for_user(built["graph"], users[emp] if emp else None, config)
            cache[emp] = GraphQueries(v, built["tracer"], built["dir"])
        return cache[emp].answer(question)
    return _ask


@pytest.fixture(scope="module")
def resolver(built, config):
    from src.graph.resolve import Resolver
    from src.graph.view import GraphView
    return Resolver(GraphView.for_user(built["graph"], None, config))


def _keys(resolver, q):
    return [m.key for m in resolver.find(q)]


# ---------------- finding entities ----------------

@pytest.mark.parametrize("question,expected", [
    ("Which teams are involved in customer onboarding?", ["process:PROC-IMPL-001"]),
    ("Who owns the procurement process?", ["process:PROC-FIN-002"]),
    ("Who handles customer escalations?", ["process:PROC-CS-001"]),                            # plural
    ("Is NexFlow covered by the bug reporting and escalation process?",
     ["product:PROD-NEXFLOW", "process:PROC-ENG-002"]),
    ("Which roles take part in handling a P1 incident?", ["process:PROC-SUP-002"]),            # one distinctive word
    ("Who is the head of the Finance department?", ["department:DEPT-FIN"]),
    ("What does the Customer Success Manager do?", ["role:ROLE-026"]),          # role beats department
    ("Which processes use the CRM? Ask the AE.", ["role:ROLE-003"]),            # alias
])
def test_resolver_finds(resolver, question, expected):
    assert _keys(resolver, question) == expected


@pytest.mark.parametrize("question", [
    "What is a product defect?",                  # one-word department only with 'department'
    "What does onboarding involve?",              # customer AND employee onboarding: never guess
    "Who approves a customer implementation plan?",   # 'Implementation' is not a product name
    "How many days of annual leave do employees get?",
])
def test_resolver_does_not_guess(resolver, question):
    assert _keys(resolver, question) == []


def test_hidden_process_cannot_be_found(built, users, config):
    from src.graph.resolve import Resolver
    from src.graph.view import GraphView
    q = "Who takes part in the invoice and payment process?"
    eng = Resolver(GraphView.for_user(built["graph"], users["EMP-0040"], config))
    fin = Resolver(GraphView.for_user(built["graph"], users["EMP-0113"], config))
    assert eng.find(q) == [] and [m.key for m in fin.find(q)] == ["process:PROC-FIN-003"]


# ---------------- what the graph takes, and what it leaves to Phase 4 ----------------

PHASE4_SETS = ["eval_questions.json", "heldout_questions.json", "heldout_v2_questions.json",
               "heldout_v3_questions.json", "heldout_v4_questions.json", "heldout_v5_questions.json"]


def test_graph_leaves_phase4_questions_alone(ask):
    """81 Phase 4 questions: the graph may take only the old-version approval question (Q22),
    which Phase 4 also answers. Anything else taken would need its own justification."""
    taken = []
    for name in PHASE4_SETS:
        for q in json.loads(Path("data/evaluation", name).read_text(encoding="utf-8"))["questions"]:
            if ask(q["question"]) is not None:
                taken.append(q["id"])
    assert taken == ["Q22"]


@pytest.mark.parametrize("question", [
    "Who approves a discount of 20% on a proposal?",          # Phase 4's rule engine knows the tiers
    "Who approves bringing a dog to the office?",
    "How soon after a trip are my expenses paid back?",       # 'after' alone is not a 'next process' question
    "What happens if a customer wants to change the go-live date after the proposal discount was approved?",
])
def test_not_graph_questions(ask, question):
    assert ask(question) is None


# ---------------- the answers ----------------

def _names(answer):
    return answer.text().lower()


def test_involved_groups_roles_by_department(ask):
    a = ask("Which teams are involved in customer onboarding?")
    assert a.status == "answered" and a.kind == "involved"
    groups = {f.group for f in a.facts}
    assert {"Implementation / Consulting", "Customer Success", "Sales"} <= groups
    for role in ("implementation consultant", "implementation manager", "solutions architect",
                 "customer success manager", "account executive"):
        assert role in _names(a)


def test_owner_from_the_company_record(ask):
    a = ask("Who owns the procurement process?")
    owners = [f for f in a.facts if "recorded owner" in f.claim]
    assert {f.claim.split(" is ")[0] for f in owners} == {"Finance Manager", "Operations Manager"}
    assert all(f.tier == "record" and f.citation["file"] == "processes.csv" for f in owners)


def test_impact_separates_dependencies_from_cross_references(ask):
    a = ask("Which process could be affected if the NexConnect integration procedure changes?")
    assert a.status == "answered" and a.facts[0].claim == "Customer Onboarding uses NexConnect"
    assert a.facts[0].tier == "quote"
    assert any("No document describes what would happen" in n for n in a.notes)
    stated = [set(f.entities) for f in a.facts]
    assert all(set(p.entities) not in stated for p in a.possible)          # never repeated as 'possible'


def test_delay_is_always_partial_and_never_claims_a_cause(ask):
    from src.graph.query import NO_LIVE_STATUS
    a = ask("Why is the customer onboarding process delayed?")
    assert a.status == "partial" and NO_LIVE_STATUS in a.notes
    gap = next(f for f in a.facts if f.claim.startswith("Known gap"))
    assert "kickoff delays" in gap.citation["text"] and gap.citation["chunk_id"] == "PROC-IMPL-001_v2.0_003"
    assert any("triggers Customer Onboarding" in f.claim for f in a.facts)       # what feeds it
    assert not any("CONF-E95F0259" in n for n in a.notes)                        # an ID clash, not a cause


def test_header_links_are_phrased_as_listing_never_as_governing(ask):
    a = ask("Which policies govern the incident management process?")
    claims = [f.claim for f in a.facts]
    assert "Incident Management lists Data Security Policy as related" in claims
    # the Customer Escalation Policy's OWN header lists Incident Management: said that way round
    assert "Customer Escalation Policy lists Incident Management as related" in claims
    assert not any(re.search(r"\bgovern", c) for c in claims)


def test_trigger_found_from_the_sentence_when_no_process_is_named(ask):
    a = ask("Which process starts when a deal is closed?")
    assert [f.claim for f in a.facts] == ["Proposal and Quotation triggers Customer Onboarding"]
    assert "Deal Closed" in a.facts[0].citation["text"]


def test_department_head_and_products(ask):
    assert "CFO" in ask("Who is the head of the Finance department?").text()
    a = ask("Which products does the Product department own?")
    assert {f.entities[0] for f in a.facts} == {f"product:PROD-{p}" for p in
                                                 ("NEXVIEW", "NEXFLOW", "NEXCRM", "NEXAUTO", "NEXCONNECT")}


def test_old_version_is_flagged_and_uses_the_old_role_name(ask):
    a = ask("Who gave final approval for customer onboarding before 2025?")
    f = a.facts[0]
    assert f.historical and "version 1.0" in f.claim and '"Project Lead"' in f.claim
    assert any(n.startswith("Today: Implementation Manager approves") for n in a.notes)


def test_role_department_and_owned_processes(ask):
    assert "Implementation / Consulting" in ask("Which department does the Implementation Manager belong to?").text()
    a = ask("Which processes does the CISO own?")
    assert {f.entities[1] for f in a.facts} == {"process:PROC-IT-001", "process:PROC-IT-002"}


# ---------------- citations: every one can be found again ----------------

def _citation_found(built, c):
    t = built["tracer"]
    if c["kind"] == "quote" and c.get("chunk_id"):
        return norm_key(c["text"]) in norm_key(t.chunks[c["chunk_id"]]["chunk_text"])
    if c["kind"] == "header":
        field, value = c["text"].split(": ", 1)
        return f"**{field}:** {value}" in t.raw_text(c["file"])
    if c["kind"] == "record":
        row = next(r for r in t.tables[Path(c["file"]).stem] if r[c["key_column"]] == c["key"])
        return str(row[c["column"]]) == str(c["value"])
    return c["kind"] == "quote" and norm_key(c["text"]) in norm_key(t.raw_text(c["file"]))


def test_every_dev_answer_cites_findable_evidence(built, ask, users):
    questions = json.loads(Path("data/evaluation/graph_dev_questions.json").read_text(encoding="utf-8"))["questions"]
    checked = 0
    for q in questions:
        a = ask(q["question"], q.get("as_user"))
        for f in (a.facts if a else []):
            assert f.documented
            assert _citation_found(built, f.citation), (q["id"], f.claim, f.citation)
            checked += 1
    assert checked > 30


# ---------------- access and privacy ----------------

def test_engineering_never_sees_finance_restricted_things(built, ask):
    assert ask("Who takes part in the invoice and payment process?", "EMP-0040") is None
    assert ask("Who takes part in the invoice and payment process?", "EMP-0113").status == "answered"
    questions = json.loads(Path("data/evaluation/graph_dev_questions.json").read_text(encoding="utf-8"))["questions"]
    for q in questions:
        a = ask(q["question"], "EMP-0040")
        if a is None:
            continue
        seen = json.dumps([f.to_dict() for f in a.facts + a.possible]) + " ".join(a.notes)
        for hidden in ("PROC-FIN-003", "DOC-DEPT-FIN", "Invoice and Payment"):
            assert hidden not in seen, (q["id"], hidden)


def test_colleague_gets_directory_facts_only(ask):
    a = ask("What does Julia Zimmermann do?", "EMP-0040")
    assert len(a.facts) == 4
    assert all(any(v in f.claim for v in (" works as ", " works in ", " is based in ", " reports to "))
               for f in a.facts)
    assert not any("relocation" in f.claim.lower() or "checklist" in f.claim.lower() for f in a.facts)
    assert ask("Is Julia Zimmermann relocating to Germany?", "EMP-0040") is None

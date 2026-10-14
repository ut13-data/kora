"""Phase 4 M7: access control. A user only gets answers from documents they
may read, and a hidden document is never cited, quoted or named."""

import json

import pytest

from src.retrieval.access import AccessPolicy, User, users_from_tables
from src.retrieval.answer import KoraAnswerer
from src.retrieval.config import ConfigError, load_config, parse_config

FINANCE = User("EMP-0113", "Finance Manager", frozenset({"all", "emp:EMP-0113", "dept:DEPT-FIN", "manager"}))
ENGINEER = User("EMP-0040", "Engineer", frozenset({"all", "emp:EMP-0040", "dept:DEPT-ENG"}))


@pytest.fixture(scope="module")
def kora(intel):
    return KoraAnswerer.from_intelligence(intel)          # Nexora config: PROC-FIN-003 + DOC-DEPT-FIN restricted


# ---------------- building blocks ----------------

def test_policy_rules():
    p = AccessPolicy({"PROC-FIN-*": ["dept:DEPT-FIN"], "PROC-FIN-003": ["emp:EMP-0113"]})
    assert p.can_read(None, "PROC-FIN-003")                      # no user = no filtering
    assert p.can_read(FINANCE, "PROC-FIN-003")                   # passes both rules
    assert not p.can_read(User("X", groups=frozenset({"dept:DEPT-FIN"})), "PROC-FIN-003")   # deny wins
    assert not p.can_read(ENGINEER, "PROC-FIN-001")              # wildcard
    assert p.can_read(ENGINEER, "PROC-HR-002")                   # no rule = open
    assert p.allowed(ENGINEER, {"PROC-FIN-001", "PROC-HR-002"}) == {"PROC-HR-002"}


def test_users_get_groups_from_the_employees_table():
    users = users_from_tables({"employees": [
        {"employee_id": "E1", "first_name": "A", "last_name": "B", "role_id": "R1", "role_title": "Boss",
         "dept_id": "D1", "location_id": "L1", "manager_id": ""},
        {"employee_id": "E2", "role_id": "R2", "dept_id": "D1", "location_id": "L2", "manager_id": "E1"}]})
    assert users["E1"].groups == {"all", "emp:E1", "dept:D1", "loc:L1", "role:R1", "manager"}
    assert "manager" not in users["E2"].groups
    assert users_from_tables({}) == {}                           # a company without an employees table


def test_access_config_is_validated():
    assert parse_config({"access": {"_comment": "x", "rules": {"DOC-1": ["dept:D1", "manager"]}}}).access_rules \
        == {"DOC-1": ["dept:D1", "manager"]}
    with pytest.raises(ConfigError, match="invalid group"):
        parse_config({"access": {"rules": {"DOC-1": ["finance team"]}}})
    with pytest.raises(ConfigError, match="non-empty list"):
        parse_config({"access": {"rules": {"DOC-1": []}}})
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config({"access": {"rules": {}, "default": "deny"}})


def test_nexora_users_are_loaded(kora):
    assert "dept:DEPT-FIN" in kora.retriever.users["EMP-0113"].groups
    assert kora.access.active


# ---------------- end to end ----------------

QUESTION = "What are Nexora's standard payment terms for vendor invoices?"


def test_reader_gets_the_answer_and_others_do_not(kora):
    finance = kora.answer(QUESTION, user=FINANCE)
    assert finance.status == "answered" and "30-day" in finance.lines[0].text
    assert kora.answer(QUESTION).status == "answered"           # no user: unchanged behaviour
    engineer = kora.answer(QUESTION, user=ENGINEER)
    assert "PROC-FIN-003" not in {c.document_id for c in engineer.citations}
    assert all("30-day payment terms" not in l.text for l in engineer.lines)
    assert engineer.user == "EMP-0040"


LEAK_QUESTIONS = [
    QUESTION,
    "Who approves purchases above 10000 euros?",
    "What does the Finance department do?",
    "Who is responsible for month-end close?",
    "How are vendor invoices matched before payment?",
    "Which documents describe the invoice and payment process?",
    "What is PROC-FIN-003?",
    "How did the invoice process change from the previous version?",
    "Are there conflicting instructions about invoice payment?",
    "Who approves exceptions to HR policies?",
]


@pytest.mark.parametrize("question", LEAK_QUESTIONS)
def test_hidden_documents_never_appear_anywhere(kora, question):
    """Not as a citation, a retrieved chunk, a routed candidate, a line, a
    note, or in the log -- whatever the question."""
    hidden = {"PROC-FIN-003", "DOC-DEPT-FIN"}
    a = kora.answer(question, user=ENGINEER)
    assert not hidden & {c.document_id for c in a.citations}
    assert not hidden & {r.chunk.document_id for r in a.retrieval.results}
    plan = a.retrieval.plan
    assert not hidden & set(plan.candidate_documents + plan.direct_documents + plan.tone_profiles)
    assert not any(kora._chunks[l.chunk_id].document_id in hidden for l in a.lines if l.chunk_id)
    assert not any(h in n for h in hidden for n in a.notes + plan.reasons)
    log = json.dumps(a.to_log()["retrieval"]["results"]) + json.dumps(a.to_log()["citations"])
    assert not any(h in log for h in hidden)


def test_a_conflict_with_a_hidden_side_is_not_shown(intel):
    raw = json.loads(open("config/kora_config.json", encoding="utf-8").read())
    raw["access"]["rules"]["PROC-HR-001-ADDENDUM"] = ["dept:DEPT-HR"]
    k = KoraAnswerer.from_intelligence(intel, parse_config(raw, "test"))
    q = "Are there conflicting instructions about security training?"
    assert k.answer(q).status == "conflict"                     # full access: both sides
    a = k.answer(q, user=ENGINEER)
    assert a.status != "conflict" and not any(l.kind == "conflict_side" for l in a.lines)
    assert not any("PROC-HR-001-ADDENDUM" in n or "does not pick one" in n for n in a.notes)
    assert "PROC-HR-001-ADDENDUM" not in {c.document_id for c in a.citations}


def test_no_rules_means_no_filtering(intel):
    k = KoraAnswerer.from_intelligence(intel, parse_config({}))
    assert not k.access.active
    assert k.answer(QUESTION, user=ENGINEER).status == "answered"


def test_kora_user_lookup(tmp_path):
    from test_robustness import _write_company
    from src.index import Kora, Paths
    raw = tmp_path / "raw"
    _write_company(raw)
    kora = Kora.open(Paths(raw_dir=raw, processed_dir=tmp_path / "p", chunks_dir=tmp_path / "c",
                           intelligence_dir=tmp_path / "i", context_dir=tmp_path / "x",
                           manifest=tmp_path / "m.json", config=tmp_path / "none.json", log_dir=None))
    assert kora.user("NOBODY") is None
    assert kora.ask("How many days of annual leave do employees get?", user=None).user is None

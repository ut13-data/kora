"""
Phase 5 M0: the groundwork that makes later before/after numbers honest.

  * stable ranking   -> the same files give byte-identical Phase 3 output
  * answer log v2    -> every record says where it came from (channel) and
                        whether KORA answered or crashed (kind)
"""

import json

import pytest

from src.intelligence.text import top_terms


# ---------------- stable ranking ----------------

def test_top_terms_breaks_ties_alphabetically():
    terms = ["success", "customer success", "csm", "zeta"]
    scores = [0.5, 0.5, 0.9, 0.1]
    assert top_terms(terms, scores, 3) == ["csm", "customer success", "success"]


def test_top_terms_treats_float_noise_as_a_tie():
    # 0.1 + 0.2 is 0.30000000000000004: still the same score as 0.3
    assert top_terms(["b", "a"], [0.1 + 0.2, 0.3], 2) == ["a", "b"]


def test_top_terms_positive_only_drops_zero_scores():
    assert top_terms(["a", "b", "c"], [0.0, 0.4, 0.0], 5, positive_only=True) == ["b"]
    assert top_terms(["a", "b", "c"], [0.0, 0.4, 0.0], 5) == ["b", "a", "c"]


def _paths(tmp_path, raw, name):
    from src.index import Paths
    base = tmp_path / name
    return Paths(raw_dir=raw, processed_dir=base / "processed", chunks_dir=base / "chunks",
                 intelligence_dir=base / "intelligence", context_dir=base / "context",
                 manifest=base / "manifest.json", config=tmp_path / "no_config.json", log_dir=None)


def test_two_rebuilds_give_identical_intelligence_files(tmp_path):
    from test_robustness import _write_company
    from src.index import rebuild
    raw = tmp_path / "raw"
    _write_company(raw)
    paths = _paths(tmp_path, raw, "build")          # same folders both times: paths appear in the output
    rebuild(paths)
    first = {p.name: p.read_bytes() for p in paths.intelligence_dir.glob("*.json")}
    rebuild(paths)
    second = {p.name: p.read_bytes() for p in paths.intelligence_dir.glob("*.json")}
    assert first, "the rebuild wrote no intelligence files"
    assert first.keys() == second.keys()
    assert [n for n in first if first[n] != second[n]] == []


def test_loaded_context_paths_use_forward_slashes(tmp_path):
    """'loaded data\\context\\x.json' on Windows vs 'data/context/x.json' on Linux
    made the same build differ between machines."""
    from src.intelligence.communication import load_preferences
    from src.intelligence.germany import load_progress
    win_style = tmp_path / "context"
    win_style.mkdir()
    for name in ("communication_preferences.json", "checklist_progress.json"):
        (win_style / name).write_text('{"employees": {}}', encoding="utf-8")
    for loader, name in ((load_preferences, "communication_preferences.json"),
                         (load_progress, "checklist_progress.json")):
        _, source = loader(win_style / name)
        assert "\\" not in source and source.endswith(f"context/{name}")


# ---------------- answer log, schema 2 ----------------

@pytest.fixture
def logged(tmp_path):
    from test_robustness import _write_company
    from src.index import Paths
    raw = tmp_path / "raw"
    _write_company(raw)
    return Paths(raw_dir=raw, processed_dir=tmp_path / "processed", chunks_dir=tmp_path / "chunks",
                 intelligence_dir=tmp_path / "intelligence", context_dir=tmp_path / "context",
                 manifest=tmp_path / "manifest.json", config=tmp_path / "no_config.json",
                 log_dir=tmp_path / "logs")


def _records(paths):
    path = paths.log_dir / "answers.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_answer_record_has_schema_kind_and_channel(logged):
    from src.index import Kora
    kora = Kora.open(logged, channel="ui")
    a = kora.ask("How many days of annual leave do employees get?")
    [rec] = _records(logged)
    assert rec["schema"] == 2 and rec["kind"] == "answer" and rec["channel"] == "ui"
    assert rec["status"] == a.status and rec["question"] == a.question


def test_default_channel_is_python(logged):
    from src.index import Kora
    Kora.open(logged).ask("How many days of annual leave do employees get?")
    assert _records(logged)[0]["channel"] == "python"


def test_unknown_channel_is_refused(logged):
    from src.index import Kora
    with pytest.raises(ValueError):
        Kora.open(logged, channel="slack")


def test_crash_while_answering_is_logged_as_error_and_still_raised(logged, monkeypatch):
    from src.index import Kora
    kora = Kora.open(logged, channel="cli")

    def broken(*args, **kwargs):
        raise KeyError("chunk X-1 missing")
    monkeypatch.setattr(kora.answerer, "answer", broken)
    with pytest.raises(KeyError):
        kora.ask("How many days of annual leave do employees get?", user=None)
    [rec] = _records(logged)
    assert rec["kind"] == "error" and rec["stage"] == "answer" and rec["error_type"] == "KeyError"
    assert rec["channel"] == "cli" and "status" not in rec      # never counted as an answer status
    assert "chunk X-1 missing" in rec["message"]


def test_failed_rebuild_is_logged_as_index_error(logged, monkeypatch):
    import src.index as index
    from src.index import IndexBuildError, Kora
    kora = Kora.open(logged)
    (logged.raw_dir / "new.md").write_text("# New\n\nA new policy text for the test.\n", encoding="utf-8")

    def failing_rebuild(paths):
        raise IndexBuildError("phase 2 failed on new.md")
    monkeypatch.setattr(index, "rebuild", failing_rebuild)
    with pytest.raises(IndexBuildError):
        kora.ask("What is the new policy?")
    [rec] = _records(logged)
    assert rec["kind"] == "error" and rec["stage"] == "index" and rec["error_type"] == "IndexBuildError"


def test_error_record_names_the_user_id_only(logged, monkeypatch):
    from src.index import Kora
    from src.retrieval.access import User
    kora = Kora.open(logged)
    monkeypatch.setattr(kora.answerer, "answer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        kora.ask("q", user=User("EMP-0001", "Ana Example, Analyst", frozenset({"all"})))
    assert _records(logged)[0]["user"] == "EMP-0001"


def test_no_log_dir_writes_nothing(logged, monkeypatch):
    from dataclasses import replace
    from src.index import Kora
    quiet = replace(logged, log_dir=None)
    Kora.open(quiet).ask("How many days of annual leave do employees get?")
    assert not (logged.log_dir / "answers.jsonl").exists()


# ---------------- evaluation: names, leaks, baseline comparison ----------------

def _answer(text="", notes=(), cite_title="", question="q"):
    from src.retrieval.answer import Answer, AnswerLine, Citation
    lines = [AnswerLine("quote", text, "C-1")] if text else []
    cites = [Citation(1, "C-1", "DOC-X", cite_title, "1.0", "Active", "2025-01-01", "s", "f.md", "")] \
        if cite_title else []
    return Answer(question, "answered" if text else "not_found", lines, cites, notes=list(notes))


def test_names_found_is_a_share_and_ignores_case():
    from src.evaluation.run_eval import names_found
    a = _answer("The finance manager and the CFO approve it.")
    assert names_found(a, ["Finance Manager", "Operations Manager"]) == 0.5
    assert names_found(a, []) is None


def test_leak_in_lines_notes_or_citations_is_caught():
    from src.evaluation.run_eval import leaks
    assert leaks(_answer("Ask the AP/AR Specialist."), ["AP/AR Specialist"]) == ["AP/AR Specialist"]
    assert leaks(_answer(cite_title="Invoice and Payment Process"), ["Invoice and Payment"]) == ["Invoice and Payment"]
    assert leaks(_answer(notes=["An older version of PROC-FIN-003 exists."]), ["PROC-FIN-003"]) == ["PROC-FIN-003"]


def test_echo_of_the_users_own_words_in_a_note_is_not_a_leak():
    from src.evaluation.run_eval import leaks
    a = _answer(notes=["The knowledge base never mentions julia, zimmermann."],
                question="Is Julia Zimmermann relocating to Germany?")
    assert leaks(a, ["Zimmermann"]) == []
    found = _answer("Julia Zimmermann is in the relocation program.", question="Is Julia Zimmermann relocating?")
    assert leaks(found, ["Zimmermann"]) == ["Zimmermann"]          # found data is still a leak


def _snap(status, correct, cited=(), leaks=()):
    summary = {m: None for m in ("status_accuracy", "key_fact_found", "names_found", "no_leak",
                                 "citation_precision", "faithfulness", "hit@5")}
    return {"s": {"summary": summary, "per_question": {
        "Q1": {"question": "q", "status": status, "status_correct": correct, "cited": list(cited),
               "faithful": True, "names_found": None, "leaks": list(leaks)}}}}


def test_compare_sorts_fixed_broken_changed():
    from src.evaluation.baseline import compare
    assert len(compare(_snap("not_found", False), _snap("answered", True))["fixed"]) == 1
    assert len(compare(_snap("answered", True), _snap("not_found", False))["broken"]) == 1
    changed = compare(_snap("answered", True, ["A"]), _snap("answered", True, ["B"]))
    assert len(changed["changed"]) == 1 and not changed["broken"]
    same = compare(_snap("answered", True), _snap("answered", True))
    assert all(same[k] == [] for k in ("fixed", "broken", "changed", "missing", "new"))


def test_a_new_leak_is_broken_even_if_the_status_got_fixed():
    from src.evaluation.baseline import compare
    report = compare(_snap("answered", False), _snap("not_found", True, leaks=["PROC-FIN-003"]))
    assert len(report["broken"]) == 1 and not report["fixed"]


def test_markdown_report_states_the_verdict_and_keeps_the_table():
    from src.evaluation.baseline import compare, render, report_markdown
    good = compare(_snap("not_found", False), _snap("answered", True))
    page = report_markdown(render(good), good)
    assert "No question got worse." in page and "FIXED: 1" in page and render(good) in page
    bad = compare(_snap("answered", True), _snap("not_found", False))
    assert "1 question(s) got worse" in report_markdown(render(bad), bad)


def test_graph_question_sets_are_well_formed():
    from pathlib import Path
    from src.evaluation.run_eval import load_dataset
    required = {"id", "category", "graph_kind", "question", "gold_documents", "acceptable_documents",
                "expected_status", "expected_names", "forbidden_names", "seen_in_development"}
    statuses = {"answered", "partial", "conflict", "not_found"}
    for name, seen in (("graph_dev_questions.json", True), ("graph_heldout_questions.json", False)):
        qs = load_dataset(Path("data/evaluation") / name)
        assert len({q["id"] for q in qs}) == len(qs)
        for q in qs:
            assert required <= q.keys(), q["id"]
            assert set(q["expected_status"]) <= statuses, q["id"]
            if q["seen_in_development"] is not seen:      # only with a written reason (H02, seen by mistake in M2)
                assert not seen and "SEEN during" in q.get("note", ""), q["id"]
            if q["forbidden_names"]:                 # a leak test only makes sense as a refusal
                assert q["expected_status"] == ["not_found"], q["id"]

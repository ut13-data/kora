"""
Phase 5 M6: dashboards from the answer log.

  * every line counted once; unreadable lines skipped AND counted
  * crashes are never an answer status
  * totals only: no person anywhere, question text only with enough askers and no names
  * the knowledge section follows the admin's own access
"""

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.analytics.answers_log import read_log, summarise

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def answer(q, status, user, days_ago=0, channel="ui", **extra):
    rec = {"schema": 2, "kind": "answer", "channel": channel, "question": q, "status": status, "user": user,
           "at": (NOW - timedelta(days=days_ago)).isoformat(), "timings": {"route_ms": 1.0, "search_ms": 2.0, "total_ms": 5.0},
           "startup": {"total_s": 0.3}, "citations": [], "uncovered_terms": [], "decision": {}, "retrieval": {}}
    rec.update(extra)
    return rec


def write(tmp_path, records, raw_lines=()):
    p = tmp_path / "answers.jsonl"
    p.write_text("\n".join([json.dumps(r) for r in records] + list(raw_lines)) + "\n", encoding="utf-8")
    return p


# ---------------- reading ----------------

def test_missing_and_empty_logs_are_safe(tmp_path):
    missing = read_log(tmp_path / "none.jsonl")
    assert missing.missing_file and summarise(missing, now=NOW)["answers"] == 0
    empty = summarise(read_log(write(tmp_path, [])), now=NOW)
    assert empty["answers"] == 0 and empty["status"]["share"]["answered"] is None
    assert empty["performance"]["overall"]["p50"] is None


def test_unreadable_lines_are_skipped_and_counted(tmp_path):
    ok = answer("Who owns procurement?", "answered", "EMP-1")
    no_time = dict(ok, at="yesterday")
    bad_status = dict(ok, status="maybe", at=NOW.isoformat())
    odd_kind = dict(ok, kind="debug")
    log = read_log(write(tmp_path, [ok, no_time, bad_status, odd_kind], ["{not json", "[1, 2]", ""]))
    assert len(log.records) == 1 and log.lines == 6
    assert log.skipped == {"not valid JSON": 1, "not a JSON object": 1, "no readable timestamp": 1,
                           "answer without a known status": 1, "unknown kind 'debug'": 1}


def test_exact_repeats_count_once(tmp_path):
    r = answer("Who owns procurement?", "answered", "EMP-1")
    log = read_log(write(tmp_path, [r, r, dict(r, user="EMP-2")]))
    assert len(log.records) == 2 and log.repeats == 1


def test_old_records_without_kind_or_channel(tmp_path):
    old = {"at": NOW.isoformat(), "question": "q", "status": "partial"}      # Phase 4, schema 1
    s = summarise(read_log(write(tmp_path, [old])), now=NOW)
    assert s["answers"] == 1 and s["usage"]["channels"] == {"unknown": 1}


# ---------------- status and errors ----------------

def test_errors_are_never_an_answer_status(tmp_path):
    err = {"schema": 2, "kind": "error", "at": NOW.isoformat(), "channel": "ui", "stage": "answer",
           "error_type": "KeyError", "question": "q", "user": "EMP-1"}
    recs = [answer("a", "answered", "E1"), answer("b", "not_found", "E2"), err]
    s = summarise(read_log(write(tmp_path, recs)), now=NOW)
    assert s["answers"] == 2 and sum(s["status"]["counts"].values()) == 2
    assert s["status"]["share"]["not_found"] == 0.5
    assert s["errors"]["count"] == 1 and s["errors"]["by_stage"] == {"answer": 1}


def test_filters_by_channel_and_period(tmp_path):
    recs = [answer("a", "answered", "E1", days_ago=1, channel="ui"), answer("b", "answered", "E2", days_ago=40, channel="cli")]
    log = read_log(write(tmp_path, recs))
    assert summarise(log, channel="ui", now=NOW)["answers"] == 1
    assert summarise(log, days=30, now=NOW)["answers"] == 1
    assert summarise(log, now=NOW)["answers"] == 2
    per_day = summarise(log, now=NOW)["status"]["per_day"]
    assert [d["day"] for d in per_day] == sorted(d["day"] for d in per_day)


# ---------------- privacy ----------------

def test_question_text_needs_enough_different_people(tmp_path):
    recs = [answer("What is my salary structure?", "not_found", "E1"),
            answer("What is my salary structure?", "not_found", "E1", days_ago=1),          # same person twice
            answer("Policy on pets?", "not_found", "E1"), answer("Policy on pets?", "not_found", "E2")]
    s = summarise(read_log(write(tmp_path, recs)), now=NOW, min_askers=2)
    assert [r["question"] for r in s["gaps"]["refused"]] == ["Policy on pets?"]
    assert s["gaps"]["refused_hidden"] == 1
    assert len(summarise(read_log(write(tmp_path, recs)), now=NOW, min_askers=1)["gaps"]["refused"]) == 2


def test_names_are_counted_never_listed(tmp_path):
    names = {"unknown": ["julia", "zimmermann", "pets"], "unknown_words": [
        {"word": "julia", "blocked": "a name in the question"}, {"word": "zimmermann", "blocked": "a name in the question"},
        {"word": "pets", "blocked": None}]}
    recs = [answer("Is Julia Zimmermann relocating?", "not_found", f"E{i}", decision=names,
                   uncovered_terms=["julia", "zimmermann"]) for i in range(5)]
    s = summarise(read_log(write(tmp_path, recs)), now=NOW)
    words = [w["word"] for w in s["gaps"]["unknown_words"]]
    assert words == ["pets"] and s["gaps"]["name_words_hidden"] == 10
    assert s["gaps"]["refused"] == [] and s["gaps"]["refused_hidden"] == 1     # 5 askers, still hidden: it names a person
    assert "Zimmermann" not in json.dumps(s, default=str)


def test_no_person_appears_anywhere(tmp_path):
    recs = [answer(f"q{i}", st, f"EMP-00{i}") for i, st in enumerate(["answered", "partial", "not_found", "conflict"])]
    text = json.dumps(summarise(read_log(write(tmp_path, recs)), now=NOW), default=str)
    assert "EMP-00" not in text and '"user"' not in text


# ---------------- gaps, speed, usage ----------------

def test_weak_topics_need_three_answers_and_drop_the_strength_tag(tmp_path):
    t = {"entities": ["topic:TOPIC-LEAVE (strong)"]}
    recs = [answer(f"q{i}", s, f"E{i}", retrieval=t) for i, s in enumerate(["answered", "partial", "not_found"])]
    recs.append(answer("x", "not_found", "E9", retrieval={"entities": ["topic:TOPIC-RARE (weak)"]}))
    weak = summarise(read_log(write(tmp_path, recs)), now=NOW)["gaps"]["weak_topics"]
    assert weak == [{"topic": "TOPIC-LEAVE", "answers": 3, "weak_share": 0.67}]


def test_latency_and_steps(tmp_path):
    recs = [answer(f"q{i}", "answered", f"E{i}", timings={"route_ms": 1.0, "graph_ms": 2.0, "total_ms": float(ms)})
            for i, ms in enumerate([5, 6, 7, 8, 100])]
    p = summarise(read_log(write(tmp_path, recs)), now=NOW)["performance"]
    assert p["overall"] == {"p50": 7.0, "p95": 100.0, "max": 100.0}
    assert p["steps"] == {"graph": 2.0, "route": 1.0}
    assert p["slowest"][0]["ms"] == 100.0 and p["slowest"][0]["question"] is None   # one asker: text hidden


def test_cited_documents_follow_access_and_count_records(tmp_path):
    cites = ["PROC-HR-002_v1.5_000", "PROC-FIN-003_v1.6_001", "record:processes.csv#process_id=X#owner_role"]
    recs = [answer("q", "answered", "E1", citations=cites), answer("r", "not_found", "E2", citations=cites)]
    u = summarise(read_log(write(tmp_path, recs)), now=NOW, can_read=lambda d: d != "PROC-FIN-003")["usage"]
    assert u["cited_documents"] == [{"document_id": "PROC-HR-002", "answers": 1}]     # not-found sources aren't "cited"
    assert u["company_record_citations"] == 1


def test_question_categories(tmp_path):
    recs = [answer("a", "answered", "E1", decision={"graph": {"used": True, "kind": "owner"}}),
            answer("b", "answered", "E2", retrieval={"intents": ["approval"]}), answer("c", "answered", "E3")]
    cats = summarise(read_log(write(tmp_path, recs)), now=NOW)["usage"]["categories"]
    assert cats == {"relationship (graph): owner": 1, "approval": 1, "general": 1}


# ---------------- config, API and page ----------------

def test_dashboard_config_is_validated():
    from src.retrieval.config import ConfigError, parse_config
    assert parse_config({}).dashboard_min_askers == 2
    assert parse_config({"dashboards": {"min_askers_for_question_text": 1}}).dashboard_min_askers == 1
    for bad in ({"dashboards": {"min_askers_for_question_text": 0}}, {"dashboards": {"min_askers_for_question_text": True}},
                {"dashboards": {"x": 1}}, {"dashboards": 3}):
        with pytest.raises(ConfigError):
            parse_config(bad)


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    from src.index import Kora, Paths
    from src.web.app import create_app
    if not Path("data/raw").exists():
        pytest.skip("Nexora data not found in data/raw")
    tmp = tmp_path_factory.mktemp("dash")
    config = tmp / "kora_config.json"
    shutil.copy("config/kora_config.json", config)
    paths = Paths(raw_dir=Path("data/raw"), processed_dir=tmp / "processed", chunks_dir=tmp / "chunks",
                  intelligence_dir=tmp / "intelligence", context_dir=Path("data/context"),
                  manifest=tmp / "manifest.json", config=config, log_dir=tmp / "logs")
    kora = Kora.open(paths)
    app = create_app(kora=kora, secret="s", dev_passcode="pass")

    def client(emp):
        c = TestClient(app, follow_redirects=False)
        assert c.post("/api/login", json={"employee_id": emp, "passcode": "pass"}).status_code == 200
        return c
    for emp in ("EMP-0040", "EMP-0113"):
        client(emp).post("/api/ask", json={"question": "Who takes part in the invoice and payment process?", "channel": "ui"})
    return client


def test_dashboard_is_for_admins_only(web):
    eng = web("EMP-0040")
    assert eng.get("/api/admin/dashboard").status_code == 403
    assert eng.get("/dashboard").headers["location"] == "/ask"
    admin = web("EMP-0120")
    assert admin.get("/dashboard").status_code == 200
    assert admin.get("/api/admin/dashboard?channel=slack").status_code == 422
    assert admin.get("/api/admin/dashboard?days=0").status_code == 422


def test_dashboard_from_the_real_log(web):
    d = web("EMP-0120").get("/api/admin/dashboard").json()
    assert d["answers"] == 2 and d["usage"]["channels"] == {"ui": 2}
    assert d["status"]["counts"]["answered"] == 1 and d["status"]["counts"]["not_found"] == 1
    k = d["knowledge"]
    assert k["graph"] and k["documented_share"] > 0.5 and k["documents_by_department"]
    text = json.dumps(d)
    assert "PROC-FIN-003" not in text and "EMP-0040" not in text and "EMP-0113" not in text

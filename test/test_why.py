"""Phase 4 M5b (A + B): every answer says WHY it got its status, and how
long each step took -- for the CLI now and the UI in Phase 5."""

import json

import pytest

from src.retrieval.answer import KoraAnswerer, explain_decision, format_timings


@pytest.fixture(scope="module")
def kora(intel):
    return KoraAnswerer.from_intelligence(intel)


@pytest.mark.parametrize("question, status, rule", [
    ("Who approves purchases above 10000 euros?", "answered", "covered"),
    ("What is the company policy on bringing pets to the office?", "not_found", "unknown_subject"),
    ("What is the parental leave policy for contractors in Austin?", "partial", "partly_covered"),
    ("Are there conflicting instructions about security training?", "conflict", "conflict"),
])
def test_every_status_names_the_rule_that_decided_it(kora, question, status, rule):
    a = kora.answer(question)
    assert a.status == status and a.decision["rule"] == rule
    assert a.decision["rule_text"]
    assert explain_decision(a.decision)[0].startswith(f"decided by: {rule}")


def test_decision_shows_the_scores_and_thresholds(kora):
    d = kora.answer("What is the company policy on bringing pets to the office?").decision
    assert d["word_strong_needed"] == kora.config.t("strong_evidence_score")
    assert d["meaning_strong_needed"] == kora.config.t("semantic_strong_evidence")
    assert "pets" in d["unknown"] and any(m["word"] == "pets" for m in d["unknown_words"])


def test_timings_per_step_and_startup(kora):
    a = kora.answer("Who approves purchases above 10000 euros?")
    graph = {"graph_ms"} if kora.graph is not None else set()      # Phase 5: graph step, when enabled
    assert set(a.timings) == {"route_ms", "search_ms", "compose_ms", "total_ms"} | graph
    assert all(v >= 0 for v in a.timings.values())
    assert a.timings["total_ms"] >= a.timings["route_ms"] + a.timings["search_ms"] - 0.01
    assert {"router", "word_indexes", "answer_layer", "total_s"} <= set(a.startup)
    assert "start-up" in format_timings(a.timings, a.startup)


def test_render_and_log_carry_why_and_timing(kora):
    a = kora.answer("Who approves purchases above 10000 euros?")
    text = a.render(why=True)
    assert "Why this status:" in text and "Timing:" in text
    assert "Why this status:" not in a.render()                     # only on request
    record = a.to_log()
    json.dumps(record)
    assert record["decision"]["rule"] == "covered" and record["timings"] and record["startup"]


def test_meaning_partner_is_reported(intel):
    """With the stand-in: 'refund' is never used; its closest quoted word is
    shown with its score and the threshold it needed."""
    from test_semantic import SYNONYMS, _config
    from src.retrieval.embeddings import HashingEmbedder
    k = KoraAnswerer.from_intelligence(intel, _config(semantic_quote_min=0.12, semantic_strong_evidence=0.15,
                                                      semantic_min_score=0.10))
    k.use_embedder(HashingEmbedder(synonyms=SYNONYMS))
    a = k.answer("How quickly is my cash refunded?")
    m = {x["word"]: x for x in a.decision["unknown_words"]}
    assert m["refunded"]["best"] == "reimbursed" and m["refunded"]["explained"]
    assert m["refunded"]["needed"] == k.config.t("semantic_word_min")
    assert "chunk_vectors" in a.startup and "sentence_vectors" in a.startup


def test_name_block_is_reported(intel):
    from test_semantic import _config
    from src.retrieval.embeddings import HashingEmbedder
    k = KoraAnswerer.from_intelligence(intel, _config(semantic_strong_evidence=0.15))
    k.use_embedder(HashingEmbedder())
    a = k.answer("How many leave days do employees in the Paris office get?")
    m = {x["word"]: x for x in a.decision["unknown_words"]}
    assert m["paris"]["blocked"] == "a name in the question" and not m["paris"]["explained"]
    assert any("paris" in l and "a name in the question" in l for l in explain_decision(a.decision))


def test_index_check_time_is_added_by_kora_ask(tmp_path):
    from test_robustness import _write_company
    from src.index import Kora, Paths
    raw = tmp_path / "raw"
    _write_company(raw)
    paths = Paths(raw_dir=raw, processed_dir=tmp_path / "processed", chunks_dir=tmp_path / "chunks",
                  intelligence_dir=tmp_path / "intelligence", context_dir=tmp_path / "context",
                  manifest=tmp_path / "manifest.json", config=tmp_path / "no_config.json", log_dir=None)
    kora = Kora.open(paths)
    a = kora.ask("How many days of annual leave do employees get?")
    assert "index_check_ms" in a.timings and "rebuild" not in a.startup
    (raw / "bike.md").write_text("# Bike Leasing\n\nEmployees can lease one bike through payroll "
                                 "for 36 months with HR approval.\n", encoding="utf-8")
    b = kora.ask("What is the bike leasing policy?")
    assert b.startup["rebuild"] > 0 and b.startup["total_s"] >= b.startup["rebuild"]

"""Phase 4 M5b: semantic embeddings -- plumbing, safety rules and fail-safe.

Most tests use HashingEmbedder, a deterministic STAND-IN with a synonym map
(no model, no network), to check KORA's rules. Tests marked `real_model` run
only where sentence-transformers and the model are available (UT's machine).
"""

import numpy as np
import pytest

from src.retrieval.answer import KoraAnswerer
from src.retrieval.config import load_config, parse_config
from src.retrieval.embeddings import HashingEmbedder, VectorCache, load_embedder, rrf


def _config(**thresholds):
    """The Nexora config with embeddings on and some thresholds overridden."""
    import json
    raw = json.loads(open("config/kora_config.json", encoding="utf-8").read())
    raw["thresholds"].update(thresholds)
    raw["embeddings"]["cache_file"] = ""          # tests never touch the real cache
    raw["embeddings"]["enabled"] = True           # embeddings are OPTIONAL (off in the Nexora config)
    return parse_config(raw, "test")


# The stand-in "understands" these paraphrases; nothing else.
SYNONYMS = {"refund": "reimbursed", "refunded": "reimbursed", "cash": "expenses", "outlay": "expenses",
            "journey": "travel"}


@pytest.fixture(scope="module")
def semantic_kora(intel):
    # Low thresholds on purpose (the stand-in's scores are small): the
    # off-topic tests below must still abstain -- the word-level rule, not a
    # threshold, is what protects them.
    cfg = _config(semantic_quote_min=0.12, semantic_strong_evidence=0.15, semantic_min_score=0.10)
    kora = KoraAnswerer.from_intelligence(intel, cfg)
    kora.use_embedder(HashingEmbedder(synonyms=SYNONYMS))
    return kora


# ---------------- building blocks ----------------

def test_rrf_combines_rankings_by_position():
    assert rrf([["a", "b", "c"], ["c", "a"]]) == ["a", "c", "b"]
    assert rrf([["x", "y"], []]) == ["x", "y"]          # one ranking alone keeps its order


def test_vector_cache_only_encodes_new_text(tmp_path):
    class Counting(HashingEmbedder):
        calls = 0

        def encode(self, texts):
            Counting.calls += len(texts)
            return super().encode(texts)
    path = tmp_path / "cache.npz"
    first = VectorCache(Counting(), path)
    first.vectors(["alpha beta gamma", "delta epsilon"])
    first.save(["alpha beta gamma", "delta epsilon"])
    second = VectorCache(Counting(), path)
    Counting.calls = 0
    v = second.vectors(["alpha beta gamma", "delta epsilon", "new text here"])
    assert Counting.calls == 1 and v.shape == (3, 4096)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_cache_from_another_model_is_ignored(tmp_path):
    path = tmp_path / "cache.npz"
    a = VectorCache(HashingEmbedder(dim=4096), path)
    a.vectors(["some text"]); a.save(["some text"])
    b = VectorCache(HashingEmbedder(dim=128), path)      # different name -> fresh cache
    assert b.vectors(["some text"]).shape == (1, 128)


# ---------------- fail-safe ----------------

def test_missing_model_means_words_only(intel, monkeypatch):
    monkeypatch.setattr("src.retrieval.embeddings.load_embedder",
                        lambda name: (None, "embeddings off: simulated failure"))
    cfg = _config()
    kora = KoraAnswerer.from_intelligence(intel, cfg)
    assert kora.retriever.chunks.semantic is None
    assert "simulated failure" in kora.retriever.semantic_status
    assert any("simulated failure" in w for w in cfg.warnings)
    assert kora.answer("Who approves purchases above 10000 euros?").status == "answered"


def test_embeddings_off_without_config():
    assert load_config(None).embeddings_enabled is False


def test_invalid_embeddings_config_is_rejected():
    from src.retrieval.config import ConfigError
    with pytest.raises(ConfigError, match="embeddings"):
        parse_config({"embeddings": {"enabled": "yes", "nope": 1}})


# ---------------- rules, with the stand-in ----------------

def test_meaning_finds_a_paraphrased_question(semantic_kora):
    """'refund' / 'cash' never appear in the knowledge base; by meaning they
    are 'reimbursed' / 'expenses' -- the expense policy must be found."""
    outcome = semantic_kora.retriever.retrieve("How quickly is my cash refunded?")
    assert "PROC-FIN-001" in {r.chunk.document_id for r in outcome.results}


def test_paraphrase_is_answered_with_a_visible_note(semantic_kora):
    a = semantic_kora.answer("How quickly is my cash refunded?")
    assert a.status in ("answered", "partial")
    assert any("reimbursed" in l.text for l in a.lines)
    assert any(n.startswith("Matched by meaning") for n in a.notes)


def test_words_known_elsewhere_are_still_flagged(semantic_kora):
    """Meaning never excuses a word the knowledge base HAS but the quoted
    source doesn't ('contractors' is in the data security policy)."""
    a = semantic_kora.answer("What is the parental leave policy for contractors in Austin?")
    assert a.status == "partial" and "contractors" in a.uncovered_terms


@pytest.mark.parametrize("question", [
    "What is the company policy on bringing pets to the office?",
    "Who founded the Munich office canteen?",
])
def test_off_topic_questions_still_abstain(semantic_kora, question):
    assert semantic_kora.answer(question).status == "not_found"


def test_faithfulness_holds_with_meaning_based_quotes(semantic_kora):
    for q in ["How quickly is my cash refunded?", "Who approves purchases above 10000 euros?",
              "Are there conflicting instructions about security training?"]:
        a = semantic_kora.answer(q)
        assert all(semantic_kora.verify_line(l) for l in a.lines) and not a.dropped_lines


# ---------------- names are never matched by meaning (M5b fix 3) ----------------

def test_name_helpers():
    from src.retrieval.answer import name_words, question_names
    assert name_words(["Our Berlin office. The office opens at 9.", "Travel is reimbursed. travel"]) == {"berlin", "our", "the"}
    assert question_names("How many leave days do employees in the Paris office get?") == {"paris"}
    assert question_names("how many leave days in paris?") == set()
    assert question_names("How Many Leave Days In The Paris Office?") == set()   # Title Case: no signal


@pytest.fixture(scope="module")
def city_kora(intel):
    """The stand-in is told 'paris' MEANS 'berlin' -- like a real model,
    for which two city names are close in meaning."""
    cfg = _config(semantic_quote_min=0.12, semantic_strong_evidence=0.15, semantic_min_score=0.10)
    kora = KoraAnswerer.from_intelligence(intel, cfg)
    kora.use_embedder(HashingEmbedder(synonyms={**SYNONYMS, "paris": "berlin"}))
    return kora


@pytest.mark.parametrize("question", [
    "How many leave days do employees in the Paris office get?",    # name in the question
    "how many leave days do employees in the paris office get?",    # lowercase: the source side is a name
])
def test_unknown_place_is_never_answered_by_another_place(city_kora, question):
    a = city_kora.answer(question)
    assert a.status != "answered"
    assert not any(n.startswith("Matched by meaning") and "paris" in n for n in a.notes)
    if a.status != "not_found":
        assert "paris" in a.uncovered_terms


def test_name_rule_is_what_blocks_it(city_kora):
    """Control: without the name rule, the stand-in WOULD excuse 'paris' via
    'Berlin' -- so the test above is not passing by accident."""
    from src.retrieval.answer import AnswerLine
    line = [AnswerLine("quote", "Employees at the Berlin office follow the leave policy.", "x")]
    assert not city_kora._explained_by_meaning("paris", line, {"paris"})
    assert not city_kora._explained_by_meaning("paris", line)
    saved = city_kora._kb_names
    city_kora._kb_names = set()
    try:
        assert city_kora._explained_by_meaning("paris", line)
    finally:
        city_kora._kb_names = saved


def test_common_word_paraphrase_still_works(city_kora):
    from src.retrieval.answer import AnswerLine
    line = [AnswerLine("quote", "Approved expenses are reimbursed within ten working days.", "x")]
    assert city_kora._explained_by_meaning("refund", line)


def test_a_question_word_never_explains_another_question_word(city_kora):
    """Found on Q08 with the real model: 'money' ~ 'business' (0.55), but
    'business' came from the question itself ('business trip')."""
    from src.retrieval.answer import AnswerLine
    line = [AnswerLine("quote", "Approved expenses are reimbursed within ten working days.", "x")]
    assert city_kora._meaning_partner("refund", line)["explains"]
    blocked = city_kora._meaning_partner("refund", line, question_words={"reimbursed"})
    assert not blocked["explains"] and blocked["best"] != "reimbursed"


# ---------------- the baseline stays words-only (M5b fix 1) ----------------

def test_baseline_search_ignores_meaning(semantic_kora):
    chunks = semantic_kora.retriever.chunks
    q = "How quickly is my cash refunded?"
    words_only = chunks.search(q, top_k=5, use_semantic=False)
    assert all(r.matched_by != "semantic" and r.semantic == 0.0 for r in words_only)
    saved, chunks.semantic = chunks.semantic, None
    try:
        plain = chunks.search(q, top_k=5)
    finally:
        chunks.semantic = saved
    assert [r.chunk.chunk_id for r in words_only] == [r.chunk.chunk_id for r in plain]


# ---------------- the model loads once, offline first (M5b fix 4) ----------------

def test_model_is_loaded_once_per_process(monkeypatch):
    import src.retrieval.embeddings as emb
    calls = []

    class Fake:
        def __init__(self, name, local_only=False):
            calls.append(local_only)
            self.name = name
    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", Fake)
    monkeypatch.setattr(emb, "_LOADED", {})
    a, _ = emb.load_embedder("m")
    b, _ = emb.load_embedder("m")
    assert a is b and calls == [True]            # local copy first, and only once


def test_model_is_downloaded_only_when_not_on_the_machine(monkeypatch):
    import src.retrieval.embeddings as emb
    calls = []

    class Fake:
        def __init__(self, name, local_only=False):
            calls.append(local_only)
            if local_only:
                raise OSError("not cached")
            self.name = name
    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", Fake)
    monkeypatch.setattr(emb, "_LOADED", {})
    e, warning = emb.load_embedder("m")
    assert e is not None and warning is None and calls == [True, False]


# ---------------- the real model (UT's machine only) ----------------

def _real():
    emb, warning = load_embedder(load_config().embeddings_model)
    if emb is None:
        pytest.skip(warning)
    return emb


@pytest.mark.real_model
def test_real_model_understands_synonyms():
    emb = _real()
    v = emb.encode(["sign-off", "approval", "laptop"])
    assert v.shape == (3, 384)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


@pytest.mark.real_model
def test_real_model_is_used_when_enabled(intel):
    _real()
    kora = KoraAnswerer.from_intelligence(intel, _config())
    assert kora.retriever.semantic_status.startswith("on: all-MiniLM-L6-v2")

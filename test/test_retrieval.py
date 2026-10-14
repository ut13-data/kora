"""Phase 4 M1: keyword index, TF-IDF index, and the combined retriever."""

import pytest

from src.chunking.chunker import Chunk
from src.retrieval.keyword import build_keyword_index, extract_id_tokens
from src.retrieval.tfidf import build_tfidf_index
from src.retrieval.search import ChunkRetriever, build_retriever


def _chunk(chunk_id: str, document_id: str, text: str, index: int = 0) -> Chunk:
    """Minimal Chunk for tests that don't need a real ingested document."""
    return Chunk(chunk_id=chunk_id, document_id=document_id, chunk_text=text,
                 chunk_index=index, document_title=document_id, department="Test",
                 document_type="Test", version="1.0", status="Active",
                 section="", parent_id=f"{chunk_id}_p")


# ---------------- keyword index ----------------

def test_extract_id_tokens_finds_known_shapes():
    text = "See PROC-HR-001 and the gap KG-01, also check EMP-0085 and DEPT-CS."
    assert extract_id_tokens(text) == {"PROC-HR-001", "KG-01", "EMP-0085", "DEPT-CS"}


def test_extract_id_tokens_ignores_lowercase_hyphenated_words():
    assert extract_id_tokens("the go-live sign-off happens here") == set()


def test_keyword_index_finds_real_chunk(intel):
    index = build_keyword_index(intel.phase2.chunks)
    hits = index.id_index.get("PROC-HR-001", set())
    assert hits, "expected at least one chunk mentioning PROC-HR-001"
    for chunk_id in hits:
        assert "PROC-HR-001" in index.chunks[chunk_id].chunk_text


def test_keyword_index_unknown_id_returns_empty(intel):
    index = build_keyword_index(intel.phase2.chunks)
    assert index.search_ids({"PROC-DOES-NOT-EXIST"}) == {"PROC-DOES-NOT-EXIST": set()}


# ---------------- TF-IDF index ----------------

def test_tfidf_index_ranks_relevant_chunk_higher(intel):
    index = build_tfidf_index(intel.phase2.chunks)
    results = index.search("security training deadline for new hires", top_k=5)
    assert results, "expected at least one match above min_score"
    # The current security-training rule lives in PROC-HR-001 -- it should
    # show up near the top for a question that's clearly about it.
    top_document_ids = {r.chunk.document_id for r in results}
    assert "PROC-HR-001" in top_document_ids


def test_tfidf_index_empty_for_unrelated_query(intel):
    index = build_tfidf_index(intel.phase2.chunks)
    # Every token here must be out-of-vocabulary -- a query built from real
    # English words (even stray ones like "real" or "word") can score a
    # nonzero match purely by accident, since TF-IDF has no idea the query
    # as a WHOLE is nonsense; it only ever scores shared vocabulary.
    results = index.search("zzqqxx vvbbnn qqxxzz wwyyuu", top_k=5)
    assert results == []


# ---------------- combined retriever ----------------

def test_retriever_named_document_prefers_its_own_chunks(intel):
    """The bug this tier exists to fix: asking about a specific document
    used to return every chunk that merely MENTIONED its ID (glossary,
    FAQ, org chart), possibly none of them actually from that document.
    The top results now have to come FROM the named document, and from
    its ACTIVE version over an archived one with the same document_id."""
    retriever = build_retriever(intel.phase2)
    results = retriever.search("what does PROC-IMPL-001 say about kickoff", top_k=5)
    assert results, "expected at least one match"
    assert results[0].matched_by == "from_document"
    assert results[0].chunk.document_id == "PROC-IMPL-001"
    assert results[0].chunk.version == "2.0"
    assert results[0].chunk.status == "Active"
    assert "kickoff" in results[0].chunk.chunk_text.lower()


def test_retriever_from_document_falls_back_to_chunk_order_with_no_other_words(intel):
    """A bare ID with no other content words can't be ranked by TF-IDF
    (nothing left to compare after the ID itself is stripped out) -- it
    should still return the start of the document rather than nothing."""
    retriever = build_retriever(intel.phase2)
    results = retriever.search("PROC-IMPL-001", top_k=3)
    from_doc = [r for r in results if r.matched_by == "from_document"]
    assert from_doc
    assert from_doc[0].chunk.document_id == "PROC-IMPL-001"
    assert from_doc[0].chunk.chunk_index == 0


def test_retriever_mentions_tier_never_fires_for_the_document_itself(intel):
    """KG-01 is a label used INSIDE PROC-IMPL-001's text, not itself a
    document_id, so there's no "from_document" tier for it. On the real
    Nexora data, every chunk that literally contains "KG-01" also scores
    high enough on ordinary TF-IDF to be found by tier 2 anyway (the ID
    becomes ordinary vocabulary once it's in the query) -- so this checks
    tier 3 isn't needed here, not that it fires. See the next test for
    tier 3's actual trigger condition, checked in isolation."""
    retriever = build_retriever(intel.phase2)
    results = retriever.search("What does KG-01 say?", top_k=5)
    assert not any(r.matched_by == "from_document" for r in results)
    assert any("KG-01" in r.chunk.chunk_text for r in results)


class _NothingFoundTfidf:
    """Stub standing in for TfidfChunkIndex: always finds nothing. Isolates
    the retriever's TIER-ASSIGNMENT logic (does "mentions" fill the gap
    when TF-IDF comes up empty?) from real TF-IDF scoring, which -- per
    the test above -- rarely leaves that gap open on real documents."""
    def search_in(self, query, chunk_ids, top_k=5):
        return []

    def search(self, query, top_k=5):
        return []


def test_retriever_mentions_tier_fills_in_when_tfidf_finds_nothing():
    chunks = [_chunk("DOC-B_v1.0_000", "DOC-B", "some text mentioning DOC-A in passing")]
    retriever = ChunkRetriever(keyword=build_keyword_index(chunks), tfidf=_NothingFoundTfidf())
    results = retriever.search("anything about DOC-A", top_k=5)
    assert len(results) == 1
    assert results[0].matched_by == "mentions"
    assert results[0].score == 1.0
    assert results[0].chunk.chunk_id == "DOC-B_v1.0_000"


def test_retriever_free_text_query_uses_tfidf(intel):
    retriever = build_retriever(intel.phase2)
    results = retriever.search("what should a new hire in Germany do about internet", top_k=5)
    assert results
    assert all(r.matched_by == "tfidf" for r in results)
    document_ids = {r.chunk.document_id for r in results}
    # Germany settling-in material covers internet setup explicitly.
    assert document_ids & {"DOC-CARE-002", "DOC-CARE-003"}


def test_retriever_never_exceeds_top_k_when_no_id_tokens(intel):
    retriever = build_retriever(intel.phase2)
    results = retriever.search("customer onboarding process steps", top_k=3)
    assert len(results) <= 3

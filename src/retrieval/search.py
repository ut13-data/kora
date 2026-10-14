"""
Phase 4 - combined chunk retrieval (M1, extended in M2)

Five tiers, in priority order:

  0. "conflict" (M2 fix 1) -- chunk IDs the router pinned because the
     question matches a conflict Phase 3 already detected: both sides go
     first, archived side included.
  1. "from_document" -- the question names a document ID (PROC-IMPL-001,
     ...) and the chunk IS PART of that document (Chunk.document_id ==
     the ID). Ranked by how well the REST of the question matches that
     document's own chunks -- so "what does PROC-IMPL-001 say about
     kickoff" surfaces the kickoff step, not just the first chunk.
  2. "routed" (M2) -- the router picked candidate documents for this
     question (e.g. "customer onboarding" -> PROC-IMPL-001). Only chunks
     from those documents, ranked by TF-IDF, with a relevance floor:
     being a candidate means "likely place to look", not "relevant".
     Tiers 0-2 together stop at top_k - 2 (3 of 5), so tier 3 always
     gets the last 2 slots.
  3. "tfidf" -- ordinary free-text relevance across every chunk, filling
     whatever slots are left. This is the safety net when the router
     guessed wrong or recognised nothing.
  4. "mentions" -- lowest priority: chunks that merely CONTAIN the ID
     string somewhere in their text, from a DIFFERENT document.

A chunk never appears twice -- it's kept at the HIGHEST tier it first
qualifies for.

include_archived=False (M2) removes chunks whose status is "Archived"
from every tier. The router decides this per question; the retriever
just obeys. Default True keeps M1 behaviour unchanged.
"""

from dataclasses import dataclass

from src.chunking.chunker import Chunk
from src.intelligence.loader import Phase2Data
from src.retrieval.embeddings import SemanticIndex, rrf
from src.retrieval.keyword import KeywordIndex, build_keyword_index, extract_id_tokens
from src.retrieval.tfidf import TfidfChunkIndex, build_tfidf_index

ROUTED_MIN_SCORE = 0.05     # same floor TfidfChunkIndex.search() uses
GLOBAL_RESERVED_SLOTS = 2   # agreed with UT: router fills at most top_k - 2 (3 of 5)


def _without_id_tokens(query: str, tokens: list[str]) -> str:
    """The question with its ID tokens removed. Used only when ranking a
    named document's OWN chunks: the ID is already the selection
    criterion (how we found these candidates), so leaving it in the query
    lets a chunk that just RESTATES its own ID in a cross-reference (e.g.
    "...superseded by PROC-IMPL-001 v2.0...") outscore a chunk that's
    actually about the question's real subject. An empty result (a bare
    ID with no other words) is fine -- search_in() falls back to natural
    chunk order when nothing scores above 0."""
    rest = query
    for token in tokens:
        rest = rest.replace(token, " ")
    return rest


def is_archived(chunk: Chunk) -> bool:
    return chunk.status.strip().lower() == "archived"


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float             # word-based (TF-IDF) score; 0.0 for a meaning-only hit
    matched_by: str          # "conflict", "from_document", "routed", "tfidf", "semantic" or "mentions"
    semantic: float = 0.0    # meaning-based cosine similarity (0.0 when embeddings are off)


@dataclass
class ChunkRetriever:
    keyword: KeywordIndex
    tfidf: TfidfChunkIndex
    semantic: SemanticIndex | None = None
    semantic_top_n: int = 20
    semantic_min_score: float = 0.30

    def search(self, query: str, top_k: int = 5,
               document_ids: list[str] | None = None,
               include_archived: bool = True,
               pinned_chunk_ids: list[str] | None = None,
               direct_documents: list[str] | None = None,
               routed_min_score: float = ROUTED_MIN_SCORE,
               reserved_slots: int = GLOBAL_RESERVED_SLOTS,
               use_semantic: bool = True,
               allowed_documents: set[str] | None = None) -> list[RetrievalResult]:
        """USE_SEMANTIC=False searches by words only, even when embeddings are
        loaded (the evaluation baseline must stay plain TF-IDF).
        DIRECT_DOCUMENTS: documents the router chose on purpose (named by ID,
        or a country's tone profile); searched like a named document. The
        conflict sides (pinned) are never cut; every other early tier stops
        at top_k - reserved_slots, so the whole-corpus search always gets a say."""
        results: list[RetrievalResult] = []
        seen: set[str] = set()
        id_tokens = sorted(extract_id_tokens(query))
        fill_to = max(top_k - reserved_slots, 1)
        semantic = self.semantic if use_semantic else None

        def allowed(chunk: Chunk) -> bool:
            # ALLOWED_DOCUMENTS (access control, M7): None = no filtering.
            if allowed_documents is not None and chunk.document_id not in allowed_documents:
                return False
            return include_archived or not is_archived(chunk)

        def add(chunk: Chunk, score: float, tier: str) -> bool:
            """Add one result if new and allowed. Returns True if added."""
            if chunk.chunk_id in seen or not allowed(chunk) or len(results) >= top_k:
                return False
            seen.add(chunk.chunk_id)
            results.append(RetrievalResult(chunk, score, tier))
            return True

        # Tier 0: chunks the router pinned (both sides of a detected conflict).
        # Added even if archived: the router only pins them on purpose.
        for cid in pinned_chunk_ids or []:
            chunk = self.keyword.chunks.get(cid)
            if allowed_documents is not None and chunk and chunk.document_id not in allowed_documents:
                continue
            if chunk and cid not in seen and len(results) < top_k:
                seen.add(cid)
                results.append(RetrievalResult(chunk, 1.0, "conflict"))

        # Tier 1: chunks that ARE the named / directly chosen document(s).
        rest_query = _without_id_tokens(query, id_tokens)
        for doc_id in dict.fromkeys([*id_tokens, *(direct_documents or [])]):
            own = [c.chunk_id for c in self.keyword.chunks.values()
                   if c.document_id == doc_id and allowed(c)]
            for scored in self.tfidf.search_in(rest_query, own, top_k=top_k):
                if len(results) >= fill_to:
                    break
                add(scored.chunk, scored.score, "from_document")

        # Tier 2: the router's candidate documents -- capped, so the last
        # GLOBAL_RESERVED_SLOTS always go to a search of the whole corpus.
        # A wrong guess by the router can then never block the right answer.
        # With embeddings, a candidate chunk can also qualify by meaning;
        # the two rankings are fused by position (RRF), never by raw score.
        if document_ids:
            wanted = set(document_ids)
            candidates = [c.chunk_id for c in self.keyword.chunks.values()
                          if c.document_id in wanted and allowed(c)]
            by_words = {s.chunk.chunk_id: s.score for s in
                        self.tfidf.search_in(query, candidates, top_k=len(candidates), min_score=routed_min_score)}
            by_meaning = dict(semantic.search(query, len(candidates), self.semantic_min_score, candidates)) \
                if semantic else {}
            for cid in rrf([list(by_words), list(by_meaning)]):
                if len(results) >= fill_to:
                    break
                add(self.keyword.chunks[cid], by_words.get(cid, 0.0), "routed")

        # Tier 3: the whole corpus. Words and (if available) meaning are two
        # rankings fused by RRF; a chunk found only by meaning is marked
        # "semantic". Extra results are requested because some are duplicates.
        if len(results) < top_k:
            by_words = {s.chunk.chunk_id: s.score for s in self.tfidf.search(query, top_k=top_k + len(seen) + 20)}
            by_meaning = dict(semantic.search(query, self.semantic_top_n, self.semantic_min_score)) \
                if semantic else {}
            for cid in rrf([list(by_words), list(by_meaning)]):
                add(self.keyword.chunks[cid], by_words.get(cid, 0.0), "tfidf" if cid in by_words else "semantic")
                if len(results) >= top_k:
                    break

        # Tier 4: other documents that merely mention the ID.
        for token in id_tokens:
            for cid in sorted(self.keyword.id_index.get(token, ())):
                add(self.keyword.chunks[cid], 1.0, "mentions")

        if semantic and results:
            meaning = semantic.chunk_scores(query, [r.chunk.chunk_id for r in results])
            for r in results:
                r.semantic = meaning.get(r.chunk.chunk_id, 0.0)
        return results


def build_retriever(phase2: Phase2Data) -> ChunkRetriever:
    """Build both indexes from a loaded Phase2Data (same object
    src/explain.py and src/employee.py already get from run_intelligence()
    via result.phase2)."""
    return ChunkRetriever(keyword=build_keyword_index(phase2.chunks),
                           tfidf=build_tfidf_index(phase2.chunks))

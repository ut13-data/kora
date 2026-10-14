"""
Phase 4 M1 - TF-IDF chunk index

Turns every chunk's text into a TF-IDF vector (scikit-learn is already a
Phase 3 dependency) and scores a question against every chunk by cosine
similarity. No embeddings model, no download, no network call -- this is
the "lightweight local option" the Phase 4 brief asks to start with.

TF-IDF is bad at exact ID matches (see keyword.py) and bad at pure
synonyms ("manager" and "lead" are unrelated vectors to it) -- it is good
at "which chunks share the most distinctive vocabulary with this
question", which covers most free-text questions well enough to start.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking.chunker import Chunk

DEFAULT_INDEX_PATH = Path("data/retrieval/tfidf_index.joblib")


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class TfidfChunkIndex:
    vectorizer: TfidfVectorizer
    matrix: object                  # scipy.sparse matrix, one row per chunk
    chunk_ids: list[str]            # same order as matrix rows
    chunks: dict[str, Chunk]

    def __post_init__(self) -> None:
        self._position = {cid: i for i, cid in enumerate(self.chunk_ids)}

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> list[ScoredChunk]:
        """Top-K chunks by cosine similarity to QUERY, across the WHOLE
        corpus. Chunks scoring below MIN_SCORE are dropped rather than
        padding results with noise -- an empty result here is a real
        signal ("nothing matched"), not a bug."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [ScoredChunk(self.chunks[cid], float(score))
                for cid, score in ranked[:top_k] if score >= min_score]

    def search_in(self, query: str, chunk_ids: list[str], top_k: int = 5,
                  min_score: float = 0.0) -> list[ScoredChunk]:
        """Rank a SUBSET of chunks (membership already decided by the
        caller) against QUERY.

        min_score = 0.0 (default, used for "the question NAMES this
        document"): being in the subset IS the relevance signal, TF-IDF
        only decides the ORDER. If every candidate scores 0 (the query had
        no content words beyond an ID token), falls back to the subset's
        natural chunk_index order, so a bare "what does PROC-IMPL-001 say"
        still returns the start of the document instead of nothing.

        min_score > 0 (used by the router's candidate documents): the
        subset is only a likely place to look, so chunks that share no
        vocabulary with the question are dropped, same as search()."""
        positions = [self._position[cid] for cid in chunk_ids if cid in self._position]
        if not positions:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix[positions])[0]
        candidate_ids = [self.chunk_ids[p] for p in positions]
        ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)
        if min_score > 0:
            ranked = [(cid, score) for cid, score in ranked if score >= min_score]
        elif all(score == 0.0 for _, score in ranked):
            ranked = sorted(zip(candidate_ids, scores),
                             key=lambda pair: self.chunks[pair[0]].chunk_index)
        return [ScoredChunk(self.chunks[cid], float(score)) for cid, score in ranked[:top_k]]

    def save(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        """Only the vectorizer + matrix + chunk_ids are saved (Chunk objects
        aren't picklable-safe across code changes) -- load_tfidf_index()
        reattaches them to whatever Chunk objects you pass it at load time."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix,
                     "chunk_ids": self.chunk_ids}, path)


def build_tfidf_index(chunks: list[Chunk]) -> TfidfChunkIndex:
    """Fit a fresh TF-IDF index over CHUNKS. Call this once per Phase 2 run
    (chunks.jsonl doesn't change between questions) -- never once per query."""
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.chunk_text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    return TfidfChunkIndex(vectorizer=vectorizer, matrix=matrix, chunk_ids=chunk_ids,
                            chunks={c.chunk_id: c for c in chunks})


def load_tfidf_index(chunks: list[Chunk], path: Path = DEFAULT_INDEX_PATH) -> TfidfChunkIndex:
    """Load a saved index and reattach it to CHUNKS (matched by chunk_id)."""
    saved = joblib.load(path)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    return TfidfChunkIndex(vectorizer=saved["vectorizer"], matrix=saved["matrix"],
                            chunk_ids=saved["chunk_ids"], chunks=chunks_by_id)

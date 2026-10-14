"""
Phase 4 M5b - semantic embeddings: find text by MEANING, not only by words.

TF-IDF can't connect "money back" with "reimbursed" or "pushed live" with
"deploys". An embedding model turns every text into a vector (384 numbers for
all-MiniLM-L6-v2) so that texts with similar meaning point in similar
directions; cosine similarity then measures "same meaning".

How it is used (and how it is NOT):
  * It ADDS to what was validated, it doesn't replace it. Word-based scores
    and every existing threshold keep their meaning; embeddings get their own
    thresholds (config: semantic_*), tuned on the real model.
  * Search: the whole-corpus safety net fuses two RANKINGS (words, meaning) by
    reciprocal rank fusion, so no raw scores of different scales are mixed.
  * Quotes: a sentence may be quoted when it matches by words (as before) OR
    clearly by meaning (semantic_quote_min).
  * Coverage: a question word the knowledge base has never seen ("money")
    counts as a paraphrase ONLY when the meaning match is strong
    (semantic_strong_evidence). Words that exist elsewhere but not in the
    quoted source ("contractors") are still flagged; off-topic questions
    ("pets") still abstain.
  * Fail-safe: no library, no model, no network on first download -> a
    warning, and KORA runs exactly as without embeddings.
  * Offline first, loaded once per process (see load_embedder).

Vectors are cached in data/retrieval/embeddings_cache.npz keyed by a hash of
model name + text, so a rebuild only encodes chunks/sentences that changed.
"""

import hashlib
import re
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CACHE = Path("data/retrieval/embeddings_cache.npz")


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-length float32 vectors, one row per text."""
        ...


class SentenceTransformerEmbedder:
    """The real model (runs locally on CPU). LOCAL_ONLY=True uses the copy
    already saved on this machine and never touches the network."""

    def __init__(self, model_name: str = DEFAULT_MODEL, local_only: bool = False):
        from sentence_transformers import SentenceTransformer   # optional dependency
        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu", local_files_only=local_only)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self._model.encode(list(texts), batch_size=64, show_progress_bar=False,
                                     normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Deterministic STAND-IN for tests (no model, no network): words are
    hashed into a fixed-size vector, with an optional synonym map so a test
    can simulate "same meaning, different words". It knows nothing about
    language -- never use it for real answers. (4096 slots: with 256, unrelated
    words collided and looked "similar", which made tests misleading.)"""

    def __init__(self, dim: int = 4096, synonyms: dict[str, str] | None = None):
        self.name = f"hashing-test-{dim}"
        self.dim = dim
        self.synonyms = synonyms or {}

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                word = self.synonyms.get(word, word)
                if len(word) < 3:
                    continue
                h = int(hashlib.md5(word[:6].encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


# One loaded model per process: loading takes seconds, so a rebuild or a
# second KORA instance reuses it instead of loading it again.
_LOADED: dict[str, "SentenceTransformerEmbedder"] = {}


def load_embedder(model_name: str) -> tuple[Embedder | None, str | None]:
    """(embedder, None) or (None, warning). Never raises: no embeddings means
    KORA works as it did before M5b.

    Offline first: the copy already saved on this machine is used without any
    network call (faster start, works offline, no Hugging Face warning). Only
    when there is no local copy is the model downloaded, once."""
    if model_name in _LOADED:
        return _LOADED[model_name], None
    try:
        try:
            embedder = SentenceTransformerEmbedder(model_name, local_only=True)
        except ImportError:
            raise
        except Exception:                                        # noqa: BLE001 -- not on this machine yet
            embedder = SentenceTransformerEmbedder(model_name)
        _LOADED[model_name] = embedder
        return embedder, None
    except ImportError:
        return None, ("embeddings off: the 'sentence-transformers' package is not installed "
                      "(pip install sentence-transformers); search uses words only")
    except Exception as e:                                       # noqa: BLE001
        return None, f"embeddings off: model '{model_name}' could not be loaded ({type(e).__name__}: {e})"


def _key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()


class VectorCache:
    """Text -> vector, persisted. Only texts not seen before are encoded."""

    def __init__(self, embedder: Embedder, path: Path | None):
        self.embedder = embedder
        self.path = Path(path) if path else None
        self._vectors: dict[str, np.ndarray] = {}
        self.encoded_last = 0
        if self.path and self.path.exists():
            try:
                data = np.load(self.path, allow_pickle=False)
                if str(data["model"]) == embedder.name:
                    self._vectors = dict(zip(data["keys"].tolist(), data["vectors"]))
            except Exception:                                   # noqa: BLE001 -- a bad cache is just rebuilt
                self._vectors = {}

    def vectors(self, texts: list[str]) -> np.ndarray:
        keys = [_key(self.embedder.name, t) for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self._vectors))
        self.encoded_last = len(missing)
        if missing:
            for t, v in zip(missing, self.embedder.encode(missing)):
                self._vectors[_key(self.embedder.name, t)] = v
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack([self._vectors[k] for k in keys]).astype(np.float32)

    def save(self, keep_texts: list[str]) -> None:
        """Persist only the vectors still in use (deleted documents drop out)."""
        if not self.path:
            return
        keep = [_key(self.embedder.name, t) for t in dict.fromkeys(keep_texts)]
        keep = [k for k in keep if k in self._vectors]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, model=np.array(self.embedder.name), keys=np.array(keep),
                 vectors=np.vstack([self._vectors[k] for k in keep]) if keep else np.zeros((0, 1)))
        tmp.replace(self.path)


def chunk_text_for_embedding(chunk) -> str:
    """Title and section give a short chunk its context."""
    return f"{chunk.document_title} > {chunk.section}\n{chunk.chunk_text}"


class SemanticIndex:
    """Vectors for every chunk (search) and every quotable sentence (answers)."""

    def __init__(self, embedder: Embedder, chunks: list, cache_path: Path | None = DEFAULT_CACHE):
        self.embedder = embedder
        self.cache = VectorCache(embedder, cache_path)
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._position = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self._chunk_texts = [chunk_text_for_embedding(c) for c in chunks]
        self.matrix = self.cache.vectors(self._chunk_texts)
        self.encoded_chunks = self.cache.encoded_last
        self._sentences: dict[str, np.ndarray] = {}
        self._query_cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return self.embedder.name

    def add_sentences(self, sentences: list[str]) -> int:
        """Pre-compute vectors for answer sentences (cached across runs)."""
        unique = list(dict.fromkeys(s for s in sentences if s not in self._sentences))
        if unique:
            for s, v in zip(unique, self.cache.vectors(unique)):
                self._sentences[s] = v
        return self.cache.encoded_last if unique else 0

    def save(self) -> None:
        self.cache.save(self._chunk_texts + list(self._sentences))

    def query(self, text: str) -> np.ndarray:
        if text not in self._query_cache:
            if len(self._query_cache) > 256:
                self._query_cache.clear()
            self._query_cache[text] = self.embedder.encode([text])[0]
        return self._query_cache[text]

    def chunk_scores(self, query: str, chunk_ids: list[str] | None = None) -> dict[str, float]:
        """Cosine similarity of the query with each chunk (all, or a subset)."""
        q = self.query(query)
        ids = chunk_ids if chunk_ids is not None else self.chunk_ids
        rows = [self._position[c] for c in ids if c in self._position]
        if not rows:
            return {}
        scores = self.matrix[rows] @ q
        return {self.chunk_ids[r]: float(s) for r, s in zip(rows, scores)}

    def search(self, query: str, top_n: int, min_score: float,
               chunk_ids: list[str] | None = None) -> list[tuple[str, float]]:
        scores = self.chunk_scores(query, chunk_ids)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [(cid, s) for cid, s in ranked[:top_n] if s >= min_score]

    def sentence_scores(self, query: str, sentences: list[str]) -> np.ndarray:
        missing = [s for s in sentences if s not in self._sentences]
        if missing:
            self.add_sentences(missing)
        if not sentences:
            return np.zeros(0, dtype=np.float32)
        q = self.query(query)
        return np.vstack([self._sentences[s] for s in sentences]) @ q


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion: combine rankings by POSITION (1/(k+rank)), so
    scores of different scales are never mixed. Ties keep first-seen order."""
    score: dict[str, float] = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
            order.setdefault(item, len(order))
    return sorted(score, key=lambda i: (-score[i], order[i]))

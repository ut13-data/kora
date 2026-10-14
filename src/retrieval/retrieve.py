"""
Phase 4 M2 - one call from question to evidence.

    question -> Router.route() -> RoutePlan
             -> ChunkRetriever.search(candidate documents, archived yes/no)
             -> evidence chunks + history notes + timing

    from src.retrieval.retrieve import KoraRetriever
    kora = KoraRetriever.load()                      # from saved Phase 2 + 3 files
    outcome = kora.retrieve("who approves a customer implementation plan?")
    outcome.plan.reasons     # why it looked where it looked
    outcome.results          # ranked chunks, each with its tier
    outcome.history_notes    # "an older, archived version exists" (decision C)

History notes come from Phase 3's outdated_knowledge.json: for every
archived document, which active document replaced it. When the router
excluded archived documents, every active document in the results that
has an archived predecessor gets a note, so history is never hidden.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.intelligence.loader import Phase2Data, load_phase2
from src.retrieval.config import KoraConfig, load_config
from src.retrieval.router import RoutePlan, Router
from src.retrieval.search import ChunkRetriever, RetrievalResult, build_retriever


@dataclass
class HistoryNote:
    document_id: str            # the active document found in the results
    archived_document_id: str
    archived_version: str
    archived_effective_date: str
    note: str


@dataclass
class RetrievalOutcome:
    question: str
    plan: RoutePlan
    results: list[RetrievalResult]
    history_notes: list[HistoryNote] = field(default_factory=list)
    elapsed_ms: float = 0.0
    route_ms: float = 0.0       # understanding the question (router)
    search_ms: float = 0.0      # finding chunks (words + meaning)

    def to_log(self) -> dict:
        """Flat, JSON-ready record of one retrieval: what Phase 5 can chart
        (latency, tiers used, documents retrieved, archived handling)."""
        return {
            "question": self.question,
            "intents": self.plan.intents,
            "entities": [f"{e.entity_type}:{e.entity_id} ({e.strength})" for e in self.plan.entities],
            "candidate_documents": self.plan.candidate_documents,
            "include_archived": self.plan.include_archived,
            "conflicts": self.plan.conflicts,
            "reasons": self.plan.reasons,
            "results": [{"chunk_id": r.chunk.chunk_id, "document_id": r.chunk.document_id,
                         "status": r.chunk.status, "tier": r.matched_by,
                         "score": round(r.score, 4), "semantic": round(r.semantic, 4)} for r in self.results],
            "history_notes": [asdict(n) for n in self.history_notes],
            "elapsed_ms": round(self.elapsed_ms, 2),
            "route_ms": round(self.route_ms, 2),
            "search_ms": round(self.search_ms, 2),
        }


def _predecessors(outdated: dict) -> dict[str, list[dict]]:
    """active document_id -> archived documents it replaced."""
    out: dict[str, list[dict]] = {}
    for old in outdated.get("superseded_documents", []):
        for new in old.get("superseded_by", []):
            out.setdefault(new["document_id"], []).append(old)
    return out


class KoraRetriever:
    def __init__(self, router: Router, chunks: ChunkRetriever, outdated: dict,
                 documents: list | None = None, tables: dict | None = None):
        self.router = router
        self.chunks = chunks
        self._predecessors = _predecessors(outdated)
        # (document_id, version) -> Phase 2 DocumentMetadata, for citations
        # (source file, effective date). Chunks don't carry those fields.
        self.documents = {(d.document_id, d.version): d for d in (documents or [])}
        # Who can ask (M7): employee ID -> User with groups, from Phase 2 tables.
        from src.retrieval.access import users_from_tables
        self.users = users_from_tables(tables or {})
        self.tables = tables or {}                 # Phase 5: locations for the answer layer
        self.semantic_status = "off (not enabled in config)"
        # Start-up time per step, in seconds (shown with every answer, so a
        # slow start can be traced to its cause).
        self.load_timings: dict[str, float] = {}
        self._attach_semantic()

    def _timed(self, step: str, since: float) -> float:
        now = time.perf_counter()
        self.load_timings[step] = round(self.load_timings.get(step, 0.0) + now - since, 3)
        return now

    def _attach_semantic(self, embedder=None) -> None:
        """Load the embedding model if the config enables it. Any failure is a
        warning, never an error: KORA then searches by words, exactly as
        before M5b. EMBEDDER can be passed in (tests use a stand-in)."""
        cfg = self.config
        t = cfg.t
        self.chunks.semantic_top_n = t("semantic_top_n")
        self.chunks.semantic_min_score = t("semantic_min_score")
        if embedder is None:
            if not cfg.embeddings_enabled:
                return
            t0 = time.perf_counter()
            try:                 # timed on its own: on Windows this import is most of the start-up
                import sentence_transformers  # noqa: F401
            except Exception:                                    # noqa: BLE001 -- load_embedder reports it
                pass
            t0 = self._timed("embedding_libraries", t0)
            from src.retrieval.embeddings import load_embedder
            embedder, warning = load_embedder(cfg.embeddings_model)
            self._timed("embedding_model", t0)
            if embedder is None:
                cfg.warnings.append(warning)
                self.semantic_status = warning
                return
        from src.retrieval.embeddings import SemanticIndex
        cache = Path(cfg.embeddings_cache) if cfg.embeddings_cache else None
        t0 = time.perf_counter()
        self.chunks.semantic = SemanticIndex(embedder, list(self.chunks.keyword.chunks.values()), cache)
        self._timed("chunk_vectors", t0)
        self.semantic_status = (f"on: {embedder.name} ({len(self.chunks.semantic.chunk_ids)} chunks, "
                                f"{self.chunks.semantic.encoded_chunks} newly encoded)")

    def use_embedder(self, embedder) -> None:
        """Switch to a given embedder (tests: a deterministic stand-in)."""
        self._attach_semantic(embedder)

    # ---- construction ----

    @property
    def config(self) -> KoraConfig:
        return self.router.config

    @classmethod
    def from_intelligence(cls, result, config: KoraConfig | None = None) -> "KoraRetriever":
        """From an IntelligenceResult (Phase 3 run in memory, e.g. in tests)."""
        cfg = config if config is not None else load_config()
        t0 = time.perf_counter()
        router = Router.from_outputs(result.outputs, cfg)
        t1 = time.perf_counter()
        words = build_retriever(result.phase2)
        t2 = time.perf_counter()
        obj = cls(router, words, result.outputs["outdated_knowledge"], result.phase2.documents,
                  result.phase2.tables)
        obj.load_timings = {"router": round(t1 - t0, 3), "word_indexes": round(t2 - t1, 3), **obj.load_timings}
        return obj

    @classmethod
    def load(cls, intelligence_dir: Path = Path("data/intelligence"),
             processed_dir: Path = Path("data/processed"),
             chunks_dir: Path = Path("data/chunks"),
             raw_dir: Path = Path("data/raw"),
             config: KoraConfig | None = None) -> "KoraRetriever":
        """From saved files: Phase 2 output + Phase 3 output. Phase 3 is not
        re-run, so this starts in about a second."""
        t0 = time.perf_counter()
        phase2: Phase2Data = load_phase2(processed_dir, chunks_dir, raw_dir)
        path = intelligence_dir / "outdated_knowledge.json"
        outdated = json.load(open(path, encoding="utf-8")) if path.exists() else {}
        cfg = config if config is not None else load_config()
        t1 = time.perf_counter()
        router = Router.from_files(intelligence_dir, cfg)
        t2 = time.perf_counter()
        words = build_retriever(phase2)
        t3 = time.perf_counter()
        obj = cls(router, words, outdated, phase2.documents, phase2.tables)
        obj.load_timings = {"read_files": round(t1 - t0, 3), "router": round(t2 - t1, 3),
                            "word_indexes": round(t3 - t2, 3), **obj.load_timings}
        return obj

    # ---- retrieval ----

    def retrieve(self, question: str, top_k: int | None = None,
                 allowed: set[str] | None = None) -> RetrievalOutcome:
        """ALLOWED: documents the asking user may read (access control, M7);
        None = no filtering. Hidden documents are removed from the plan too."""
        start = time.perf_counter()
        t = self.config.t
        plan = self.router.route(question)
        if allowed is not None:
            self._restrict_plan(plan, allowed)
        routed = time.perf_counter()
        results = self.chunks.search(question, top_k=top_k or t("top_k"),
                                     document_ids=plan.candidate_documents or None,
                                     include_archived=plan.include_archived,
                                     pinned_chunk_ids=plan.pinned_chunks,
                                     direct_documents=plan.direct_documents,
                                     routed_min_score=t("routed_min_score"),
                                     reserved_slots=t("global_reserved_slots"),
                                     allowed_documents=allowed)
        searched = time.perf_counter()
        notes = [] if plan.include_archived else self._history_notes(results)
        if allowed is not None:
            notes = [n for n in notes if n.archived_document_id in allowed]
        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalOutcome(question, plan, results, notes, elapsed,
                                (routed - start) * 1000, (searched - routed) * 1000)

    def _restrict_plan(self, plan: RoutePlan, allowed: set[str]) -> None:
        """Remove every trace of documents the user can't read from the plan."""
        hidden = {d for d, _ in self.documents} - allowed
        chunk_doc = {cid: c.document_id for cid, c in self.chunks.keyword.chunks.items()}
        plan.candidate_documents = [d for d in plan.candidate_documents if d in allowed]
        plan.direct_documents = [d for d in plan.direct_documents if d in allowed]
        plan.tone_profiles = [d for d in plan.tone_profiles if d in allowed]
        if any(chunk_doc.get(c) not in allowed for c in plan.pinned_chunks):
            plan.pinned_chunks, plan.conflicts = [], []      # half a conflict would hint at the hidden side
        plan.reasons = [r for r in plan.reasons if not any(h in r for h in hidden)]

    def _history_notes(self, results: list[RetrievalResult]) -> list[HistoryNote]:
        notes: list[HistoryNote] = []
        for doc_id in dict.fromkeys(r.chunk.document_id for r in results):   # ordered, unique
            for old in self._predecessors.get(doc_id, []):
                notes.append(HistoryNote(
                    document_id=doc_id,
                    archived_document_id=old["document_id"],
                    archived_version=old.get("version", "unknown"),
                    archived_effective_date=old.get("effective_date", "unknown"),
                    note=(f"An older, archived version exists: {old['document_id']} "
                          f"v{old.get('version', '?')} (effective {old.get('effective_date', '?')}). "
                          f"Ask about the previous version to see it."),
                ))
        return notes

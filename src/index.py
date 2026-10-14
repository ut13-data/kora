"""
Phase 4 - index maintenance: KORA is always answering from the current files.

    from src.index import Kora
    kora = Kora.open()                       # builds the index if missing or stale
    print(kora.ask("Who approves purchases above 10000 euros?").render())

    python -m src.index --status             # is the index current? what changed?
    python -m src.index --rebuild            # force a full rebuild

How "new files work immediately" is guaranteed:
  * A manifest (data/index_manifest.json) records every file in data/raw at
    build time: size, modified time and SHA-256 hash.
  * Before every question, Kora.ask() compares data/raw with the manifest:
    size+time first (fast); a file whose size/time changed is re-hashed, so
    merely touching a file doesn't trigger a rebuild. Added, changed or
    deleted files -> full rebuild (Phase 2 + Phase 3), reload, then answer.
  * Full rebuild, not incremental: ~6 s for 53 documents, about a minute for
    ~1,000. Simple and never inconsistent; incremental updates only pay off
    far beyond that size (noted as a later optimisation).
  * Atomic: everything is built in a temporary folder and swapped in only if
    the whole build succeeded. A failing build leaves the previous index
    untouched and raises IndexBuildError -- KORA never answers from a
    half-built index.
  * The company config (config/kora_config.json) is re-read on every rebuild
    and every open; it doesn't need a rebuild of its own.
"""

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1


class IndexBuildError(RuntimeError):
    """The rebuild failed; the previous index (if any) is still in place."""


class StaleIndexError(RuntimeError):
    """The index doesn't match data/raw and auto_rebuild is off."""


@dataclass
class Paths:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    chunks_dir: Path = Path("data/chunks")
    intelligence_dir: Path = Path("data/intelligence")
    context_dir: Path = Path("data/context")
    manifest: Path = Path("data/index_manifest.json")
    config: Path = Path("config/kora_config.json")
    log_dir: Path | None = Path("data/logs")


@dataclass
class IndexStatus:
    current: bool
    reason: str
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [self.reason]
        for name, items in (("added", self.added), ("changed", self.changed), ("removed", self.removed)):
            if items:
                parts.append(f"{name}: {', '.join(items[:10])}{' ...' if len(items) > 10 else ''}")
        return "; ".join(parts)


# ---------------- manifest ----------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _raw_files(raw_dir: Path) -> dict[str, Path]:
    """Every file under raw_dir, by its relative path ('/' separators)."""
    if not raw_dir.exists():
        return {}
    return {p.relative_to(raw_dir).as_posix(): p for p in sorted(raw_dir.rglob("*"))
            if p.is_file() and not p.name.startswith(".")}


def scan(raw_dir: Path) -> dict[str, dict]:
    return {rel: {"size": p.stat().st_size, "mtime": p.stat().st_mtime, "sha256": _sha256(p)}
            for rel, p in _raw_files(raw_dir).items()}


def check(paths: Paths) -> IndexStatus:
    """Is the index built from exactly the files now in data/raw?"""
    for needed in (paths.processed_dir / "documents.json", paths.chunks_dir / "chunks.jsonl",
                   paths.intelligence_dir / "entities.json"):
        if not needed.exists():
            return IndexStatus(False, f"index incomplete: {needed} missing")
    if not paths.manifest.exists():
        return IndexStatus(False, "no manifest: index was never built by src.index")
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return IndexStatus(False, f"manifest unreadable ({e})")
    if manifest.get("version") != MANIFEST_VERSION:
        return IndexStatus(False, "manifest from another KORA version")
    recorded: dict = manifest.get("files", {})
    current = _raw_files(paths.raw_dir)
    added = sorted(set(current) - set(recorded))
    removed = sorted(set(recorded) - set(current))
    changed = []
    for rel in sorted(set(current) & set(recorded)):
        st = current[rel].stat()
        old = recorded[rel]
        if st.st_size == old["size"] and st.st_mtime == old["mtime"]:
            continue                               # fast path: untouched
        if st.st_size != old["size"] or _sha256(current[rel]) != old["sha256"]:
            changed.append(rel)                    # touched-but-identical files are not changes
    if added or changed or removed:
        return IndexStatus(False, "data/raw changed since the last build", added, changed, removed)
    return IndexStatus(True, "index is current")


# ---------------- build ----------------

def rebuild(paths: Paths) -> dict:
    """Full Phase 2 + Phase 3 rebuild into a temporary folder, swapped in only
    on success. Returns a build report."""
    from src.build_intelligence import IntelligenceSettings, run_intelligence
    from src.ingest import IngestionSettings, run_ingestion

    if not paths.raw_dir.exists() or not _raw_files(paths.raw_dir):
        raise IndexBuildError(f"{paths.raw_dir} is empty or missing: nothing to index")
    before = scan(paths.raw_dir)
    tmp = paths.processed_dir.parent / ".index_build"
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.perf_counter()
    try:
        ingest_summary, _ = run_ingestion(IngestionSettings(
            raw_dir=paths.raw_dir, processed_dir=tmp / "processed", chunks_dir=tmp / "chunks"))
        t_phase2 = time.perf_counter() - t0
        paths.context_dir.mkdir(parents=True, exist_ok=True)
        intel_summary, _ = run_intelligence(IntelligenceSettings(
            processed_dir=tmp / "processed", chunks_dir=tmp / "chunks", raw_dir=paths.raw_dir,
            context_dir=paths.context_dir, output_dir=tmp / "intelligence"))
        t_phase3 = time.perf_counter() - t0 - t_phase2
    except Exception as e:                         # noqa: BLE001 -- any failure keeps the old index
        shutil.rmtree(tmp, ignore_errors=True)
        raise IndexBuildError(f"rebuild failed, previous index kept: {type(e).__name__}: {e}") from e

    if scan(paths.raw_dir) != before:
        shutil.rmtree(tmp, ignore_errors=True)
        raise IndexBuildError("data/raw changed during the rebuild; previous index kept -- try again")

    # Swap in: each target is replaced only after everything was built.
    for name, target in (("processed", paths.processed_dir), ("chunks", paths.chunks_dir),
                         ("intelligence", paths.intelligence_dir)):
        shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / name), str(target))
    shutil.rmtree(tmp, ignore_errors=True)

    report = {
        "version": MANIFEST_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "files": before,
        "documents": ingest_summary.documents,
        "chunks": ingest_summary.chunks,
        "phase2_errors": ingest_summary.errors,
        "phase3_validation": intel_summary.validation,
        "seconds": {"phase2": round(t_phase2, 2), "phase3": round(t_phase3, 2)},
    }
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------- the always-current KORA ----------------

class Kora:
    """KORA that checks its index before every question and rebuilds when
    data/raw changed. Every answer is appended to data/logs/answers.jsonl."""

    def __init__(self, paths: Paths, auto_rebuild: bool = True):
        self.paths = paths
        self.auto_rebuild = auto_rebuild
        self.answerer = None
        self.last_build: dict | None = None
        self.events: list[str] = []
        self._ensure_current()

    @classmethod
    def open(cls, paths: Paths | None = None, auto_rebuild: bool = True) -> "Kora":
        return cls(paths or Paths(), auto_rebuild)

    def _load(self) -> None:
        from src.retrieval.answer import KoraAnswerer
        from src.retrieval.config import load_config
        p = self.paths
        config = load_config(p.config)
        self.answerer = KoraAnswerer.load(intelligence_dir=p.intelligence_dir, config=config,
                                          processed_dir=p.processed_dir, chunks_dir=p.chunks_dir,
                                          raw_dir=p.raw_dir)
        for w in config.warnings:
            self.events.append(f"config: {w}")

    def _ensure_current(self) -> IndexStatus:
        status = check(self.paths)
        if not status.current:
            if not self.auto_rebuild:
                raise StaleIndexError(status.summary())
            self.events.append(f"rebuilding: {status.summary()}")
            self.last_build = rebuild(self.paths)
            self.answerer = None
        if self.answerer is None:
            self._load()
        return status

    def status(self) -> IndexStatus:
        return check(self.paths)

    def user(self, employee_id: str):
        """The User for an employee ID (groups from the employees table), or None."""
        self._ensure_current()
        return self.answerer.retriever.users.get(employee_id)

    def ask(self, question: str, top_k: int | None = None, user=None):
        """Answer from the CURRENT files (rebuilds first if data/raw changed).
        answer.timings gains index_check_ms; if this question triggered a
        rebuild, answer.startup gains its seconds (and they count in total_s)."""
        t0 = time.perf_counter()
        rebuilt_before = self.last_build
        self._ensure_current()
        check_ms = (time.perf_counter() - t0) * 1000
        answer = self.answerer.answer(question, top_k=top_k, user=user)
        answer.timings = {"index_check_ms": round(check_ms, 2), **answer.timings,
                          "total_ms": round(answer.timings.get("total_ms", 0.0) + check_ms, 2)}
        if self.last_build is not rebuilt_before and self.last_build:
            secs = self.last_build.get("seconds", {})
            rebuild_s = round(sum(secs.values()), 3)
            answer.startup = {"rebuild": rebuild_s, **answer.startup,
                              "total_s": round(answer.startup.get("total_s", 0.0) + rebuild_s, 3)}
        self._log(answer)
        return answer

    def _log(self, answer) -> None:
        if self.paths.log_dir is None:
            return
        try:
            self.paths.log_dir.mkdir(parents=True, exist_ok=True)
            record = {"at": datetime.now(timezone.utc).isoformat(), **answer.to_log()}
            with open(self.paths.log_dir / "answers.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:                       # logging must never break answering
            self.events.append(f"log write failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KORA index maintenance.")
    parser.add_argument("--status", action="store_true", help="Is the index current?")
    parser.add_argument("--rebuild", action="store_true", help="Force a full rebuild")
    args = parser.parse_args()
    paths = Paths()
    if args.rebuild:
        report = rebuild(paths)
        print(f"Rebuilt: {report['documents']} documents, {report['chunks']} chunks, "
              f"Phase 2 {report['seconds']['phase2']} s, Phase 3 {report['seconds']['phase3']} s")
    status = check(paths)
    print(("CURRENT: " if status.current else "STALE: ") + status.summary())


if __name__ == "__main__":
    main()

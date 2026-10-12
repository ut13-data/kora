"""
Phase 3 input: load what Phase 2 saved to disk.

Phase 3 reads Phase 2's OUTPUT FILES, not its code paths. That keeps the
phases loosely coupled: Phase 2 can change how it chunks or extracts, and
Phase 3 keeps working as long as the saved JSON keeps its shape.

The JSON is turned back into Phase 2's own dataclasses (DocumentMetadata,
Chunk, Relationship, EntityRegistry), so Phase 3 can reuse Phase 2 helpers
such as role resolution instead of rewriting them.

One extra input: the ORIGINAL document text, read through Phase 2's own
extractor. Chunks are built for retrieval, not for structure: the Markdown
splitter strips every line and drops blank lines, so inside a chunk a
step's wrapped second line looks exactly like the paragraph after the list.
Structure (steps, role paragraphs, the org chart) is therefore parsed from
the original text, and every finding is still mapped back to chunk IDs.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from src.chunking.chunker import Chunk, _strip_metadata_header
from src.extraction.extractor import ExtractedTable, ExtractionError, extract_file
from src.ingestion.file_intake import FileInfo, FileType, UnsupportedFileTypeError, identify_file
from src.metadata.document_metadata import DocumentMetadata
from src.metadata.registry import EntityRegistry, build_registry
from src.metadata.relationships import Relationship, RelationType

from src.intelligence.text import norm_key


class Phase2OutputMissing(FileNotFoundError):
    """Raised when Phase 2 hasn't been run yet (or wrote somewhere else)."""


@dataclass
class Section:
    parent_id: str
    name: str            # e.g. "Customer Onboarding Process > Steps" (same format as Chunk.section)
    text: str            # original Markdown of the section, indentation and blank lines kept
    chunk_ids: list[str]


@dataclass
class Phase2Data:
    documents: list[DocumentMetadata]
    chunk_counts: dict[str, int]                 # source_filename -> number of chunks
    chunks: list[Chunk]
    relationships: list[Relationship]
    tables: dict[str, list[dict]]                # "employees" -> rows, unchanged
    registry: EntityRegistry
    unresolved_references: list[str] = field(default_factory=list)
    phase2_issues: list[dict] = field(default_factory=list)
    source_texts: dict[str, str] = field(default_factory=dict)   # filename (lower) -> body, header removed
    load_notes: list[str] = field(default_factory=list)

    # ---- chunks ----

    def chunks_for(self, document_id: str, version: str | None = None) -> list[Chunk]:
        """All chunks of one document, optionally of one version only."""
        return [c for c in self.chunks
                if c.document_id == document_id and (version is None or c.version == version)]

    def find_chunks(self, document_id: str, version: str, statement: str) -> list[str]:
        """Chunk IDs whose text contains the statement (compared on letters and
        digits only, so line wrapping and bold markers don't matter)."""
        chunks = self.chunks_for(document_id, version)
        key = norm_key(statement)
        for probe in (key[:60], key[-60:]):
            if len(probe) < 12:
                continue
            hits = [c.chunk_id for c in chunks if probe in norm_key(c.chunk_text)]
            if hits:
                return hits
        return []

    # ---- sections ----

    def _filename_for(self, document_id: str, version: str) -> str | None:
        for m in self.documents:
            if m.document_id == document_id and m.version == version:
                return m.source_filename
        return None

    def sections(self, document_id: str, version: str | None = None) -> list[Section]:
        """The document split at its headings, from the original text.
        Falls back to rebuilding sections from chunks if the file wasn't found."""
        if version is None:
            versions = [m.version for m in self.documents if m.document_id == document_id]
            version = versions[0] if versions else None
        filename = self._filename_for(document_id, version) if version else None
        text = self.source_texts.get(filename.lower()) if filename else None
        if text is None:
            return self._sections_from_chunks(document_id, version)

        chunks = self.chunks_for(document_id, version)
        out: list[Section] = []
        for index, (name, body) in enumerate(_split_sections(text)):
            ids = [c.chunk_id for c in chunks if c.section == name]
            if not ids:   # heading text differs slightly: locate by content instead
                ids = self.find_chunks(document_id, version, body[:200])
            out.append(Section(f"{document_id}_v{version}_s{index:02d}", name, body, ids))
        return out

    def _sections_from_chunks(self, document_id: str, version: str | None) -> list[Section]:
        grouped: dict[str, list[Chunk]] = {}
        for chunk in sorted(self.chunks_for(document_id, version), key=lambda c: c.chunk_index):
            grouped.setdefault(chunk.parent_id, []).append(chunk)
        return [Section(pid, cs[0].section, "\n\n".join(c.chunk_text for c in cs), [c.chunk_id for c in cs])
                for pid, cs in grouped.items()]


# Same heading levels Phase 2 splits on.
_MAX_LEVEL = 3


def _heading_text(raw: str) -> str:
    """Phase 2's rule: a heading that wrapped onto two lines leaves an
    unclosed '(' -- keep only the part before it, so names match Chunk.section."""
    if "(" in raw and ")" not in raw:
        raw = raw.split("(")[0].strip()
    return raw.strip()


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_name, section_text) pairs, ignoring '#' inside code blocks."""
    out: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    current: list[str] = []
    in_code = False

    def flush():
        body = "\n".join(current).strip("\n")
        if body.strip():
            out.append((" > ".join(h for _, h in stack), body))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        hashes = len(stripped) - len(stripped.lstrip("#"))
        if not in_code and 1 <= hashes <= _MAX_LEVEL and stripped[hashes:hashes + 1] == " ":
            flush()
            current = []
            while stack and stack[-1][0] >= hashes:
                stack.pop()
            stack.append((hashes, _heading_text(stripped[hashes:])))
            continue
        current.append(line)
    flush()
    return out


# ---------------- Loading ----------------

def _read_json(path: Path):
    if not path.exists():
        raise Phase2OutputMissing(f"{path} not found. Run Phase 2 first:  python -m src.ingest")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _only_known_fields(cls, data: dict) -> dict:
    """Keep only the keys the dataclass accepts.
    documents.json adds 'chunk_count', which DocumentMetadata doesn't have."""
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def _load_source_texts(raw_dir: Path, documents: list[DocumentMetadata]) -> tuple[dict[str, str], list[str]]:
    """Original text of each ingested document, via Phase 2's extractor.
    Not cleaned: cleaning collapses the indentation that marks a list item's
    continuation and the org chart's levels. Nothing here changes any words."""
    texts: dict[str, str] = {}
    notes: list[str] = []
    if not raw_dir.exists():
        return texts, [f"{raw_dir} not found: structure is rebuilt from chunks instead"]
    files = {p.name.lower(): p for p in raw_dir.rglob("*") if p.is_file()}
    for m in documents:
        path = files.get(m.source_filename.lower())
        if path is None:
            notes.append(f"{m.source_filename}: original file not found in {raw_dir}; using chunks")
            continue
        try:
            extracted = extract_file(identify_file(path))
        except (UnsupportedFileTypeError, ExtractionError) as e:
            notes.append(f"{m.source_filename}: {e}; using chunks")
            continue
        texts[m.source_filename.lower()] = _strip_metadata_header(extracted.text.replace("\xa0", " "))
    return texts, notes


def load_phase2(
    processed_dir: Path = Path("data/processed"),
    chunks_dir: Path = Path("data/chunks"),
    raw_dir: Path = Path("data/raw"),
) -> Phase2Data:
    """Load every Phase 2 output and rebuild Phase 2's objects."""
    raw_docs = _read_json(processed_dir / "documents.json")
    documents = [DocumentMetadata(**_only_known_fields(DocumentMetadata, d)) for d in raw_docs]
    chunk_counts = {d["source_filename"]: d.get("chunk_count", 0) for d in raw_docs}

    relationships = [
        Relationship(r["source"], RelationType(r["relation_type"]), r["target"], r["evidence"])
        for r in _read_json(processed_dir / "relationships.json")
    ]

    # One JSON file per CSV. Rebuilt as ExtractedTable so Phase 2's
    # build_registry() produces exactly the registry Phase 2 used.
    tables: dict[str, list[dict]] = {}
    extracted: list[ExtractedTable] = []
    tables_dir = processed_dir / "tables"
    if not tables_dir.exists():
        raise Phase2OutputMissing(f"{tables_dir} not found. Run Phase 2 first:  python -m src.ingest")
    for path in sorted(tables_dir.glob("*.json")):
        data = _read_json(path)
        fi = data["file_info"]
        info = FileInfo(filename=fi["filename"], file_type=FileType(fi["file_type"]),
                        size_bytes=fi["size_bytes"], source_path=fi["source_path"])
        extracted.append(ExtractedTable(file_info=info, rows=data["rows"]))
        tables[path.stem] = data["rows"]
    registry = build_registry(extracted)

    chunks_path = chunks_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise Phase2OutputMissing(f"{chunks_path} not found. Run Phase 2 first:  python -m src.ingest")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]

    report = _read_json(processed_dir / "validation_report.json")
    source_texts, notes = _load_source_texts(raw_dir, documents)

    return Phase2Data(
        documents=documents,
        chunk_counts=chunk_counts,
        chunks=chunks,
        relationships=relationships,
        tables=tables,
        registry=registry,
        unresolved_references=report.get("unresolved_references", []),
        phase2_issues=report.get("issues", []),
        source_texts=source_texts,
        load_notes=notes,
    )

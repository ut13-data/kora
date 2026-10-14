"""
Check every piece of relationship evidence against its source AGAIN, and give
each relationship a provenance tier (schema.py).

Phase 3 recorded where each link came from; this module re-opens that source
and confirms it still says so. Same idea as Phase 4's runtime check of answer
lines, applied to the graph:

  quote    the quoted sentence is in the cited chunk (letters and digits
           compared, so wrapping and **bold** don't matter); org-chart quotes,
           which have no chunk, are looked up in the raw file
  header   the header line ("**Owner/Role:** ...") is in the raw file and
           names the other end of the link
  record   the table row exists (first column = one end's ID) and a named
           column holds the other end's ID, name or alias
  mention  the cited chunks exist, belong to the cited document, and name
           the other end
  derived  KORA's own analysis (similarity, topic anchors, computed rules):
           nothing to re-check, and never stated as a fact

A failed check doesn't delete anything: the evidence is marked unverified
with the reason, the quality report lists it, and answers won't use it.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.graph.schema import DERIVED, HEADER, MENTION, QUOTE, RECORD, stronger
from src.intelligence.graph import KnowledgeGraph
from src.intelligence.relationships import Evidence, KnowledgeRelationship
from src.intelligence.text import norm_key

# header field named in an evidence detail -> the label used in the file
HEADER_FIELDS = {
    "owner": "Owner/Role", "department": "Department", "related roles": "Related Roles",
    "related processes": "Related Processes", "document type": "Document Type",
}
_HEADER_LINE = re.compile(r"^\*\*(?P<label>[^*:]+):\*\*\s*(?P<value>.*)$")


def words(text: str) -> str:
    """'Owner/Role: Finance Manager' -> ' owner role finance manager ' (whole-word matching)."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", str(text).lower())) + " "


@dataclass
class Provenance:
    tier: str                 # quote | header | record | mention | derived
    verified: bool
    reason: str               # what was checked, or why it failed
    evidence: Evidence
    citation: dict            # what a user is shown: kind + where + the exact text/value

    def to_dict(self) -> dict:
        return {"tier": self.tier, "verified": self.verified, "reason": self.reason, "citation": self.citation,
                "source_file": self.evidence.source_file, "detail": self.evidence.detail}


class Tracer:
    def __init__(self, graph: KnowledgeGraph, chunks: dict[str, dict], tables: dict[str, list[dict]],
                 raw_files: dict[str, Path]):
        self.graph = graph
        self.chunks = chunks          # chunk_id -> {"document_id", "chunk_text", ...}
        self.tables = tables          # "processes" -> rows (CSV column order kept)
        self.raw_files = raw_files    # filename (lower) -> path
        self._raw_cache: dict[str, str | None] = {}
        self._cache: dict[str, list[Provenance]] = {}

    @classmethod
    def from_dirs(cls, graph: KnowledgeGraph, processed_dir: Path = Path("data/processed"),
                  chunks_dir: Path = Path("data/chunks"), raw_dir: Path = Path("data/raw")) -> "Tracer":
        chunks = {}
        path = chunks_dir / "chunks.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        chunks[c["chunk_id"]] = c
        tables = {}
        tdir = processed_dir / "tables"
        for p in sorted(tdir.glob("*.json")) if tdir.exists() else []:
            with open(p, encoding="utf-8") as f:
                tables[p.stem] = json.load(f).get("rows", [])
        raw = {p.name.lower(): p for p in raw_dir.rglob("*") if p.is_file()} if raw_dir.exists() else {}
        return cls(graph, chunks, tables, raw)

    # ---------------- helpers ----------------

    def raw_text(self, filename: str) -> str | None:
        key = (filename or "").lower()
        if key not in self._raw_cache:
            path = self.raw_files.get(key)
            text = None
            if path is not None:
                if path.suffix.lower() in (".md", ".txt"):
                    text = path.read_text(encoding="utf-8", errors="replace")
                else:                              # DOCX / PDF: same extractor as Phase 2
                    try:
                        from src.extraction.extractor import extract_file
                        from src.ingestion.file_intake import identify_file
                        text = extract_file(identify_file(path)).text
                    except Exception:              # noqa: BLE001 -- unreadable file = unverifiable, reported
                        text = None
            self._raw_cache[key] = text
        return self._raw_cache[key]

    def names(self, key: str) -> list[str]:
        """Every way the documents might name an entity: ID, name, aliases, header ID."""
        e = self.graph.node(key)
        if e is None:
            return [key.split(":", 1)[-1]]
        out = [e.entity_id, e.name, *e.aliases]
        for attr in ("header_document_id", "source_filename"):   # archived copies; "see x.md" references
            if e.attributes.get(attr):
                out.append(e.attributes[attr])
        anchor = e.attributes.get("anchor")
        if e.entity_type == "topic" and anchor and anchor != key:  # a topic is named by what it is about
            out += self.names(anchor)
        return [n for n in out if n and len(words(n).strip()) > 0]

    def _named(self, text: str, keys: list[str]) -> bool:
        haystack = words(text)
        return any(words(n) in haystack for k in keys for n in self.names(k))

    def _document_ids(self, document_id: str) -> set[str]:
        """A document's own ID plus the ID in its header: the archived customer
        onboarding file is PROC-IMPL-001-V1 in documents.csv but its header (and
        so its chunks) say PROC-IMPL-001 -- a known source-data finding."""
        e = self.graph.node(f"document:{document_id}")
        return {document_id} | ({e.attributes["header_document_id"]}
                                if e and e.attributes.get("header_document_id") else set())

    def _other_ends(self, rel: KnowledgeRelationship, ev: Evidence) -> list[str]:
        """The end(s) the evidence must name: not the document the evidence comes from."""
        ends = [rel.source_key, rel.target_key]
        others = [k for k in ends if k.split(":", 1)[1] != ev.source_document]
        return others or ends

    # ---------------- per evidence ----------------

    def check(self, rel: KnowledgeRelationship, ev: Evidence) -> Provenance:
        method = ev.method
        if method in ("derived", "tfidf_similarity", "title_match"):
            return Provenance(DERIVED, True, f"KORA analysis ({method}): {ev.detail}", ev,
                              {"kind": "derived", "detail": ev.detail})
        if method == "table":
            return self._check_record(rel, ev)
        if method == "header":
            return self._check_header(rel, ev)
        if ev.quote and not ev.quote.startswith("similarity "):
            return self._check_quote(rel, ev)
        if ev.chunk_ids:
            return self._check_mention(rel, ev)
        raw = self.raw_text(ev.source_file)              # no chunk recorded: the whole file must name it
        if raw is not None and self._named(raw, self._other_ends(rel, ev)):
            return Provenance(MENTION, True, f"{ev.source_file} names it", ev,
                              {"kind": "mention", "file": ev.source_file, "detail": ev.detail})
        return Provenance(MENTION, False, f"nothing to check, or {ev.source_file} doesn't name it ({method})", ev,
                          {"kind": "mention", "detail": ev.detail})

    def _check_quote(self, rel, ev) -> Provenance:
        key = norm_key(ev.quote)
        for cid in ev.chunk_ids:
            chunk = self.chunks.get(cid)
            if chunk and key in norm_key(chunk["chunk_text"]):
                return Provenance(QUOTE, True, f"quote found in chunk {cid}", ev,
                                  {"kind": "quote", "chunk_id": cid, "document_id": chunk["document_id"],
                                   "text": ev.quote})
        if not ev.chunk_ids:                       # e.g. org chart: no chunk, check the raw file
            raw = self.raw_text(ev.source_file)
            if raw is not None and key in norm_key(raw):
                return Provenance(QUOTE, True, f"quote found in {ev.source_file}", ev,
                                  {"kind": "quote", "file": ev.source_file, "document_id": ev.source_document,
                                   "text": ev.quote})
        if "table" in ev.detail.lower():                  # rule read from a table row: cite the row itself
            parts = [norm_key(p) for p in re.split(r":\s*approved by\s*", ev.quote) if norm_key(p)]
            for cid in ev.chunk_ids:
                chunk = self.chunks.get(cid)
                for line in (chunk["chunk_text"].splitlines() if chunk else []):
                    if line.strip().startswith("|") and parts and all(p in norm_key(line) for p in parts):
                        row = " | ".join(c.strip() for c in line.strip().strip("|").split("|"))
                        return Provenance(QUOTE, True, f"table row found in chunk {cid}", ev,
                                          {"kind": "quote", "chunk_id": cid, "document_id": chunk["document_id"],
                                           "text": row})
        missing = [c for c in ev.chunk_ids if c not in self.chunks]
        why = f"chunk(s) {missing} not in the index" if missing else "quote not found in its source"
        return Provenance(QUOTE, False, why, ev, {"kind": "quote", "text": ev.quote})

    def _check_header(self, rel, ev) -> Provenance:
        detail = ev.detail.lower()
        if "topic's own document" in detail:       # topic anchoring, not a header line
            return Provenance(DERIVED, True, f"KORA analysis: {ev.detail}", ev,
                              {"kind": "derived", "detail": ev.detail})
        label = next((v for k, v in HEADER_FIELDS.items() if k in detail),
                     "Related Processes" if "related" in detail else None)
        raw = self.raw_text(ev.source_file)
        if label is None or raw is None:
            why = f"header field not recognised in '{ev.detail}'" if label is None else \
                f"file {ev.source_file} not found"
            return Provenance(HEADER, False, why, ev, {"kind": "header", "file": ev.source_file})
        for line in raw.splitlines():
            m = _HEADER_LINE.match(line.strip())
            if m and m.group("label").strip().lower() == label.lower():
                value = m.group("value").strip()
                cite = {"kind": "header", "file": ev.source_file, "document_id": ev.source_document,
                        "field": label, "text": f"{label}: {value}"}
                if self._named(value, self._other_ends(rel, ev)) or label == "Document Type":
                    return Provenance(HEADER, True, f"header line '{label}' names it", ev, cite)
                return Provenance(HEADER, False, f"header line '{label}' doesn't name "
                                  f"{', '.join(self._other_ends(rel, ev))}", ev, cite)
        return Provenance(HEADER, False, f"no '{label}' header line in {ev.source_file}", ev,
                          {"kind": "header", "file": ev.source_file})

    def _check_record(self, rel, ev) -> Provenance:
        table = Path(ev.source_file).stem
        rows = self.tables.get(table)
        if ev.source_file.lower().endswith(".md"):          # a table INSIDE a document: check the file
            raw = self.raw_text(ev.source_file)
            ok = raw is not None and self._named(raw, self._other_ends(rel, ev))
            return Provenance(RECORD if ok else MENTION, ok, f"document table row ({ev.detail})", ev,
                              {"kind": "record", "file": ev.source_file, "detail": ev.detail})
        if not rows:
            return Provenance(RECORD, False, f"table {ev.source_file} not loaded", ev,
                              {"kind": "record", "file": ev.source_file})
        pk = next(iter(rows[0]))
        columns = [c for c in rows[0] if re.search(rf"\b{re.escape(c)}\b", ev.detail)]
        for end, other in ((rel.source_key, rel.target_key), (rel.target_key, rel.source_key)):
            end_id = end.split(":", 1)[1]
            row = next((r for r in rows if str(r.get(pk, "")) == end_id), None)
            if row is None:
                continue
            for col in columns:
                if self._named(str(row.get(col) or ""), [other]):
                    return Provenance(RECORD, True, f"{ev.source_file} row {end_id}: {col}", ev,
                                      {"kind": "record", "file": ev.source_file, "key_column": pk, "key": end_id,
                                       "column": col, "value": row.get(col),
                                       "text": f"{ev.source_file}, row {end_id}: {col} = {row.get(col)}"})
            if columns:                                      # row + columns exist, link is computed from them
                values = ", ".join(f"{c} = {row.get(c)}" for c in columns)
                return Provenance(DERIVED, True, f"computed from {ev.source_file} row {end_id} ({values})", ev,
                                  {"kind": "derived", "detail": f"{ev.source_file}, row {end_id}: {values}"})
        return Provenance(RECORD, False, f"no {ev.source_file} row whose {pk} is either end, "
                          f"or no column named in '{ev.detail}'", ev, {"kind": "record", "file": ev.source_file})

    def _check_mention(self, rel, ev) -> Provenance:
        texts = []
        for cid in ev.chunk_ids:
            chunk = self.chunks.get(cid)
            if chunk is None:
                return Provenance(MENTION, False, f"chunk {cid} not in the index", ev, {"kind": "mention"})
            if ev.source_document and chunk["document_id"] not in self._document_ids(ev.source_document):
                return Provenance(MENTION, False, f"chunk {cid} is not from {ev.source_document}", ev,
                                  {"kind": "mention"})
            texts.append(chunk["chunk_text"])
        others = self._other_ends(rel, ev)
        if self._named(" ".join(texts), others):
            return Provenance(MENTION, True, "chunk names it", ev,
                              {"kind": "mention", "chunk_id": ev.chunk_ids[0], "detail": ev.detail})
        raw = self.raw_text(ev.source_file)              # e.g. "header names the process": headers aren't in chunks
        if raw is not None and self._named(raw, others):
            return Provenance(MENTION, True, f"{ev.source_file} names it (not in the cited chunk)", ev,
                              {"kind": "mention", "file": ev.source_file, "detail": ev.detail})
        return Provenance(MENTION, False, "neither the chunk nor the file names the other end", ev,
                          {"kind": "mention", "chunk_id": ev.chunk_ids[0], "detail": ev.detail})

    # ---------------- per relationship ----------------

    def trace(self, rel: KnowledgeRelationship) -> list[Provenance]:
        if rel.relationship_id not in self._cache:
            self._cache[rel.relationship_id] = [self.check(rel, ev) for ev in rel.evidence]
        return self._cache[rel.relationship_id]

    def tier(self, rel: KnowledgeRelationship, evidence: list[Evidence] | None = None) -> str | None:
        """Strongest VERIFIED tier; None if nothing could be verified. Pass the
        evidence a user may see (view.py) to get the tier for that user."""
        allowed = None if evidence is None else {id(e) for e in evidence}
        best = None
        for p in self.trace(rel):
            if p.verified and (allowed is None or id(p.evidence) in allowed):
                best = stronger(best, p.tier)
        return best

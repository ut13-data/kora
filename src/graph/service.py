"""
Phase 5 M3 - the graph inside KORA's answer flow.

GraphService holds what every request needs: the saved graph, the evidence
tracer, Phase 3's process and conflict lists, and the company config. It
hands out per-user GraphQueries (cached by user), and it turns a graph fact
into an answer line that Phase 4's runtime verifier can check again.

Citation IDs for graph evidence that isn't a chunk (they live in
AnswerLine.chunk_id, so numbering, logging and evaluation keep working):

    header:<file>#<Field>                   text: "<Field>: <value>"
    record:<file>#<key column>=<key>#<column>   text: "<file>, row <key>: <column> = <value>"
    file:<file>                             text: a quote found in the raw file (org chart)

Each is re-checked from the ID and the text alone, at answer time:
header line still in the file, record value still in the table, quote still
in the file. A line that fails is dropped, exactly like a Phase 4 quote.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from src.graph.load import load_graph
from src.graph.query import GraphFact, GraphQueries, _read_list
from src.graph.trace import Tracer
from src.graph.view import GraphView
from src.intelligence.text import norm_key

PSEUDO = ("header:", "record:", "file:")
_RECORD_ID = re.compile(r"^record:(?P<file>[^#]+)#(?P<kcol>[^=#]+)=(?P<key>[^#]+)#(?P<col>.+)$")


@dataclass
class GraphLine:
    """A graph fact as an answer line: TEXT is what is verified, CLAIM is KORA's statement."""
    kind: str            # relation | possible
    text: str
    chunk_id: str        # a real chunk ID or a pseudo ID (see module docstring)
    claim: str
    tier: str
    historical: bool


class GraphService:
    def __init__(self, graph, tracer: Tracer, config, processes: list[dict], conflicts: list[dict]):
        self.graph = graph
        self.tracer = tracer
        self.config = config
        self.processes = processes
        self.conflicts = conflicts
        self._queries: dict[str | None, GraphQueries] = {}

    @classmethod
    def from_files(cls, config, intelligence_dir=Path("data/intelligence"), processed_dir=Path("data/processed"),
                   chunks_dir=Path("data/chunks"), raw_dir=Path("data/raw")) -> "GraphService":
        loaded = load_graph(intelligence_dir)
        tracer = Tracer.from_dirs(loaded.graph, processed_dir, chunks_dir, raw_dir)
        return cls(loaded.graph, tracer, config, _read_list(intelligence_dir / "processes.json", "processes"),
                   _read_list(intelligence_dir / "conflicts.json", "conflicts"))

    @classmethod
    def from_intelligence(cls, result, config, raw_dir=Path("data/raw")) -> "GraphService":
        """From a Phase 3 run in memory (tests)."""
        chunks = {c.chunk_id: {"chunk_id": c.chunk_id, "document_id": c.document_id, "chunk_text": c.chunk_text}
                  for c in result.phase2.chunks}
        raw = {p.name.lower(): p for p in Path(raw_dir).rglob("*") if p.is_file()} if Path(raw_dir).exists() else {}
        tracer = Tracer(result.graph, chunks, result.phase2.tables, raw)
        out = result.outputs
        processes = out.get("processes") or []
        return cls(result.graph, tracer, config, processes if isinstance(processes, list) else
                   processes.get("processes", []), (out.get("conflicts") or {}).get("conflicts", []))

    def queries(self, user) -> GraphQueries:
        key = user.user_id if user else None
        if key not in self._queries:
            view = GraphView.for_user(self.graph, user, self.config)
            self._queries[key] = GraphQueries(view, self.tracer, None, self.processes, self.conflicts)
        return self._queries[key]

    # ---------------- facts -> answer lines ----------------

    @staticmethod
    def line(fact: GraphFact, kind: str) -> GraphLine | None:
        c = fact.citation
        text = str(c.get("text") or "")
        if not text:
            return None
        if c.get("chunk_id"):
            chunk_id = c["chunk_id"]
        elif c["kind"] == "header":
            chunk_id = f"header:{c['file']}#{c['field']}"
        elif c["kind"] == "record":
            chunk_id = f"record:{c['file']}#{c['key_column']}={c['key']}#{c['column']}"
        elif c["kind"] == "quote" and c.get("file"):
            chunk_id = f"file:{c['file']}"
        else:
            return None                      # mentions/derived without a checkable source: never shown
        claim = f"{fact.group}: {fact.claim}" if fact.group else fact.claim
        return GraphLine(kind, text, chunk_id, claim, fact.tier, fact.historical)

    # ---------------- runtime re-check of non-chunk evidence ----------------

    def verify(self, chunk_id: str, text: str) -> bool:
        if chunk_id.startswith("header:"):
            file, _, field = chunk_id[len("header:"):].partition("#")
            raw = self.tracer.raw_text(file)
            label, sep, value = text.partition(": ")
            return bool(raw and sep and label == field) and any(
                line.strip() == f"**{field}:** {value}".strip() for line in raw.splitlines())
        if chunk_id.startswith("record:"):
            m = _RECORD_ID.match(chunk_id)
            if not m:
                return False
            rows = self.tracer.tables.get(Path(m["file"]).stem) or []
            row = next((r for r in rows if str(r.get(m["kcol"])) == m["key"]), None)
            return row is not None and m["col"] in row and \
                text == f"{m['file']}, row {m['key']}: {m['col']} = {row[m['col']]}"
        if chunk_id.startswith("file:"):
            raw = self.tracer.raw_text(chunk_id[len("file:"):])
            return bool(raw) and bool(norm_key(text)) and norm_key(text) in norm_key(raw)
        return False

    def describe(self, chunk_id: str) -> dict:
        """What a citation for a pseudo ID shows (file, document, section)."""
        if chunk_id.startswith("record:"):
            m = _RECORD_ID.match(chunk_id)
            return {"kind": "record", "file": m["file"], "section": f"row {m['key']}, column {m['col']}"}
        if chunk_id.startswith("header:"):
            file, _, field = chunk_id[len("header:"):].partition("#")
            return {"kind": "header", "file": file, "section": f"document header: {field}"}
        return {"kind": "file", "file": chunk_id[len("file:"):], "section": "(whole file)"}

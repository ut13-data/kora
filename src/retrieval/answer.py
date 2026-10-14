"""
Phase 4 - grounded answers with citations, and honest uncertainty.

    question -> KoraRetriever.retrieve()      where to look, what was found
             -> evidence lines                 quotes, rules, conflict sides, checklist items,
                                               version changes, document lists
             -> VERIFY every line              a line not found in its cited chunk is dropped
             -> status + notes                 answered | partial | conflict | not_found,
                                               gaps, history, caveats, unknown metadata

    from src.retrieval.answer import KoraAnswerer
    kora = KoraAnswerer.load()
    print(kora.answer("Who approves purchases above 10000 euros?").render())

No language model (decided with UT): KORA never writes a sentence about the
company. Every line is copied from a source chunk (or is one of Phase 3's
structured facts extracted from a chunk), and is VERIFIED against that chunk
at runtime before it is shown -- not only in tests. Lines that fail are
removed and recorded in Answer.dropped_lines.

Precision first (decided with UT): when the evidence is weak, KORA says so
("partial") or abstains ("not_found") instead of guessing.

Company-specific behaviour (relocation checklist, tone profiles, notes) and
all thresholds come from config/kora_config.json via the retriever.
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.config import KoraConfig, load_config
from src.retrieval.retrieve import KoraRetriever, RetrievalOutcome
from src.retrieval.router import normalise

# Line kinds that are complete structured facts: never cut by max_lines.
STRUCTURED_KINDS = {"conflict_side", "checklist", "change", "document"}

STATUS_MEANING = {
    "superseded_per_source": "the sources themselves say the newer document replaced the older one",
    "open": "no source says which statement is correct",
}

# Words that make a question an approval/escalation question but say nothing
# about WHAT is approved. Removed before comparing rules with the question
# (otherwise "Direct manager approves." matches every "who approves" question).
_RULE_NOISE = re.compile(
    r"\b(who|whom|approv\w*|sign\w*|off|authority|authori\w+|final|requires?|required|"
    r"escalat\w*|process|procedure|what|is|the|for|of|a|an|do|does|i|to)\b", re.I)

# Generic English words never worth reporting as "not covered": question
# words, common verbs/adverbs and generic business words. Company-independent.
GENERIC_WORDS = set(ENGLISH_STOP_WORDS) | {
    # question / request words
    "tell", "know", "explain", "describe", "describes", "list", "show", "mean", "means", "say", "says",
    "said", "ask", "asked", "question", "questions", "answer", "info", "information", "details",
    "contact", "contacts", "contacted",
    # common verbs
    "need", "needs", "needed", "happen", "happens", "happened", "make", "makes", "made", "doing", "done",
    "does", "did", "do",
    "give", "gives", "given", "giving", "get", "gets", "getting", "got", "go", "goes", "going", "went",
    "keep", "keeps", "kept", "look", "looks", "looking", "follow", "follows", "following", "use", "used",
    "uses", "using", "take", "takes", "taking", "took", "come", "comes", "put", "puts", "handle", "handles",
    "handled", "deal", "work", "works", "working", "worked", "want", "wants", "like", "help", "helps",
    "start", "starting", "started", "finish", "finished", "complete", "completed", "prepare", "prepared",
    "move", "moving", "moved", "join", "joining", "joined", "happen", "allowed", "allow", "able",
    "supposed", "expected", "expect", "require", "requires", "required", "apply", "applies", "valid",
    # adverbs / quantities / time words
    "usually", "normally", "typically", "generally", "quickly", "long", "much", "many", "often",
    "still", "currently", "current", "actually", "exactly", "really", "just", "also", "first", "next",
    "time", "times", "day", "days", "week", "weeks", "month", "months", "year", "years", "today", "now",
    # generic business words
    "company", "document", "documents", "process", "processes", "policy", "policies", "procedure",
    "procedures", "rule", "rules", "employee", "employees", "person", "people", "someone", "anyone",
    "responsible", "responsibility", "responsibilities", "owns", "owner",
    "anybody", "team", "teams", "colleague", "colleagues", "staff", "new", "old", "previous",
    "previously", "earlier", "older", "version", "versions", "change", "changed", "archived",
    "history", "historical", "approve", "approves", "approval", "approved", "escalate", "escalating",
    "escalation", "conflict", "conflicting", "instruction", "instructions", "exist", "exists",
    # pieces of contractions after normalising ("don't" -> "don t")
    "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won", "wouldn", "shouldn", "couldn",
    "can", "t", "s", "ve", "ll", "re", "d", "m",
}

# Symbols the word cleanup removes, so "euros" can't be matched as a word.
_SYMBOL_WORDS = {"euro": "€", "euros": "€", "eur": "€", "dollar": "$", "dollars": "$", "usd": "$"}

# Time windows in relocation questions -> "due by day N" from the checklist.
_WINDOWS = [
    (re.compile(r"\b(?:first|1st) (?:two|2) weeks\b|\bfirst 14 days\b"), 14),
    (re.compile(r"\b(?:first|1st) (?:three|3) months\b|\bfirst 90 days\b"), 90),
    (re.compile(r"\b(?:first|1st) (?:two|2) months\b|\bfirst 60 days\b"), 60),
    (re.compile(r"\b(?:first|1st) month\b|\bfirst 30 days\b"), 30),
    (re.compile(r"\b(?:first|1st) week\b|\bweek (?:one|1)\b"), 7),
]
_N_DAYS = re.compile(r"\bfirst (\d+) days\b")


def time_window(norm_question: str) -> int | None:
    """Days covered by 'first week' / 'first month' / 'first 90 days' ..."""
    for regex, days in _WINDOWS:
        if regex.search(norm_question):
            return days
    m = _N_DAYS.search(norm_question)
    return int(m.group(1)) if m else None


# ---------------- data shapes ----------------

@dataclass
class Citation:
    number: int
    chunk_id: str
    document_id: str
    title: str
    version: str
    status: str
    effective_date: str
    section: str
    source_filename: str
    excerpt: str
    id_source: str = ""
    title_source: str = ""
    kind: str = "chunk"       # Phase 5: chunk | header | record | file (graph evidence that isn't a chunk)

    def label(self) -> str:
        if self.kind == "record":
            return f"[{self.number}] Company record: {self.source_filename} > {self.section}"
        if self.kind == "file":
            return f"[{self.number}] {self.title} > {self.section} -- file {self.source_filename}"
        status = self.status if self.status.lower() != "unknown" else "status not stated"
        version = f"v{self.version}" if self.version.lower() != "unknown" else "version not stated"
        date = self.effective_date if self.effective_date.lower() != "unknown" else "no effective date"
        where = "document header" if self.kind == "header" else f"chunk {self.chunk_id}"
        return (f"[{self.number}] {self.title} ({self.document_id} {version}, {status}, {date}) "
                f"> {self.section or '(top)'} -- {where}, file {self.source_filename}")


@dataclass
class AnswerLine:
    kind: str                 # quote | rule | conflict_side | checklist | change | document | gap
                              # Phase 5: relation (a documented graph link) | possible (only cross-referenced)
    text: str
    chunk_id: str | None      # the source chunk; None only for a conflict side from Phase 3
                              # analysis with no chunk (e.g. employees.csv vs org chart)
    detail: str = ""          # may contain "{ref}": replaced by ref_chunk_id's citation number
    score: float = 1.0        # how well it matched the question (structured facts: 1.0)
    ref_chunk_id: str | None = None   # a second source mentioned in the detail (old version text)
    citation: int | None = None       # assigned once, after verification
    semantic: float = 0.0             # meaning match with the question (0.0 when embeddings are off)


@dataclass
class Answer:
    question: str
    status: str                                   # answered | partial | conflict | not_found
    lines: list[AnswerLine]
    citations: list[Citation]
    notes: list[str] = field(default_factory=list)
    uncovered_terms: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    dropped_lines: list[dict] = field(default_factory=list)   # failed runtime verification
    retrieval: RetrievalOutcome | None = None
    elapsed_ms: float = 0.0
    # WHY this status: every score and threshold the status rules used, and
    # which rule decided (see KoraAnswerer.answer, step 6).
    decision: dict = field(default_factory=dict)
    # Time per step for THIS answer (ms), and KORA's start-up per step (s).
    timings: dict = field(default_factory=dict)
    startup: dict = field(default_factory=dict)
    user: str | None = None                       # who asked (M7); None = no access filtering

    def render(self, why: bool = False) -> str:
        out = [f"Q: {self.question}", f"Status: {self.status}", ""]
        out.append("I couldn't find this in the knowledge base." if self.status == "not_found"
                   else "Answer (documented in the sources):")
        prefixes = {"quote": "-", "rule": "- Rule:", "conflict_side": "- Says:", "checklist": "- To do:",
                    "change": "- Changed:", "document": "- Document:", "gap": "- Not documented:"}
        shown_possible = False
        for line in self.lines:
            ref = f" [{line.citation}]" if line.citation else " [no chunk: Phase 3 analysis]"
            if line.kind in ("relation", "possible"):          # Phase 5 graph: KORA's statement, then its evidence
                if line.kind == "possible" and not shown_possible:
                    out.append("  Possibly related (listed or mentioned together, not stated as a relationship):")
                    shown_possible = True
                out.append(f"  - {line.detail}{ref}")
                out.append(f"      \"{line.text}\"")
                continue
            out.append(f"  {prefixes[line.kind]} {line.text}{ref}")
            if line.detail:
                out.append(f"      ({line.detail})")
        if self.notes:
            out += ["", "Notes:"] + [f"  ! {n}" for n in self.notes]
        if self.citations:
            out += ["", "Sources:" if self.status != "not_found" else "Closest sources (not an answer):"]
            out += [f"  {c.label()}" for c in self.citations]
        if why and self.decision:
            out += ["", "Why this status:"] + [f"  {l}" for l in explain_decision(self.decision)]
        if self.timings:
            out += ["", "Timing: " + format_timings(self.timings, self.startup)]
        return "\n".join(out)

    def to_log(self) -> dict:
        return {
            "question": self.question,
            "status": self.status,
            "lines": [asdict(l) for l in self.lines],
            "citations": [c.chunk_id for c in self.citations],
            "notes": self.notes,
            "uncovered_terms": self.uncovered_terms,
            "rule_ids": self.rule_ids,
            "dropped_lines": self.dropped_lines,
            "semantic_max": round(max((l.semantic for l in self.lines), default=0.0), 4),
            "retrieval": self.retrieval.to_log() if self.retrieval else None,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "user": self.user,
            "decision": self.decision,
            "timings": self.timings,
            "startup": self.startup,
        }


# ---------------- "why" and timing, in plain words ----------------

def explain_decision(d: dict) -> list[str]:
    """The status decision as short readable lines (CLI / UI)."""
    out = [f"decided by: {d.get('rule', '?')} -- {d.get('rule_text', '')}".rstrip(" -")]
    g = d.get("graph")
    if g:
        out.append(f"knowledge graph: {g['kind']} about {', '.join(g['subjects']) or 'trigger sentences'}; "
                   f"{g['facts']} documented link(s), {g['possible']} possibly related")
        out.append(f"Phase 4 alone would have said: {d.get('phase4_status')} ({d.get('phase4_rule')})")
        return out
    if "word_strength" in d:
        out.append(f"best word match {d['word_strength']:.2f} (strong from {d['word_strong_needed']:.2f})")
    if d.get("meaning_strength") is not None:
        out.append(f"best meaning match {d['meaning_strength']:.2f} (strong from {d['meaning_strong_needed']:.2f})")
    elif "word_strength" in d:
        out.append("meaning match: off (embeddings not loaded)")
    if d.get("important"):
        out.append(f"main words: {', '.join(d['important'])}")
    if d.get("uncovered"):
        out.append(f"not in the quoted sources: {', '.join(d['uncovered'])}")
    for m in d.get("unknown_words", []):
        if m.get("blocked"):
            out.append(f"'{m['word']}' never appears in the knowledge base; not matched by meaning ({m['blocked']})")
        elif m.get("best"):
            verdict = "explained" if m.get("explained") else "not explained"
            gate = "" if d.get("meaning_strength") is None or m.get("explained") or \
                d["meaning_strength"] >= d["meaning_strong_needed"] else ", answer match not strong enough"
            out.append(f"'{m['word']}' never appears in the knowledge base; closest quoted word "
                       f"'{m['best']}' {m['score']:.2f} (needs {m['needed']:.2f}{gate}) -> {verdict}")
        else:
            out.append(f"'{m['word']}' never appears in the knowledge base")
    return out


def format_timings(timings: dict, startup: dict | None = None) -> str:
    """'36 ms (route 2, search 8, compose 26) | start-up 38.2 s (...)'."""
    parts = [f"{k.removesuffix('_ms').replace('_', ' ')} {v:.0f}" for k, v in timings.items() if k != "total_ms"]
    text = f"{timings.get('total_ms', 0):.0f} ms ({', '.join(parts)})"
    if startup:
        total = startup.get("total_s", sum(v for k, v in startup.items() if k != "total_s"))
        steps = ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in startup.items() if k != "total_s")
        text += f" | start-up {total:.1f} s ({steps})"
    return text


# ---------------- text helpers ----------------

_UNIT_START = re.compile(r"^(#+\s|[-*]\s|\d+\.\s|\*\*Q:|\*\*Scenario|>|\|)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"*(])")


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def name_words(texts) -> set[str]:
    """Lowercased words that are written ONLY capitalised across the texts
    (never in lower case): place, person and product names ("Berlin",
    "Nexora", "CFO"). A word seen lowercase anywhere ("travel") is not a name."""
    upper, lower = set(), set()
    for text in texts:
        for tok in _TOKEN.findall(text):
            if tok[0].isalpha():
                (upper if tok[0].isupper() else lower).add(tok.lower())
    return upper - lower


def question_names(question: str) -> set[str]:
    """Words the asker capitalised mid-sentence ("in the Paris office").
    The first word is skipped (always capitalised); a question typed in
    Title Case or ALL CAPS gives no signal, so it yields nothing."""
    tokens = [t for t in _TOKEN.findall(question) if t[0].isalpha()]
    caps = [t for t in tokens[1:] if t[0].isupper()]
    if not tokens or len(caps) > len(tokens) / 2:
        return set()
    return {t.lower() for t in caps}


def split_sentences(chunk_text: str, min_words: int = 5) -> list[str]:
    """Sentences of a chunk. Wrapped lines are joined back into paragraphs /
    list items (a new unit starts at a heading, bullet, numbered step, table
    row or FAQ question; the line after a heading always starts a new unit),
    then split at sentence ends. Headings and table rows are dropped (tables
    are handled as rules), Markdown markers removed, words never changed.
    An FAQ entry stays whole, so a question is never quoted without its
    answer. Fragments under MIN_WORDS words are dropped."""
    units: list[str] = []
    for line in chunk_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        after_heading = bool(units) and units[-1].startswith("#")
        if _UNIT_START.match(stripped) or not units or after_heading:
            units.append(stripped)
        else:
            units[-1] += " " + stripped
    sentences = []
    for unit in units:
        if unit.startswith("|"):
            row = table_row(unit)
            if row and len(row.split()) >= min_words:
                sentences.append(row)
            continue
        if unit.startswith("#"):
            continue
        unit = re.sub(r"^([-*]|\d+\.|>)\s+", "", unit)
        unit = unit.replace("**", "").replace("`", "")
        unit = re.sub(r"(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)", "", unit)       # lone italic markers
        parts = [unit] if unit.startswith("Q:") else _SENTENCE_SPLIT.split(unit)
        for s in parts:
            s = s.strip()
            if len(s.split()) >= min_words:
                sentences.append(s)
    return sentences


def table_row(line: str) -> str | None:
    """A Markdown table row as 'cell | cell | cell' (verbatim cells), or None
    for the separator row (|---|---|)."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if not any(cells) or all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
        return None
    return " | ".join(cells)


def table_header(chunk_text: str, row: str) -> str | None:
    """The header row of the table that ROW belongs to (first row of its table)."""
    header = None
    for line in chunk_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            header = None
            continue
        rendered = table_row(stripped)
        if rendered is None:
            continue
        if header is None:
            header = rendered
        if rendered == row:
            return header if header != row else None
    return None


def subject_words(question: str) -> str:
    """The question without generic words, for scoring QUOTES. Bug found in
    M5: "policy on bringing pets to the office" quoted a leave sentence
    because it says "company policy" -- filler, not the subject."""
    words = [w for w in normalise(question).split() if w not in GENERIC_WORDS]
    return " ".join(words) if words else question


def _stem(word: str) -> str:
    """Crude suffix strip, so "paying" finds "pay"/"payment" and "approves"
    finds "approval". -ing keeps 3 letters ("paying" -> "pay"); the others 4,
    so short words aren't cut to nothing ("uses" stays "uses")."""
    if word.endswith("ing") and len(word) - 3 >= 3:
        return word[:-3]
    for suffix in ("es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


# Lower-case hyphenated words ("part-time", "go-live"); IDs like PROC-IT-001 are upper case.
_COMPOUND = re.compile(r"(?<![\w-])[a-z]+(?:-[a-z]+)+(?![\w-])")


def _stem_phrase(phrase: str) -> str:
    """'money back' -> 'money back'; 'reimbursed' -> 'reimburs' (last word stemmed)."""
    words = phrase.split()
    return " ".join(words[:-1] + [_stem(words[-1])]) if words else phrase


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------- answerer ----------------

class KoraAnswerer:
    def __init__(self, retriever: KoraRetriever, outputs: dict):
        """OUTPUTS: Phase 3 output dicts by file stem (entities, roles,
        conflicts, knowledge_gaps, outdated_knowledge, topics, and the
        relocation checklist file if configured). Any of them may be missing
        or empty (e.g. a company with documents only): the matching feature
        simply finds nothing."""
        self._init_start = time.perf_counter()
        self.retriever = retriever
        self.config: KoraConfig = retriever.config
        self._vectorizer = retriever.chunks.tfidf.vectorizer
        self._chunks = retriever.chunks.keyword.chunks
        self._chunk_norm = {cid: normalise(c.chunk_text) for cid, c in self._chunks.items()}

        roles = outputs.get("roles") or {}
        self._rules = {"approval": roles.get("approval_rules", []), "escalation": roles.get("escalation_rules", [])}
        self._rule_matrix = {kind: self._vectorizer.transform(
            [_RULE_NOISE.sub(" ", f"{r['statement']} {r.get('context') or ''}") for r in rules])
            if rules else None for kind, rules in self._rules.items()}
        entities = (outputs.get("entities") or {}).get("entities", [])
        self._role_names = {e["entity_id"]: e["name"] for e in entities if e["entity_type"] == "role"}
        self._header_id = {e["entity_id"]: e["attributes"].get("header_document_id") or e["entity_id"]
                           for e in entities if e["entity_type"] == "document"}
        self._conflicts = {c["conflict_id"]: c for c in (outputs.get("conflicts") or {}).get("conflicts", [])}
        self._gaps = [g for g in (outputs.get("knowledge_gaps") or {}).get("gaps", [])
                      if g.get("severity") in set(self.config.gap_severities)]
        # Chunks that ARE a documented gap (declared by the source itself).
        self._gap_chunks = {cid for g in self._gaps if g.get("gap_type") == "declared_gap"
                            for e in g.get("evidence", []) for cid in e.get("chunk_ids", [])}
        self._superseded = (outputs.get("outdated_knowledge") or {}).get("superseded_documents", [])
        self._topics = {t["topic_id"]: t for t in (outputs.get("topics") or {}).get("topics", [])}

        reloc = self.config.relocation
        checklist = outputs.get("checklist") or {}
        self._checklist_items = checklist.get("items", []) if reloc and reloc.checklist_file else []
        self._relocation_disclaimer = (checklist.get("program", {}).get("not_advice")
                                       if reloc and reloc.disclaimer else None)
        self._relocation_docs = set(self._topics.get(reloc.topic_id, {}).get("home_documents", [])) \
            if reloc and reloc.topic_id else set()

        self._document_raw = self._build_document_text()
        self._document_text = {k: normalise(v) for k, v in self._document_raw.items()}
        self._kb_words = sorted({w for t in self._document_text.values() for w in t.split()})
        self._kb_names = name_words(self._document_raw.values())
        self._synonyms = [(re.compile(r"\b" + re.escape(phrase) + r"s?\b"), phrase, words)
                          for phrase, words in self.config.synonyms.items()]
        from src.retrieval.access import AccessPolicy
        self.access = AccessPolicy(self.config.access_rules)
        self._all_documents = {d for d, _ in retriever.documents} | {c.document_id for c in self._chunks.values()}
        self.load_timings: dict[str, float] = dict(retriever.load_timings)
        self.load_timings["answer_layer"] = round(time.perf_counter() - self._init_start, 3)
        self.graph = None                      # Phase 5: GraphService, attached by load()/from_intelligence()
        self._precompute_sentences()

    @property
    def _semantic(self):
        return self.retriever.chunks.semantic

    def _precompute_sentences(self) -> None:
        """Encode every quotable sentence once (cached on disk across runs),
        so answering only has to encode the question."""
        if self._semantic is None:
            return
        t0 = time.perf_counter()
        min_words = self.config.t("min_quote_words")
        self._semantic.add_sentences([s for c in self._chunks.values()
                                      for s in split_sentences(c.chunk_text, min_words)])
        try:
            self._semantic.save()
        except OSError as e:                 # a read-only disk must not stop answering
            self.config.warnings.append(f"embedding cache not saved: {e}")
        self.load_timings["sentence_vectors"] = round(time.perf_counter() - t0, 3)

    def use_embedder(self, embedder) -> None:
        """Switch to a given embedder (tests use a deterministic stand-in)."""
        self.retriever.use_embedder(embedder)
        self.load_timings.update(self.retriever.load_timings)
        self._precompute_sentences()

    @classmethod
    def from_intelligence(cls, result, config: KoraConfig | None = None) -> "KoraAnswerer":
        cfg = config if config is not None else load_config()
        outputs = dict(result.outputs)
        if cfg.relocation and cfg.relocation.checklist_file:
            outputs["checklist"] = result.outputs.get(cfg.relocation.checklist_file, {})
        obj = cls(KoraRetriever.from_intelligence(result, cfg), outputs)
        if cfg.graph_enabled:
            from src.graph.service import GraphService
            t0 = time.perf_counter()
            obj.graph = GraphService.from_intelligence(result, cfg)
            obj.load_timings["graph"] = round(time.perf_counter() - t0, 3)
        return obj

    @classmethod
    def load(cls, intelligence_dir: Path = Path("data/intelligence"), config: KoraConfig | None = None,
             **dirs) -> "KoraAnswerer":
        """From saved Phase 2 + Phase 3 files (Phase 3 is not re-run). For
        an always-fresh index use src.index.open_kora() instead."""
        cfg = config if config is not None else load_config()
        outputs = {stem: _read_json(intelligence_dir / f"{stem}.json")
                   for stem in ("entities", "roles", "conflicts", "knowledge_gaps", "outdated_knowledge", "topics")}
        if cfg.relocation and cfg.relocation.checklist_file:
            outputs["checklist"] = _read_json(intelligence_dir / f"{cfg.relocation.checklist_file}.json")
        obj = cls(KoraRetriever.load(intelligence_dir=intelligence_dir, config=cfg, **dirs), outputs)
        if cfg.graph_enabled and (intelligence_dir / "relationships.json").exists():
            from src.graph.service import GraphService
            t0 = time.perf_counter()
            obj.graph = GraphService.from_files(cfg, intelligence_dir, **{k: v for k, v in dirs.items()
                                                                          if k in ("processed_dir", "chunks_dir", "raw_dir")})
            obj.load_timings["graph"] = round(time.perf_counter() - t0, 3)
        return obj

    def _build_document_text(self) -> dict[tuple, str]:
        """(document_id, version) -> full text, for the coverage check."""
        text: dict[tuple, list[str]] = {}
        for c in sorted(self._chunks.values(), key=lambda c: c.chunk_index):
            text.setdefault((c.document_id, c.version), []).append(c.chunk_text)
        return {k: " ".join(v) for k, v in text.items()}

    # ---------------- the pipeline ----------------

    def answer(self, question: str, top_k: int | None = None, user=None) -> Answer:
        """Answer QUESTION. The result also says WHY it got its status
        (answer.decision) and how long each step took (answer.timings, plus
        KORA's start-up in answer.startup).

        USER (access control, M7): a src.retrieval.access.User. Only documents
        that user may read are searched, quoted, cited or mentioned. None = no
        filtering (backend tools and evaluation; the UI must pass a user)."""
        start = time.perf_counter()
        decision: dict = {}
        allowed = self.access.allowed(user, self._all_documents) if self.access.active else None
        answer = self._compose(question, top_k, decision, start, allowed)
        graph_ms = 0.0
        if self.graph is not None:                 # Phase 5: relationship questions from the graph
            t0 = time.perf_counter()
            graph_answer = self.graph.queries(user).answer(question)
            if graph_answer is not None:
                answer = self._from_graph(graph_answer, answer, decision)
            graph_ms = (time.perf_counter() - t0) * 1000
        if allowed is not None:
            self._scrub_hidden(answer, allowed)
        answer.decision = decision
        answer.user = user.user_id if user else None
        r = answer.retrieval
        total = (time.perf_counter() - start) * 1000
        answer.timings = {"route_ms": round(r.route_ms, 2) if r else 0.0,
                          "search_ms": round(r.search_ms, 2) if r else 0.0,
                          "compose_ms": round(total - (r.elapsed_ms if r else 0.0) - graph_ms, 2),
                          **({"graph_ms": round(graph_ms, 2)} if self.graph is not None else {}),
                          "total_ms": round(total, 2)}
        answer.elapsed_ms = total
        answer.startup = self.startup_summary()
        return answer

    def _scrub_hidden(self, answer: "Answer", allowed: set[str]) -> None:
        """Last safety net (M7): no citation, line or note may name a document
        the user can't read -- by ID or by title."""
        hidden = self._all_documents - allowed
        if not hidden:
            return
        hidden_titles = {m.title for (d, _), m in self.retriever.documents.items()
                         if d in hidden and getattr(m, "title", "") and m.title.lower() != "unknown"}
        hidden_titles |= {c.document_title for c in self._chunks.values() if c.document_id in hidden}
        keep = {c.number for c in answer.citations if c.document_id in allowed
                or (c.kind == "record" and not any(h in c.chunk_id for h in hidden))}
        answer.citations = [c for c in answer.citations if c.number in keep]
        answer.lines = [l for l in answer.lines if l.citation is None or l.citation in keep]
        answer.notes = [n for n in answer.notes
                        if not any(h in n for h in hidden) and not any(t in n for t in hidden_titles)]
        if not answer.lines and answer.status != "not_found":
            answer.status = "not_found"
            answer.notes = ["No sentence in the retrieved sources answers this question. It may not be documented."]

    def startup_summary(self) -> dict:
        """KORA's start-up time per step, in seconds, plus the total."""
        steps = {k: v for k, v in self.load_timings.items()}
        return {**steps, "total_s": round(sum(steps.values()), 3)}

    def expand(self, question: str) -> tuple[str, dict[str, list[str]]]:
        """(text to search with, {phrase: words}) using the company word list."""
        q = normalise(question)
        used = {phrase: words for pattern, phrase, words in self._synonyms if pattern.search(q)}
        if not used:
            return question, {}
        extra = " ".join(dict.fromkeys(w for ws in used.values() for w in ws))
        return f"{question} {extra}", used

    def _compose(self, question: str, top_k: int | None, decision: dict, start: float,
                 allowed: set[str] | None = None) -> Answer:
        t = self.config.t
        # Company word list: "trip" -> also search for "travel". The question
        # itself is never changed; only what KORA searches and scores with.
        search_q, synonyms_used = self.expand(question)
        decision["synonyms"] = synonyms_used
        outcome = self.retriever.retrieve(search_q, top_k=top_k, allowed=allowed)
        if allowed is not None:            # a conflict with a hidden side is not shown at all
            outcome.plan.conflicts = [c for c in outcome.plan.conflicts if all(
                not side.get("document_id") or side["document_id"] in allowed
                for side in self._conflicts.get(c, {}).get("sides", []))]
        # A place named in the question is also matched by its code when
        # choosing sentences and checking coverage (the leave policy says
        # "UK-based employees (LOC-LON)", never "London"). Search is unchanged.
        codes = {e.matched_text: normalise(e.entity_id) for e in outcome.plan.entities
                 if e.entity_type == "location"}
        location_ids = [e.entity_id for e in outcome.plan.entities if e.entity_type == "location"]
        score_q = " ".join([search_q, *dict.fromkeys(location_ids)])
        plan = outcome.plan
        lines: list[AnswerLine] = []
        notes: list[str] = []

        # 1. Structured facts (never cut by max_lines).
        lines += self._conflict_lines(plan, notes)
        lines += self._document_lines(plan, outcome, notes)
        lines += self._change_lines(plan, notes)
        lines += self._checklist_lines(question, plan, notes)
        n_structured = sum(l.kind in STRUCTURED_KINDS for l in lines)
        extra = 1 if plan.conflicts else 0
        limit = max(t("max_lines"), n_structured + extra) if n_structured else t("max_lines")

        # 2. Tone-profile lines -- only when the router chose a country's profile.
        for chunk_id, sentence in self._profile_sentences(question, plan):
            if len(lines) >= limit:
                break
            lines.append(AnswerLine("quote", sentence, chunk_id))

        # 3. Phase 3 approval / escalation rules.
        rule_ids = []
        for line, rule_id in self._rule_lines(search_q, plan):
            if len(lines) >= limit:
                break
            lines.append(line)
            rule_ids.append(rule_id)

        # 4. Quotes from retrieved chunks (table rows get their column names
        # in the detail). A quote from a documented Known Gap
        # brings the whole gap paragraph (verbatim): its later sentences
        # ("handled case-by-case") rarely share words with the question.
        used = {l.text for l in lines}
        for chunk_id, sentence, score, meaning in self._best_sentences(score_q, outcome, used):
            if chunk_id in self._gap_chunks:
                if not any(l.kind == "gap" and l.chunk_id == chunk_id for l in lines):
                    gap_lines = [AnswerLine("gap", s, chunk_id, score=score, semantic=meaning) for s in
                                 split_sentences(self._chunks[chunk_id].chunk_text, t("min_quote_words"))]
                    lines = [l for l in lines if not (l.kind == "quote" and l.chunk_id == chunk_id)] + gap_lines
                    limit = max(limit, sum(l.kind in STRUCTURED_KINDS for l in lines))
                continue
            if len(lines) >= limit:
                break
            header = table_header(self._chunks[chunk_id].chunk_text, sentence) if " | " in sentence else None
            lines.append(AnswerLine("quote", sentence, chunk_id, f"table columns: {header}" if header else "",
                                    score=score, semantic=meaning))

        # 5. Runtime verification, THEN citation numbers (assigned once, in
        # line order, so no number ever has to change afterwards).
        if allowed is not None:            # access control: no line from a hidden document
            lines = [l for l in lines if l.chunk_id is None or self._chunks[l.chunk_id].document_id in allowed]
        lines, dropped = self._verify(lines)
        citations = self._number_citations(lines)
        if dropped:
            notes.append(f"{len(dropped)} candidate line(s) were removed because they could not be "
                         f"verified word-for-word in their cited source.")

        # 6. Status (precision first).
        structured = any(l.kind in STRUCTURED_KINDS for l in lines)
        aliases = dict(synonyms_used)
        for text, code in codes.items():             # "london" is covered where the source says LOC-LON
            aliases[text] = aliases.get(text, []) + [code]
        important, uncovered, unknown = self._coverage(question, citations.values(), aliases)
        strength = max((l.score for l in lines), default=0.0)
        if self._semantic is not None and lines:
            for line, meaning in zip(lines, self._semantic.sentence_scores(question, [l.text for l in lines])):
                line.semantic = max(line.semantic, float(meaning))
        meaning_strength = max((l.semantic for l in lines), default=0.0)
        strong_meaning = self._semantic is not None and meaning_strength >= t("semantic_strong_evidence")
        asked_names = question_names(question)
        # A question word can't explain another question word (found on Q08:
        # "money" ~ "business" 0.55 -- "business" was from "business trip").
        q_words = set(normalise(question).split())
        partners = [self._meaning_partner(w, lines, asked_names, q_words) for w in unknown]
        explained = [m["word"] for m in partners if m["explains"]] if strong_meaning else []
        for m in partners:
            m["explained"] = m["word"] in explained
        decision.update({
            "word_strength": round(strength, 4), "word_strong_needed": t("strong_evidence_score"),
            "meaning_strength": round(meaning_strength, 4) if self._semantic is not None else None,
            "meaning_strong_needed": t("semantic_strong_evidence"),
            "important": list(important), "uncovered": list(uncovered), "unknown": list(unknown),
            "unknown_words": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
                              for m in partners],
            "lines": [{"kind": l.kind, "words": round(l.score, 4), "meaning": round(l.semantic, 4)}
                      for l in lines],
        })
        if explained:
            # A word the knowledge base NEVER uses ("trip") is a paraphrase only
            # when the quoted lines say it in other words: the answer matches
            # strongly by meaning AND that specific word has a close match in
            # the quoted lines ("travel"). A good overall match is not enough
            # -- found in testing: "pets" was excused by an "office" sentence.
            # Words that exist elsewhere but not in the quoted source
            # ("contractors") are never excused.
            notes.append(f"Matched by meaning: the sources never use the word(s) {', '.join(explained)}; "
                         f"the quoted lines say it in other words (meaning match {meaning_strength:.2f}).")
            important = [w for w in important if w not in explained]
            uncovered = [w for w in uncovered if w not in explained]
            unknown = [w for w in unknown if w not in explained]
        if not lines:
            why = "No sentence in the retrieved sources answers this question."
            decision.update(rule="no_lines", rule_text=why)
            return self._not_found(question, outcome, why, uncovered, dropped, start)
        if not structured and important and len(uncovered) == len(important):
            decision.update(rule="no_main_word_covered",
                            rule_text="none of the main words appear in the closest sources")
            return self._not_found(question, outcome, f"None of the main words of the question appear in "
                                   f"the closest sources ({', '.join(uncovered)}).", uncovered, dropped, start)
        covered_share = (len(important) - len(uncovered)) / len(important) if important else 1.0
        weak_words = strength < t("strong_evidence_score")
        weak = weak_words and not strong_meaning
        # Unknown words that meaning could NOT explain: only word evidence
        # counts, so a context match ("office") can't rescue "pets".
        if not structured and unknown and important:
            # ONE simple rule: if the question uses a word the documents NEVER
            # use ("pets", "canteen", "Madrid", "room"), KORA does not guess --
            # an answer about the other words would not be about that. Found on
            # held-out v4: "leave days in the Madrid office" quoted other
            # offices' leave rules as "partial". (Replaces the older share /
            # weak-evidence rule.) If the word means something the documents do
            # say ("boss" -> "manager"), the fix is one line in the company word
            # list, and the "why" view names the word.
            decision.update(rule="unknown_subject", rule_text=(
                f"the documents never use: {', '.join(unknown)} (if a word means something the documents "
                f"do say, add it to the company word list in the config)"))
            return self._not_found(question, outcome, f"The knowledge base never mentions "
                                   f"{', '.join(unknown)}.", uncovered, dropped, start)
        if not structured and weak and important:
            # Weak evidence: more than half of the main words must appear
            # TOGETHER in one quoted line (with its heading), not scattered
            # over lines or whole documents. Found on UT's own questions:
            # "What is my salary structure?" was 'answered' from an org-chart
            # line ("structure") + an insurance line that happened to say "salary".
            def share(line):
                text = normalise(f"{(self._chunks[line.chunk_id].section or '').split(' > ')[-1]} {line.text}")
                _, missing, _ = self._coverage(question, [], aliases, texts=[text])
                return (len(important) - len([w for w in missing if w in important])) / len(important)
            line_share = max((share(l) for l in lines if l.chunk_id in self._chunks), default=0.0)
            if line_share <= 0.5:
                decision.update(rule="weak_and_scattered",
                                rule_text=f"weak evidence; the best quoted line holds only {line_share:.0%} "
                                          f"of the main words")
                return self._not_found(question, outcome, f"The closest sources match only weakly and don't "
                                       f"answer this directly.", uncovered, dropped, start)
        if not structured and weak and important and covered_share <= 0.5:
            # Weak evidence that covers at most half of the question
            # ("stock options" -> inventory "safety stock"): abstain.
            decision.update(rule="weak_and_half_covered",
                            rule_text=f"weak evidence covering {covered_share:.0%} of the main words")
            return self._not_found(question, outcome, f"The closest sources match only weakly and don't "
                                   f"mention {', '.join(uncovered)}.", uncovered, dropped, start)

        # 7. Notes.
        quoted_chunks = {c.chunk_id for c in citations.values()}
        quoted_docs = {c.document_id for c in citations.values()}
        evidence_chunks = quoted_chunks | {r.chunk.chunk_id for r in self._quotable(outcome.results, search_q)}
        if synonyms_used and lines:
            notes.append("Searched with the company word list: " + "; ".join(
                f"{k} -> {', '.join(v)}" for k, v in synonyms_used.items()) + ".")
        notes += [n for n in self._gap_notes(evidence_chunks) if n not in notes]
        notes += [n.note for n in outcome.history_notes if n.document_id in quoted_docs]
        if self._relocation_disclaimer and quoted_docs & self._relocation_docs:
            notes.append(self._relocation_disclaimer)
        comm = self.config.communication
        if comm and comm.note and ("communication" in plan.intents or quoted_docs & set(comm.profiles.values())):
            notes.append(comm.note)
        archived = [c for c in citations.values() if c.status.lower() == "archived"]
        if archived and not plan.conflicts:
            notes.append("Some quoted sources are ARCHIVED (" + ", ".join(
                f"[{c.number}] {c.document_id} v{c.version}" for c in archived) +
                "): they describe a past rule, not the current one.")
        notes += self._metadata_notes(citations.values())
        if uncovered and not structured:
            notes.append(f"Not covered by the quoted sources: {', '.join(uncovered)}. The answer above "
                         f"may not apply to that part of your question.")

        if plan.conflicts and any(l.kind == "conflict_side" for l in lines):
            status = "conflict"
            decision.update(rule="conflict", rule_text="the sources disagree; both sides shown")
        elif uncovered and not structured:
            status = "partial"
            decision.update(rule="partly_covered", rule_text=f"not covered: {', '.join(uncovered)}")
        else:
            status = "answered"
            decision.update(rule="covered", rule_text="every main word is covered by the quoted sources"
                            if not structured else "structured facts from Phase 3 analysis")
        elapsed = (time.perf_counter() - start) * 1000
        return Answer(question, status, lines, list(citations.values()), notes, uncovered,
                      rule_ids, dropped, outcome, elapsed)

    def _not_found(self, question, outcome, why, uncovered, dropped, start) -> Answer:
        citations: dict[str, Citation] = {}
        for r in outcome.results[:3]:
            if r.chunk.chunk_id not in citations:
                citations[r.chunk.chunk_id] = self._citation(len(citations) + 1, r.chunk.chunk_id)
        elapsed = (time.perf_counter() - start) * 1000
        return Answer(question, "not_found", [], list(citations.values()),
                      [why + " It may not be documented."], uncovered, [], dropped, outcome, elapsed)

    # ---------------- verification and citations ----------------

    def verify_line(self, line: AnswerLine) -> bool:
        """Is this line found word-for-word in its source chunk? (Letters and
        digits compared, so Markdown and line breaks don't matter.) A conflict
        side without a chunk (employees.csv vs org chart) is shown as Phase 3
        analysis and is the only line allowed without a source chunk."""
        if line.chunk_id is None:
            return line.kind == "conflict_side"
        if line.chunk_id.startswith(("header:", "record:", "file:")):     # Phase 5 graph evidence
            return self.graph is not None and line.kind in ("relation", "possible") and \
                self.graph.verify(line.chunk_id, line.text)
        if line.chunk_id not in self._chunks:
            return False
        if line.kind == "document":
            return self._chunks[line.chunk_id].document_id in line.text
        text = line.text.split(" now: ", 1)[-1] if line.kind == "change" else line.text
        key = normalise(text)
        return bool(key) and key in self._chunk_norm[line.chunk_id]

    def _verify(self, lines: list[AnswerLine]) -> tuple[list[AnswerLine], list[dict]]:
        kept, dropped = [], []
        for line in lines:
            if self.verify_line(line):
                kept.append(line)
            else:
                dropped.append({"kind": line.kind, "text": line.text, "chunk_id": line.chunk_id})
        return kept, dropped

    def _citation(self, number: int, chunk_id: str) -> Citation:
        if chunk_id.startswith(("header:", "record:", "file:")):
            return self._graph_citation(number, chunk_id)
        c = self._chunks[chunk_id]
        meta = self.retriever.documents.get((c.document_id, c.version))
        return Citation(number, chunk_id, c.document_id, c.document_title, c.version, c.status,
                        getattr(meta, "effective_date", "unknown"), c.section,
                        getattr(meta, "source_filename", "unknown"), c.chunk_text[:200],
                        getattr(meta, "id_source", ""), getattr(meta, "title_source", ""))

    def _graph_citation(self, number: int, chunk_id: str) -> Citation:
        d = self.graph.describe(chunk_id)
        if d["kind"] == "record":
            return Citation(number, chunk_id, d["file"], f"Company record ({d['file']})", "n/a", "n/a", "n/a",
                            d["section"], d["file"], "", kind="record")
        meta = next((m for m in self.retriever.documents.values()
                     if getattr(m, "source_filename", "").lower() == d["file"].lower()), None)
        if meta is None:
            return Citation(number, chunk_id, d["file"], d["file"], "unknown", "unknown", "unknown",
                            d["section"], d["file"], "", kind=d["kind"])
        return Citation(number, chunk_id, meta.document_id, meta.title, meta.version, meta.status,
                        getattr(meta, "effective_date", "unknown"), d["section"], meta.source_filename, "",
                        getattr(meta, "id_source", ""), getattr(meta, "title_source", ""), kind=d["kind"])

    # ---------------- Phase 5: a graph answer as a KORA answer ----------------

    def _from_graph(self, g, phase4: "Answer", decision: dict) -> "Answer":
        """Turn a GraphAnswer into an Answer: same citations, same runtime check,
        same notes. Phase 4's own answer is kept when it found a CONFLICT (KORA
        never hides a conflict behind a graph fact); the graph facts are added."""
        lines = []
        for kind, facts in (("relation", g.facts), ("possible", g.possible)):
            for f in facts:
                gl = self.graph.line(f, kind)
                if gl:
                    lines.append(AnswerLine(kind, gl.text, gl.chunk_id, gl.claim))
        kept, dropped = self._verify(lines)
        decision.update({"phase4_status": phase4.status, "phase4_rule": decision.get("rule"),
                         "graph": {"kind": g.kind, "subjects": g.subjects,
                                   "facts": sum(l.kind == "relation" for l in kept),
                                   "possible": sum(l.kind == "possible" for l in kept)}})
        if not any(l.kind == "relation" for l in kept):          # nothing survived the check: Phase 4 stands
            decision["graph"]["used"] = False
            phase4.dropped_lines += dropped
            return phase4
        decision["graph"]["used"] = True
        notes = list(g.notes)
        if any(f.historical for f in g.facts):
            notes.append("Some of this comes from an archived (older) version; it is marked as history.")
        status = g.status
        if phase4.status == "conflict":
            kept = [l for l in phase4.lines if l.kind == "conflict_side"] + kept
            notes = phase4.notes + notes
            status = "conflict"
        for l in kept:
            l.citation = None
        citations = self._number_citations(kept)
        decision["rule"] = "graph_" + g.kind
        decision["rule_text"] = {"answered": "documented links in the knowledge graph answer the question",
                                 "partial": "the knowledge graph documents part of the answer (see notes)",
                                 "conflict": "the sources conflict; graph links shown alongside"}[status]
        return Answer(phase4.question, status, kept, list(citations.values()), notes, [], [],
                      phase4.dropped_lines + dropped, phase4.retrieval, phase4.elapsed_ms)

    def _number_citations(self, lines: list[AnswerLine]) -> dict[str, Citation]:
        """Number sources in the order lines use them; fill {ref} in details."""
        citations: dict[str, Citation] = {}

        def number(chunk_id):
            if chunk_id not in citations:
                citations[chunk_id] = self._citation(len(citations) + 1, chunk_id)
            return citations[chunk_id].number

        for line in lines:
            if line.chunk_id:
                line.citation = number(line.chunk_id)
            if "{ref}" in line.detail:
                ref = f"[{number(line.ref_chunk_id)}]" if line.ref_chunk_id in self._chunks else "[not located]"
                line.detail = line.detail.replace("{ref}", ref)
        return citations

    # ---------------- structured facts ----------------

    def _conflict_lines(self, plan, notes) -> list[AnswerLine]:
        lines = []
        for conflict_id in plan.conflicts:
            conflict = self._conflicts.get(conflict_id)
            if not conflict:
                continue
            for side in conflict["sides"]:
                chunk_ids = side.get("chunk_ids") or []
                where = (f"{side['document_id']} v{side['version']}, {side['status']}, "
                         f"effective {side['effective_date']}") if side.get("document_id") else "Phase 3 analysis"
                lines.append(AnswerLine("conflict_side", side["statement"],
                                        chunk_ids[0] if chunk_ids else None, where))
            meaning = STATUS_MEANING.get(conflict["status"], conflict["status"])
            notes.append(f"Conflict {conflict_id} ({conflict['conflict_type']}): sources disagree on "
                         f"'{conflict['topic']}'. Status: {conflict['status']}, i.e. {meaning}. "
                         f"KORA shows every side and does not pick one.")
            if conflict.get("detail"):
                notes.append(f"{conflict_id}: {conflict['detail']}")
        return lines

    def _first_chunk(self, document_id: str, include_archived: bool) -> str | None:
        """First chunk of a document's current version (archived only if allowed)."""
        chunks = [c for c in self._chunks.values() if c.document_id == document_id
                  and (include_archived or c.status.lower() != "archived")]
        if not chunks:
            return None
        chunks.sort(key=lambda c: (c.status.lower() == "archived", c.chunk_index))
        return chunks[0].chunk_id

    def _document_lines(self, plan, outcome, notes) -> list[AnswerLine]:
        """'Which documents describe X?': list documents from the topic
        catalogue (how each is linked, with Phase 3 confidence). Without a
        matched topic, the documents found by search are listed instead and
        labelled as such."""
        if "document_listing" not in plan.intents:
            return []
        entries: dict[str, tuple[float, str]] = {}
        related: dict[str, set[str]] = {}        # how linked -> document IDs (shown in a note)
        for topic_id in plan.topics:
            for d in self._topics.get(topic_id, {}).get("documents", []):
                conf = float(d.get("confidence") or 0)
                if conf < self.config.t("document_list_min_confidence"):
                    related.setdefault(str(d.get("detail") or d.get("method")), set()).add(d["document_id"])
                    continue
                how = f"{d.get('method')}: {d.get('detail')}, confidence {d.get('confidence')}"
                if d["document_id"] not in entries or entries[d["document_id"]][0] < conf:
                    entries[d["document_id"]] = (conf, how)
        source = "topic catalogue"
        if not entries:
            source = "search results"
            for r in self._quotable(outcome.results):
                entries.setdefault(r.chunk.document_id, (r.score, f"found by search (score {r.score:.2f})"))
        lines = []
        for doc_id, (conf, how) in sorted(entries.items(), key=lambda kv: (-kv[1][0], kv[0])):
            chunk_id = self._first_chunk(doc_id, plan.include_archived)
            if chunk_id is None:
                continue
            chunk = self._chunks[chunk_id]
            lines.append(AnswerLine("document", f"{chunk.document_title} ({doc_id})", chunk_id, how))
        if lines:
            notes.append(f"Documents listed from the {source}"
                         + (" (Phase 3 links, strongest first)." if source == "topic catalogue" else
                            ": no topic matched, so this list is less certain."))
            groups = [f"{how}: {', '.join(sorted(ids - set(entries)))}" for how, ids in sorted(related.items())
                      if ids - set(entries)]
            if groups:
                notes.append("Related documents that don't describe it themselves -- " + "; ".join(groups) + ".")
        return lines

    def _find_chunk(self, document_id: str, text: str, version: str | None = None) -> str | None:
        """The chunk of DOCUMENT_ID (optionally one version) containing TEXT."""
        key = normalise(text)
        probes = [p for p in (key[:50], key[-50:]) if len(p) >= 12] or ([key] if key else [])
        for c in sorted(self._chunks.values(), key=lambda c: c.chunk_index):
            if c.document_id != document_id or (version and c.version != version):
                continue
            if any(p in self._chunk_norm[c.chunk_id] for p in probes):
                return c.chunk_id
        return None

    def _change_lines(self, plan, notes) -> list[AnswerLine]:
        """Version questions: Phase 3's step-by-step changes between an archived
        document and its replacement -- only for documents the ROUTER chose
        (fix: retrieved noise used to trigger changes of unrelated documents)."""
        if "version" not in plan.intents:
            return []
        relevant = set(plan.candidate_documents) | set(plan.direct_documents)
        lines = []
        for old in self._superseded:
            if not old.get("changes"):
                continue
            old_header = self._header_id.get(old["document_id"], old["document_id"])
            for new in old.get("superseded_by", []):
                if not ({old["document_id"], old_header, new["document_id"]} & relevant):
                    continue
                notes.append(f"What changed from {old['document_id']} v{old['version']} "
                             f"(effective {old['effective_date']}, archived) to {new['document_id']} "
                             f"v{new['version']} (effective {new['effective_date']}, current), step by step, "
                             f"as compared by Phase 3.")
                for ch in old["changes"]:
                    after_id = self._find_chunk(new["document_id"], ch["after"], new["version"])
                    before_id = self._find_chunk(old_header, ch["before"], old["version"])
                    detail = f"before (v{old['version']}): \"{ch['before']}\" "
                    detail += "{ref}" if before_id else "[old text too short to locate]"
                    if ch.get("roles_added"):
                        detail += f"; roles added: {', '.join(ch['roles_added'])}"
                    if ch.get("roles_removed"):
                        detail += f"; roles removed: {', '.join(ch['roles_removed'])}"
                    lines.append(AnswerLine("change", f"Step {ch['step']} now: {ch['after']}",
                                            after_id, detail, ref_chunk_id=before_id))
        return lines

    def _checklist_lines(self, question, plan, notes) -> list[AnswerLine]:
        """Relocation questions with a time window ('first week') list the
        checklist items due in that window, each citing its source sentence."""
        if "relocation" not in plan.intents or not self._checklist_items:
            return []
        window = time_window(normalise(question))
        if window is None:
            return []
        lines = []
        for item in self._checklist_items:
            due = [p for p in item.get("pace", []) if p.get("due_by_day") is not None
                   and p["due_by_day"] <= window]
            if not due:
                continue
            pace = min(due, key=lambda p: p["due_by_day"])
            chunk_id = self._find_chunk(pace["document_id"], pace["statement"])
            detail = f"by day {pace['due_by_day']}; checklist item '{item['title']}'"
            contacts = [f"{c['name']} ({c['location_qualifier']})" if c.get("location_qualifier") else c["name"]
                        for c in item.get("nexora_contacts", [])]
            if contacts:
                detail += f"; contact: {', '.join(contacts)}"
            lines.append(AnswerLine("checklist", pace["statement"], chunk_id, detail))
            if item.get("official_guidance_note"):
                notes.append(f"{item['title']}: {item['official_guidance_note']}")
        if lines:
            notes.insert(0, f"Checklist items due within the first {window} days "
                            f"({self.config.relocation.name}). Nothing is marked done: "
                            f"progress only counts when you report it.")
        return lines

    # ---------------- quotes and rules ----------------

    def _profile_sentences(self, question, plan) -> list[tuple[str, str]]:
        """Lines from the tone profile(s) the router chose for a named country.
        A line like "Feedback: Tends to be specific..." comes first when its
        label shares a word with the question; others follow in document order."""
        if not plan.tone_profiles:
            return []
        profiles = set(plan.tone_profiles)
        stems = {_stem(w) for w in normalise(question).split() if len(w) >= 4}
        labelled, others = [], []
        min_words = self.config.t("min_quote_words")
        for c in sorted(self._chunks.values(), key=lambda c: (c.document_id, c.chunk_index)):
            if c.document_id not in profiles or c.status.lower() == "archived":
                continue
            for sentence in split_sentences(c.chunk_text, min_words):
                label, sep, _ = sentence.partition(":")
                label_words = normalise(label).split() if sep and len(label) <= 40 else []
                if not label_words:
                    continue          # intro / reminder sentences: covered by the communication note
                hit = any(w.startswith(st) or st.startswith(w) for w in label_words for st in stems
                          if len(w) >= 4)
                (labelled if hit else others).append((c.chunk_id, sentence))
        return labelled + others

    def _render_rule(self, rule: dict, chunk_id: str) -> tuple[str, str] | None:
        """The rule as it appears in its chunk. Phase 3 rebuilds table rows as
        sentences ("X: approved by Y"); those are shown as the original table
        row instead, so the line is verbatim. None if neither can be found."""
        chunk_norm = self._chunk_norm[chunk_id]
        if normalise(rule["statement"]) in chunk_norm:
            return rule["statement"], ""
        first_cell = normalise(rule["statement"].split(":", 1)[0])
        for row in self._chunks[chunk_id].chunk_text.splitlines():
            if not row.strip().startswith("|"):
                continue
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if cells and first_cell and normalise(cells[0]) == first_cell:
                return " | ".join(cells), "from a table row"
        return None

    def _rule_lines(self, question, plan):
        """Phase 3 approval/escalation rules matching the question's SUBJECT.
        Archived rules only when the router allowed archived documents."""
        t = self.config.t
        out = []
        query = self._vectorizer.transform([_RULE_NOISE.sub(" ", question)])
        for kind in ("approval", "escalation"):
            if kind not in plan.intents or self._rule_matrix[kind] is None:
                continue
            scores = cosine_similarity(query, self._rule_matrix[kind])[0]
            for score, rule in sorted(zip(scores, self._rules[kind]), key=lambda p: -p[0]):
                if score < t("rule_min_score") or len(out) >= t("max_rules"):
                    break
                if (rule.get("historical") and not plan.include_archived) or not rule.get("chunk_ids"):
                    continue
                rendered = self._render_rule(rule, rule["chunk_ids"][0]) if rule["chunk_ids"][0] in self._chunks else None
                if rendered is None:
                    continue
                text, origin = rendered
                who = ", ".join(self._role_names.get(r, r) for r in rule.get("role_ids", [])) \
                    or ("no approval needed" if rule.get("no_approval_required") else "role not named")
                detail = (f"{'approver' if kind == 'approval' else 'escalates to'}: {who}; "
                          f"Phase 3 rule {rule['rule_id']}, confidence {rule.get('confidence')}")
                if origin:
                    detail += f"; {origin}"
                out.append((AnswerLine("rule", text, rule["chunk_ids"][0], detail), rule["rule_id"]))
        return out

    def _quotable(self, results, question: str | None = None):
        """Which retrieved chunks may be QUOTED. If the router aimed anywhere
        (conflict sides, named/direct documents, candidate documents), only
        those chunks. Otherwise only whole-corpus chunks scoring at least
        unrouted_quote_ratio of the best one."""
        t = self.config.t
        semantic_on = self._semantic is not None

        def clear_meaning(r):
            return semantic_on and r.semantic >= t("semantic_quote_min")

        unrouted = ("tfidf", "semantic")
        targeted = [r for r in results if r.matched_by in ("conflict", "from_document", "routed")]
        if targeted:
            # The whole-corpus slots exist for router mistakes: a chunk there
            # that scores HIGHER than everything the router chose may be
            # quoted too (bug found on held-out data: "legacy NexFlow" routed
            # to the NexFlow product page, 0.09, while the answer -- the KG-02
            # gap in the bug process -- scored 0.30 and could not be quoted).
            # With embeddings the same holds for a clearly better MEANING match.
            others = [r for r in targeted if r.matched_by != "conflict"]
            best_words = max((r.score for r in others), default=1.0)
            best_meaning = max((r.semantic for r in others), default=1.0)
            return targeted + [r for r in results if r.matched_by in unrouted and
                               (r.score > best_words or (clear_meaning(r) and r.semantic > best_meaning))]
        scored = [r for r in results if r.matched_by in unrouted]
        if not scored:
            return results
        best = max((r.score for r in scored if r.matched_by == "tfidf"), default=0.0)
        keep = [r for r in scored if (r.matched_by == "tfidf" and r.score >= t("unrouted_quote_ratio") * best)
                or clear_meaning(r)]
        if question:
            # A chunk can score low overall and still hold THE sentence
            # (bug found on Q37: "Chat -- default for quick questions" lost to
            # a tone page that only shared "colleague"). Such a chunk may be
            # quoted when one sentence covers at least 2 of the question's
            # main words and more of them than any sentence already allowed.
            words = {_stem(w) for w in subject_words(question).split() if len(w) >= 3}
            best_kept = max((self._sentence_cover(r, words) for r in keep), default=0)
            keep += [r for r in scored if r not in keep and
                     self._sentence_cover(r, words) >= 2 and self._sentence_cover(r, words) > best_kept]
        return keep

    def _sentence_cover(self, result, stems: set[str]) -> int:
        """Most main question words found in any ONE sentence of the chunk."""
        best = 0
        section = result.chunk.section or ""         # a sentence under "## Channels" is about channels
        for sentence in split_sentences(result.chunk.chunk_text, self.config.t("min_quote_words")):
            text = normalise(f"{section} {sentence}")
            best = max(best, sum(1 for st in stems if re.search(r"\b" + re.escape(st), text)))
        return best

    def _best_sentences(self, question, outcome, already_used):
        """Sentences from the quotable chunks, most similar to the question
        first; repeats skipped; at most max_lines_per_chunk per chunk."""
        t = self.config.t
        candidates = [(r.chunk.chunk_id, s) for r in self._quotable(outcome.results, question)
                      for s in split_sentences(r.chunk.chunk_text, t("min_quote_words"))]
        if not candidates:
            return []
        # Each sentence is scored together with its section heading: a bullet
        # under "## Annual Leave" is about annual leave even if it only says
        # "25 days/year". (The quote itself is the sentence alone.)
        def heading(chunk_id):
            return (self._chunks[chunk_id].section or "").split(" > ")[-1]
        words = cosine_similarity(self._vectorizer.transform([subject_words(question)]),
                                  self._vectorizer.transform([f"{heading(c)} {s}" for c, s in candidates]))[0]
        meaning = self._semantic.sentence_scores(question, [s for _, s in candidates]) \
            if self._semantic is not None else [0.0] * len(candidates)
        best_words = float(max(words)) if len(words) else 0.0

        def by_words(i):        # eligible by words, and not much weaker than the best word match
            return words[i] >= t("sentence_min_score") and words[i] >= t("quote_relative_floor") * best_words

        def by_meaning(i):      # eligible by a clear meaning match
            return self._semantic is not None and meaning[i] >= t("semantic_quote_min")

        word_rank = [i for i in sorted(range(len(candidates)), key=lambda i: -words[i]) if by_words(i)]
        meaning_rank = [i for i in sorted(range(len(candidates)), key=lambda i: -meaning[i]) if by_meaning(i)]
        from src.retrieval.embeddings import rrf
        order = rrf([word_rank, meaning_rank])     # positions, never raw scores of different scales
        seen_text = {normalise(x) for x in already_used}
        per_chunk: dict[str, int] = {}
        out = []
        for i in order:
            chunk_id, sentence = candidates[i]
            key = normalise(sentence)
            repeat = any(key == s or (len(s) > 30 and s in key) or (len(key) > 30 and key in s) for s in seen_text)
            if repeat or per_chunk.get(chunk_id, 0) >= t("max_lines_per_chunk"):
                continue
            seen_text.add(key)
            per_chunk[chunk_id] = per_chunk.get(chunk_id, 0) + 1
            out.append((chunk_id, sentence, float(words[i]), float(meaning[i])))
        return out

    # ---------------- notes and coverage ----------------

    def _explained_by_meaning(self, word: str, lines: list[AnswerLine],
                              asked_names: set[str] = frozenset()) -> bool:
        return self._meaning_partner(word, lines, asked_names)["explains"]

    def _meaning_partner(self, word: str, lines: list[AnswerLine],
                         asked_names: set[str] = frozenset(), question_words: set[str] = frozenset()) -> dict:
        """Does an unknown question word have a close meaning match with a
        (non-generic) word in the quoted lines?

        NAMES are never matched by meaning, on either side: a name can't be
        paraphrased. "Paris" is close in meaning to "Berlin" (both cities), so
        without this rule "vacation days in the Paris office" could be
        answered with another office's rules. A name in the question
        ("Paris") is never explained; a lowercase "paris" can't be explained
        by a name in the sources ("Berlin", "Munich") either."""
        needed = self.config.t("semantic_word_min")
        out = {"word": word, "best": None, "score": 0.0, "needed": needed, "blocked": None, "explains": False}
        if word in asked_names:
            out["blocked"] = "a name in the question"
            return out
        if self._semantic is None:
            out["blocked"] = "embeddings off"
            return out
        words = sorted({w for l in lines for w in normalise(l.text).split()
                        if len(w) >= 3 and w not in GENERIC_WORDS and not w.isdigit()
                        and w not in self._kb_names and w not in question_words})
        if not words:
            return out
        scores = self._semantic.sentence_scores(word, words)
        i = int(scores.argmax())
        out.update(best=words[i], score=float(scores[i]), explains=float(scores[i]) >= needed)
        return out

    def _gap_notes(self, chunk_ids: set[str]) -> list[str]:
        notes = []
        for gap in self._gaps:
            gap_chunks = {cid for e in gap.get("evidence", []) for cid in e.get("chunk_ids", [])}
            if gap_chunks & chunk_ids:
                notes.append(f"Known gap {gap['gap_id']} ({gap['severity']}): {gap['description']}")
        return notes

    def _metadata_notes(self, citations) -> list[str]:
        """Precision first: say when a source doesn't state what a reader would
        assume (status, version), or when KORA had to derive its ID/title."""
        notes = []
        for c in citations:
            missing = [what for what, v in (("status", c.status), ("version", c.version)) if v.lower() == "unknown"]
            if missing:
                notes.append(f"[{c.number}] {c.document_id} does not state its {' or '.join(missing)}, "
                             f"so KORA cannot confirm it is the current version.")
            if c.id_source == "derived_from_path":
                notes.append(f"[{c.number}] has no document ID in its header; {c.document_id} was derived "
                             f"from its file path ({c.source_filename}).")
        return notes

    def _known_to_kb(self, stem: str) -> bool:
        """Does any word in the whole knowledge base start with STEM?"""
        import bisect
        i = bisect.bisect_left(self._kb_words, stem)
        return i < len(self._kb_words) and self._kb_words[i].startswith(stem)

    def _coverage(self, question: str, citations, synonyms_used: dict[str, list[str]] | None = None,
                  texts: list[str] | None = None, skip: set | None = None) -> tuple[list[str], list[str], list[str]]:
        """(important words, those no QUOTED document mentions, those the
        whole knowledge base never mentions). Whole documents are checked,
        not just the quoted chunk."""
        keys = [(c.document_id, c.version) for c in citations]
        texts_given = texts is not None
        if not texts_given:
            texts = [self._document_text.get(k, "") for k in keys]
        raws = texts if texts_given else [self._document_raw.get(k, "") for k in keys]
        skip = GENERIC_WORDS if skip is None else skip
        important, uncovered, unknown = [], [], []
        # Hyphenated words are one idea: "part-time" must appear as such
        # (its parts "part" and "time" are generic and would vanish).
        compounds = list(dict.fromkeys(normalise(c) for c in _COMPOUND.findall(question)))
        in_compound = {w for c in compounds for w in c.split()}
        for phrase in compounds:
            important.append(phrase)
            if not any(re.search(r"\b" + re.escape(phrase) + r"\b", t) for t in texts):
                uncovered.append(phrase)
                if not any(re.search(r"\b" + re.escape(phrase) + r"\b", t) for t in self._document_text.values()):
                    unknown.append(phrase)
        for word in dict.fromkeys(normalise(question).split()):
            if word in in_compound:
                continue
            if word in _SYMBOL_WORDS:
                symbol = _SYMBOL_WORDS[word]
                important.append(word)
                if texts and not any(symbol in raw or re.search(r"\b" + word, t) for raw, t in zip(raws, texts)):
                    uncovered.append(word)
                continue
            if len(word) < 3 or word in skip or word.isdigit():
                continue
            important.append(word)
            stem = _stem(word)
            # A word from a company-word-list phrase ("trip") counts as covered
            # / known through the words the documents use ("travel").
            # Whole phrases: "loc lon" must appear as such, not "loc..." anywhere.
            alternatives = [_stem_phrase(alt) for phrase, alts in (synonyms_used or {}).items()
                            if word in phrase.split() for alt in alts]
            if not any(re.search(r"\b" + re.escape(st), t) for st in [stem, *alternatives] for t in texts):
                uncovered.append(word)
                if not any(all(self._known_to_kb(w) for w in st.split()) for st in [stem, *alternatives]):
                    unknown.append(word)
        if not important and skip is GENERIC_WORDS:
            # Every word was "generic" ("What is the procedure to change the
            # team?"): nothing would be checked and anything could pass (found
            # on UT's questions). Check the plain content words instead.
            return self._coverage(question, citations, synonyms_used, texts if texts_given else None,
                                  set(ENGLISH_STOP_WORDS))
        return important, uncovered, unknown

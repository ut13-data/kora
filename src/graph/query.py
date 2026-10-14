"""
Phase 5 M2 - relationship questions answered from the knowledge graph.

    q = GraphQueries(view, tracer, intelligence_dir)
    answer = q.answer("Which teams are involved in customer onboarding?")   # GraphAnswer or None

None means "not a relationship question I can recognise": the caller (M3:
the normal answer flow) handles it exactly as in Phase 4. The graph never
guesses: no recognised intent AND entity -> None.

Every statement is built from the link's VERIFIED evidence (trace.py) and
phrased by what that evidence actually says:

    tier     example evidence                                 KORA says
    quote    "The Implementation Manager provides final       Implementation Manager approves Customer
              go-live approval ..."                           Onboarding  + the quote
    header   "Related Processes: PROC-IT-002 (Data Security)" Incident Management lists Data Security
                                                              Policy as related  (NOT "is governed by")
    record   processes.csv, row PROC-FIN-002: owner_role =    Finance Manager is the recorded owner of
             Finance Manager / Operations Manager             Procurement Process

Only quote / header / record are FACTS. Mentions and KORA's own analysis
are listed separately as "possibly related", never as facts.

Status:
    answered   documented facts answer the question
    partial    only part is documented (e.g. a "why delayed" question: the
               documents can name known causes, never the cause of one case)
    not_found  the subject is known but nothing documented answers it
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.graph.resolve import Match, Resolver, tokens
from src.graph.schema import DOCUMENTED, QUOTE, TIER_LABELS
from src.graph.trace import Tracer
from src.intelligence.graph import IN, OUT
from src.intelligence.text import norm_key

# ---------------- intents (checked in this order; first match wins) ----------------
HISTORY = re.compile(r"\b(before (19|20)\d\d|previous(ly)?|old(er)? version|earlier version|used to|"
                     r"in the past|originally|archived)\b")
INTENTS = [
    ("delay", re.compile(r"\bwhy\b.*\b(delay\w*|late|slow\w*|stuck|block\w*|inconsistent\w*|taking so long|"
                         r"held up|not (moving|working))\b")),
    ("impact", re.compile(r"\b(affect\w*|impact\w*|break\w*|depend\w* on)\b")),
    ("head_of_owner", re.compile(r"\b(head|heads|leads|runs)\b.*\b(department|team)\b.*\bown")),
    ("involved", re.compile(r"\binvolved\b|\btakes? part\b|\bparticipat\w*|\bpart of\b|"
                            r"\bwhich (teams|roles|people)\b|\bwho works on\b")),
    ("policies", re.compile(r"\bgovern\w*|\bwhich polic(y|ies)\b|\bpolic(y|ies) (apply|applies)\b")),
    ("which_department", re.compile(r"\b(which|what) (department|team)s?\b")),
    ("approver", re.compile(r"\bapprov\w*|\bsign[- ]?off\b|\bsigns off\b")),
    ("owner", re.compile(r"\bown(s|er|ers|ed)?\b|\bresponsible for\b|\bin charge of\b")),
    ("next", re.compile(r"\bwhat (comes|happens|starts) (after|next)\b|\bstarts? when\b|\btrigger\w*|"
                        r"\bwhich process (starts|follows|comes)\b|\bnext (step|process)\b|\bcomes? after\b")),
    ("uses", re.compile(r"\buse[sd]?\b|\busing\b")),
    ("about", re.compile(r"\bhead\b|\bwho (is|leads|runs)\b|\bwhat does\b|\bwhich (products|processes|roles)\b|"
                         r"\bbelong\w*\b")),
]
# Left to Phase 4 on purpose (it does them better): approval questions (its rule engine knows
# amounts and thresholds: "a 20% discount" -> Sales Manager only) unless they ask about an OLD
# version, and "what does a <role> do" (Phase 4 quotes the role's own description).
_ROLE_ABOUT = re.compile(r"\bwhich (process|processes)\b|\bdepartment\b|\bteam\b|\bbelong")

# Conflicts about document IDs say nothing about how a process runs: not shown with "why delayed".
HOUSEKEEPING_CONFLICTS = {"identifier"}
NO_LIVE_STATUS = ("KORA has no live status information, so it can't say why a particular case is delayed. "
                  "Below is what the documents say about this process's dependencies and known problems.")
NO_IMPACT_TEXT = ("No document describes what would happen if this changed. These are the documented "
                  "dependencies and cross-references, so the owners can check each one.")


@dataclass
class GraphFact:
    claim: str                       # KORA's short statement; its verb matches the evidence
    tier: str                        # quote | header | record | mention | derived
    citation: dict                   # the evidence a user sees (trace.Provenance.citation)
    relationship_id: str | None = None
    entities: list[str] = field(default_factory=list)
    historical: bool = False
    group: str = ""                  # e.g. the department a role belongs to

    @property
    def documented(self) -> bool:
        return self.tier in DOCUMENTED

    def to_dict(self) -> dict:
        return {"claim": self.claim, "tier": self.tier, "citation": self.citation,
                "relationship_id": self.relationship_id, "entities": self.entities,
                "historical": self.historical, "group": self.group}


@dataclass
class GraphAnswer:
    question: str
    kind: str                        # the intent that answered
    subjects: list[str]              # entity keys the question is about
    status: str                      # answered | partial | not_found
    facts: list[GraphFact] = field(default_factory=list)       # documented
    possible: list[GraphFact] = field(default_factory=list)    # mentioned / derived / cross-referenced
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        """Everything a user would read (for evaluation: names_found, leaks)."""
        parts = [f.claim + " " + str(f.citation.get("text", "")) for f in self.facts + self.possible]
        return " ".join(parts + self.notes)

    def render(self) -> str:
        out = [f"Q: {self.question}", f"Status: {self.status}  (graph: {self.kind})", ""]
        groups: dict[str, list[GraphFact]] = {}
        for f in self.facts:
            groups.setdefault(f.group, []).append(f)
        for group, facts in groups.items():
            if group:
                out.append(f"  {group}:")
            for f in facts:
                out.append(f"  - {f.claim}{'  [history]' if f.historical else ''}")
                if f.citation.get("text"):
                    out.append(f"      {TIER_LABELS[f.tier]}: \"{f.citation['text']}\"  ({_where(f.citation)})")
        if self.possible:
            out += ["", "Possibly related (not stated as a relationship):"]
            out += [f"  - {f.claim}  ({TIER_LABELS[f.tier]})" for f in self.possible]
        if self.notes:
            out += ["", "Notes:"] + [f"  ! {n}" for n in self.notes]
        return "\n".join(out)


def _where(c: dict) -> str:
    if c.get("chunk_id"):
        return f"{c.get('document_id')}, chunk {c['chunk_id']}"
    if c.get("kind") == "record":
        return f"{c.get('file')}, row {c.get('key')}"
    return f"{c.get('document_id') or ''} {c.get('file') or ''}".strip()


# ---------------- phrasing: the verb follows the evidence ----------------

def claim_for(rel, tier: str, name, header_document: str | None = None) -> str:
    s, t, r = name(rel.source_key), name(rel.target_key), rel.relationship
    if tier not in DOCUMENTED:                 # never more than "they appear together"
        return f"{s} and {t} are mentioned together"
    if tier == "header" and header_document and r in ("governed_by", "related_to"):
        # A header line belongs to ONE document: it is that document listing the other one.
        # "Customer Escalation Policy lists Incident Management", never the reverse.
        ends = [rel.source_key, rel.target_key]
        own = next((k for k in ends if k.split(":", 1)[1] == header_document), None)
        if own:
            other = ends[1] if own == ends[0] else ends[0]
            return f"{name(own)} lists {name(other)} as related"
    if r == "approves":
        return f"{s} approves {t}"
    if r == "owns":
        return f"{s} is the recorded owner of {t}" if tier == "record" else f"{t} names {s} as its owner"
    if r == "performs":
        # Phase 3 links every role NAMED in a step; the step may be done by someone else
        # ("Consultant reviews the AE's handoff record"), so say exactly that much.
        return f"{s} is named in a step of {t}"
    if r == "participates_in":
        return f"{t} lists {s} as a related role" if tier == "header" else f"{s} takes part in {t}"
    if r == "handles_escalations_for":
        return f"{s} handles escalations for {t}"
    if r == "uses":
        return f"{s} uses {t}"
    if r == "triggers":
        return f"{s} triggers {t}"
    if r == "governed_by":
        return f"{s} lists {t} as a related policy" if tier == "header" else f"{s} refers to {t}"
    if r == "related_to":
        return f"{s} lists {t} as related" if tier == "header" else f"{s} mentions {t}"
    if r == "owned_by":
        return f"{s} belongs to the {t} department" + (" (company record)" if tier == "record" else "")
    if r == "headed_by":
        return f"The {s} department is headed by the {t}"
    if r == "in_department":
        return f"{s} is part of the {t} department"
    if r == "has_role":
        return f"{s} works as {t}"
    if r == "works_in":
        return f"{s} works in the {t} department"
    if r == "located_in":
        return f"{s} is based in {t}"
    if r == "reports_to":
        return f"{s} reports to {t}"
    return f"{s} {r.replace('_', ' ')} {t}"


class GraphQueries:
    def __init__(self, view, tracer: Tracer, intelligence_dir: Path | None = Path("data/intelligence"),
                 processes: list[dict] | None = None, conflicts: list[dict] | None = None):
        """PROCESSES / CONFLICTS: Phase 3 output lists; read from INTELLIGENCE_DIR when not given."""
        self.view = view
        self.tracer = tracer
        self.resolver = Resolver(view)
        if processes is None:
            processes = _read_list(intelligence_dir / "processes.json", "processes") if intelligence_dir else []
        if conflicts is None:
            conflicts = _read_list(intelligence_dir / "conflicts.json", "conflicts") if intelligence_dir else []
        self.processes = {p["process_id"]: p for p in processes}
        self.conflicts = conflicts

    # ---------------- building blocks ----------------

    def name(self, key: str) -> str:
        e = self.view.node(key)
        return e.name if e else key.split(":", 1)[1]

    def fact(self, rel, group: str = "") -> GraphFact | None:
        """The link as a fact, from its strongest VERIFIED visible evidence (None if nothing verifies)."""
        best = None
        for p in self.tracer.trace(rel):
            if p.verified and p.evidence in rel.evidence:
                from src.graph.schema import TIERS
                if best is None or TIERS.index(p.tier) < TIERS.index(best.tier):
                    best = p
        if best is None:
            return None
        return GraphFact(claim_for(rel, best.tier, self.name, best.evidence.source_document), best.tier,
                         best.citation, rel.relationship_id,
                         [rel.source_key, rel.target_key], rel.historical, group)

    def links(self, key: str, relationship: str, direction: str, source_type: str | None = None,
              current_only: bool = True):
        hops = self.view.hops(key, relationship, direction, current_only=current_only)
        return [h for h in hops if source_type is None or h.node.startswith(source_type + ":")]

    def department_of_role(self, role_key: str) -> str:
        hops = self.links(role_key, "in_department", OUT)
        return self.name(hops[0].node) if hops else "Department not recorded"

    def _split(self, facts):
        facts = [f for f in facts if f is not None]
        return [f for f in facts if f.documented], [f for f in facts if not f.documented]

    def _answer(self, question, kind, subjects, facts, possible=(), notes=(), status=None) -> GraphAnswer:
        documented, weak = self._split(facts)
        possible = list(possible) + weak
        if status is None:
            status = "answered" if documented else ("partial" if possible else "not_found")
        documented = _dedupe(documented)
        stated = [set(f.entities) for f in documented]
        # a pair already given as a fact is not repeated as "possibly related"
        possible = [f for f in _dedupe(possible) if set(f.entities) not in stated]
        return GraphAnswer(question, kind, subjects, status, documented, possible, list(notes))

    # ---------------- entry point ----------------

    def intent(self, question: str) -> str | None:
        q = " ".join(question.lower().split())
        return next((name for name, pattern in INTENTS if pattern.search(q)), None)

    def answer(self, question: str) -> GraphAnswer | None:
        """A graph answer, or None to leave the question to Phase 4. The graph
        never answers "not found" itself: if it has nothing documented, Phase 4
        searches the documents as before."""
        a = self._route(question)
        return None if a is None or a.status == "not_found" else a

    def _route(self, question: str) -> GraphAnswer | None:
        kind = self.intent(question)
        if kind is None:
            return None
        matches = self.resolver.find(question)
        by_type: dict[str, list[Match]] = {}
        for m in matches:
            by_type.setdefault(m.entity_type, []).append(m)
        history = bool(HISTORY.search(question.lower()))
        first = lambda *types: next((m.key for m in matches if m.entity_type in types), None)  # noqa: E731
        proc = first("process", "policy")

        if kind == "delay" and proc:
            return self.delay(question, proc)
        if kind == "impact" and (proc or first("product")):
            return self.impact(question, proc or first("product"))
        if kind == "head_of_owner" and (first("product", "process", "policy")):
            return self.head_of_owner(question, first("product", "process", "policy"))
        if kind == "which_department":
            target = first("role", "process", "policy", "product")
            if target:
                return self.which_department(question, target)
        if kind == "policies" and proc:
            return self.policies(question, proc)
        if kind == "approver":
            return self.approvers(question, proc, history) if proc and history else None
        if kind == "owner":
            if proc:
                return self.owners(question, proc, history)
            if first("role"):
                return self.role_processes(question, first("role"), ("owns",))
            if first("department"):
                return self.department(question, first("department"))
        if kind == "involved":
            if proc:
                return self.involved(question, proc)
            if first("role"):
                return self.role_processes(question, first("role"))
        if kind == "next":
            return self.next_after(question, proc)
        if kind == "uses" and first("product"):
            return self.users_of(question, first("product"))
        if kind in ("about", "uses", "owner", "involved"):
            if first("department"):
                return self.department(question, first("department"))
            if first("role") and (kind != "about" or _ROLE_ABOUT.search(question.lower())):
                return self.role_processes(question, first("role"))
            if first("employee") and kind == "about":
                return self.person(question, first("employee"))
        return None

    # ---------------- query types ----------------

    def approvers(self, question, proc, history=False) -> GraphAnswer:
        notes = []
        if history:
            older = self._older_versions(proc)
            facts = [self._historical(self.fact(h.relationship), k) for k in older
                     for h in self.links(k, "approves", IN, "role", current_only=False)]
            if not any(facts):
                facts = [self._historical(self.fact(h.relationship), k) for k in older
                         for h in self.links(k, "owns", IN, "role", current_only=False)]
                if any(facts):
                    notes.append("The archived version records an owner; it doesn't name a separate approver.")
            if not older:
                notes.append(f"KORA has no archived version of {self.name(proc)}.")
            current = [self.fact(h.relationship) for h in self.links(proc, "approves", IN, "role")]
            if any(current):
                notes.append("Today: " + "; ".join(f.claim for f in current if f) + ".")
            status = None if any(f and f.documented for f in facts) else ("partial" if current else "not_found")
            return self._answer(question, "approver", [proc], facts, notes=notes, status=status)
        facts = [self.fact(h.relationship) for h in self.links(proc, "approves", IN, "role")]
        if not any(f and f.documented for f in facts):
            notes.append(f"No document names who approves {self.name(proc)}.")
        return self._answer(question, "approver", [proc], facts, notes=notes)

    def policies(self, question, proc) -> GraphAnswer:
        facts = [self.fact(h.relationship) for h in self.links(proc, "governed_by", OUT, "policy")]
        notes = ["The documents list these policies as related; they don't spell out how each one applies."] \
            if any(f and f.tier == "header" for f in facts) else []
        return self._answer(question, "policies", [proc], facts, notes=notes)

    def owners(self, question, proc, history=False) -> GraphAnswer:
        keys = [proc] + (self._older_versions(proc) if history else [])
        facts = [self.fact(h.relationship) for k in keys for h in self.links(k, "owns", IN, "role", not history)]
        facts += [self.fact(h.relationship) for h in self.links(proc, "owned_by", OUT, None)]
        return self._answer(question, "owner", [proc], facts)

    def involved(self, question, proc) -> GraphAnswer:
        best: dict[str, GraphFact] = {}
        for rel_name in ("performs", "approves", "owns", "participates_in", "handles_escalations_for"):
            for h in self.links(proc, rel_name, IN, "role"):
                f = self.fact(h.relationship, group=self.department_of_role(h.node))
                if f and (h.node not in best or _better(f, best[h.node])):
                    best[h.node] = f
        facts = sorted(best.values(), key=lambda f: (f.group, f.claim))
        notes = ["Teams are the departments these roles belong to (roles.csv)."] if facts else \
            [f"No document names the roles involved in {self.name(proc)}."]
        return self._answer(question, "involved", [proc], facts, notes=notes)

    def impact(self, question, subject) -> GraphAnswer:
        facts, possible = [], []
        same_id = [k for k in (f"policy:{subject.split(':', 1)[1]}", f"process:{subject.split(':', 1)[1]}")
                   if self.view.node(k)] or [subject]
        for key in same_id:
            facts += [self.fact(h.relationship) for h in self.links(key, "uses", IN, "process")]          # X is used by
            facts += [self.fact(h.relationship) for h in self.links(key, "triggers", OUT, "process")]     # X starts
            facts += [self.fact(h.relationship) for h in self.links(key, "governed_by", IN, "process")]   # lists X
            for h in self.links(key, "related_to", IN, "process") + self.links(key, "related_to", OUT, "process"):
                f = self.fact(h.relationship)
                if f:
                    possible.append(f)
        documented, weak = self._split(facts)
        dependencies = [f for f in documented if f.tier == QUOTE]
        status = "answered" if dependencies else ("partial" if documented or possible or weak else "not_found")
        return self._answer(question, "impact", [subject], documented, possible + weak,
                            [NO_IMPACT_TEXT] if status != "not_found" else
                            [f"No documented dependency on {self.name(subject)} was found."], status)

    def delay(self, question, proc) -> GraphAnswer:
        facts = [self.fact(h.relationship) for h in self.links(proc, "owns", IN, "role")]
        facts += [self.fact(h.relationship) for h in self.links(proc, "triggers", IN, "process")]   # what feeds it
        facts += self._known_gaps(proc)
        notes = [NO_LIVE_STATUS]
        doc_ids = {h.node.split(":", 1)[1] for h in self.links(proc, "documented_by", OUT)}
        for c in self.conflicts:
            involved = set(c.get("documents_involved", []))
            if c.get("conflict_type") in HOUSEKEEPING_CONFLICTS:
                continue
            if involved & doc_ids and c.get("status") != "resolved" and all(self.view.can_read(d) for d in involved):
                notes.append(f"Open conflict {c['conflict_id']} involves this process's documents: {c.get('detail', '')}")
        if not any(f and f.claim.startswith("Known gap") for f in facts):
            notes.append(f"No known problem is documented for {self.name(proc)}.")
        return self._answer(question, "delay", [proc], facts, notes=notes, status="partial")

    def next_after(self, question, proc=None) -> GraphAnswer:
        facts, possible = [], []
        if proc:
            facts += [self.fact(h.relationship) for h in self.links(proc, "triggers", OUT, "process")]
        if not facts:                                    # no process named: match the trigger sentences
            q = set(tokens(question)) - _COMMON
            for rel in self.view.relationships:
                if rel.relationship != "triggers":
                    continue
                f = self.fact(rel)
                if f and len(q & set(tokens(f.citation.get("text", "")))) >= 2:
                    facts.append(f)
        if proc and not facts:
            step = self._last_step(proc)
            if step:
                facts.append(step)
            possible += [f for f in (self.fact(h.relationship) for h in self.links(proc, "related_to", OUT, "process"))
                         if f]
            notes = ["No document says which process starts after this one. The last step is shown, and processes "
                     "the document lists as related (not stated as the next step)."]
            return self._answer(question, "next", [proc], facts, possible, notes, status="partial" if facts else None)
        return self._answer(question, "next", [proc] if proc else [], facts, possible,
                            [] if facts else ["No documented step or trigger matches this question."])

    def users_of(self, question, product) -> GraphAnswer:
        facts = [self.fact(h.relationship) for h in self.links(product, "uses", IN, "process")]
        possible = [f for f in (self.fact(h.relationship) for h in self.links(product, "related_to", IN, "process"))
                    if f]
        return self._answer(question, "uses", [product], facts, possible)

    def which_department(self, question, subject) -> GraphAnswer:
        rel_name = "in_department" if subject.startswith("role:") else "owned_by"
        facts = [self.fact(h.relationship) for h in self.links(subject, rel_name, OUT, "department")]
        return self._answer(question, "which_department", [subject], facts)

    def head_of_owner(self, question, subject) -> GraphAnswer:
        facts = []
        for h in self.links(subject, "owned_by", OUT, "department"):
            facts.append(self.fact(h.relationship))
            facts += [self.fact(x.relationship) for x in self.links(h.node, "headed_by", OUT, "role")]
        return self._answer(question, "head_of_owner", [subject], facts)

    def department(self, question, dept) -> GraphAnswer:
        q = question.lower()
        want = {k for k, pat in (("head", r"\bhead|\blead|\bruns\b"), ("products", r"\bproduct"),
                                  ("processes", r"\bprocess|\bown"), ("roles", r"\brole|\bwho works"))
                if re.search(pat, q)} or {"head", "products", "processes"}
        facts = []
        if "head" in want:
            facts += [self.fact(h.relationship) for h in self.links(dept, "headed_by", OUT, "role")]
        if "products" in want:
            facts += [self.fact(h.relationship) for h in self.links(dept, "owned_by", IN, "product")]
        if "processes" in want and not ("products" in want and "process" not in q):
            facts += [self.fact(h.relationship) for h in self.links(dept, "owned_by", IN, "process")]
        if "roles" in want:
            facts += [self.fact(h.relationship) for h in self.links(dept, "in_department", IN, "role")]
        return self._answer(question, "department", [dept], facts)

    def role_processes(self, question, role, relationships=("owns", "approves", "performs", "participates_in",
                                                              "handles_escalations_for")) -> GraphAnswer:
        q = question.lower()
        if relationships != ("owns",) and re.search(r"\bdepartment|\bteam|\bbelong", q):
            return self.which_department(question, role)
        best: dict[str, GraphFact] = {}
        for rel_name in relationships:
            for h in self.links(role, rel_name, OUT, "process"):
                f = self.fact(h.relationship)
                if f and (h.node not in best or _better(f, best[h.node])):
                    best[h.node] = f
        return self._answer(question, "role", [role], sorted(best.values(), key=lambda f: f.claim))

    def person(self, question, emp) -> GraphAnswer:
        facts = [self.fact(h.relationship) for rel_name in ("has_role", "works_in", "located_in", "reports_to")
                 for h in self.links(emp, rel_name, OUT)]
        return self._answer(question, "person", [emp], facts)

    # ---------------- helpers ----------------

    def _lineage(self, proc) -> set[str]:
        """The header document ID(s) behind a process: an archived copy keeps the
        same header ID (PROC-IMPL-001-V1's file says PROC-IMPL-001)."""
        out = set()
        for h in self.view.hops(proc, "documented_by", OUT):
            doc = self.view.node(h.node)
            if doc:
                out.add(doc.attributes.get("header_document_id") or doc.entity_id)
        return out or {proc.split(":", 1)[1]}

    def _older_versions(self, proc) -> list[str]:
        """Archived processes from the same document lineage (PROC-IMPL-001-V1 for PROC-IMPL-001)."""
        mine = self._lineage(proc)
        return [f"process:{p.entity_id}" for p in self.view.entities("process")
                if f"process:{p.entity_id}" != proc and (p.attributes.get("status") or "").lower() == "archived"
                and self._lineage(f"process:{p.entity_id}") & mine]

    def _historical(self, fact, proc_key):
        if fact is None:
            return None
        e = self.view.node(proc_key)
        version = e.attributes.get("version") if e else None
        fact.claim += f", in version {version}" if version else ", in an archived version"
        quote = str(fact.citation.get("text", "")).lower()
        for key in fact.entities:                 # the old document may use an older role name
            role = self.view.node(key)
            if role and role.entity_type == "role" and role.name.lower() not in quote:
                old = [a for a in role.aliases if a.lower() in quote]
                if old:
                    fact.claim += f'; this version calls the role "{old[0]}"'
        fact.historical = True
        return fact

    def _known_gaps(self, proc) -> list[GraphFact]:
        out = []
        for gap in self.processes.get(proc.split(":", 1)[1], {}).get("known_gaps", []):
            text = gap["text"].rstrip(" -").strip()
            for cid in gap.get("chunk_ids", []):
                chunk = self.tracer.chunks.get(cid)
                if chunk and self.view.can_read(chunk["document_id"]) and norm_key(text) in norm_key(chunk["chunk_text"]):
                    label = gap.get("heading") or "Known gap"
                    out.append(GraphFact(f"Known gap ({label}) documented for {self.name(proc)}", QUOTE,
                                         {"kind": "quote", "chunk_id": cid, "document_id": chunk["document_id"],
                                          "text": text}, None, [proc]))
                    break
        return out

    def _last_step(self, proc) -> GraphFact | None:
        steps = self.processes.get(proc.split(":", 1)[1], {}).get("steps", [])
        for step in reversed(steps):
            for cid in step.get("chunk_ids", []):
                chunk = self.tracer.chunks.get(cid)
                if chunk and self.view.can_read(chunk["document_id"]) and \
                        norm_key(step["text"]) in norm_key(chunk["chunk_text"]):
                    title = f"{step.get('title')}: " if step.get("title") else ""
                    return GraphFact(f"The last documented step of {self.name(proc)} is step {step['number']}",
                                     QUOTE, {"kind": "quote", "chunk_id": cid, "document_id": chunk["document_id"],
                                             "text": f"{title}{step['text']}"}, None, [proc])
        return None


_COMMON = set(tokens("what which who when where how does do is are a an the of to in on for and or after "
                     "process processes happen happens start starts started trigger triggers next"))


def _better(a: GraphFact, b: GraphFact) -> bool:
    """Prefer a quote (it says what the role DOES) over a header or record line."""
    from src.graph.schema import TIERS
    return TIERS.index(a.tier) < TIERS.index(b.tier)


def _dedupe(facts: list[GraphFact]) -> list[GraphFact]:
    seen, out = set(), []
    for f in facts:
        k = (f.claim, f.citation.get("text"))
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def _read_list(path: Path, key: str) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key, []) if isinstance(data, dict) else data

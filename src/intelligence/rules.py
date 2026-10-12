"""
Approval and escalation rules: who approves what, who receives escalations.

Read from AUTHORITATIVE documents only: process documents, policies and
department descriptions. The FAQ and scenario documents restate these rules;
Milestone 3 compares them against the rules found here instead of counting
them twice.

How an approver is recognised (each is a fixed text pattern):
    "requires Sales Manager approval"              role directly before "approval"
    "Direct manager approves" / "provides ... approval"
    "approval from the Data & Analytics Manager"
    "**Sales Manager** — ... is the approval authority for ..."   (role paragraph)
    | System | Approver | tables
A generic "manager" or "department head" is kept as written: it means
"the person's own manager", which only an employee record can resolve.

Nothing is invented: a sentence that clearly states an approval rule but
names no approver is kept with approver_identified = False.
"""

import hashlib
import re
from dataclasses import asdict, dataclass, field

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.matchers import generic_roles, role_matcher
from src.intelligence.models import CONF_DOCUMENTED, CONF_PATTERN, DOCUMENT, PROCESS
from src.intelligence.text import Block, amounts, blocks, durations, id_refs, percents, sentences, tables

APPROVAL = "approval"
ESCALATION = "escalation"

# ---- approval patterns (looked for right after / before a role name) ----
_AFTER_APPROVAL = re.compile(r"^(?:'s)?\s*(?:\+[^.;]*?\s)?(?:pre-)?approval\b", re.I)
_AFTER_APPROVES = re.compile(r"^\s+(?:\w+\s+)?(?:approves?|signs? off)\b", re.I)
_AFTER_PROVIDES = re.compile(r"^\s+provides\s+(?:[\w-]+\s+){0,3}?(?:approval|sign-off)\b", re.I)
_AFTER_AUTHORITY = re.compile(r"^[^.;]{0,40}?\bis\s+(?:the\s+)?(?:final\s+)?approval authority\b", re.I)
_BEFORE_FROM = re.compile(r"(?:approval|approved|sign-off)\s+(?:from|by)\s+(?:the\s+)?$", re.I)
_SUBJECT_APPROVES = re.compile(r"\bapproves?\b|\bapproval authority\b|\bfinal approval\b", re.I)
_GENERIC_APPROVAL = re.compile(
    r"(?P<who>(?:[Dd]irect |[Hh]iring |[Ll]ine )?[Mm]anager|[Dd]epartment heads?)"
    r"(?:'s)?\s+(?:may\s+|can\s+|must\s+)?(?:pre-)?approv")
_NO_APPROVAL = re.compile(r"\bno (?:additional )?(?:pre-)?approval\b|\bwithout (?:pre-)?approval\b", re.I)
_APPROVAL_WORD = re.compile(r"approv|sign-off", re.I)
_REQUIRE = re.compile(r"\brequire[sd]?\b|\bneeds?\b|\bmust\b", re.I)

# ---- escalation patterns ----
_ESCALATION_WORD = re.compile(r"escalat|flagged to|routed (?:jointly )?to|reported to", re.I)
_BEFORE_TO = re.compile(
    r"(?:escalat\w*|flagged|routed(?:\s+jointly)?|reported)\s+(?:[\w-]+\s+){0,3}?to\s+"
    r"(?:the\s+)?(?:\w+'s\s+)?(?:assigned\s+|relevant\s+)?$", re.I)
_CHAIN = re.compile(r"^\s*(?:,|and|or|/|\+)\s*(?:the\s+)?$", re.I)
_ESC_POINT = re.compile(r"escalation point\b", re.I)
_ABSENCE = re.compile(r"\bno (?:formal |documented )?(?:escalation|fallback)\b|\bno formal\b|\bno documented\b", re.I)
_TO_TEXT = re.compile(r"\bescalat\w*\s+(?:\w+\s+){0,2}?to\s+([A-Z][\w/&-]*(?:,?\s+(?:then\s+)?[A-Z][\w/&-]*)*)")
_LEVEL = re.compile(r"^Level\s*(\d+)\b", re.I)

_ROLE_TITLE = re.compile(r"^(.+?)\s*\([A-Z]{2,6}\)$")   # '... (CSM)' -> '...'


@dataclass
class Rule:
    rule_id: str
    rule_type: str                         # approval | escalation
    statement: str                         # the exact clause, markup removed
    context: str                           # step title or section it came from
    role_ids: list[str]                    # approvers / escalation targets (role entity IDs)
    generic_roles: list[str]               # 'Manager', 'Department heads' (resolved per employee later)
    unresolved_targets: list[str]          # named but not a known role, e.g. 'Executive (CEO/CTO)'
    mentioned_role_ids: list[str]          # other roles in the clause (requesters, escalating roles)
    process_ids: list[str]
    document_id: str
    source_filename: str
    section: str
    chunk_ids: list[str]
    method: str
    confidence: float
    historical: bool
    conditions: dict = field(default_factory=dict)    # amounts, percents, durations
    level: int | None = None                           # escalation level, if stated
    no_approval_required: bool = False
    states_absence: bool = False                       # "no formal escalation path exists"

    @property
    def identified(self) -> bool:
        return bool(self.role_ids or self.generic_roles or self.unresolved_targets or self.no_approval_required)

    def to_dict(self) -> dict:
        return {**asdict(self), "identified": self.identified}


def _rule_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}-{digest}"


def _ids(keys: list[str]) -> list[str]:
    return [k.split(":", 1)[1] for k in keys]


class _Extractor:
    def __init__(self, data: Phase2Data, index: EntityIndex):
        self.data, self.index = data, index
        self.roles = role_matcher(index)
        self.rules: list[Rule] = []
        # document entity id -> process ids it documents (version-aware, from Milestone 1)
        self.doc_processes: dict[str, list[str]] = {}
        for p in index.of_type(PROCESS):
            if p.attributes.get("document_id"):
                self.doc_processes.setdefault(p.attributes["document_id"], []).append(p.entity_id)

    # ---------- documents to read ----------

    def source_documents(self):
        wanted = set(self.doc_processes)
        for doc in self.index.of_type(DOCUMENT):
            t = (doc.attributes.get("document_type") or "").lower()
            if "policy" in t or "department description" in t:
                wanted.add(doc.entity_id)
        return [self.index.get(DOCUMENT, d) for d in sorted(wanted) if self.index.get(DOCUMENT, d)]

    # ---------- helpers ----------

    def _subject_role(self, doc, section_name: str, block: Block) -> str | None:
        """In a department description's Roles section, the bold heading is the subject."""
        if "department description" not in (doc.attributes.get("document_type") or "").lower():
            return None
        if not section_name.lower().endswith("roles") or not block.title:
            return None
        title = block.title
        m = _ROLE_TITLE.match(title)
        keys = self.roles.keys(m.group(1) if m else title)
        return keys[0] if len(keys) == 1 else None

    def _processes(self, doc_id: str, text: str) -> list[str]:
        out = list(self.doc_processes.get(doc_id, []))
        for ref in id_refs(text):
            if self.index.get(PROCESS, ref) and ref not in out:
                out.append(ref)
        return out

    def _make(self, rule_type, doc, section, clause, context, method, confidence, **kw) -> Rule:
        a = doc.attributes
        prefix = "APR" if rule_type == APPROVAL else "ESC"
        chunk_ids = self.data.find_chunks(a["header_document_id"], a["version"], clause) or section.chunk_ids
        return Rule(
            rule_id=_rule_id(prefix, doc.entity_id, clause),
            rule_type=rule_type,
            statement=clause,
            context=context,
            process_ids=self._processes(doc.entity_id, clause),
            document_id=doc.entity_id,
            source_filename=a["source_filename"],
            section=section.name,
            chunk_ids=chunk_ids,
            method=method,
            confidence=confidence,
            historical=(a.get("status") or "").lower() == "archived",
            conditions={"amounts": amounts(clause), "percents": percents(clause), "durations": durations(clause)},
            **kw,
        )

    # ---------- approvals ----------

    def _approval(self, doc, section, clause: str, context: str, subject: str | None) -> Rule | None:
        if not _APPROVAL_WORD.search(clause):
            return None
        mentions = self.roles.find(clause)
        approvers = []
        for m in mentions:
            after, before = clause[m.end:m.end + 80], clause[max(0, m.start - 40):m.start]
            if (_AFTER_APPROVAL.match(after) or _AFTER_APPROVES.match(after) or _AFTER_PROVIDES.match(after)
                    or _AFTER_AUTHORITY.match(after) or _BEFORE_FROM.search(before)):
                approvers.append(m.key)
        method = "approval_pattern"
        if not approvers and subject and _SUBJECT_APPROVES.search(clause):
            approvers, method = [subject], "role_paragraph"

        masked = self.roles.mask(clause)
        generic = [generic_roles(m.group("who"))[0] for m in _GENERIC_APPROVAL.finditer(masked)]
        no_approval = bool(_NO_APPROVAL.search(clause))
        if not (approvers or generic or no_approval or _REQUIRE.search(clause)):
            return None    # 'Approved invoices are paid...' mentions approval but states no rule

        approvers = list(dict.fromkeys(approvers))
        others = [k for k in dict.fromkeys(m.key for m in mentions) if k not in approvers]
        if subject and subject not in approvers and subject not in others:
            others.insert(0, subject)
        return self._make(APPROVAL, doc, section, clause, context, method, CONF_PATTERN,
                          role_ids=_ids(approvers), generic_roles=list(dict.fromkeys(generic)),
                          unresolved_targets=[], mentioned_role_ids=_ids(others),
                          no_approval_required=no_approval)

    # ---------- escalations ----------

    def _escalation(self, doc, section, clause: str, context: str, subject: str | None,
                    level: int | None = None, level_targets: list[str] | None = None,
                    level_text: str | None = None) -> Rule | None:
        if level is None and not _ESCALATION_WORD.search(clause):
            return None
        mentions = self.roles.find(clause)
        targets: list[str] = []
        previous_target = None
        for m in mentions:
            before = clause[max(0, m.start - 60):m.start]
            chained = previous_target is not None and _CHAIN.match(clause[previous_target.end:m.start])
            if _BEFORE_TO.search(before) or chained:
                targets.append(m.key)
                previous_target = m
        method = "escalation_pattern"
        if not targets and subject and _ESC_POINT.search(clause):
            targets, method = [subject], "role_paragraph"
        if level is not None:
            method = "escalation_level"
            targets = targets or list(level_targets or []) or [m.key for m in mentions]

        unresolved: list[str] = []
        if not targets:
            to_text = _TO_TEXT.search(clause)
            if to_text:
                unresolved.append(to_text.group(1))
            elif level is not None and level_text:
                unresolved.append(level_text)

        targets = list(dict.fromkeys(targets))
        others = [k for k in dict.fromkeys(m.key for m in mentions) if k not in targets]
        return self._make(ESCALATION, doc, section, clause, context, method, CONF_PATTERN,
                          role_ids=_ids(targets), generic_roles=generic_roles(self.roles.mask(clause)) if not targets else [],
                          unresolved_targets=unresolved, mentioned_role_ids=_ids(others),
                          level=level, states_absence=bool(_ABSENCE.search(clause)))

    # ---------- approver tables ----------

    def _approver_table(self, doc, section, rows: list[dict]) -> None:
        for row in rows:
            approver_col = next((k for k in row if k.strip().lower() == "approver"), None)
            if approver_col is None:
                continue
            what_col = next(k for k in row if k != approver_col)
            approver_text = row[approver_col]
            keys = self.roles.keys(approver_text)
            leftover = [p.strip() for p in re.split(r"\+|,| and ", self.roles.mask(approver_text))
                        if p.strip() and set(p.strip()) != {"_"}]
            clause = f"{row[what_col]}: approved by {approver_text}"
            self.rules.append(self._make(
                APPROVAL, doc, section, clause, f"table: {what_col} / {approver_col}",
                "approver_table", CONF_DOCUMENTED,
                role_ids=_ids(keys), generic_roles=[], unresolved_targets=leftover,
                mentioned_role_ids=[]))

    # ---------- run ----------

    def run(self) -> list[Rule]:
        for doc in self.source_documents():
            a = doc.attributes
            for section in self.data.sections(a["header_document_id"], a["version"]):
                all_blocks = blocks(section.text)
                for rows in tables(all_blocks):
                    self._approver_table(doc, section, rows)

                is_escalation_section = "escalation" in section.name.lower()
                for block in all_blocks:
                    if block.kind not in ("numbered", "bullet", "paragraph"):
                        continue
                    subject = self._subject_role(doc, section.name, block)
                    context = block.title or section.name.split(" > ")[-1]

                    level_match = _LEVEL.match(block.title or "") if is_escalation_section else None
                    if level_match:
                        statement = f"{block.title}: {' '.join(sentences(block.body))}"
                        rule = self._escalation(doc, section, statement, context, subject,
                                                level=int(level_match.group(1)),
                                                level_targets=self.roles.keys(block.title),
                                                level_text=block.title.split("—")[-1].strip())
                        self.rules.append(rule)
                        continue

                    for sentence in sentences(block.body):
                        for clause in (c.strip() for c in re.split(r";\s+", sentence)):
                            if not clause:
                                continue
                            for rule in (self._approval(doc, section, clause, context, subject),
                                         self._escalation(doc, section, clause, context, subject)):
                                if rule:
                                    self.rules.append(rule)

        # One rule per (document, statement, type): a clause can't count twice.
        unique: dict[str, Rule] = {}
        for rule in self.rules:
            unique.setdefault(rule.rule_id + rule.rule_type, rule)
        return sorted(unique.values(), key=lambda r: (r.rule_type, r.document_id, r.rule_id))


def extract_rules(data: Phase2Data, index: EntityIndex) -> list[Rule]:
    return _Extractor(data, index).run()

"""
Process discovery: what each registered process actually contains.

Starting point is processes.csv (Nexora's own register), NOT "every
document". Each process is then read from its own document (the right
VERSION of it) and classified from the document's declared type:

    SOP                -> procedure          (expects numbered steps)
    Policy / Addendum  -> policy             (rules, not steps)
    Policy/SOP         -> policy_and_procedure
    Guideline          -> guideline

Inputs and outputs are only recorded when the text states them:
  * Purpose sentence "from X to/into Y"         (Customer Onboarding: signed contract -> live production usage)
  * a quoted artifact handed between steps        ('"Deal Closed" handoff record')
Otherwise they stay empty and inputs_outputs_stated is False.
"""

import re
from dataclasses import asdict, dataclass, field

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.matchers import generic_roles, product_matcher, role_matcher
from src.intelligence.models import (
    CONF_KEYWORD, CONF_PATTERN, CONF_STRUCTURED, DOCUMENT, POLICY, PROCESS,
)
from src.intelligence.normalize import resolve_roles, unique
from src.intelligence.rules import Rule
from src.intelligence.text import blocks, durations, gap_codes, id_refs, plain, sentences, tables

_KIND_BY_TYPE = [                  # checked in order: most specific first
    ("policy/sop", "policy_and_procedure"),
    ("policy addendum", "policy"),
    ("policy", "policy"),
    ("sop", "procedure"),
    ("guideline", "guideline"),
]
_FROM_TO = re.compile(r"\bfrom\s+(.+?)\s+(?:to|into)\s+(.+?)(?:\s+(?:across|within|for|in)\s+.*)?[.]?$", re.I)
_ARTIFACT = re.compile(r"[\"“]([^\"”]+)[\"”]\s+((?:handoff\s+)?(?:record|form|document|template|ticket|report|request))\b", re.I)
_OUTPUT_VERBS = re.compile(r"\b(?:trigger|complete|send|deliver|issue|create|produce|submit)s?\b", re.I)
_INPUT_VERBS = re.compile(r"\b(?:review|receive|use|assume|check)s?\b", re.I)
_TRIGGER_WORDS = re.compile(r"\btrigger(?:s|ed)?\b|\bhandoff\b|\bhand(?:s|ed)? off\b", re.I)
_GAP_SECTION = re.compile(r"known gap", re.I)
_HISTORY_SECTION = re.compile(r"version history", re.I)


@dataclass
class ProcessRecord:
    process_id: str
    name: str
    kind: str
    status: str
    version: str
    effective_date: str
    historical: bool
    document_id: str | None
    source_filename: str | None
    owner_roles: list[dict] = field(default_factory=list)
    department: dict | None = None
    supporting_departments: list[dict] = field(default_factory=list)
    participants: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    inputs_outputs_stated: bool = False
    approval_rule_ids: list[str] = field(default_factory=list)
    escalation_rule_ids: list[str] = field(default_factory=list)
    triggers: list[dict] = field(default_factory=list)          # this process starts another
    related_processes: list[dict] = field(default_factory=list)
    related_products: list[dict] = field(default_factory=list)
    related_policies: list[dict] = field(default_factory=list)
    related_documents: list[dict] = field(default_factory=list)
    timing_rules: list[dict] = field(default_factory=list)
    tables: list[list[dict]] = field(default_factory=list)
    known_gaps: list[dict] = field(default_factory=list)
    version_notes: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _kind(document_type: str) -> str:
    # 'Policy / SOP' (header) and 'Policy/SOP' (documents.csv) must compare equal.
    t = re.sub(r"\s*/\s*", "/", (document_type or "").lower())
    for word, kind in _KIND_BY_TYPE:
        if word in t:
            return kind
    return "unclassified"


class _Analyzer:
    def __init__(self, data: Phase2Data, index: EntityIndex, rules: list[Rule]):
        self.data, self.index, self.rules = data, index, rules
        self.roles = role_matcher(index)
        self.products = product_matcher(index)
        self.dept_names = {d.entity_id: d.name for d in index.of_type("department")}

    def _role(self, role_id: str) -> str:
        e = self.index.get("role", role_id)
        return e.name if e else role_id

    def _ref_types(self, ref: str) -> list[str]:
        return [t for t in (PROCESS, POLICY, DOCUMENT) if self.index.get(t, ref)]

    # ---------- one process ----------

    def analyze(self, proc) -> ProcessRecord:
        a = proc.attributes
        doc = self.index.get(DOCUMENT, a["document_id"]) if a.get("document_id") else None
        da = doc.attributes if doc else {}
        rec = ProcessRecord(
            process_id=proc.entity_id, name=proc.name,
            kind=_kind(da.get("document_type", "")),
            status=a.get("status") or "unknown", version=a.get("version") or "unknown",
            effective_date=da.get("effective_date", "unknown"),
            historical=(a.get("status") or "").lower() == "archived",
            document_id=doc.entity_id if doc else None,
            source_filename=da.get("source_filename"),
        )
        self._ownership(rec, proc, doc)
        if doc is None:
            rec.notes.append("No ingested document for this process: steps, participants and links unavailable")
            return rec

        sections = self.data.sections(da["header_document_id"], da["version"])
        self._read_sections(rec, doc, sections)
        self._participants(rec, doc)
        self._purpose_flow(rec, doc, sections)
        rec.approval_rule_ids = [r.rule_id for r in self.rules if r.rule_type == "approval" and proc.entity_id in r.process_ids]
        rec.escalation_rule_ids = [r.rule_id for r in self.rules if r.rule_type == "escalation" and proc.entity_id in r.process_ids]
        self._header_links(rec, doc)
        return rec

    def _ownership(self, rec: ProcessRecord, proc, doc) -> None:
        a = proc.attributes
        for role_id in a.get("owner_role_ids", []):
            rec.owner_roles.append({"role_id": role_id, "name": self._role(role_id),
                                    "source": "processes.csv: owner_role", "confidence": CONF_STRUCTURED})
        if not a.get("owner_role_ids"):
            rec.notes.append(f"Owner '{a.get('owner_role')}' in processes.csv does not resolve to a role")
        if doc:
            header_owner = doc.attributes.get("owner_role_ids", [])
            if set(header_owner) != set(a.get("owner_role_ids", [])):
                rec.notes.append(f"Document header owner '{doc.attributes.get('owner_role')}' differs "
                                 f"from processes.csv owner '{a.get('owner_role')}'")

        dept_id = a.get("owning_dept_id")
        if dept_id:
            rec.department = {"dept_id": dept_id, "name": self.dept_names.get(dept_id, dept_id),
                              "source": "processes.csv: owning_dept_id", "confidence": CONF_STRUCTURED}
        if doc:
            for other in doc.attributes.get("department_ids", []):
                if other != dept_id:
                    rec.supporting_departments.append({
                        "dept_id": other, "name": self.dept_names.get(other, other),
                        "source": f"{doc.attributes['source_filename']}: Department header",
                        "confidence": CONF_STRUCTURED})

    def _read_sections(self, rec: ProcessRecord, doc, sections) -> None:
        da = doc.attributes
        for section in sections:
            heading = section.name.split(" > ")[-1]
            all_blocks = blocks(section.text)
            body = plain(" ".join(b.text for b in all_blocks if b.kind != "heading"))

            if _GAP_SECTION.search(heading):
                rec.known_gaps.append({"heading": heading, "gap_codes": gap_codes(heading + " " + body),
                                       "text": body, "chunk_ids": section.chunk_ids})
            elif _HISTORY_SECTION.search(heading):
                rec.version_notes.append({"heading": heading, "text": body, "chunk_ids": section.chunk_ids})

            for rows in tables(all_blocks):
                rec.tables.append(rows)

            for block in all_blocks:
                if block.kind == "numbered" and not _GAP_SECTION.search(heading):
                    rec.steps.append(self._step(block, heading, doc))
                if block.kind in ("numbered", "bullet", "paragraph"):
                    for sentence in sentences(block.body):
                        found = durations(sentence)
                        if found:
                            rec.timing_rules.append({
                                "statement": sentence, "context": block.title or heading,
                                "durations": found,
                                "chunk_ids": self.data.find_chunks(da["header_document_id"], da["version"], sentence)
                                             or section.chunk_ids})

    def _step(self, block, heading: str, doc) -> dict:
        da = doc.attributes
        text = plain(block.body)
        quote = plain(block.text)
        refs = [{"id": r, "types": self._ref_types(r)} for r in id_refs(text)]
        return {
            "number": block.number,
            "title": block.title,
            "text": text,
            "section": heading,
            "role_ids": [k.split(":", 1)[1] for k in self.roles.keys(quote)],
            "generic_roles": generic_roles(self.roles.mask(quote)),
            "product_ids": [k.split(":", 1)[1] for k in self.products.keys(quote)],
            "references": refs,
            "durations": durations(text),
            "artifacts": [f'"{m.group(1)}" {m.group(2)}' for m in _ARTIFACT.finditer(text)],
            "chunk_ids": self.data.find_chunks(da["header_document_id"], da["version"], quote),
        }

    def _participants(self, rec: ProcessRecord, doc) -> None:
        """Roles taking part: listed in the header (1.0) or acting in a step (0.85)."""
        found: dict[str, dict] = {}
        for raw in doc.attributes.get("related_roles", []):
            role_ids, _ = resolve_roles(raw, self.data.registry)
            for role_id in role_ids:
                found.setdefault(role_id, {"role_id": role_id, "name": self._role(role_id), "how": [],
                                           "confidence": CONF_STRUCTURED})
                found[role_id]["how"].append("document header: Related Roles")
        for step in rec.steps:
            for role_id in step["role_ids"]:
                entry = found.setdefault(role_id, {"role_id": role_id, "name": self._role(role_id), "how": [],
                                                   "confidence": CONF_PATTERN})
                entry["how"].append(f"named in step {step['number']} ({step['title'] or step['section']})")
        rec.participants = sorted(found.values(), key=lambda p: p["role_id"])

    def _purpose_flow(self, rec: ProcessRecord, doc, sections) -> None:
        for section in sections:
            if section.name.split(" > ")[-1].lower() != "purpose":
                continue
            for sentence in sentences(section.text):
                m = _FROM_TO.search(sentence)
                if m:
                    ev = {"quote": sentence, "method": "purpose_from_to", "confidence": CONF_PATTERN,
                          "chunk_ids": section.chunk_ids}
                    rec.inputs.append({"item": m.group(1).strip(), **ev})
                    rec.outputs.append({"item": m.group(2).strip().rstrip("."), **ev})

        for step in rec.steps:
            for artifact in step["artifacts"]:
                ev = {"item": artifact, "quote": step["text"], "method": "step_artifact",
                      "confidence": CONF_PATTERN, "chunk_ids": step["chunk_ids"], "step": step["number"]}
                if _OUTPUT_VERBS.search(step["text"]):
                    rec.outputs.append(ev)
                elif _INPUT_VERBS.search(step["text"]):
                    rec.inputs.append(ev)
        rec.inputs_outputs_stated = bool(rec.inputs or rec.outputs)

    def _header_links(self, rec: ProcessRecord, doc) -> None:
        for raw in doc.attributes.get("related_processes", []):
            ref = raw.split("(")[0].strip()
            if self.index.get(PROCESS, ref) and ref != rec.process_id:
                rec.related_processes.append({"process_id": ref, "how": "document header: Related Processes",
                                              "confidence": CONF_STRUCTURED})
        for step in rec.steps:
            for ref in step["references"]:
                if PROCESS in ref["types"] and ref["id"] != rec.process_id and _TRIGGER_WORDS.search(step["text"]):
                    rec.triggers.append({"process_id": ref["id"], "step": step["number"], "quote": step["text"],
                                         "chunk_ids": step["chunk_ids"], "confidence": CONF_PATTERN})


# ---------- cross-process links (need every record first) ----------

def _link_products(records: list[ProcessRecord], data: Phase2Data) -> None:
    from src.metadata.relationships import RelationType
    keyword_hits: dict[str, set[str]] = {}
    for rel in data.relationships:
        if rel.relation_type == RelationType.PROCESS_RELATED_TO_PRODUCT:
            keyword_hits.setdefault(rel.source, set()).add(rel.target)
    for rec in records:
        in_steps: dict[str, list[int]] = {}
        for step in rec.steps:
            for product_id in step["product_ids"]:
                in_steps.setdefault(product_id, []).append(step["number"])
        for product_id, numbers in sorted(in_steps.items()):
            rec.related_products.append({"product_id": product_id, "how": f"named in step(s) {numbers}",
                                         "relationship": "uses", "confidence": CONF_PATTERN})
        for product_id in sorted(keyword_hits.get(rec.process_id, set()) - set(in_steps)):
            rec.related_products.append({"product_id": product_id, "how": "mentioned elsewhere in the process document",
                                         "relationship": "related_to", "confidence": CONF_KEYWORD})


def _link_policies(records: list[ProcessRecord], index: EntityIndex) -> None:
    """Process -> governed_by -> Policy, only for processes that are not policies themselves.

    Confidence reflects what the text actually says:
      a step explicitly points at the policy ("see PROC-IT-001")   0.85
      a header merely lists them as related                         0.7
    'Related' is not the same as 'governs', so header-only links stay lower.
    """
    def add(found: dict, policy_id: str, how: str, confidence: float) -> None:
        entry = found.setdefault(policy_id, {"policy_id": policy_id, "how": [], "confidence": confidence})
        entry["how"].append(how)
        entry["confidence"] = max(entry["confidence"], confidence)

    for rec in records:
        if rec.kind == "policy":
            continue
        found: dict[str, dict] = {}
        doc = index.get(DOCUMENT, rec.document_id) if rec.document_id else None
        for raw in (doc.attributes.get("related_processes", []) if doc else []):
            ref = raw.split("(")[0].strip()
            if index.get(POLICY, ref) and ref != rec.document_id:
                add(found, ref, "process document header lists the policy as related", CONF_KEYWORD)
        for step in rec.steps:
            for ref in step["references"]:
                if POLICY in ref["types"] and ref["id"] != rec.document_id:
                    add(found, ref["id"], f"step {step['number']} refers to the policy", CONF_PATTERN)
        for policy in index.of_type(POLICY):
            pdoc = index.get(DOCUMENT, policy.entity_id)
            for raw in (pdoc.attributes.get("related_processes", []) if pdoc else []):
                if raw.split("(")[0].strip() == rec.process_id and policy.entity_id != rec.document_id:
                    add(found, policy.entity_id, "policy header lists this process as related", CONF_KEYWORD)
        rec.related_policies = sorted(found.values(), key=lambda p: p["policy_id"])


def _link_handoffs(records: list[ProcessRecord]) -> None:
    """An artifact output by one process and input to another is a handoff."""
    outputs = [(r, o) for r in records for o in r.outputs if o["method"] == "step_artifact"]
    inputs = [(r, i) for r in records for i in r.inputs if i["method"] == "step_artifact"]
    for src, out in outputs:
        for dst, inp in inputs:
            if src.process_id != dst.process_id and out["item"].lower() == inp["item"].lower():
                src.triggers.append({"process_id": dst.process_id, "step": out["step"],
                                     "quote": out["quote"], "chunk_ids": out["chunk_ids"],
                                     "via_artifact": out["item"], "confidence": CONF_PATTERN})


def _link_documents(records: list[ProcessRecord], data: Phase2Data, index: EntityIndex) -> None:
    from src.metadata.relationships import RelationType
    kinds = {RelationType.DOCUMENT_RELATED_TO_PROCESS: ("header Related Processes", CONF_STRUCTURED),
             RelationType.DOCUMENT_MENTIONS_PROCESS: ("mentioned in text", CONF_KEYWORD)}
    for rec in records:
        found: dict[str, dict] = {}
        if rec.document_id:
            found[rec.document_id] = {"document_id": rec.document_id, "how": ["the process's own document"],
                                      "confidence": CONF_STRUCTURED}
        for rel in data.relationships:
            if rel.relation_type in kinds and rel.target == rec.process_id:
                doc_id = index.document_for_relationship(rel)
                if doc_id and doc_id != rec.document_id:
                    how, conf = kinds[rel.relation_type]
                    entry = found.setdefault(doc_id, {"document_id": doc_id, "how": [], "confidence": conf})
                    entry["how"].append(how)
                    entry["confidence"] = max(entry["confidence"], conf)
        rec.related_documents = sorted(found.values(), key=lambda d: d["document_id"])


def analyze_processes(data: Phase2Data, index: EntityIndex, rules: list[Rule]) -> list[ProcessRecord]:
    analyzer = _Analyzer(data, index, rules)
    records = [analyzer.analyze(p) for p in sorted(index.of_type(PROCESS), key=lambda e: e.entity_id)]
    _link_products(records, data)
    _link_policies(records, index)
    _link_handoffs(records)
    _link_documents(records, data, index)
    for rec in records:
        rec.triggers = list({(t["process_id"], t["step"]): t for t in rec.triggers}.values())
        if rec.kind in ("procedure", "policy_and_procedure") and not rec.steps:
            rec.notes.append("Procedure document has no numbered steps")
    # unique() keeps first-seen order for any list that might repeat
    for rec in records:
        rec.notes = unique(rec.notes)
    return records

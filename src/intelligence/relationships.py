"""
The relationship layer: Entity -> relationship -> Entity, every link with evidence.

Two sources:
  1. Phase 2's relationships, lifted into Phase 3's shape (typed ends, a
     plain-English relationship name, confidence, the exact source document).
  2. New Phase 3 relationships from processes.py, roles.py, rules.py and the tables.

One record per (source, relationship, target). If several sources support
the same link, they all stay in `evidence`, and confidence is the highest.

A link is only created when BOTH ends exist in the entity index. Anything
else goes to `dangling` and is reported by validation, never silently dropped.

Relationship IDs are a hash of (source, relationship, target), so the same
link keeps the same ID on every run: later feedback can point at REL-xxxx.
"""

import hashlib
import re
from dataclasses import asdict, dataclass, field

from src.metadata.relationships import RelationType as RT

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.models import (
    AUDIENCE, CHECKLIST, CONF_DOCUMENTED, CONF_KEYWORD, CONF_PATTERN, CONF_STRUCTURED,
    CUSTOMER, DEPARTMENT, DOCUMENT, EMPLOYEE, LOCATION, POLICY, PROCESS, PRODUCT, ROLE,
    TERMINOLOGY, TOPIC, entity_key,
)
from src.intelligence.processes import ProcessRecord
from src.intelligence.roles import RoleRecord
from src.intelligence.rules import Rule
from src.intelligence.text import sentences

# ---------------- Evidence ----------------

TABLE, HEADER, TEXT_MENTION, TITLE_MATCH, STEP, RULE, ORG_CHART, SENTENCE, DERIVED = (
    "table", "header", "text_mention", "title_match", "step_pattern", "rule_pattern",
    "org_chart", "sentence_pattern", "derived")


@dataclass
class Evidence:
    source_file: str                 # 'employees.csv' or 'customer_onboarding.md'
    source_document: str | None      # document entity ID, when a document is the source
    detail: str                      # 'column role_id', 'header Related Roles', 'step 5', ...
    method: str
    confidence: float
    chunk_ids: list[str] = field(default_factory=list)
    quote: str | None = None
    historical: bool = False         # comes from an archived document


@dataclass
class KnowledgeRelationship:
    relationship_id: str
    source_id: str
    source_type: str
    relationship: str
    target_id: str
    target_type: str
    evidence: list[Evidence] = field(default_factory=list)
    endpoint_archived: bool = False

    @property
    def source_key(self) -> str:
        return entity_key(self.source_type, self.source_id)

    @property
    def target_key(self) -> str:
        return entity_key(self.target_type, self.target_id)

    @property
    def confidence(self) -> float:
        return max(e.confidence for e in self.evidence)

    @property
    def source_document(self) -> str | None:
        """The document behind the strongest evidence (None if only tables support it)."""
        best = sorted(self.evidence, key=lambda e: (-e.confidence, e.source_document is None))
        return next((e.source_document for e in best if e.source_document), None)

    @property
    def historical(self) -> bool:
        """True if the link only reflects history: an archived end, or archived-only evidence."""
        return self.endpoint_archived or all(e.historical for e in self.evidence)

    def to_dict(self) -> dict:
        return {
            "relationship_id": self.relationship_id,
            "source_key": self.source_key, "source_id": self.source_id, "source_type": self.source_type,
            "relationship": self.relationship,
            "target_key": self.target_key, "target_id": self.target_id, "target_type": self.target_type,
            "source_document": self.source_document,
            "confidence": self.confidence,
            "historical": self.historical,
            "evidence": [asdict(e) for e in self.evidence],
        }


def relationship_id(source_key: str, relationship: str, target_key: str) -> str:
    digest = hashlib.sha1(f"{source_key}|{relationship}|{target_key}".encode("utf-8")).hexdigest()
    return f"REL-{digest[:10].upper()}"


class RelationshipBuilder:
    def __init__(self, index: EntityIndex):
        self.index = index
        self.by_key: dict[tuple[str, str, str], KnowledgeRelationship] = {}
        self.dangling: list[dict] = []

    def _archived(self, entity_type: str, entity_id: str) -> bool:
        e = self.index.get(entity_type, entity_id)
        return bool(e) and (e.attributes.get("status") or "").lower() == "archived"

    def add(self, source_type: str, source_id: str, relationship: str,
            target_type: str, target_id: str, evidence: Evidence) -> None:
        if not source_id or not target_id:
            return
        s_key, t_key = entity_key(source_type, source_id), entity_key(target_type, target_id)
        missing = [k for k in (s_key, t_key) if k not in self.index.entities]
        if missing:
            self.dangling.append({"source": s_key, "relationship": relationship, "target": t_key,
                                  "missing": missing, "evidence": asdict(evidence)})
            return
        if s_key == t_key:
            return
        key = (s_key, relationship, t_key)
        rel = self.by_key.get(key)
        if rel is None:
            rel = KnowledgeRelationship(
                relationship_id(s_key, relationship, t_key), source_id, source_type, relationship,
                target_id, target_type,
                endpoint_archived=self._archived(source_type, source_id) or self._archived(target_type, target_id))
            self.by_key[key] = rel
        if evidence not in rel.evidence:
            rel.evidence.append(evidence)

    def doc_evidence(self, doc_id: str, detail: str, method: str, confidence: float,
                     chunk_ids: list[str] | None = None, quote: str | None = None) -> Evidence:
        doc = self.index.get(DOCUMENT, doc_id)
        a = doc.attributes if doc else {}
        return Evidence(a.get("source_filename") or doc_id, doc_id, detail, method, confidence,
                        list(chunk_ids or []), quote, (a.get("status") or "").lower() == "archived")

    def results(self) -> list[KnowledgeRelationship]:
        return sorted(self.by_key.values(), key=lambda r: (r.relationship, r.source_key, r.target_key))


# ---------------- 1. Phase 2 relationships, lifted ----------------

# Phase 2 type -> (source type, relationship, target type, method, confidence).
# PROCESS_DOCUMENTED_IN is rebuilt version-aware below instead: Phase 2 linked
# the archived process PROC-IMPL-001-V1 to document ID 'PROC-IMPL-001', which
# names the CURRENT file.
_PHASE2 = {
    RT.EMPLOYEE_HAS_ROLE: (EMPLOYEE, "has_role", ROLE, TABLE, CONF_STRUCTURED),
    RT.EMPLOYEE_AT_LOCATION: (EMPLOYEE, "located_in", LOCATION, TABLE, CONF_STRUCTURED),
    RT.EMPLOYEE_REPORTS_TO: (EMPLOYEE, "reports_to", EMPLOYEE, TABLE, CONF_STRUCTURED),
    RT.ROLE_IN_DEPARTMENT: (ROLE, "in_department", DEPARTMENT, TABLE, CONF_STRUCTURED),
    RT.ROLE_RELATED_TO_PROCESS: (ROLE, "owns", PROCESS, TABLE, CONF_STRUCTURED),
    RT.PROCESS_OWNED_BY_DEPARTMENT: (PROCESS, "owned_by", DEPARTMENT, TABLE, CONF_STRUCTURED),
    RT.PRODUCT_OWNED_BY_DEPARTMENT: (PRODUCT, "owned_by", DEPARTMENT, TABLE, CONF_STRUCTURED),
    RT.PRODUCT_DOCUMENTED_IN: (PRODUCT, "documented_by", DOCUMENT, TABLE, CONF_STRUCTURED),
    RT.DOCUMENT_RELATED_TO_PROCESS: (DOCUMENT, "related_to", PROCESS, HEADER, CONF_STRUCTURED),
    RT.DOCUMENT_RELATED_TO_DOCUMENT: (DOCUMENT, "references", DOCUMENT, HEADER, CONF_STRUCTURED),
    RT.DOCUMENT_RELATED_TO_ROLE: (DOCUMENT, "relevant_to", ROLE, HEADER, CONF_STRUCTURED),
    RT.DOCUMENT_FOR_AUDIENCE: (DOCUMENT, "for_audience", AUDIENCE, HEADER, CONF_STRUCTURED),
    RT.DOCUMENT_MENTIONS_PROCESS: (DOCUMENT, "mentions", PROCESS, TEXT_MENTION, CONF_KEYWORD),
    RT.DOCUMENT_MENTIONS_DOCUMENT: (DOCUMENT, "mentions", DOCUMENT, TEXT_MENTION, CONF_KEYWORD),
    RT.DOCUMENT_MENTIONS_PRODUCT: (DOCUMENT, "mentions", PRODUCT, TEXT_MENTION, CONF_KEYWORD),
    RT.DOCUMENT_MENTIONS_ROLE: (DOCUMENT, "mentions", ROLE, TEXT_MENTION, CONF_KEYWORD),
    RT.DOCUMENT_MENTIONS_LOCATION: (DOCUMENT, "mentions", LOCATION, TEXT_MENTION, CONF_KEYWORD),
    RT.DOCUMENT_HAS_TOPIC: (DOCUMENT, "uses_term", TERMINOLOGY, TEXT_MENTION, CONF_KEYWORD),
    RT.PROCESS_RELATED_TO_PRODUCT: (PROCESS, "related_to", PRODUCT, TEXT_MENTION, CONF_KEYWORD),
    RT.LOCATION_HAS_CHECKLIST: (LOCATION, "has_checklist", CHECKLIST, TITLE_MATCH, CONF_PATTERN),
    RT.CULTURE_TONE_GUIDANCE: (LOCATION, "has_tone_guidance", DOCUMENT, TITLE_MATCH, CONF_PATTERN),
}
_FIRST_CHUNK = re.compile(r"first in (\S+)$")


def _lift_phase2(b: RelationshipBuilder, data: Phase2Data, index: EntityIndex) -> None:
    term_by_phase2 = {e.attributes.get("phase2_topic_id"): e.entity_id
                      for e in index.of_type(TERMINOLOGY) if e.attributes.get("phase2_topic_id")}
    for rel in data.relationships:
        spec = _PHASE2.get(rel.relation_type)
        if spec is None:
            continue
        s_type, name, t_type, method, conf = spec
        source_file, _, detail = rel.evidence.partition(":")
        doc_id = index.document_for_relationship(rel)
        chunk = _FIRST_CHUNK.search(rel.evidence)

        source, target = rel.source, rel.target
        if s_type == DOCUMENT:
            source = doc_id or rel.source      # header ID -> the exact file (fixes the archived v1.0)
        if t_type == TERMINOLOGY:
            target = term_by_phase2.get(rel.target, rel.target)
        if t_type in (DOCUMENT, CHECKLIST) and method == TITLE_MATCH:
            target = doc_id or rel.target

        evidence = (b.doc_evidence(doc_id, detail.strip(), method, conf, [chunk.group(1)] if chunk else [])
                    if doc_id else
                    Evidence(source_file.strip(), None, detail.strip(), method, conf))
        b.add(s_type, source, name, t_type, target, evidence)


# ---------------- 2. New Phase 3 relationships ----------------

_PRESENCE = re.compile(r"\b(?:based|headquartered|concentrated|located|pod|hub)\b", re.I)


def _from_tables(b: RelationshipBuilder, data: Phase2Data, index: EntityIndex) -> None:
    for row in data.tables.get("employees", []):
        b.add(EMPLOYEE, row.get("employee_id"), "works_in", DEPARTMENT, row.get("dept_id"),
              Evidence("employees.csv", None, "column dept_id", TABLE, CONF_STRUCTURED))
    for dept in index.of_type(DEPARTMENT):
        for rid in dept.attributes.get("head_role_ids", []):
            b.add(DEPARTMENT, dept.entity_id, "headed_by", ROLE, rid,
                  Evidence("departments.csv", None, "column head_role", TABLE, CONF_STRUCTURED))
        b.add(DEPARTMENT, dept.entity_id, "head_located_in", LOCATION, dept.attributes.get("head_location_id"),
              Evidence("departments.csv", None, "column head_location", TABLE, CONF_STRUCTURED))
    for proc in index.of_type(PROCESS):
        b.add(PROCESS, proc.entity_id, "documented_by", DOCUMENT, proc.attributes.get("document_id"),
              Evidence("processes.csv", None, "columns doc_id + version (matched to the exact file)",
                       TABLE, CONF_STRUCTURED))


def _from_documents(b: RelationshipBuilder, data: Phase2Data, index: EntityIndex) -> None:
    for doc in index.of_type(DOCUMENT):
        a = doc.attributes
        for dept_id in a.get("department_ids", []):
            b.add(DOCUMENT, doc.entity_id, "belongs_to", DEPARTMENT, dept_id,
                  b.doc_evidence(doc.entity_id, f"header Department '{a.get('department')}'", HEADER, CONF_STRUCTURED))
        for rid in a.get("owner_role_ids", []):
            b.add(ROLE, rid, "owns", DOCUMENT, doc.entity_id,
                  b.doc_evidence(doc.entity_id, f"header Owner '{a.get('owner_role')}'", HEADER, CONF_STRUCTURED))

        # Department description: "Based in Munich (LOC-MUC)" -> Department present_in Location
        if "department description" in (a.get("document_type") or "").lower() and len(a.get("department_ids", [])) == 1:
            dept_id = a["department_ids"][0]
            locations = {loc.entity_id: loc for loc in index.of_type(LOCATION)}
            for section in data.sections(a["header_document_id"], a["version"]):
                if section.name.lower().endswith("roles"):
                    continue
                for sentence in sentences(section.text):
                    # Clause level: in "Based in Munich (LOC-MUC), working closely with
                    # Engineering (Bengaluru)", only Munich is where Product is based.
                    clauses = [c for c in re.split(r"[,;—]|\s+and\s+", sentence) if _PRESENCE.search(c)]
                    for loc_id, loc in locations.items():
                        if any(re.search(rf"\b{re.escape(loc_id)}\b|\b{re.escape(loc.name)}\b", c) for c in clauses):
                            b.add(DEPARTMENT, dept_id, "present_in", LOCATION, loc_id,
                                  b.doc_evidence(doc.entity_id, f"section '{section.name}'", SENTENCE, CONF_PATTERN,
                                                 data.find_chunks(a["header_document_id"], a["version"], sentence),
                                                 sentence))

    for policy in index.of_type(POLICY):
        b.add(POLICY, policy.entity_id, "documented_by", DOCUMENT, policy.entity_id,
              b.doc_evidence(policy.entity_id, "Document Type header", HEADER, CONF_STRUCTURED))
    for checklist in index.of_type(CHECKLIST):
        b.add(CHECKLIST, checklist.entity_id, "documented_by", DOCUMENT, checklist.entity_id,
              b.doc_evidence(checklist.entity_id, "Document Type header", HEADER, CONF_STRUCTURED))


def _from_terminology_and_customers(b: RelationshipBuilder, data: Phase2Data, index: EntityIndex) -> None:
    for term in index.of_type(TERMINOLOGY):
        glossary_doc = term.source_documents[0] if term.source_documents else None
        if glossary_doc:
            b.add(TERMINOLOGY, term.entity_id, "defined_in", DOCUMENT, glossary_doc,
                  b.doc_evidence(glossary_doc, term.sources[0], TABLE if term.attributes["kind"] == "glossary_term"
                                 else SENTENCE, CONF_STRUCTURED, term.attributes.get("chunk_ids")))
        if term.attributes.get("kind") == "terminology_note":
            for ref in term.attributes.get("referenced_ids", []):
                if index.get(PROCESS, ref):
                    b.add(TERMINOLOGY, term.entity_id, "disambiguates", PROCESS, ref,
                          b.doc_evidence(glossary_doc, term.sources[0], SENTENCE, CONF_DOCUMENTED,
                                         term.attributes.get("chunk_ids"), term.attributes.get("note_text")))

    by_chunk = {c.chunk_id: c for c in data.chunks}
    for customer in index.of_type(CUSTOMER):
        pattern = re.compile(rf"(?<![\w-]){re.escape(customer.name)}(?![\w-])")
        for chunk_id in customer.attributes.get("mentioned_in_chunks", []):
            chunk = by_chunk[chunk_id]
            doc_id = index.document_for_chunk(chunk)
            if not doc_id:
                continue
            quote = next((s for s in sentences(chunk.chunk_text) if pattern.search(s)), None)
            b.add(DOCUMENT, doc_id, "mentions", CUSTOMER, customer.entity_id,
                  b.doc_evidence(doc_id, f"chunk {chunk_id}", TEXT_MENTION, CONF_KEYWORD, [chunk_id], quote))
            # Product documentation naming a customer as a customer -> Product used_by Customer
            products = [p for p in index.of_type(PRODUCT) if p.attributes.get("document_id") == doc_id]
            if products and quote and re.search(r"\bcustomer\b", quote, re.I):
                for product in products:
                    b.add(PRODUCT, product.entity_id, "used_by", CUSTOMER, customer.entity_id,
                          b.doc_evidence(doc_id, f"section '{chunk.section}'", SENTENCE, CONF_PATTERN, [chunk_id], quote))


def _from_processes(b: RelationshipBuilder, processes: list[ProcessRecord]) -> None:
    for proc in processes:
        if not proc.document_id:
            continue
        for p in proc.participants:
            if any(h.startswith("document header") for h in p["how"]):
                b.add(ROLE, p["role_id"], "participates_in", PROCESS, proc.process_id,
                      b.doc_evidence(proc.document_id, "header Related Roles", HEADER, CONF_STRUCTURED))
        for step in proc.steps:
            for rid in step["role_ids"]:
                b.add(ROLE, rid, "performs", PROCESS, proc.process_id,
                      b.doc_evidence(proc.document_id, f"step {step['number']} ({step['title'] or step['section']})",
                                     STEP, CONF_PATTERN, step["chunk_ids"], step["text"]))
            for pid in step["product_ids"]:
                b.add(PROCESS, proc.process_id, "uses", PRODUCT, pid,
                      b.doc_evidence(proc.document_id, f"step {step['number']}", STEP, CONF_PATTERN,
                                     step["chunk_ids"], step["text"]))
        for link in proc.related_processes:
            b.add(PROCESS, proc.process_id, "related_to", PROCESS, link["process_id"],
                  b.doc_evidence(proc.document_id, link["how"], HEADER, CONF_STRUCTURED))
        for t in proc.triggers:
            detail = f"step {t['step']}" + (f", hands over {t['via_artifact']}" if t.get("via_artifact") else "")
            b.add(PROCESS, proc.process_id, "triggers", PROCESS, t["process_id"],
                  b.doc_evidence(proc.document_id, detail, STEP, t["confidence"], t["chunk_ids"], t["quote"]))
        for pol in proc.related_policies:
            for how in pol["how"]:
                conf = CONF_PATTERN if how.startswith("step") else CONF_KEYWORD
                b.add(PROCESS, proc.process_id, "governed_by", POLICY, pol["policy_id"],
                      b.doc_evidence(pol["policy_id"] if how.startswith("policy header") else proc.document_id,
                                     how, STEP if how.startswith("step") else HEADER, conf))


def _from_rules(b: RelationshipBuilder, rules: list[Rule]) -> None:
    for rule in rules:
        name = "approves" if rule.rule_type == "approval" else "handles_escalations_for"
        for rid in rule.role_ids:
            for pid in rule.process_ids:
                b.add(ROLE, rid, name, PROCESS, pid,
                      b.doc_evidence(rule.document_id, f"{rule.rule_type} rule {rule.rule_id} ({rule.context})",
                                     RULE, rule.confidence, rule.chunk_ids, rule.statement))


def _from_roles(b: RelationshipBuilder, roles: list[RoleRecord], index: EntityIndex) -> None:
    for role in roles:
        for line in role.reports_to:
            if not line["role_id"]:
                continue
            filename = line["source"].split(":")[0]
            doc_id = index.doc_by_filename.get(filename.lower())
            ev = (b.doc_evidence(doc_id, line["source"], ORG_CHART if "Reporting" in line["source"] else SENTENCE,
                                 line["confidence"], [], line["quote"])
                  if doc_id else Evidence(filename, None, line["source"], ORG_CHART, line["confidence"], [], line["quote"]))
            b.add(ROLE, role.role_id, "reports_to", ROLE, line["role_id"], ev)


_TOPIC_METHOD = {"home_document": HEADER, "linked": TEXT_MENTION, "text_similarity": "tfidf_similarity"}


def _from_topics(b: RelationshipBuilder, topics) -> None:
    for topic in topics:
        anchor_type, anchor_id = topic.anchor_key.split(":", 1)
        b.add(TOPIC, topic.topic_id, "about", anchor_type, anchor_id,
              Evidence("topics.py", None, f"{topic.level} topic anchored on {topic.anchor_key}", DERIVED, CONF_STRUCTURED))
        for d in topic.documents:
            sim = d.get("similarity")
            quote = (f"similarity {sim['similarity']}; shared terms: {', '.join(sim['shared_terms'])}" if sim else None)
            b.add(DOCUMENT, d["document_id"], "discusses", TOPIC, topic.topic_id,
                  b.doc_evidence(d["document_id"], d["detail"], _TOPIC_METHOD[d["method"]], d["confidence"],
                                 sim["chunk_ids"] if sim else [], quote))


def build_relationships(data: Phase2Data, index: EntityIndex, processes: list[ProcessRecord],
                        roles: list[RoleRecord], rules: list[Rule], topics=None) -> RelationshipBuilder:
    b = RelationshipBuilder(index)
    _lift_phase2(b, data, index)
    _from_tables(b, data, index)
    _from_documents(b, data, index)
    _from_terminology_and_customers(b, data, index)
    _from_processes(b, processes)
    _from_rules(b, rules)
    _from_roles(b, roles, index)
    _from_topics(b, topics or [])
    return b

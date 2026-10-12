"""
Entity discovery: every person, team, role, process, product, customer,
policy, document, location, checklist and term Nexora's knowledge contains.

Order of trust, highest first:
  1. CSV tables and document header fields   (structured, CONF_STRUCTURED)
  2. What a Nexora document explicitly says   (e.g. glossary "Same role.")
  3. Fixed text patterns in a known section   (e.g. bold names under "Customers")

Nothing is created from similarity or guessing. Values that can't be
resolved go into EntityIndex.notes, so a person can see them.

Topics are added later by topics.py (Milestone 3).
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.metadata.discovery import _PRODUCT_ALIASES, _TABLE_ROW
from src.metadata.relationships import _ROLE_ALIASES, Relationship, RelationType

from src.intelligence.loader import Phase2Data
from src.intelligence.models import (
    AUDIENCE, CHECKLIST, CONF_DOCUMENTED, CONF_PATTERN, CONF_STRUCTURED,
    CUSTOMER, DEPARTMENT, DOCUMENT, EMPLOYEE, LOCATION, POLICY, PROCESS,
    PRODUCT, ROLE, TERMINOLOGY, Entity, entity_key,
)
from src.intelligence.normalize import (
    DepartmentResolver, resolve_roles, slug, unique, version_key,
)
from src.intelligence.text import blocks

UNKNOWN = "unknown"

# 'Customer Success Manager (CSM)' -> ('Customer Success Manager', 'CSM'); acronyms only.
_ROLE_HEADING = re.compile(r"^(.+?)\s*\(([A-Z]{2,6})\)$")


@dataclass
class EntityIndex:
    """All discovered entities, looked up by key, type, filename or header ID."""
    entities: dict[str, Entity] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)          # unresolved, reported
    collisions: list[str] = field(default_factory=list)     # same key added twice

    # Document lookups. Chunks and Phase 2 relationships use the HEADER ID,
    # which two files can share (PROC-IMPL-001 v1.0 and v2.0), so filename
    # and (header ID, version) are the reliable ways to find the right file.
    doc_by_filename: dict[str, str] = field(default_factory=dict)            # filename (lower) -> entity_id
    doc_by_header: dict[tuple[str, str], str] = field(default_factory=dict)  # (header_id, version) -> entity_id

    # Unresolved names, by the kind of entity they were supposed to be.
    # duplicates.py looks for candidates for each of them.
    unresolved_names: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    # Parsed glossary terminology notes (TN-xxx), reused by duplicates.py.
    glossary_notes: list[dict] = field(default_factory=list)

    def document_for_chunk(self, chunk) -> str | None:
        """Chunks carry the header ID; (header ID, version) pins the file."""
        return self.doc_by_header.get((chunk.document_id, version_key(chunk.version)))

    def add(self, entity: Entity) -> None:
        if entity.key in self.entities:
            # Reported, never overwritten: the first record wins and the
            # validator (Milestone 5) flags the clash.
            self.collisions.append(f"{entity.key} added twice (sources: {entity.sources})")
            return
        self.entities[entity.key] = entity

    def get(self, entity_type: str, entity_id: str) -> Entity | None:
        return self.entities.get(entity_key(entity_type, entity_id))

    def of_type(self, entity_type: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def document_for_relationship(self, rel: Relationship) -> str | None:
        """Phase 2 evidence for document-based links starts with the filename
        ('customer_onboarding.md: Related Roles'), which pins the exact file."""
        filename = rel.evidence.split(":")[0].strip().lower()
        return self.doc_by_filename.get(filename)

    def to_dict(self) -> dict:
        counts = Counter(e.entity_type for e in self.entities.values())
        return {
            "counts_by_type": dict(sorted(counts.items())),
            "entities": [e.to_dict() for e in sorted(self.entities.values(), key=lambda e: e.key)],
            "notes": self.notes,
            "unresolved_names": {kind: sorted(names) for kind, names in sorted(self.unresolved_names.items())},
            "collisions": self.collisions,
        }


# ---------------- Documents ----------------

def _discover_documents(data: Phase2Data, index: EntityIndex, depts: DepartmentResolver) -> None:
    """One entity per document FILE, identified by its documents.csv ID.

    documents.csv is the master index, so its ID is the stable one.
    If a file's header disagrees (the archived onboarding file says
    PROC-IMPL-001, the index says PROC-IMPL-001-V1), the header ID is kept
    as an alias and the clash is reported by duplicates.py.
    """
    registry = data.registry
    csv_by_file = registry.by_name.get("document_files", {})
    csv_rows = {r["doc_id"]: r for r in data.tables.get("documents", [])}
    seen: set[str] = set()

    for m in data.documents:
        csv_id = csv_by_file.get(m.source_filename.lower())
        entity_id = csv_id or (m.document_id if m.document_id != UNKNOWN else None)
        if entity_id is None:
            index.notes.append(f"{m.source_filename}: no Document ID in header or documents.csv, skipped")
            continue

        aliases = []
        if csv_id and m.document_id not in (csv_id, UNKNOWN):
            aliases.append(m.document_id)

        dept_raw = m.department if m.department != UNKNOWN else csv_rows.get(entity_id, {}).get("department", "")
        dept_ids, dept_missing = depts.resolve(dept_raw)
        for piece in dept_missing:
            index.unresolved_names[DEPARTMENT].add(piece)
            index.notes.append(f"{m.source_filename}: department '{piece}' is not in departments.csv")

        owner_ids, owner_missing = resolve_roles(m.owner_role, registry)
        for name in owner_missing:
            index.unresolved_names[ROLE].add(name)
            index.notes.append(f"{m.source_filename}: owner '{name}' is not in roles.csv")

        sources = [f"{m.source_filename}: document header"]
        if csv_id:
            sources.append("documents.csv: matched on file_path")

        index.add(Entity(
            entity_id=entity_id,
            entity_type=DOCUMENT,
            name=m.title,
            aliases=aliases,
            source_documents=[entity_id],
            sources=sources,
            confidence=CONF_STRUCTURED,
            attributes={
                "header_document_id": m.document_id,
                "source_filename": m.source_filename,
                "document_type": m.document_type,
                "department": dept_raw,
                "department_ids": dept_ids,
                "owner_role": m.owner_role,
                "owner_role_ids": owner_ids,
                "version": m.version,
                "effective_date": m.effective_date,
                "status": m.status,
                "related_processes": m.related_processes,
                "related_roles": m.related_roles,
                "chunk_count": data.chunk_counts.get(m.source_filename, 0),
                "ingested": True,
            },
        ))
        seen.add(entity_id)
        index.doc_by_filename[m.source_filename.lower()] = entity_id
        index.doc_by_header[(m.document_id, version_key(m.version))] = entity_id

    # Listed in documents.csv but not ingested (e.g. Phase 2 ran with
    # --exclude-archived, or the file is missing). Kept, clearly flagged.
    for doc_id, row in csv_rows.items():
        if doc_id in seen:
            continue
        index.notes.append(f"{doc_id}: listed in documents.csv but no ingested file ({row.get('file_path')})")
        dept_ids, _ = depts.resolve(row.get("department", ""))
        owner_ids, _ = resolve_roles(row.get("owner_role", ""), registry)
        index.add(Entity(
            entity_id=doc_id, entity_type=DOCUMENT, name=row.get("title", doc_id),
            source_documents=[doc_id], sources=["documents.csv"], confidence=CONF_STRUCTURED,
            attributes={
                "header_document_id": None, "source_filename": row.get("file_path"),
                "document_type": row.get("doc_type"), "department": row.get("department"),
                "department_ids": dept_ids, "owner_role": row.get("owner_role"),
                "owner_role_ids": owner_ids, "version": row.get("version"),
                "effective_date": UNKNOWN, "status": row.get("status"),
                "related_processes": [], "related_roles": [], "chunk_count": 0,
                "ingested": False,
            },
        ))


def _docs_by_type(index: EntityIndex, *type_words: str) -> list[Entity]:
    """Documents whose header Document Type contains any of the given words."""
    return [d for d in index.of_type(DOCUMENT)
            if any(w in (d.attributes.get("document_type") or "").lower() for w in type_words)]


# ---------------- Glossary: terms, terminology notes, documented aliases ----------------

# '**TN-006 — "Project Lead" vs. "Implementation Manager."**'
_TN_HEADING = re.compile(r"\*\*(TN-\d+)\s*[—–-]+\s*(.+?)\*\*", re.S)
_QUOTED = re.compile(r'"([^"]+)"')
_DOC_IDS = re.compile(r"\b(?:PROC|DOC)-[A-Z]+(?:-[A-Z0-9]+)*\b")

# How each note relates its terms, decided by the note's OWN wording.
# Checked in order; the phrase that matched is kept as evidence.
_NOTE_RULES = [
    ("overloaded", re.compile(r"used for two different things|overloaded", re.I)),
    ("not_interchangeable", re.compile(r"not (?:always )?(?:perfectly )?interchangeable", re.I)),
    ("equivalent", re.compile(r"\bsame role\b|same underlying concept|\bsame concept\b", re.I)),
]


@dataclass
class _Glossary:
    document_id: str | None = None
    rows: list[tuple[str, str]] = field(default_factory=list)           # (term, definition)
    notes: list[dict] = field(default_factory=list)                     # parsed TN-xxx notes


def _read_glossary(data: Phase2Data, index: EntityIndex) -> _Glossary:
    glossary = _Glossary()
    docs = _docs_by_type(index, "glossary")
    if not docs:
        index.notes.append("No document of type 'Glossary' found: no terminology discovered")
        return glossary
    doc = docs[0]
    glossary.document_id = doc.entity_id
    sections = data.sections(doc.attributes["header_document_id"], doc.attributes["version"])

    for section in sections:
        for line in section.text.splitlines():
            match = _TABLE_ROW.match(line.strip())
            if not match:
                continue
            term, definition = match.group(1).strip(), match.group(2).strip()
            if term.lower() == "term" or set(term) <= set("-: "):
                continue   # header row and |---|---| separator
            glossary.rows.append((term, definition))

        headings = list(_TN_HEADING.finditer(section.text))
        for i, h in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(section.text)
            heading, body = h.group(2).strip(), " ".join(section.text[h.end():end].split())
            kind, phrase = "unclassified", None
            for rule_kind, pattern in _NOTE_RULES:
                found = pattern.search(heading + " " + body)
                if found:
                    kind, phrase = rule_kind, found.group(0)
                    break
            glossary.notes.append({
                "note_id": h.group(1),
                "heading": heading,
                "body": body,
                "terms": unique(t.strip(" .,") for t in _QUOTED.findall(heading)),
                "relation": kind,
                "matched_phrase": phrase,
                "referenced_ids": unique(_DOC_IDS.findall(body)),
                "chunk_ids": section.chunk_ids,
            })
    return glossary


def _discover_terminology(glossary: _Glossary, data: Phase2Data, index: EntityIndex) -> None:
    if glossary.document_id is None:
        return
    where = index.get(DOCUMENT, glossary.document_id).attributes["source_filename"]

    # Phase 2 linked documents to glossary terms as TOPIC-<slug>.
    # Phase 3 calls them terminology (what a word MEANS) and keeps "topic"
    # for what a document is ABOUT. The Phase 2 ID is kept as an attribute.
    mentioned_in: dict[str, set[str]] = defaultdict(set)
    for rel in data.relationships:
        if rel.relation_type == RelationType.DOCUMENT_HAS_TOPIC:
            doc_id = index.document_for_relationship(rel)
            if doc_id:
                mentioned_in[rel.target].add(doc_id)

    for term, definition in glossary.rows:
        phase2_id = f"TOPIC-{slug(term)}"
        parts = [t.strip() for t in term.split("/") if t.strip()]
        index.add(Entity(
            entity_id=f"TERM-{slug(term)}",
            entity_type=TERMINOLOGY,
            name=term,
            aliases=parts if len(parts) > 1 else [],
            source_documents=unique([glossary.document_id, *sorted(mentioned_in[phase2_id])]),
            sources=[f"{where}: glossary table row '{term}'"],
            confidence=CONF_STRUCTURED,
            attributes={"kind": "glossary_term", "definition": definition, "phase2_topic_id": phase2_id},
        ))

    for note in glossary.notes:
        index.add(Entity(
            entity_id=f"TERM-{note['note_id']}",
            entity_type=TERMINOLOGY,
            name=note["heading"],
            aliases=note["terms"],
            source_documents=unique([glossary.document_id, *note["referenced_ids"]]),
            sources=[f"{where}: {note['note_id']}"],
            confidence=CONF_DOCUMENTED if note["relation"] != "unclassified" else CONF_PATTERN,
            attributes={
                "kind": "terminology_note",
                "relation": note["relation"],
                "matched_phrase": note["matched_phrase"],
                "terms": note["terms"],
                "referenced_ids": note["referenced_ids"],
                "note_text": note["body"],
                "chunk_ids": note["chunk_ids"],
            },
        ))


def _documented_role_aliases(glossary: _Glossary, data: Phase2Data, index: EntityIndex) -> dict[str, dict[str, list[str]]]:
    """role_id -> {alias: [evidence, ...]}, from three documented sources."""
    role_by_title = data.registry.by_name.get("roles", {})
    found: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    where = (index.get(DOCUMENT, glossary.document_id).attributes["source_filename"]
             if glossary.document_id else "glossary")

    # 1. Glossary rows whose definition IS a role title: 'AE' = 'Account Executive'.
    for term, definition in glossary.rows:
        role_id = role_by_title.get(definition.strip().lower())
        if role_id:
            found[role_id][term].append(f"{where}: '{term}' defined as '{definition}'")

    # 2. Terminology notes that say two terms are the same: TN-006 'Same role.'
    for note in glossary.notes:
        if note["relation"] != "equivalent":
            continue
        role_terms = [t for t in note["terms"] if t.lower() in role_by_title]
        other_terms = [t for t in note["terms"] if t.lower() not in role_by_title]
        for role_term in role_terms:
            for alias in other_terms:
                found[role_by_title[role_term.lower()]][alias].append(
                    f"{where}: {note['note_id']} says '{note['matched_phrase']}'")

    # 3. Department documents introduce acronyms in role headings:
    #    '**HR Business Partner (HRBP)** — Partners with department heads...'
    for doc in _docs_by_type(index, "department description"):
        a = doc.attributes
        for section in data.sections(a["header_document_id"], a["version"]):
            if not section.name.lower().endswith("roles"):
                continue
            for block in blocks(section.text):
                match = _ROLE_HEADING.match(block.title or "")
                if not match:
                    continue
                role_id = role_by_title.get(match.group(1).strip().lower())
                if role_id:
                    found[role_id][match.group(2)].append(
                        f"{a['source_filename']}: role heading '{block.title}'")

    # 4. Phase 2's alias table, for aliases no document row states directly
    #    (e.g. 'Demand Gen Manager', which appears in marketing.md's header).
    for alias, title in _ROLE_ALIASES.items():
        role_id = role_by_title.get(title.lower())
        if not role_id:
            continue
        shown = alias.upper() if len(alias) <= 3 else alias.title()
        if not any(a.lower() == alias for a in found[role_id]):
            found[role_id][shown].append("Phase 2 alias table (src/metadata/relationships.py)")
    return found


# ---------------- Tables: departments, roles, employees, locations, products, processes ----------------

def _docs_linked(data: Phase2Data, index: EntityIndex, types: set[RelationType],
                 target: str | None = None, source: str | None = None) -> list[str]:
    """Documents that Phase 2 linked to an entity (as target or source)."""
    out = []
    for rel in data.relationships:
        if rel.relation_type not in types:
            continue
        if (target is not None and rel.target == target) or (source is not None and rel.source == source):
            doc_id = index.document_for_relationship(rel)
            if doc_id:
                out.append(doc_id)
    return sorted(set(out))


def _discover_departments(data: Phase2Data, index: EntityIndex, depts: DepartmentResolver) -> None:
    employees_per_dept = Counter(r.get("dept_id") for r in data.tables.get("employees", []))
    for row in data.tables.get("departments", []):
        dept_id = row["dept_id"]
        head_ids, head_missing = resolve_roles(row.get("head_role", ""), data.registry)
        for name in head_missing:
            index.unresolved_names[ROLE].add(name)
        docs = [d.entity_id for d in index.of_type(DOCUMENT) if dept_id in d.attributes.get("department_ids", [])]
        index.add(Entity(
            entity_id=dept_id,
            entity_type=DEPARTMENT,
            name=row["name"],
            aliases=sorted(depts.variants_seen.get(dept_id, set())),   # spellings seen in headers
            source_documents=sorted(docs),
            sources=["departments.csv: dept_id"],
            attributes={
                "head_role": row.get("head_role"),
                "head_role_ids": head_ids,
                "head_location_id": row.get("head_location"),
                "parent_dept_id": row.get("parent_dept_id") or None,
                # Two different numbers, kept side by side, never reconciled:
                "approx_headcount_stated": row.get("approx_headcount"),
                "employees_in_employees_csv": employees_per_dept.get(dept_id, 0),
            },
        ))


def _discover_roles(data: Phase2Data, index: EntityIndex, aliases: dict[str, dict[str, list[str]]]) -> None:
    employees_per_role = Counter(r.get("role_id") for r in data.tables.get("employees", []))
    role_links = {RelationType.DOCUMENT_RELATED_TO_ROLE, RelationType.DOCUMENT_MENTIONS_ROLE}
    for row in data.tables.get("roles", []):
        role_id = row["role_id"]
        role_aliases = aliases.get(role_id, {})
        index.add(Entity(
            entity_id=role_id,
            entity_type=ROLE,
            name=row["title"],
            aliases=sorted(role_aliases),
            source_documents=_docs_linked(data, index, role_links, target=role_id),
            sources=["roles.csv: role_id"],
            attributes={
                "dept_id": row.get("dept_id"),
                "employee_count": employees_per_role.get(role_id, 0),
                "alias_evidence": {a: ev for a, ev in sorted(role_aliases.items())},
            },
        ))


_EMPLOYEE_FIELDS = ("role_id", "role_title", "dept_id", "location_id", "employment_type",
                    "join_date", "onboarding_stage", "manager_id", "international_relocation_germany")


def _discover_employees(data: Phase2Data, index: EntityIndex) -> None:
    for row in data.tables.get("employees", []):
        index.add(Entity(
            entity_id=row["employee_id"],
            entity_type=EMPLOYEE,
            name=f"{row.get('first_name', '').strip()} {row.get('last_name', '').strip()}".strip(),
            sources=["employees.csv: employee_id"],
            attributes={f: (row.get(f) or None) for f in _EMPLOYEE_FIELDS},
        ))


def _discover_locations(data: Phase2Data, index: EntityIndex) -> None:
    links = {RelationType.DOCUMENT_MENTIONS_LOCATION}
    guidance = {RelationType.LOCATION_HAS_CHECKLIST, RelationType.CULTURE_TONE_GUIDANCE}
    for row in data.tables.get("locations", []):
        loc_id = row["location_id"]
        docs = _docs_linked(data, index, links, target=loc_id) + _docs_linked(data, index, guidance, source=loc_id)
        index.add(Entity(
            entity_id=loc_id,
            entity_type=LOCATION,
            name=row["city"],
            source_documents=sorted(set(docs)),
            sources=["locations.csv: location_id"],
            attributes={"country": row.get("country"), "function": row.get("function")},
        ))


def _discover_products(data: Phase2Data, index: EntityIndex) -> None:
    links = {RelationType.DOCUMENT_MENTIONS_PRODUCT}
    for row in data.tables.get("products", []):
        prod_id, name = row["product_id"], row["name"]
        aliases = [short for short, full in _PRODUCT_ALIASES.items() if full == name.lower()]
        index.add(Entity(
            entity_id=prod_id,
            entity_type=PRODUCT,
            name=name,
            aliases=aliases,
            source_documents=sorted(set([row.get("doc_id")] + _docs_linked(data, index, links, target=prod_id)) - {None, ""}),
            sources=["products.csv: product_id"],
            attributes={"category": row.get("category"), "owning_dept_id": row.get("owning_dept_id"),
                        "document_id": row.get("doc_id"),
                        "alias_evidence": {a: ["Phase 2 product alias table (src/metadata/discovery.py)"] for a in aliases}},
        ))


def _discover_processes(data: Phase2Data, index: EntityIndex) -> None:
    """Every row of processes.csv: Nexora's own register of processes.

    Some of these are policies or guidelines rather than step-by-step
    procedures; processes.py (Milestone 2) classifies them from their
    documents instead of assuming every row is a procedure.
    """
    links = {RelationType.DOCUMENT_RELATED_TO_PROCESS, RelationType.DOCUMENT_MENTIONS_PROCESS}
    for row in data.tables.get("processes", []):
        proc_id = row["process_id"]
        # (doc_id, version) picks the right file when two share a header ID.
        doc_entity_id = index.doc_by_header.get((row.get("doc_id"), version_key(row.get("version"))))
        if doc_entity_id is None:
            index.notes.append(f"processes.csv {proc_id}: no ingested document with "
                               f"ID {row.get('doc_id')} version {row.get('version')}")
        owner_ids, owner_missing = resolve_roles(row.get("owner_role", ""), data.registry)
        for name in owner_missing:
            index.unresolved_names[ROLE].add(name)
            index.notes.append(f"processes.csv {proc_id}: owner '{name}' is not in roles.csv")
        docs = _docs_linked(data, index, links, target=proc_id)
        index.add(Entity(
            entity_id=proc_id,
            entity_type=PROCESS,
            name=row["name"],
            source_documents=sorted(set(([doc_entity_id] if doc_entity_id else []) + docs)),
            sources=["processes.csv: process_id"],
            attributes={
                "owning_dept_id": row.get("owning_dept_id"),
                "owner_role": row.get("owner_role"),
                "owner_role_ids": owner_ids,
                "document_id": doc_entity_id,
                "version": row.get("version"),
                "status": row.get("status"),
            },
        ))


# ---------------- Document-derived: policies, checklists, audiences, customers ----------------

def _discover_policies_and_checklists(index: EntityIndex) -> None:
    """Policies and checklists are documents of a declared type.
    The type comes from the document's own header, not from its wording."""
    for doc in _docs_by_type(index, "policy"):
        a = doc.attributes
        index.add(Entity(
            entity_id=doc.entity_id, entity_type=POLICY, name=doc.name,
            source_documents=[doc.entity_id],
            sources=[f"{a['source_filename']}: Document Type '{a['document_type']}'"],
            attributes={k: a.get(k) for k in ("document_type", "version", "effective_date", "status",
                                               "owner_role_ids", "department_ids")} | {"document_id": doc.entity_id},
        ))
    for doc in _docs_by_type(index, "checklist"):
        a = doc.attributes
        index.add(Entity(
            entity_id=doc.entity_id, entity_type=CHECKLIST, name=doc.name,
            source_documents=[doc.entity_id],
            sources=[f"{a['source_filename']}: Document Type '{a['document_type']}'"],
            attributes={k: a.get(k) for k in ("version", "effective_date", "status", "related_roles")}
                       | {"document_id": doc.entity_id},
        ))


_AUDIENCE_TEXT = re.compile(r"Related Roles '(.+)'")


def _discover_audiences(data: Phase2Data, index: EntityIndex) -> None:
    """Groups of people (Phase 2's audience decision), with their original wording."""
    names: dict[str, str] = {}
    docs: dict[str, set[str]] = defaultdict(set)
    for rel in data.relationships:
        if rel.relation_type != RelationType.DOCUMENT_FOR_AUDIENCE:
            continue
        match = _AUDIENCE_TEXT.search(rel.evidence)
        names.setdefault(rel.target, match.group(1) if match else rel.target)
        doc_id = index.document_for_relationship(rel)
        if doc_id:
            docs[rel.target].add(doc_id)
    for aud_id, name in sorted(names.items()):
        index.add(Entity(
            entity_id=aud_id, entity_type=AUDIENCE, name=name,
            source_documents=sorted(docs[aud_id]),
            sources=["document headers: Related Roles (Phase 2 audience)"],
        ))


_BOLD_NAME = re.compile(r"\*\*([A-Z][A-Za-z&.' -]+?)\*\*")


def _discover_customers(data: Phase2Data, index: EntityIndex) -> None:
    """Customers are only taken from the company overview's 'Customers'
    section, where they are listed by name in bold. Anywhere else, a bold
    phrase is just emphasis, so it isn't treated as a customer."""
    overviews = _docs_by_type(index, "company overview")
    found: dict[str, list[str]] = {}
    for doc in overviews:
        a = doc.attributes
        for section in data.sections(a["header_document_id"], a["version"]):
            if "customer" not in section.name.lower():
                continue
            for name in _BOLD_NAME.findall(section.text):
                found.setdefault(name.strip(), []).append(
                    f"{a['source_filename']}: section '{section.name}' (chunks {', '.join(section.chunk_ids)})")

    for name, evidence in found.items():
        pattern = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        hits = [c for c in data.chunks if pattern.search(c.chunk_text)]
        index.add(Entity(
            entity_id=f"CUST-{slug(name)}",
            entity_type=CUSTOMER,
            name=name,
            source_documents=sorted({index.document_for_chunk(c) or c.document_id for c in hits}),
            sources=evidence,
            confidence=CONF_PATTERN,
            attributes={"mentioned_in_chunks": sorted(c.chunk_id for c in hits)},
        ))


# ---------------- Everything ----------------

def discover_entities(data: Phase2Data) -> EntityIndex:
    """Run every discovery step in dependency order."""
    index = EntityIndex()
    depts = DepartmentResolver(data.registry)

    _discover_documents(data, index, depts)           # first: everything else points at documents
    glossary = _read_glossary(data, index)
    _discover_terminology(glossary, data, index)
    _discover_departments(data, index, depts)          # after documents, so header spellings are known
    _discover_roles(data, index, _documented_role_aliases(glossary, data, index))
    _discover_employees(data, index)
    _discover_locations(data, index)
    _discover_products(data, index)
    _discover_processes(data, index)
    _discover_policies_and_checklists(index)
    _discover_audiences(data, index)
    _discover_customers(data, index)

    index.glossary_notes = glossary.notes   # for duplicates.py
    return index

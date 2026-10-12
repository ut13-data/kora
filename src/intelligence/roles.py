"""
Role and responsibility intelligence.

For every role in roles.csv, brings together:
  department            roles.csv                                     1.0
  employees             employees.csv                                 1.0
  responsibilities      its paragraph in the department description   0.85
  reports_to            organization.md org chart, role paragraphs    0.85
  owns / performs / participates in processes   (processes.py)
  approvals / escalations                       (rules.py)
  documents it owns or is listed on             (document headers)

Names that can't be resolved ('CTO', 'Marketing Managers') are recorded as
written and reported, never mapped to the "closest" role.
"""

import re
from dataclasses import asdict, dataclass, field

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.matchers import role_matcher
from src.intelligence.models import CONF_PATTERN, DOCUMENT, EMPLOYEE, ROLE
from src.intelligence.normalize import text_key
from src.intelligence.processes import ProcessRecord
from src.intelligence.rules import Rule
from src.intelligence.text import blocks, sentences

_ROLE_TITLE = re.compile(r"^(.+?)\s*\([A-Z]{2,6}\)$")
_REPORTS_TO = re.compile(r"\breports? (?:directly )?to (?:the )?([A-Z][\w&/ -]*?)(?=[,.;]|$)")
_TAGS = [
    ("owns", re.compile(r"\bowns?\b", re.I)),
    ("approves", re.compile(r"approv", re.I)),
    ("escalation", re.compile(r"escalation", re.I)),
    ("manages", re.compile(r"\bmanages?\b", re.I)),
    ("reports_to", re.compile(r"\breports? (?:directly )?to\b", re.I)),
]


@dataclass
class RoleRecord:
    role_id: str
    name: str
    aliases: list[str]
    department_id: str | None
    department_name: str | None
    employee_ids: list[str] = field(default_factory=list)
    responsibilities: list[dict] = field(default_factory=list)
    reports_to: list[dict] = field(default_factory=list)
    direct_report_roles: list[dict] = field(default_factory=list)
    owns_processes: list[dict] = field(default_factory=list)
    performs_in_processes: list[dict] = field(default_factory=list)
    participates_in_processes: list[dict] = field(default_factory=list)
    approval_rule_ids: list[str] = field(default_factory=list)
    escalation_rule_ids: list[str] = field(default_factory=list)
    owned_documents: list[str] = field(default_factory=list)
    listed_on_documents: list[str] = field(default_factory=list)
    heads_departments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**asdict(self), "employee_count": len(self.employee_ids)}


class _OrgChart:
    """Parse the reporting-lines diagram:

        CEO
        ├── CTO
        │   ├── VP Engineering → Engineering Managers → Software Engineers / QA / DevOps

    Depth comes from where the name starts (4 characters per level).
    '→' chains a line; ' / ' separates siblings; 'Support Engineers (L1/L2/L3)'
    expands to the three levels. Plural names are matched to singular titles.
    """

    def __init__(self, index: EntityIndex):
        self.by_key: dict[str, str] = {}
        for role in index.of_type(ROLE):
            for label in [role.name, *role.aliases]:
                self.by_key.setdefault(text_key(label), role.entity_id)

    def resolve(self, name: str) -> tuple[list[str], list[str]]:
        """Return (role_ids, unresolved_names) for one diagram node."""
        expanded = [name]
        m = re.match(r"^(.*?)\s*\(([^)]*/[^)]*)\)$", name)
        if m:
            base = m.group(1)
            base = base[:-1] if base.endswith("s") else base
            expanded = [f"{base} {level.strip()}" for level in m.group(2).split("/")]
        ids, missing = [], []
        for candidate in expanded:
            for form in (candidate, candidate[:-1] if candidate.endswith("s") else None):
                if form and text_key(form) in self.by_key:
                    ids.append(self.by_key[text_key(form)])
                    break
            else:
                # 'Marketing Managers' is reported as 'Marketing Manager'; single words
                # ('DevOps', 'QA') are kept exactly as written.
                plural = " " in candidate and candidate.endswith("s")
                missing.append(candidate[:-1] if plural else candidate)
        return ids, missing

    def edges(self, code_lines: list[str]) -> list[dict]:
        out = []
        stack: dict[int, list[str]] = {}
        for line in code_lines:
            first = re.search(r"[A-Za-z]", line)
            if not first:
                continue
            depth = first.start() // 4
            chain = [part.strip() for part in line[first.start():].split("→")]
            nodes = [[n.strip() for n in re.split(r"\s+/\s+", part) if n.strip()] for part in chain]
            for i, names in enumerate(nodes):
                parents = nodes[i - 1] if i else stack.get(depth - 1, [])
                for child in names:
                    for parent in parents:
                        out.append({"child": child, "parent": parent, "quote": line.strip()})
            stack[depth] = nodes[0]
            for deeper in [d for d in stack if d > depth]:
                del stack[deeper]
        return out


def analyze_roles(data: Phase2Data, index: EntityIndex, processes: list[ProcessRecord],
                  rules: list[Rule]) -> tuple[list[RoleRecord], dict]:
    roles = role_matcher(index)
    def dept_name(dept_id: str | None) -> str | None:
        dept = index.get("department", dept_id) if dept_id else None
        return dept.name if dept else None

    records = {r.entity_id: RoleRecord(r.entity_id, r.name, r.aliases, r.attributes.get("dept_id"),
                                       dept_name(r.attributes.get("dept_id")))
               for r in sorted(index.of_type(ROLE), key=lambda e: e.entity_id)}
    org: dict = {"edges": [], "unresolved": []}

    for emp in index.of_type(EMPLOYEE):
        rid = emp.attributes.get("role_id")
        if rid in records:
            records[rid].employee_ids.append(emp.entity_id)

    for dept in index.of_type("department"):
        for rid in dept.attributes.get("head_role_ids", []):
            if rid in records:
                records[rid].heads_departments.append(dept.entity_id)

    # ---- responsibilities + stated reporting lines, from department descriptions ----
    for doc in index.of_type(DOCUMENT):
        a = doc.attributes
        if "department description" not in (a.get("document_type") or "").lower():
            continue
        for section in data.sections(a["header_document_id"], a["version"]):
            if not section.name.lower().endswith("roles"):
                continue
            for block in blocks(section.text):
                if not block.title:
                    continue
                m = _ROLE_TITLE.match(block.title)
                keys = roles.keys(m.group(1) if m else block.title)
                if len(keys) != 1:
                    index.notes.append(f"{a['source_filename']}: role heading '{block.title}' did not resolve to one role")
                    continue
                rec = records[keys[0].split(":", 1)[1]]
                for sentence in sentences(block.body):
                    for clause in (c.strip() for c in re.split(r";\s+", sentence) if c.strip()):
                        rec.responsibilities.append({
                            "statement": clause,
                            "tags": [tag for tag, pattern in _TAGS if pattern.search(clause)],
                            "source_document": doc.entity_id,
                            "source_filename": a["source_filename"],
                            "chunk_ids": data.find_chunks(a["header_document_id"], a["version"], clause) or section.chunk_ids,
                            "confidence": CONF_PATTERN,
                        })
                        for target in _REPORTS_TO.findall(clause):
                            target_ids = roles.keys(target)
                            rec.reports_to.append({
                                "role_id": target_ids[0].split(":", 1)[1] if target_ids else None,
                                "name": target.strip(), "source": f"{a['source_filename']}: role paragraph",
                                "quote": clause, "confidence": CONF_PATTERN})
                            if not target_ids:
                                index.unresolved_names[ROLE].add(target.strip())

    # ---- org chart ----
    chart = _OrgChart(index)
    for doc in index.of_type(DOCUMENT):
        a = doc.attributes
        if "organization" not in (a.get("document_type") or "").lower():
            continue
        for section in data.sections(a["header_document_id"], a["version"]):
            code = [b.raw_lines[0] for b in blocks(section.text) if b.kind == "code"]
            if not code:
                continue
            for edge in chart.edges(code):
                child_ids, child_missing = chart.resolve(edge["child"])
                parent_ids, parent_missing = chart.resolve(edge["parent"])
                for name in child_missing + parent_missing:
                    index.unresolved_names[ROLE].add(name)
                entry = {**edge, "child_role_ids": child_ids, "parent_role_ids": parent_ids,
                         "unresolved": child_missing + parent_missing,
                         "source_document": doc.entity_id, "source_filename": a["source_filename"],
                         "section": section.name, "chunk_ids": section.chunk_ids, "confidence": CONF_PATTERN}
                org["edges"].append(entry)
                if entry["unresolved"]:
                    org["unresolved"].append(entry)
                for child in child_ids:
                    for parent in parent_ids:
                        records[child].reports_to.append({
                            "role_id": parent, "name": records[parent].name,
                            "source": f"{a['source_filename']}: {section.name}",
                            "quote": edge["quote"], "confidence": CONF_PATTERN})
                        records[parent].direct_report_roles.append({"role_id": child, "name": records[child].name})
                for child in child_ids:
                    for parent in parent_missing:
                        records[child].reports_to.append({
                            "role_id": None, "name": parent,
                            "source": f"{a['source_filename']}: {section.name}",
                            "quote": edge["quote"], "confidence": CONF_PATTERN})

    # ---- processes ----
    for proc in processes:
        for owner in proc.owner_roles:
            records[owner["role_id"]].owns_processes.append(
                {"process_id": proc.process_id, "source": owner["source"], "historical": proc.historical})
        for p in proc.participants:
            target = (records[p["role_id"]].participates_in_processes
                      if any(h.startswith("document header") for h in p["how"]) else None)
            if target is not None:
                target.append({"process_id": proc.process_id, "historical": proc.historical})
            steps = [s["number"] for s in proc.steps if p["role_id"] in s["role_ids"]]
            if steps:
                records[p["role_id"]].performs_in_processes.append(
                    {"process_id": proc.process_id, "steps": steps, "historical": proc.historical})

    # ---- rules ----
    for rule in rules:
        for rid in rule.role_ids:
            if rid in records:
                (records[rid].approval_rule_ids if rule.rule_type == "approval"
                 else records[rid].escalation_rule_ids).append(rule.rule_id)

    # ---- documents ----
    for doc in index.of_type(DOCUMENT):
        for rid in doc.attributes.get("owner_role_ids", []):
            if rid in records:
                records[rid].owned_documents.append(doc.entity_id)
    from src.metadata.relationships import RelationType
    for rel in data.relationships:
        if rel.relation_type == RelationType.DOCUMENT_RELATED_TO_ROLE and rel.target in records:
            doc_id = index.document_for_relationship(rel)
            if doc_id and doc_id not in records[rel.target].listed_on_documents:
                records[rel.target].listed_on_documents.append(doc_id)

    # de-duplicate reporting lines (org chart and paragraph may agree)
    for rec in records.values():
        seen, unique_lines = set(), []
        for line in rec.reports_to:
            key = (line["role_id"], line["name"].lower(), line["source"])
            if key not in seen:
                seen.add(key)
                unique_lines.append(line)
        rec.reports_to = unique_lines
        rec.employee_ids.sort()
        rec.owned_documents.sort()
        rec.listed_on_documents.sort()

    return list(records.values()), org

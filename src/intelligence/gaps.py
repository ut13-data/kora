"""
Knowledge gap detection: what Nexora's knowledge is missing or leaves unclear.

Never fills a gap. Each gap says what is missing, where that was noticed, and
how serious it is:
    high     a process can stall or a customer can be affected (declared Known Gaps)
    medium   a responsibility, approver, escalation target or owner can't be identified
    low      data quality, terminology, limitations, documentation polish

Every check is listed in `checks`, including the ones that found nothing,
so "no gaps of this type" is visible rather than silent.

Later milestones add their own gap producers (e.g. Germany checklist items
without internal guidance), merged through add_gaps().
"""

import hashlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.models import (
    CONF_DOCUMENTED, CONF_KEYWORD, CONF_PATTERN, CONF_STRUCTURED,
    CUSTOMER, DEPARTMENT, DOCUMENT, EMPLOYEE, PRODUCT, ROLE, TERMINOLOGY,
)
from src.intelligence.processes import ProcessRecord
from src.intelligence.roles import RoleRecord
from src.intelligence.rules import Rule
from src.intelligence.text import gap_codes, plain, sentences

HIGH, MEDIUM, LOW = "high", "medium", "low"
_IMPLEMENTATION_HEADING = re.compile(r"configur|implement|set ?up|onboard|dependenc", re.I)
_LIMITATION = re.compile(r"known limitation", re.I)
_MISSING_SOURCE = re.compile(r"not included in this knowledge base", re.I)


@dataclass
class Gap:
    gap_id: str
    gap_type: str
    description: str
    related_entities: list[str]
    source_documents: list[str]
    severity: str
    confidence: float
    evidence: list[dict] = field(default_factory=list)
    suggestion: str | None = None      # where an answer might be found -- never the answer itself

    def to_dict(self) -> dict:
        return asdict(self)


def gap_id(gap_type: str, *parts: str) -> str:
    return "GAP-" + hashlib.sha1("|".join([gap_type, *parts]).encode("utf-8")).hexdigest()[:8].upper()


class GapReport:
    def __init__(self):
        self.gaps: dict[str, Gap] = {}
        self.checks: list[dict] = []

    def check(self, name: str, description: str, found: int) -> None:
        self.checks.append({"check": name, "description": description, "gaps_found": found})

    def add(self, gap: Gap) -> None:
        if gap.gap_id in self.gaps:
            existing = self.gaps[gap.gap_id]
            existing.evidence += [e for e in gap.evidence if e not in existing.evidence]
            existing.source_documents = sorted(set(existing.source_documents) | set(gap.source_documents))
        else:
            self.gaps[gap.gap_id] = gap

    def to_dict(self) -> dict:
        order = {HIGH: 0, MEDIUM: 1, LOW: 2}
        gaps = sorted(self.gaps.values(), key=lambda g: (order[g.severity], g.gap_type, g.gap_id))
        counts = defaultdict(int)
        for g in gaps:
            counts[g.severity] += 1
        return {"summary": dict(counts), "gaps": [g.to_dict() for g in gaps], "checks": self.checks}


# ---------------- individual checks ----------------

def _declared(report: GapReport, data: Phase2Data, index: EntityIndex, processes: list[ProcessRecord]) -> None:
    """'Known Gap' sections the documents declare themselves, plus every other place citing the same KG code."""
    n = 0
    cited: dict[str, set[str]] = defaultdict(set)
    for chunk in data.chunks:
        for code in gap_codes(chunk.chunk_text):
            doc_id = index.document_for_chunk(chunk)
            if doc_id:
                cited[code].add(doc_id)
    for proc in processes:
        for g in proc.known_gaps:
            code = g["gap_codes"][0] if g["gap_codes"] else f"{proc.process_id}:{g['heading']}"
            report.add(Gap(
                gap_id("declared", code), "declared_gap",
                f"{proc.name}: {g['text']}",
                [f"process:{proc.process_id}"],
                sorted({proc.document_id, *cited.get(code, set())}),
                HIGH, CONF_DOCUMENTED,
                [{"document_id": proc.document_id, "section": g["heading"], "chunk_ids": g["chunk_ids"],
                  "gap_code": code if g["gap_codes"] else None}],
                "Declared by the source document itself; the fix belongs to the process owner.",
            ))
            n += 1
    report.check("declared_gap", "'Known Gap' sections in process documents (and every document citing the KG code)", n)


def _limitations(report: GapReport, data: Phase2Data, index: EntityIndex) -> None:
    n = 0
    for doc in index.of_type(DOCUMENT):
        a = doc.attributes
        for section in data.sections(a["header_document_id"], a["version"]):
            heading = section.name.split(" > ")[-1]
            text = plain(section.text)
            if _LIMITATION.search(heading):
                report.add(Gap(gap_id("limitation", doc.entity_id, heading), "documented_limitation",
                               f"{doc.name}: {text}", [doc.key], [doc.entity_id], LOW, CONF_DOCUMENTED,
                               [{"document_id": doc.entity_id, "section": section.name, "chunk_ids": section.chunk_ids}],
                               "A product limitation, documented on purpose: answer with it, don't work around it."))
                n += 1
            for sentence in sentences(section.text):
                if _MISSING_SOURCE.search(sentence):
                    report.add(Gap(gap_id("missing_source", doc.entity_id, sentence), "referenced_source_missing",
                                   f"{doc.entity_id} refers to material that is not in the knowledge base: \"{sentence}\"",
                                   [doc.key], [doc.entity_id], LOW, CONF_PATTERN,
                                   [{"document_id": doc.entity_id, "section": section.name, "chunk_ids": section.chunk_ids}]))
                    n += 1
    report.check("documented_limitation", "'Known Limitation' sections and references to material outside the knowledge base", n)


def _undefined_names(report: GapReport, data: Phase2Data, index: EntityIndex, duplicates: dict,
                     rules: list[Rule], org_chart: dict) -> None:
    """Role and department names used in documents that match nothing in the tables."""
    usage: dict[str, list[str]] = defaultdict(list)
    for doc in index.of_type(DOCUMENT):
        if doc.attributes.get("owner_role") and not doc.attributes.get("owner_role_ids"):
            usage[doc.attributes["owner_role"].split("(")[0].strip()].append(f"owner of {doc.entity_id}")
    for rule in rules:
        for target in rule.unresolved_targets:
            usage[target].append(f"{rule.rule_type} target in {rule.rule_id} ({rule.document_id})")
    for edge in org_chart.get("unresolved", []):
        for name in edge["unresolved"]:
            line = f"org chart line '{edge['quote']}'"
            if line not in usage[name]:
                usage[name].append(line)
    # Phase 2 recorded where each unresolved header value came from.
    for note in data.unresolved_references:
        m = re.match(r"(?P<file>[^:]+): Related (?P<field>Roles|Processes) value '(?P<name>.+?)' not found", note)
        if m:
            usage[m.group("name")].append(f"Related {m.group('field')} header of {m.group('file')}")

    # Names the duplicate finder could only SUGGEST a match for are still undefined:
    # the suggestion goes into the gap, it is not applied.
    suggestions: dict[str, list[str]] = defaultdict(list)
    suggested_kind: dict[str, str] = {}
    for dup in duplicates.get("potential_duplicates", []):
        if dup["method"] == "unresolved_reference":
            name = dup["entity_b"].split(":", 1)[1]
            suggestions[name].append(dup["entity_a"])
            suggested_kind[name] = dup["entity_a"].split(":", 1)[0]

    n = 0
    names = {(u["name"], u["expected_type"]) for u in duplicates.get("unmatched_names", [])}
    names |= {(name, kind) for name, kind in suggested_kind.items()}
    for name, kind in sorted(names):
        where = usage.get(name, [])
        critical = any(w.startswith(("owner", "approval", "escalation")) for w in where)
        unparsed = len(name.split()) > 4 or name[:1].islower()
        report.add(Gap(
            gap_id("undefined", kind, name),
            "unparsed_header_value" if unparsed else f"undefined_{kind}",
            (f"Header value '{name}' could not be read as a {kind}." if unparsed else
             f"'{name}' is used as a {kind} but is not in {kind}s.csv." +
             (" It is an owner, approver or escalation target, so questions that depend on it can't be routed." if critical else "")),
            [], [], MEDIUM if critical else LOW, CONF_STRUCTURED,
            [{"used_as": w} for w in where] or [{"used_as": "named in a document header or text"}],
            ("Possible matches to review: " + ", ".join(sorted(set(suggestions[name])))) if suggestions.get(name) else None,
        ))
        n += 1
    report.check("undefined_name", "Role/department names in documents that match no table row", n)


def _structure(report: GapReport, index: EntityIndex, processes: list[ProcessRecord], roles: list[RoleRecord]) -> None:
    n_owner = n_steps = n_doc = 0
    for p in processes:
        if p.historical:
            continue
        if not p.owner_roles:
            report.add(Gap(gap_id("no_owner", p.process_id), "process_owner_missing",
                           f"{p.name} has no owner that resolves to a role.", [f"process:{p.process_id}"],
                           [p.document_id] if p.document_id else [], MEDIUM, CONF_STRUCTURED))
            n_owner += 1
        if not p.document_id:
            report.add(Gap(gap_id("no_doc", p.process_id), "process_without_document",
                           f"{p.name} is in processes.csv but has no ingested document.", [f"process:{p.process_id}"],
                           [], MEDIUM, CONF_STRUCTURED))
            n_doc += 1
        if p.kind in ("procedure", "policy_and_procedure") and not p.steps:
            report.add(Gap(gap_id("no_steps", p.process_id), "procedure_without_steps",
                           f"{p.name} is a procedure but its document has no numbered steps.",
                           [f"process:{p.process_id}"], [p.document_id], MEDIUM, CONF_PATTERN))
            n_steps += 1
    report.check("process_owner_missing", "Every current process has an owner role", n_owner)
    report.check("process_without_document", "Every current process has a document", n_doc)
    report.check("procedure_without_steps", "Every procedure has numbered steps", n_steps)

    n_resp = n_emp = 0
    for r in roles:
        if not r.responsibilities:
            report.add(Gap(gap_id("no_resp", r.role_id), "role_without_responsibilities",
                           f"{r.name} exists in roles.csv but no document describes its responsibilities.",
                           [f"role:{r.role_id}"], [], MEDIUM, CONF_STRUCTURED))
            n_resp += 1
        if not r.employee_ids:
            report.add(Gap(gap_id("no_emp", r.role_id), "role_without_employees",
                           f"No employee in employees.csv holds the role {r.name}.", [f"role:{r.role_id}"], [], LOW,
                           CONF_STRUCTURED))
            n_emp += 1
    report.check("role_without_responsibilities", "Every role has documented responsibilities", n_resp)
    report.check("role_without_employees", "Every role is held by at least one employee", n_emp)

    heads = {rid for d in index.of_type(DEPARTMENT) for rid in d.attributes.get("head_role_ids", [])}
    n_mgr = 0
    by_role: dict[str, list[str]] = defaultdict(list)
    for e in index.of_type(EMPLOYEE):
        if not e.attributes.get("manager_id") and e.attributes.get("role_id") not in heads:
            by_role[e.attributes.get("role_id")].append(e.entity_id)
    for rid, emps in sorted(by_role.items()):
        role = index.get(ROLE, rid)
        report.add(Gap(gap_id("no_mgr", rid), "employee_without_manager",
                       f"{len(emps)} {role.name if role else rid}(s) have no manager_id, and the role does not head a department.",
                       [f"employee:{e}" for e in sorted(emps)], [], LOW, CONF_STRUCTURED,
                       [{"source": "employees.csv: manager_id"}]))
        n_mgr += 1
    report.check("employee_without_manager", "Employees without a manager are department heads", n_mgr)


def _rules(report: GapReport, rules: list[Rule]) -> None:
    n = 0
    for rule in rules:
        if rule.identified and not (rule.rule_type == "escalation" and not rule.role_ids and rule.unresolved_targets):
            continue
        if rule.states_absence:
            continue    # "no formal escalation path exists": covered by the declared gap
        if rule.rule_type == "escalation" and not rule.role_ids and not rule.unresolved_targets:
            continue    # a statement ABOUT escalation (e.g. "does not automatically trigger"), not a route
        # A formal escalation LEVEL whose target isn't a known role can't be routed: medium.
        severity = MEDIUM if rule.level is not None else LOW
        report.add(Gap(gap_id("rule", rule.rule_id), f"{rule.rule_type}_target_unclear",
                       f"{rule.rule_type.capitalize()} statement without an identifiable role: \"{rule.statement}\"",
                       [f"process:{p}" for p in rule.process_ids], [rule.document_id], severity, CONF_PATTERN,
                       [{"rule_id": rule.rule_id, "chunk_ids": rule.chunk_ids,
                         "unresolved": rule.unresolved_targets}]))
        n += 1
    report.check("rule_target_unclear", "Approval/escalation statements name who approves or receives them", n)


def _products(report: GapReport, data: Phase2Data, index: EntityIndex, processes: list[ProcessRecord]) -> None:
    """A software product should say how it is implemented or configured.
    Professional-services entries (products.csv category) are services, not software, and are skipped."""
    # Implementation processes: those owned by a department whose name says Implementation.
    impl_depts = {d.entity_id for d in index.of_type(DEPARTMENT) if "implementation" in d.name.lower()}
    impl_processes = {p.process_id for p in processes if p.department and p.department["dept_id"] in impl_depts}
    in_steps = {pid for p in processes if p.process_id in impl_processes and not p.historical
                for s in p.steps for pid in s["product_ids"]}
    n = 0
    for prod in sorted(index.of_type(PRODUCT), key=lambda e: e.entity_id):
        if "services" in (prod.attributes.get("category") or "").lower():
            continue
        doc = index.get(DOCUMENT, prod.attributes.get("document_id") or "")
        evidence = []
        if prod.entity_id in in_steps:
            evidence.append("named in an implementation process step")
        if doc:
            a = doc.attributes
            for section in data.sections(a["header_document_id"], a["version"]):
                heading = section.name.split(" > ")[-1]
                if _IMPLEMENTATION_HEADING.search(heading):
                    evidence.append(f"section '{heading}'")
                if any(pid in section.text for pid in impl_processes):
                    evidence.append(f"section '{heading}' refers to an implementation process")
        if evidence:
            continue
        report.add(Gap(gap_id("impl", prod.entity_id), "product_without_implementation_guidance",
                       f"{prod.name} has product documentation, but nothing explains how it is configured or "
                       f"implemented for a customer (no configuration section, not named in Customer Onboarding steps).",
                       [prod.key], [doc.entity_id] if doc else [], MEDIUM, CONF_PATTERN,
                       suggestion="Ask the product's Product Manager or the Implementation Manager."))
        n += 1
    report.check("product_without_implementation_guidance", "Each software product has implementation/configuration guidance", n)


def _customers(report: GapReport, index: EntityIndex, relationships) -> None:
    linked = {r.target_key for r in relationships if r.relationship == "used_by"}
    n = 0
    for c in sorted(index.of_type(CUSTOMER), key=lambda e: e.entity_id):
        if c.key in linked:
            continue
        report.add(Gap(gap_id("cust", c.entity_id), "customer_without_product",
                       f"{c.name} is named as a customer, but no document says which products it uses.",
                       [c.key], c.source_documents, LOW, CONF_KEYWORD))
        n += 1
    report.check("customer_without_product", "Named customers are linked to the products they use", n)


def _terminology(report: GapReport, index: EntityIndex) -> None:
    n = 0
    for t in index.of_type(TERMINOLOGY):
        if t.attributes.get("kind") != "terminology_note" or t.attributes.get("relation") == "equivalent":
            continue
        report.add(Gap(gap_id("term", t.entity_id), "terminology_ambiguity",
                       f"{t.name} {t.attributes.get('note_text', '')}".strip(),
                       [t.key], t.source_documents, LOW, CONF_DOCUMENTED,
                       suggestion="Disambiguate by context (department, document) before answering."))
        n += 1
    report.check("terminology_ambiguity", "Terms the glossary itself flags as ambiguous or overloaded", n)


def add_gaps(report: GapReport, gaps: list[dict], check: str, description: str) -> None:
    """Merge gaps produced by another module (e.g. germany.py), keeping the same shape."""
    for g in gaps:
        related = [f"location:{g['location_id']}"] if g.get("location_id") else []
        report.add(Gap(gap_id(g["gap_type"], g.get("item_id", ""), g.get("location_id") or ""), g["gap_type"],
                       g["description"], related, g.get("source_documents", []), g["severity"], CONF_PATTERN,
                       [{"item_id": g.get("item_id"), "chunk_ids": g.get("chunk_ids", [])}]))
    report.check(check, description, len(gaps))


def detect_gaps(data: Phase2Data, index: EntityIndex, processes: list[ProcessRecord], roles: list[RoleRecord],
                rules: list[Rule], duplicates: dict, org_chart: dict, relationships) -> GapReport:
    report = GapReport()
    _declared(report, data, index, processes)
    _limitations(report, data, index)
    _undefined_names(report, data, index, duplicates, rules, org_chart)
    _structure(report, index, processes, roles)
    _rules(report, rules)
    _products(report, data, index, processes)
    _customers(report, index, relationships)
    _terminology(report, index)
    return report

"""
Phase 3 validation: checks every intelligence output, and reports problems instead of hiding them.

Each check returns PASS, WARN or FAIL:
    FAIL  KORA itself produced something broken (a dangling link, a cited chunk that
          doesn't exist, a confidence of 1.7). Must be fixed in code.
    WARN  The SOURCE DATA has an issue KORA found and reports (e.g. a document owner
          that isn't a role). Expected, visible, not KORA's bug.
    PASS  Checked and clean.

Checks are run on the outputs as written to JSON, the same shape a later retrieval
layer or UI will read, so a bug in serialisation is caught too.
"""

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from src.intelligence.models import (
    AUDIENCE, CHECKLIST, CUSTOMER, DEPARTMENT, DOCUMENT, EMPLOYEE, ENTITY_TYPES, LOCATION, POLICY,
    PROCESS, PRODUCT, ROLE, TERMINOLOGY, TOPIC,
)

PASS, WARN, FAIL = "pass", "warn", "fail"

# What a valid ID looks like for each entity type (Nexora's own formats, plus KORA-made prefixes).
ID_FORMATS = {
    EMPLOYEE: r"EMP-\d{4}", ROLE: r"ROLE-\d{3}", DEPARTMENT: r"DEPT-[A-Z]+", LOCATION: r"LOC-[A-Z]{3}",
    PRODUCT: r"PROD-[A-Z]+", PROCESS: r"PROC-[A-Z]+-\d{3}(?:-[A-Z0-9]+)?",
    DOCUMENT: r"(?:DOC|PROC)-[A-Z0-9]+(?:-[A-Z0-9]+)*", POLICY: r"(?:DOC|PROC)-[A-Z0-9]+(?:-[A-Z0-9]+)*",
    CHECKLIST: r"DOC-[A-Z]+-\d{3}", TOPIC: r"TOPIC-[A-Z0-9-]+", TERMINOLOGY: r"TERM-[A-Z0-9-]+",
    CUSTOMER: r"CUST-[A-Z0-9-]+", AUDIENCE: r"AUD-[A-Z0-9-]+",
}


@dataclass
class Check:
    check: str
    description: str
    status: str
    count: int = 0
    details: list = field(default_factory=list)


class _Validator:
    def __init__(self, outputs: dict, chunk_ids: set[str]):
        self.o = outputs
        self.chunk_ids = chunk_ids
        self.entities = {e["key"]: e for e in outputs["entities"]["entities"]}
        self.doc_ids = {e["entity_id"] for e in outputs["entities"]["entities"] if e["entity_type"] == DOCUMENT}
        self.checks: list[Check] = []

    def add(self, name: str, description: str, problems: list, status_if_problems: str = FAIL) -> None:
        self.checks.append(Check(name, description, status_if_problems if problems else PASS,
                                 len(problems), problems[:25]))

    # ---------- entities ----------

    def entities_valid(self) -> None:
        bad = []
        for e in self.entities.values():
            pattern = ID_FORMATS.get(e["entity_type"])
            if e["entity_type"] not in ENTITY_TYPES:
                bad.append(f"{e['key']}: unknown entity type")
            elif not e["entity_id"] or (pattern and not re.fullmatch(pattern, e["entity_id"])):
                bad.append(f"{e['key']}: ID does not match {pattern}")
        self.add("entity_ids_valid", "Every entity has an ID in the expected format for its type", bad)

        dupes = self.o["entities"].get("collisions", [])
        self.add("no_duplicate_ids", "No entity key (type:ID) is used twice", dupes)

        shared = Counter(e["entity_id"] for e in self.entities.values())
        reuse = sorted(f"{i}: {', '.join(sorted(k.split(':')[0] for k in self.entities if k.split(':', 1)[1] == i))}"
                       for i, n in shared.items() if n > 1)
        self.checks.append(Check("id_shared_across_types", "IDs Nexora reuses across kinds (e.g. a process and its "
                                 "document). Allowed by design: keys are type:ID.", PASS, len(reuse), reuse[:25]))

        unsourced = [k for k, e in self.entities.items() if not e["sources"]]
        self.add("entities_have_sources", "Every entity records where it came from", unsourced)

        missing_docs = [f"{k}: {d}" for k, e in self.entities.items() for d in e["source_documents"]
                        if d not in self.doc_ids]
        self.add("entity_source_documents_exist", "Every source document an entity cites exists", missing_docs)

    def not_fabricated(self, chunks_text: dict[str, str]) -> None:
        """Entities that don't come from a table must be visible in the source text."""
        problems = []
        for e in self.entities.values():
            if e["entity_type"] == CUSTOMER:
                if not any(e["name"] in chunks_text.get(c, "") for c in e["attributes"].get("mentioned_in_chunks", [])):
                    problems.append(f"{e['key']}: name not found in its cited chunks")
            if e["entity_type"] == TOPIC and e["attributes"].get("anchor") not in self.entities:
                problems.append(f"{e['key']}: anchor {e['attributes'].get('anchor')} does not exist")
            if e["entity_type"] == TERMINOLOGY and not e["source_documents"]:
                problems.append(f"{e['key']}: no glossary source")
        self.add("no_fabricated_entities",
                 "Text-derived entities are traceable: customers appear in their chunks, topics have a real anchor, "
                 "terms have a glossary source", problems)

    # ---------- relationships ----------

    def relationships(self) -> None:
        rels = self.o["relationships"]["relationships"]
        dangling = [f"{r['source_key']} --{r['relationship']}--> {r['target_key']}"
                    for r in rels if r["source_key"] not in self.entities or r["target_key"] not in self.entities]
        dangling += [f"{d['source']} --{d['relationship']}--> {d['target']} (not created)"
                     for d in self.o["relationships"].get("dangling", [])]
        self.add("relationships_reference_entities", "Both ends of every relationship exist", dangling)

        ids = Counter(r["relationship_id"] for r in rels)
        self.add("relationship_ids_unique", "Relationship IDs are unique", [i for i, n in ids.items() if n > 1])

        no_evidence = [r["relationship_id"] for r in rels if not r["evidence"]]
        self.add("relationships_have_evidence", "Every relationship keeps at least one piece of evidence", no_evidence)

        doc_methods = {"header", "text_mention", "title_match", "step_pattern", "rule_pattern", "org_chart",
                       "sentence_pattern", "tfidf_similarity"}
        lost = [f"{r['relationship_id']} ({r['relationship']}): {e['method']} evidence without a source document"
                for r in rels for e in r["evidence"] if e["method"] in doc_methods and not e["source_document"]]
        self.add("document_evidence_keeps_source", "Evidence read from a document names that document", lost)

        checklist = [f"{r['source_key']} --has_checklist--> {r['target_key']}" for r in rels
                     if r["relationship"] == "has_checklist" and r["source_type"] != LOCATION]
        checklist += [f"{r['source_key']} --eligible_for--> {r['target_key']}: employee's location has no checklist"
                      for r in rels if r["relationship"] == "eligible_for" and not self._location_has_checklist(r)]
        self.add("checklist_links_valid", "Checklist links start at a location, and eligible employees sit at a "
                 "location that has the checklist", checklist)

    def _location_has_checklist(self, rel: dict) -> bool:
        emp = self.entities.get(rel["source_key"])
        loc = emp["attributes"].get("location_id") if emp else None
        return any(r["relationship"] == "has_checklist" and r["source_id"] == loc and r["target_key"] == rel["target_key"]
                   for r in self.o["relationships"]["relationships"])

    # ---------- structure ----------

    def references(self) -> None:
        problems = []
        for p in self.o["processes"]:
            for owner in p["owner_roles"]:
                if f"role:{owner['role_id']}" not in self.entities:
                    problems.append(f"{p['process_id']}: owner {owner['role_id']} is not a role")
            if p["department"] and f"department:{p['department']['dept_id']}" not in self.entities:
                problems.append(f"{p['process_id']}: department {p['department']['dept_id']} does not exist")
        self.add("processes_reference_valid_roles_departments", "Process owners are roles; departments exist", problems)

        problems = []
        for e in self.entities.values():
            if e["entity_type"] != EMPLOYEE:
                continue
            a = e["attributes"]
            for t, key in ((ROLE, "role_id"), (DEPARTMENT, "dept_id"), (LOCATION, "location_id")):
                if f"{t}:{a.get(key)}" not in self.entities:
                    problems.append(f"{e['key']}: {key}={a.get(key)} does not exist")
            if a.get("manager_id") and f"employee:{a['manager_id']}" not in self.entities:
                problems.append(f"{e['key']}: manager {a['manager_id']} does not exist")
            role = self.entities.get(f"role:{a.get('role_id')}")
            if role and role["attributes"].get("dept_id") != a.get("dept_id"):
                problems.append(f"{e['key']}: role belongs to {role['attributes'].get('dept_id')}, employee to {a.get('dept_id')}")
        self.add("employees_reference_valid_entities",
                 "Employees point at existing roles, departments, locations and managers", problems)

        unresolved = [f"{e['entity_id']}: owner '{e['attributes'].get('owner_role')}' is not a role"
                      for e in self.entities.values()
                      if e["entity_type"] == DOCUMENT and e["attributes"].get("owner_role")
                      and not e["attributes"].get("owner_role_ids")]
        bad = [f"{e['entity_id']}: owner {r} missing" for e in self.entities.values() if e["entity_type"] == DOCUMENT
               for r in e["attributes"].get("owner_role_ids", []) if f"role:{r}" not in self.entities]
        self.add("document_owners_valid", "Resolved document owners exist", bad)
        self.add("document_owners_resolved", "Document owners that don't resolve to a role (a source-data issue, "
                 "also reported as a knowledge gap)", unresolved, WARN)

    def conflicts(self) -> None:
        problems = [f"{c['conflict_id']}: {d} does not exist" for c in self.o["conflicts"]["conflicts"]
                    for d in c["documents_involved"] if d not in self.doc_ids]
        problems += [f"{c['conflict_id']}: fewer than two sides" for c in self.o["conflicts"]["conflicts"]
                     if len(c["sides"]) < 2]
        problems += [f"{c['conflict_id']}: status '{c['status']}' looks like a verdict" for c in self.o["conflicts"]["conflicts"]
                     if c["status"] not in ("superseded_per_source", "signals_point_one_way", "open")]
        self.add("conflicts_reference_documents", "Conflicts cite real documents, have two sides, and never "
                 "claim a resolution", problems)

        gaps = [f"{g['gap_id']}: {d} does not exist" for g in self.o["knowledge_gaps"]["gaps"]
                for d in g["source_documents"] if d not in self.doc_ids]
        self.add("gaps_reference_documents", "Knowledge gaps cite real documents", gaps)

    # ---------- confidence + traceability ----------

    def confidence(self) -> None:
        bad = []

        def check(where: str, value) -> None:
            if not isinstance(value, (int, float)) or not 0 < value <= 1:
                bad.append(f"{where}: {value}")

        for k, e in self.entities.items():
            check(k, e["confidence"])
        for r in self.o["relationships"]["relationships"]:
            check(r["relationship_id"], r["confidence"])
            for ev in r["evidence"]:
                check(f"{r['relationship_id']} evidence", ev["confidence"])
        for d in self.o["potential_duplicates"]["potential_duplicates"]:
            check(d["duplicate_id"], d["confidence"])
        for c in self.o["conflicts"]["conflicts"]:
            check(c["conflict_id"], c["confidence"])
        for g in self.o["knowledge_gaps"]["gaps"]:
            check(g["gap_id"], g["confidence"])
        for rule in self.o["roles"]["approval_rules"] + self.o["roles"]["escalation_rules"]:
            check(rule["rule_id"], rule["confidence"])
        self.add("confidence_values_valid", "Every confidence is a number in (0, 1]", bad)

    def chunk_citations(self) -> None:
        """Every chunk ID cited anywhere must exist in chunks.jsonl: this is what lets
        KORA say 'this comes from PROC-IMPL-001_v2.0_002' and be right."""
        cited: set[str] = set()

        def walk(obj) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("chunk_ids", "mentioned_in_chunks") and isinstance(v, list):
                        cited.update(x for x in v if isinstance(x, str))
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for x in obj:
                    walk(x)

        walk(self.o)
        missing = sorted(cited - self.chunk_ids)
        self.add("cited_chunks_exist", f"Every cited chunk ID exists in chunks.jsonl ({len(cited)} distinct cited)", missing)

        rules = self.o["roles"]["approval_rules"] + self.o["roles"]["escalation_rules"]
        no_source = [r["rule_id"] for r in rules if not r["document_id"] or not r["chunk_ids"]]
        self.add("rules_keep_sources", "Every approval/escalation rule keeps its document and chunk IDs", no_source)

    # ---------- people ----------

    def people(self) -> None:
        g = self.o["germany_checklists"]
        problems = []
        loc_ids = {l["location_id"] for l in g["locations"]}
        for loc in loc_ids:
            if f"location:{loc}" not in self.entities:
                problems.append(f"{loc}: location does not exist")
        for emp in g["employees"]:
            for item in emp.get("plan", {}).get("items", []):
                if item["status"] not in ("reported_complete", "not_reported"):
                    problems.append(f"{emp['employee_id']} {item['item_id']}: status '{item['status']}'")
        if "unknown item IDs" in g["progress_source"]:
            problems.append(g["progress_source"])
        self.add("germany_checklist_valid", "Checklist locations exist; completion is only ever 'reported' or "
                 "'not_reported'; progress file uses known item IDs", problems)

        c = self.o["communication_guidance"]
        self.add("tone_profiles_hedged", "Cultural profiles describe tendencies with hedged language",
                 [f"{u['profile_id']}: {u['text']}" for u in c["hedging_check"]["unhedged_statements"]], WARN)

        problems = []
        for e in self.o["employee_context"]["employees"]:
            row = self.entities.get(f"employee:{e['employee_id']}")
            if row is None:
                problems.append(f"{e['employee_id']}: not an employee")
                continue
            a = row["attributes"]
            if (e["role"]["role_id"], e["department"]["dept_id"], e["location"]["location_id"]) != \
                    (a.get("role_id"), a.get("dept_id"), a.get("location_id")):
                problems.append(f"{e['employee_id']}: context disagrees with employees.csv")
        for h in self.o["employee_context"]["hypothetical_examples"]:
            if not h["hypothetical"] or h["employee_id"] is not None:
                problems.append(f"hypothetical profile '{h['name']}' is not labelled as hypothetical")
        self.add("employee_context_matches_records", "Employee context never contradicts employees.csv; "
                 "hypothetical profiles are labelled", problems)

    def run(self, chunks_text: dict[str, str]) -> dict:
        self.entities_valid()
        self.not_fabricated(chunks_text)
        self.relationships()
        self.references()
        self.conflicts()
        self.confidence()
        self.chunk_citations()
        self.people()
        status = Counter(c.status for c in self.checks)
        return {"summary": {"checks": len(self.checks), "passed": status[PASS], "warnings": status[WARN],
                            "failed": status[FAIL]},
                "checks": [asdict(c) for c in self.checks]}


def validate_outputs(outputs: dict, chunks) -> dict:
    """outputs: the dicts exactly as written to data/intelligence/*.json (keyed by file stem)."""
    return _Validator(outputs, {c.chunk_id for c in chunks}).run({c.chunk_id: c.chunk_text for c in chunks})

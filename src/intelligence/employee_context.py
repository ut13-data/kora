"""
Employee context: everything in Nexora's knowledge that is relevant to ONE person.

    Employee + Role + Department + Location + Joining stage
      -> processes they own / perform / take part in
      -> approvals they give, and approvals they will need (with the actual approver)
      -> policies that apply to them
      -> onboarding timeline, with dates computed from their join date
      -> Germany settling-in plan (if eligible)
      -> communication context (individual preference first, location profile as hypothesis)
      -> a short reading list, each document with the reason it's on it

Every item says WHY it is relevant ("your role performs step 2 of PROC-IMPL-001",
"policy for All Employees"), so KORA's answers stay traceable.

What it never does: infer anything about a person from their name, nationality
or location beyond what employees.csv states; mark anything complete that
nobody reported; invent dates. Computed dates (e.g. a training deadline in
working days) say how they were computed.

Hypothetical profiles ("a new Data Analyst in Munich, week 2") use the same code
with no employee record, and are labelled as hypothetical.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

from src.intelligence.communication import CommunicationIntel
from src.intelligence.entities import EntityIndex
from src.intelligence.germany import PROGRAM_HORIZON_DAYS, GermanyCare, day_number, stage_for
from src.intelligence.models import AUDIENCE, DEPARTMENT, DOCUMENT, EMPLOYEE, LOCATION, POLICY, ROLE, Entity
from src.intelligence.processes import ProcessRecord
from src.intelligence.relationships import KnowledgeRelationship
from src.intelligence.roles import RoleRecord
from src.intelligence.rules import Rule

_ALL_EMPLOYEES = re.compile(r"all employees", re.I)
_STEP_DAY = [(re.compile(r"pre-?start", re.I), 0), (re.compile(r"\bday\s*1\b", re.I), 1),
             (re.compile(r"first week", re.I), 7), (re.compile(r"(\d+)-day", re.I), None)]
_WORKING_DAYS = re.compile(r"(\d+)\s+working days", re.I)
_PLAIN_MANAGER = {"manager", "direct manager", "line manager", "hiring manager"}


def add_working_days(start: date, n: int) -> date:
    """n working days counted from the start date as day 1, skipping Saturdays and Sundays.
    Public holidays are NOT considered: the result says so wherever it is shown."""
    d, counted = start, 0
    while True:
        if d.weekday() < 5:
            counted += 1
            if counted == n:
                return d
        d += timedelta(days=1)


@dataclass
class _Ctx:
    index: EntityIndex
    processes: list[ProcessRecord]
    roles: dict[str, RoleRecord]
    rules: list[Rule]
    relationships: list[KnowledgeRelationship]
    germany: GermanyCare
    comms: CommunicationIntel
    conflicts: dict
    as_of: date


def _name(index: EntityIndex, entity_type: str, entity_id: str | None) -> str | None:
    e = index.get(entity_type, entity_id or "")
    return e.name if e else None


def _audience_documents(ctx: _Ctx) -> set[str]:
    """Documents whose header audience is 'All Employees'."""
    out = set()
    for rel in ctx.relationships:
        if rel.relationship == "for_audience":
            aud = ctx.index.get(AUDIENCE, rel.target_id)
            if aud and _ALL_EMPLOYEES.search(aud.name):
                out.add(rel.source_id)
    return out


def _processes(ctx: _Ctx, role_id: str, dept_id: str | None) -> list[dict]:
    role = ctx.roles.get(role_id)
    found: dict[str, dict] = {}

    def add(pid: str, why: str, strength: int):
        proc = next((p for p in ctx.processes if p.process_id == pid), None)
        if proc is None or proc.historical:
            return
        entry = found.setdefault(pid, {"process_id": pid, "name": proc.name, "why": [], "strength": 0})
        entry["why"].append(why)
        entry["strength"] = max(entry["strength"], strength)

    if role:
        for x in role.owns_processes:
            add(x["process_id"], "your role owns it (processes.csv)", 3)
        for x in role.performs_in_processes:
            add(x["process_id"], f"your role acts in step(s) {x['steps']}", 2)
        for x in role.participates_in_processes:
            add(x["process_id"], "your role is listed in its header", 2)
    for p in ctx.processes:
        if dept_id and p.department and p.department["dept_id"] == dept_id:
            add(p.process_id, "owned by your department", 1)
    return sorted(found.values(), key=lambda e: (-e["strength"], e["process_id"]))


def _policies(ctx: _Ctx, role_id: str, processes: list[dict]) -> list[dict]:
    everyone = _audience_documents(ctx)
    found: dict[str, dict] = {}
    for pol in ctx.index.of_type(POLICY):
        if (pol.attributes.get("status") or "").lower() == "archived":
            continue
        why = []
        if pol.entity_id in everyone:
            why.append("applies to All Employees (document header)")
        if role_id in ctx.roles and pol.entity_id in ctx.roles[role_id].listed_on_documents:
            why.append("your role is listed in its header")
        for p in processes:
            proc = next((x for x in ctx.processes if x.process_id == p["process_id"]), None)
            if proc and any(r["policy_id"] == pol.entity_id for r in proc.related_policies):
                why.append(f"governs {proc.name}, which is relevant to you")
        if why:
            found[pol.entity_id] = {"policy_id": pol.entity_id, "name": pol.name, "why": why}
    return sorted(found.values(), key=lambda e: e["policy_id"])


def _people_with_role(ctx: _Ctx, role_id: str, prefer_location: str | None) -> list[dict]:
    people = [e for e in ctx.index.of_type(EMPLOYEE) if e.attributes.get("role_id") == role_id]
    people.sort(key=lambda e: (e.attributes.get("location_id") != prefer_location, e.entity_id))
    return [{"employee_id": e.entity_id, "name": e.name, "location_id": e.attributes.get("location_id")}
            for e in people[:3]]


def _approvals(ctx: _Ctx, role_id: str, manager: dict | None, location_id: str | None) -> dict:
    give, need, via_manager = [], [], []
    for r in ctx.rules:
        if r.rule_type != "approval" or r.historical:
            continue
        if role_id in r.role_ids:
            give.append({"rule_id": r.rule_id, "statement": r.statement, "document_id": r.document_id})
        elif role_id in r.mentioned_role_ids and r.role_ids:
            need.append({"rule_id": r.rule_id, "statement": r.statement, "document_id": r.document_id,
                         "approvers": [{"role_id": a, "name": _name(ctx.index, ROLE, a),
                                        "people": _people_with_role(ctx, a, location_id)} for a in r.role_ids]})
        # 'manager approval' rules (expenses, purchases, leave) apply to everyone: resolve 'manager'
        # to THIS person's manager from employees.csv.
        generic = [g for g in r.generic_roles if g.lower() in _PLAIN_MANAGER]
        if generic:
            via_manager.append({
                "rule_id": r.rule_id, "statement": r.statement, "document_id": r.document_id,
                "written_as": generic[0],
                "your_manager": manager,
                "note": None if generic[0].lower() != "hiring manager" else
                        "'Hiring Manager' is not a role in roles.csv; for an existing employee this is most "
                        "likely the manager in employees.csv. Confirm with HR."})
    return {"you_approve": give, "you_may_need_approval_from": need, "your_manager_approves": via_manager}


def _onboarding(ctx: _Ctx, join: date | None, onboarding_stage: str | None) -> dict | None:
    proc = next((p for p in ctx.processes if not p.historical and p.name.lower() == "employee onboarding"), None)
    if proc is None or join is None:
        return None
    conflict = next((c for c in ctx.conflicts.get("conflicts", []) if proc.document_id in c["documents_involved"]
                     and c["conflict_type"] == "numeric_fact"), None)
    completed_per_csv = (onboarding_stage or "").lower() == "completed"
    steps = []
    for step in proc.steps:
        due, how = None, None
        wd = _WORKING_DAYS.search(step["text"])
        if wd:
            due = add_working_days(join, int(wd.group(1)))
            how = f"{wd.group(1)} working days from the join date (weekends skipped; public holidays not considered)"
        else:
            for pattern, day in _STEP_DAY:
                m = pattern.search(step["title"] or "")
                if m:
                    offset = int(m.group(1)) if day is None else day
                    due = join + timedelta(days=max(offset - 1, -1))
                    how = f"day {offset} of employment, from the step title '{step['title']}'"
                    break
        if completed_per_csv:
            status = "completed_per_employees_csv"
        elif due is None:
            status = "no_date_in_source"
        elif due < ctx.as_of:
            status = "date_passed_completion_not_recorded"
        else:
            status = "upcoming"
        steps.append({"step": step["number"], "title": step["title"], "text": step["text"],
                      "date": due.isoformat() if due else None, "date_basis": how, "status": status,
                      "roles": step["role_ids"], "generic_roles": step["generic_roles"]})
    return {"process_id": proc.process_id, "document_id": proc.document_id,
            "onboarding_stage_in_employees_csv": onboarding_stage, "steps": steps,
            "note": ("A known conflict exists about one of these deadlines: " + conflict["conflict_id"] +
                     " (the current document is used; the archived one is history).") if conflict else None}


def _reading_list(ctx: _Ctx, dept_id, role_id, processes, policies, germany_status, location_id) -> list[dict]:
    out: list[dict] = []

    def add(doc_id: str | None, why: str):
        doc = ctx.index.get(DOCUMENT, doc_id or "")
        if doc and doc_id not in {x["document_id"] for x in out}:
            out.append({"document_id": doc_id, "title": doc.name, "why": why})

    for doc in ctx.index.of_type(DOCUMENT):
        if "department description" in (doc.attributes.get("document_type") or "").lower() and \
                dept_id in doc.attributes.get("department_ids", []):
            add(doc.entity_id, "your department and its roles")
    for p in processes[:4]:
        proc = next(x for x in ctx.processes if x.process_id == p["process_id"])
        add(proc.document_id, f"{p['name']}: {p['why'][0]}")
    if germany_status in ("active", "pre_start"):
        add(ctx.germany.program.get("overview_document"), "you are relocating to Germany: the settling-in program")
        for doc_id in ctx.germany.program.get("documents", []):
            add(doc_id, "settling-in checklist")
    profile = ctx.comms.profile_for_location(location_id)
    if profile:
        add(profile.document_id, "communication tendencies at your office (a starting point, not a rule)")
    for p in policies:
        if any("All Employees" in w for w in p["why"]):
            add(p["policy_id"], "a policy for all employees")
    return out


def context_for(ctx: _Ctx, emp: Entity, hypothetical: bool = False) -> dict:
    a = emp.attributes
    role_id, dept_id, location_id = a.get("role_id"), a.get("dept_id"), a.get("location_id")
    loc = ctx.index.get(LOCATION, location_id or "")
    mgr_entity = ctx.index.get(EMPLOYEE, a.get("manager_id") or "")
    manager = ({"employee_id": mgr_entity.entity_id, "name": mgr_entity.name,
                "role": _name(ctx.index, ROLE, mgr_entity.attributes.get("role_id"))} if mgr_entity else None)
    day = day_number(a.get("join_date"), ctx.as_of)
    try:
        join = date.fromisoformat(a.get("join_date") or "")
    except ValueError:
        join = None

    processes = _processes(ctx, role_id, dept_id)
    policies = _policies(ctx, role_id, processes)
    germany = ctx.germany.eligibility(emp, ctx.index)
    germany_block = {"eligibility": germany}
    if germany["status"] in ("active", "pre_start"):
        germany_block["plan"] = ctx.germany.plan(emp.entity_id, germany["day"], loc.name if loc else None)

    is_new = day is not None and 1 <= day <= PROGRAM_HORIZON_DAYS     # joined within the last 90 days
    return {
        "employee_id": None if hypothetical else emp.entity_id,
        "hypothetical": hypothetical,
        "name": emp.name,          # for a hypothetical profile this is its label, e.g. "new Data Analyst, Munich, week 2"
        "role": {"role_id": role_id, "name": _name(ctx.index, ROLE, role_id)},
        "department": {"dept_id": dept_id, "name": _name(ctx.index, DEPARTMENT, dept_id)},
        "location": {"location_id": location_id, "city": loc.name if loc else None,
                     "country": loc.attributes.get("country") if loc else None},
        "manager": manager,
        "join_date": a.get("join_date"),
        "day": day,
        "joining_stage": stage_for(day) if day is not None else None,
        "is_new_joiner": is_new,
        "onboarding_stage_in_employees_csv": a.get("onboarding_stage"),
        "relevant_processes": processes,
        "relevant_policies": policies,
        "approvals": _approvals(ctx, role_id, manager, location_id),
        "escalations_you_handle": [{"rule_id": r.rule_id, "statement": r.statement, "document_id": r.document_id}
                                   for r in ctx.rules if r.rule_type == "escalation" and role_id in r.role_ids
                                   and not r.historical],
        "onboarding": _onboarding(ctx, join, a.get("onboarding_stage")) if is_new or
                      (a.get("onboarding_stage") or "").lower() == "in progress" else None,
        "germany_care": germany_block,
        "communication": ctx.comms.for_employee(emp.entity_id, location_id),
        "reading_list": _reading_list(ctx, dept_id, role_id, processes, policies, germany["status"], location_id),
        "sources": ["employees.csv row" if not hypothetical else "hypothetical profile (no employee record)",
                    "roles.json, processes.json, roles approval/escalation rules",
                    "germany_checklists.json", "communication_guidance.json"],
    }


def hypothetical_profile(index: EntityIndex, role_id: str, location_id: str, join_date: date,
                         label: str) -> Entity:
    role = index.get(ROLE, role_id)
    return Entity(entity_id=f"HYPOTHETICAL-{label}", entity_type=EMPLOYEE, name=label,
                  attributes={"role_id": role_id, "dept_id": role.attributes.get("dept_id") if role else None,
                              "location_id": location_id, "join_date": join_date.isoformat(),
                              "onboarding_stage": "In Progress", "manager_id": None,
                              "international_relocation_germany": "Unknown"})


def build_employee_context(index: EntityIndex, processes: list[ProcessRecord], roles: list[RoleRecord],
                           rules: list[Rule], relationships: list[KnowledgeRelationship], germany: GermanyCare,
                           comms: CommunicationIntel, conflicts: dict, as_of: date) -> dict:
    ctx = _Ctx(index, processes, {r.role_id: r for r in roles}, rules, relationships, germany, comms, conflicts, as_of)
    employees = [context_for(ctx, e) for e in sorted(index.of_type(EMPLOYEE), key=lambda e: e.entity_id)]

    # The spec's example. No such employee exists, so it is a labelled hypothetical, not an invented person.
    analyst = next((r for r in index.of_type(ROLE) if r.name == "Data Analyst"), None)
    munich = next((l for l in index.of_type(LOCATION) if l.name == "Munich"), None)
    examples = []
    if analyst and munich:
        week2_join = as_of - timedelta(days=9)          # as_of is then day 10: week 2
        examples.append(context_for(ctx, hypothetical_profile(
            index, analyst.entity_id, munich.entity_id, week2_join, "new Data Analyst, Munich, week 2"),
            hypothetical=True))

    return {
        "as_of": as_of.isoformat(),
        "as_of_note": "Joining stages and dates are computed relative to as_of (configurable).",
        "stages": ["pre_start", "week_1", "week_2", "month_1", "month_2", "month_3", "settled"],
        "new_joiners": [e["employee_id"] for e in employees if e["is_new_joiner"]],
        "joining_after_as_of": [e["employee_id"] for e in employees if e["day"] is not None and e["day"] < 1],
        "onboarding_in_progress_per_csv": [e["employee_id"] for e in employees
                                           if (e["onboarding_stage_in_employees_csv"] or "").lower() == "in progress"],
        "employees": employees,
        "hypothetical_examples": examples,
    }

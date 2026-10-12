"""
The Phase 3 test questions: questions KORA must eventually answer, each answered
here from the intelligence layer with plain lookups. No RAG, no model.

The point is to prove the STRUCTURE is there. A later retrieval/LLM phase will
phrase answers naturally; these functions show which facts, rules and sources
it will build on. Every answer lists its sources (document IDs, rule IDs,
relationship IDs) and the output files it came from.
"""

from dataclasses import dataclass
from typing import Callable

GERMAN_NEW_JOINER = "EMP-0085"      # relocated to Munich, week 2 at the default as_of


@dataclass
class Question:
    qid: str
    category: str
    question: str
    answer_fn: Callable
    supported_by: list[str]


def _names(result, keys) -> list[str]:
    return [result.graph.label(k) for k in keys]


# ---------------- organization ----------------

def q_who_works_in_data(result):
    keys = result.graph.neighbors("department:DEPT-DATA", "works_in", "in")
    people = [result.entities.entities[k] for k in keys]
    rows = [f"{p.name} ({p.entity_id}, {p.attributes['role_title']}, {p.attributes['location_id']})" for p in people]
    return {"answer": f"{len(rows)} people work in Data & Analytics (employees.csv).",
            "items": rows, "sources": ["employees.csv: dept_id"]}


def q_who_owns_customer_onboarding(result):
    p = result.process("PROC-IMPL-001")
    owners = [o["role_id"] for o in p.owner_roles]
    people = [e.name + f" ({e.entity_id})" for e in result.entities.of_type("employee")
              if e.attributes.get("role_id") in owners]
    return {"answer": f"{', '.join(o['name'] for o in p.owner_roles)} owns Customer Onboarding; held by {', '.join(people)}.",
            "items": people, "sources": ["processes.csv: owner_role", p.document_id]}


def q_which_department_implementation(result):
    dept = result.entities.get("department", "DEPT-IMPL")
    procs = result.graph.neighbors(dept.key, "owned_by", "in", current_only=True)   # archived versions left out
    return {"answer": f"{dept.name} ({dept.entity_id}); it owns {', '.join(_names(result, procs))}.",
            "items": procs, "sources": ["departments.csv", "processes.csv: owning_dept_id", "DOC-DEPT-IMPL"]}


# ---------------- processes ----------------

def q_who_approves_proposal(result):
    rules = [r for r in result.rules if r.rule_type == "approval" and "PROC-SALES-002" in r.process_ids
             and r.document_id == "PROC-SALES-002"]
    lines = []
    for r in rules:
        who = ", ".join(result.entities.get("role", x).name for x in r.role_ids) or \
              ("no approval needed" if r.no_approval_required else "unclear")
        lines.append(f"{r.statement} -> {who}")
    return {"answer": "Discount approval on a proposal is tiered: " + " | ".join(lines),
            "items": lines, "sources": [r.rule_id for r in rules] + ["PROC-SALES-002"]}


def q_customer_onboarding_steps(result):
    p = result.process("PROC-IMPL-001")
    steps = [f"{s['number']}. {s['title']}: {s['text']}" for s in p.steps]
    return {"answer": f"{len(steps)} steps, from kickoff to the handoff to Customer Success "
                      f"(input: {', '.join(i['item'] for i in p.inputs)}).",
            "items": steps, "sources": [p.document_id] + [c for s in p.steps for c in s["chunk_ids"]][:6],
            "known_gaps": [g["text"] for g in p.known_gaps]}


def q_documents_incident_management(result):
    p = result.process("PROC-SUP-002")
    docs = [f"{d['document_id']} ({', '.join(d['how'])})" for d in p.related_documents]
    topic = next(t for t in result.topics.topics if t.name == "Incident Management")
    also = sorted({d["document_id"] for d in topic.documents} - {d["document_id"] for d in p.related_documents})
    return {"answer": f"{len(docs)} documents describe or reference it; the topic layer adds {len(also)} more.",
            "items": docs + [f"{d} (topic: Incident Management)" for d in also],
            "sources": [p.document_id, topic.topic_id]}


# ---------------- relationships ----------------

def q_products_in_onboarding(result):
    p = result.process("PROC-IMPL-001")
    items = [f"{result.entities.get('product', x['product_id']).name}: {x['relationship']} ({x['how']}, "
             f"confidence {x['confidence']})" for x in p.related_products]
    return {"answer": f"{len(items)} products are linked to Customer Onboarding.", "items": items,
            "sources": [p.document_id]}


def q_policies_govern_employee_onboarding(result):
    p = result.process("PROC-HR-001")
    current = [x for x in p.related_policies
               if (result.entities.get("policy", x["policy_id"]).attributes.get("status") or "").lower() != "archived"]
    items = [f"{x['policy_id']} {result.entities.get('policy', x['policy_id']).name}: {'; '.join(x['how'])} "
             f"(confidence {x['confidence']})" for x in sorted(current, key=lambda x: -x["confidence"])]
    return {"answer": f"{len(items)} policies are linked to Employee Onboarding; the strongest link is a step "
                      f"that points at the policy directly.", "items": items, "sources": [p.document_id]}


def q_roles_interact_with_cs(result):
    cs_roles = {r.role_id for r in result.roles if r.department_id == "DEPT-CS"}
    shared: dict[str, set[str]] = {}
    for p in result.processes:
        if p.historical:
            continue
        involved = {x["role_id"] for x in p.participants} | {o["role_id"] for o in p.owner_roles}
        if involved & cs_roles:
            for r in involved - cs_roles:
                shared.setdefault(r, set()).add(p.process_id)
    items = [f"{result.entities.get('role', r).name}: via {', '.join(sorted(ps))}" for r, ps in sorted(shared.items())]
    return {"answer": f"{len(items)} roles outside Customer Success share processes with it.",
            "items": items, "sources": sorted({p for ps in shared.values() for p in ps})}


# ---------------- knowledge quality ----------------

def q_conflicting_instructions(result):
    main = [c for c in result.conflicts["conflicts"] if c["conflict_type"] != "org_vs_employees"]
    items = [f"{c['conflict_id']} [{c['conflict_type']}, {c['status']}] {c['topic']}: {c['detail']}" for c in main]
    return {"answer": f"{len(main)} conflicts (plus {len(result.conflicts['conflicts']) - len(main)} low-confidence "
                      f"reporting-line differences). KORA reports both sides and the signals; it does not pick one.",
            "items": items, "sources": [d for c in main for d in c["documents_involved"]]}


def q_outdated_documents(result):
    items = [f"{s['document_id']} v{s['version']} ({s['status']}) -> replaced by "
             f"{', '.join(t['document_id'] + ' v' + t['version'] for t in s['superseded_by'])}"
             for s in result.outdated["superseded_documents"]]
    return {"answer": f"{len(items)} documents are archived and replaced; no active document uses outdated "
                      f"terminology.", "items": items,
            "sources": [s["document_id"] for s in result.outdated["superseded_documents"]]}


def q_missing_information(result):
    gaps = [g for g in result.gaps.to_dict()["gaps"] if g["severity"] in ("high", "medium")]
    items = [f"[{g['severity']}] {g['gap_type']}: {g['description'][:160]}" for g in gaps]
    return {"answer": f"{len(gaps)} high/medium gaps (and more low ones in knowledge_gaps.json).",
            "items": items, "sources": sorted({d for g in gaps for d in g["source_documents"]})}


# ---------------- Germany ----------------

def _plan(result):
    e = result.employee(GERMAN_NEW_JOINER)
    return e, e["germany_care"]["plan"], {i["item_id"]: i for i in e["germany_care"]["plan"]["items"]}


def q_new_in_germany(result):
    e, plan, items = _plan(result)
    now = [i for i in plan["items"] if i["timing"] in ("past_suggested_pace", "this_stage")]
    return {"answer": f"{e['name']} is in {e['joining_stage']} in {e['location']['city']}. "
                      f"Start with: {', '.join(items[i]['title'] for i in plan['next_steps'])}.",
            "items": [f"{i['title']} (suggested by day {i['due_by_day']})"
                      + (f" -- {i['official_guidance_note']}" if i["official_guidance_note"] else "") for i in now],
            "sources": result.germany.program["documents"], "tone": plan["tone"]}


def q_relevant_settling_tasks(result):
    e, plan, _ = _plan(result)
    return {"answer": f"{len(plan['items'])} settling-in tasks apply (relocating to Germany, per employees.csv).",
            "items": [f"{i['title']} [{i['timing']}]" for i in plan["items"]],
            "sources": result.germany.program["documents"] + ["employees.csv: international_relocation_germany"]}


def q_not_completed(result):
    e, plan, items = _plan(result)
    open_items = [i for i in plan["items"] if i["status"] == "not_reported"]
    lead = ("Nothing has been reported as done yet, so I can't say what's finished. "
            if not plan["progress_reported"] else "")
    return {"answer": lead + f"{len(open_items)} tasks are not reported as done.",
            "items": [i["title"] for i in open_items], "sources": ["data/context/checklist_progress.json"]}


def q_possibly_forgotten(result):
    e, plan, items = _plan(result)
    return {"answer": "Worth a gentle check: " + ", ".join(items[i]["title"] for i in plan["possibly_forgotten"][:3]) + ".",
            "items": [f"{items[i]['title']}" + (" (often forgotten)" if items[i]["commonly_forgotten"] else "")
                      for i in plan["possibly_forgotten"]],
            "sources": ["DOC-CARE-002: Three Things People Commonly Forget", "DOC-CARE-004"]}


def q_general_steps(result):
    steps = result.germany.to_dict(result.entities)["general_steps"]
    return {"answer": f"{len(steps)} steps, in dependency and pace order.",
            "items": [f"{s['title']} (by day {s['due_by_day']})" if s["due_by_day"] else s["title"] for s in steps],
            "sources": result.germany.program["documents"]}


# ---------------- communication ----------------

def q_german_workplace_message(result):
    out = result.communication.present("Please complete the security training by Friday.",
                                       "COMM-PROFILE-GERMANY", "direct")
    return {"answer": out["rendered"], "items": [f"{c['check']}: {c['why']}" for c in out["checks"]],
            "sources": ["DOC-COMM-002", "DOC-COMM-001"]}


def q_change_tone(result):
    fact = "Please complete the security training by Friday."
    items = [f"{t}: {result.communication.present(fact, None, t)['rendered']}" for t in ("neutral", "direct", "friendly")]
    return {"answer": "Yes: neutral, direct or friendly (relationship-oriented). Only phrasing changes; the fact "
                      "and deadline stay identical.", "items": items, "sources": ["DOC-COMM-001", "DOC-COMM-006"]}


def q_policy_vs_culture(result):
    fw = result.communication.framework
    items = [f"{layer['rank']}. {layer['layer']}: {layer['definition']}" for layer in fw["layers"]]
    items.append(f"Company layer example: {result.communication.company_layer[1]['statement']}")
    germany = next(p for p in result.communication.profiles if p.profile_id == "COMM-PROFILE-GERMANY")
    items.append(f"Cultural layer example: {germany.statements[0].text}")
    return {"answer": fw["precedence"], "items": items, "sources": [fw["document_id"], "PROC-HR-003", "DOC-COMM-002"]}


# ---------------- extra ----------------

def q_warehouse_access(result):
    rule = next(r for r in result.rules if r.method == "approver_table" and "warehouse" in r.statement)
    people = [f"{e.name} ({e.entity_id})" for e in result.entities.of_type("employee")
              if e.attributes.get("role_id") in rule.role_ids]
    return {"answer": f"{rule.statement}. Today that is {', '.join(people)}.",
            "items": [rule.statement] + rule.unresolved_targets, "sources": [rule.rule_id, rule.document_id] + rule.chunk_ids}


def q_csm_escalation(result):
    levels = sorted((r for r in result.rules if r.rule_type == "escalation" and r.level), key=lambda r: r.level)
    items = [f"Level {r.level}: {', '.join(result.entities.get('role', x).name for x in r.role_ids) or ', '.join(r.unresolved_targets)}"
             f" -- {r.statement}" for r in levels]
    return {"answer": "Escalate by level; Level 3 names executives who are not roles in roles.csv (a gap).",
            "items": items, "sources": [r.rule_id for r in levels] + ["PROC-CS-001"]}


def q_security_training_deadline(result):
    c = next(c for c in result.conflicts["conflicts"] if c["conflict_type"] == "numeric_fact")
    current = next(s for s in c["sides"] if s["status"] == "Active")
    old = next(s for s in c["sides"] if s["status"] == "Archived")
    return {"answer": f"{current['statement']} (current, {current['document_id']} v{current['version']}). "
                      f"An archived document said {old['values'][0]}s; both documents state the change.",
            "items": [current["statement"], old["statement"]], "sources": [c["conflict_id"]] + c["documents_involved"]}


def q_onboarding_meaning(result):
    note = result.entities.get("terminology", "TERM-TN-002")
    return {"answer": "Two different things: employee onboarding (HR, PROC-HR-001) and customer onboarding "
                      "(Implementation, PROC-IMPL-001). Ask which one is meant.",
            "items": [note.attributes["note_text"]], "sources": note.source_documents}


def q_project_lead(result):
    role = result.entities.get("role", "ROLE-022")
    return {"answer": f"'Project Lead' is the pre-2025-03-01 title of the {role.name} ({role.entity_id}).",
            "items": role.attributes["alias_evidence"]["Project Lead"], "sources": ["DOC-CORP-003", "DOC-CORP-002"]}


def q_new_data_analyst(result):
    [h] = result.employee_context["hypothetical_examples"]
    need = [f"{x['statement']} -> {', '.join(p['name'] for a in x['approvers'] for p in a['people'])}"
            for x in h["approvals"]["you_may_need_approval_from"]]
    return {"answer": f"Hypothetical profile ({h['name']}): request warehouse access early; it needs the Data & "
                      f"Analytics Manager's approval and is not automatic.",
            "items": need + [f"read: {r['document_id']} ({r['why']})" for r in h["reading_list"]],
            "sources": ["employee_context.json: hypothetical_examples"]}


QUESTIONS = [
    Question("Q01", "organization", "Who works in Data & Analytics?", q_who_works_in_data, ["entities.json", "relationships.json"]),
    Question("Q02", "organization", "Who owns customer onboarding?", q_who_owns_customer_onboarding, ["processes.json"]),
    Question("Q03", "organization", "Which department handles implementation?", q_which_department_implementation, ["relationships.json"]),
    Question("Q04", "processes", "Who approves a customer proposal?", q_who_approves_proposal, ["roles.json: approval_rules"]),
    Question("Q05", "processes", "What happens during customer onboarding?", q_customer_onboarding_steps, ["processes.json"]),
    Question("Q06", "processes", "Which documents describe incident management?", q_documents_incident_management, ["processes.json", "topics.json"]),
    Question("Q07", "relationships", "Which products are involved in customer onboarding?", q_products_in_onboarding, ["processes.json"]),
    Question("Q08", "relationships", "Which policies govern employee onboarding?", q_policies_govern_employee_onboarding, ["processes.json"]),
    Question("Q09", "relationships", "Which roles interact with Customer Success?", q_roles_interact_with_cs, ["processes.json", "roles.json"]),
    Question("Q10", "knowledge_quality", "Are there conflicting instructions?", q_conflicting_instructions, ["conflicts.json"]),
    Question("Q11", "knowledge_quality", "Which document appears outdated?", q_outdated_documents, ["outdated_knowledge.json"]),
    Question("Q12", "knowledge_quality", "What information is missing?", q_missing_information, ["knowledge_gaps.json"]),
    Question("Q13", "germany", "I'm a new employee in Germany. What should I take care of?", q_new_in_germany, ["germany_checklists.json", "employee_context.json"]),
    Question("Q14", "germany", "What settling-in tasks are relevant to me?", q_relevant_settling_tasks, ["germany_checklists.json"]),
    Question("Q15", "germany", "What have I not completed yet?", q_not_completed, ["germany_checklists.json"]),
    Question("Q16", "germany", "What have I potentially forgotten?", q_possibly_forgotten, ["germany_checklists.json"]),
    Question("Q17", "germany", "Give me the general steps.", q_general_steps, ["germany_checklists.json"]),
    Question("Q18", "communication", "How should this message be communicated in a German workplace context?", q_german_workplace_message, ["communication_guidance.json"]),
    Question("Q19", "communication", "Can I change the communication tone?", q_change_tone, ["communication_guidance.json"]),
    Question("Q20", "communication", "What is company policy versus cultural communication guidance?", q_policy_vs_culture, ["communication_guidance.json"]),
    Question("Q21", "processes", "Who approves access to the analytics data warehouse?", q_warehouse_access, ["roles.json: approval_rules"]),
    Question("Q22", "processes", "Who should a CSM escalate an at-risk account to?", q_csm_escalation, ["roles.json: escalation_rules"]),
    Question("Q23", "knowledge_quality", "What is the security training deadline for new hires?", q_security_training_deadline, ["conflicts.json"]),
    Question("Q24", "terminology", "What does 'onboarding' mean at Nexora?", q_onboarding_meaning, ["entities.json"]),
    Question("Q25", "terminology", "Who is the 'Project Lead'?", q_project_lead, ["entities.json", "potential_duplicates.json"]),
    Question("Q26", "employee_context", "I'm a new Data Analyst in Munich, week 2. What do I need?", q_new_data_analyst, ["employee_context.json"]),
]


def answer_all(result) -> list[dict]:
    out = []
    for q in QUESTIONS:
        try:
            a = q.answer_fn(result)
            status = "answered" if a.get("items") or a.get("answer") else "no_answer"
        except Exception as exc:            # reported, never hidden
            a, status = {"answer": None, "items": [], "sources": [], "error": f"{type(exc).__name__}: {exc}"}, "error"
        out.append({"id": q.qid, "category": q.category, "question": q.question, "status": status,
                    "supported_by": q.supported_by, **a})
    return out

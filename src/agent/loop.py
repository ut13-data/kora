"""
The agent loop, Stage 2: Goal -> State -> Plan -> Act -> Verify -> Stop.

Read it top to bottom, it is the whole agent. Every decision below is plain
Python; no language model is involved yet. Each step is written in the task's
notebook (AgentTask.steps), so `python -m src.agent ... ` can show all of it.

Rules from the Stage 1 lesson, enforced here:
    - Ask ONE question per run, then stop and wait.        (calm, never nagging)
    - Never mark anything done. Only the employee's answer does that (Stage 3).
    - A dependency ("Bank Account needs Housing") only shapes what to ASK.
    - Nexora's suggested pace is quoted as a company suggestion with its source;
      no legal deadline is stated, because none is documented.
    - Eligibility comes from the company record and is explained politely.
    - A tool error stops the task. The agent never retries the same failing step.
    - A hard step budget: the loop can never run forever.
"""

import re

from src.agent.task import AgentTask, StepLimit
from src.agent.tools import ToolContext, call_tool

# Goals this agent accepts. Stage 2 is deliberately narrow: one goal, done well.
# (The relocation patterns from kora_config.json, plus everyday "settle" / "checklist".)
GOAL_PATTERNS = [r"\bgerman", r"\brelocat", r"\bsettl", r"\banmeldung\b", r"\bchecklist\b", r"\bnew joiner\b"]

MAX_STEPS = 10      # the normal path uses 8; the spare 2 are headroom, not a target


def _goal_supported(goal: str) -> bool:
    return any(re.search(p, goal, re.I) for p in GOAL_PATTERNS)


def _pace_line(item: dict) -> dict | None:
    """The suggested-pace statement that sets the item's due day, with its document."""
    return next((p for p in item["pace"] if p["due_by_day"] == item["due_by_day"]), None)


def _unblocks(item_id: str, items: dict, open_ids: set[str]) -> list[str]:
    """Open items that list ITEM_ID as something they depend on."""
    return [i["title"] for i in items.values()
            if i["item_id"] in open_ids and any(d["item_id"] == item_id for d in i["depends_on"])]


def _compose_question(first_name: str, day: int, city: str, item: dict, unblocks: list[str],
                      greet: bool = True) -> tuple[str, list[dict]]:
    """The question text, plus every company statement in it and the source field it came from."""
    claims = []
    lines = ([f"Hi {first_name}! You're on day {day} in {city}. One quick question to help you settle in:"] if greet
             else ["Next one:"]) + [f"Have you sorted out {item['title']} yet?"]
    pace = _pace_line(item)
    if pace:
        lines.append(f"Why I'm asking: Nexora's checklist suggests \"{pace['statement']}\" "
                     f"(by day {pace['due_by_day']}, {pace['document_id']}). This is a company suggestion, not a legal deadline.")
        claims.append({"text": pace["statement"], "field": "pace", "document_id": pace["document_id"]})
    if unblocks:
        lines.append(f"It usually comes before: {', '.join(unblocks)}.")
    if item["official_guidance_note"]:
        lines.append(item["official_guidance_note"])
        claims.append({"text": item["official_guidance_note"], "field": "official_guidance_note",
                       "document_id": item["source_document"]})
    lines.append("No rush, and nothing is marked as done until you tell me.")
    return "\n".join(lines), claims


def _verify(claims: list[dict], item: dict) -> list[str]:
    """Every company statement in the question must be found word for word in the item's source fields."""
    problems = []
    for c in claims:
        source = item.get(c["field"])
        texts = [p["statement"] for p in source] if c["field"] == "pace" else [source]
        if c["text"] not in texts:
            problems.append(f"not found in {c['field']}: {c['text'][:60]}")
    return problems


def _not_eligible_message(status: str, record: dict) -> str:
    where = f"{record['city']}, {record['country']}" if record.get("city") else "outside Germany"
    if status == "not_applicable":
        return (f"Your company record shows you're based in {where} and you're not on the Germany relocation "
                f"program, so the Germany settling-in checklist doesn't apply to you. If you're planning a move, "
                f"HR can update your record. Meanwhile, you can still ask me general questions about the Germany guides.")
    if status == "germany_based_not_relocating":
        return (f"Your company record shows you're based in {where}, but not as an international relocation, so "
                f"the settling-in checklist isn't set up for you. If that's not right, HR can update your record. "
                f"You can still ask me anything about the Germany guides.")
    if status == "past_program_window":
        return (f"Your record shows you joined on {record['join_date']}. The settling-in program covers the first "
                f"90 days, so I won't walk you through the checklist, but the guides are all still there if you "
                f"want to look something up.")
    return f"The settling-in checklist isn't available for you right now (status: {status})."


def run(goal: str, actor: str, ctx: ToolContext | None = None, max_steps: int = MAX_STEPS) -> AgentTask:
    return start(goal, actor, ctx, max_steps)[0]


def start(goal: str, actor: str, ctx: ToolContext | None = None, max_steps: int = MAX_STEPS):
    """Run the loop once. Returns (task, record, care); record/care are None if it stopped early."""
    ctx = ctx or ToolContext()
    task = AgentTask(goal=goal, actor=actor, max_steps=max_steps)
    record = care = None
    try:
        record, care = _loop(task, ctx)
    except StepLimit:
        task.finish("stopped", f"step limit reached ({max_steps} steps); stopped safely")
    return task, record, care


def _tool(task: AgentTask, ctx: ToolContext, name: str, **args):
    """Call one READ tool and write the observation down. On failure, stop the task (no retry)."""
    result = call_tool(name, ctx, allowed_level="read", **args)
    if result.ok:
        task.log("state", f"Read with {name}: ok", tool=name)
        return result.data
    task.errors.append(f"{name}: {result.error}")
    task.log("state", f"{name} refused or failed: {result.error}", tool=name)
    task.message = f"I couldn't continue: {result.error}."
    task.finish("stopped", f"tool '{name}' failed; not retried")
    return None


def _loop(task: AgentTask, ctx: ToolContext):
    # GOAL: is this something this agent is built for?
    if not _goal_supported(task.goal):
        task.log("goal", "Goal not recognised as a Germany settling-in goal")
        task.message = "Right now I can only help with settling in to Germany. For other questions, use Ask."
        task.finish("stopped", "goal not supported by this agent")
        return None, None
    task.log("goal", f"Goal accepted: \"{task.goal}\"")

    # STATE: what do the company record and the checklist say? (read-only tools)
    record = _tool(task, ctx, "read_employee_record", actor=task.actor, employee_id=task.actor)
    if record is None:
        return None, None
    care = _tool(task, ctx, "read_settling_in_plan", actor=task.actor, employee_id=task.actor)
    if care is None:
        return None, None

    status = care["eligibility"]["status"]
    if status not in ("active", "pre_start"):
        task.log("plan", f"Not eligible for the checklist: status '{status}'",
                 evidence=care["eligibility"].get("evidence", []))
        task.message = _not_eligible_message(status, record)
        task.finish("stopped", f"not eligible ({status})")
        return None, None

    ask_next(task, record, care)
    return record, care


def ask_next(task: AgentTask, record: dict, care: dict, exclude=(), greet: bool = True) -> None:
    """PLAN -> ACT -> VERIFY -> STOP for one question. EXCLUDE: items the employee said 'not yet' to."""
    # PLAN: which ONE item to ask about? Phase 3's own order: unblocked, then pace.
    plan, items = care["plan"], care["items"]
    open_ids = {p["item_id"] for p in plan["items"] if p["status"] == "not_reported"}
    reported = plan["reported_complete"]
    task.log("plan", f"{len(reported)} item(s) reported done, {len(open_ids)} not reported yet",
             reported_complete=reported, progress_source=care["progress_source"])
    candidates = [i for i in plan["next_steps"] if i not in exclude]
    if not plan["next_steps"]:
        task.message = "Everything on your settling-in checklist is reported as done. Nice work!"
        task.finish("done", "no open checklist items")
        return
    if not candidates:
        task.message = "That's everything I'd suggest for now. We can pick it up again any time."
        task.finish("done", "no more items to ask about in this session")
        return
    item_id = candidates[0]
    item = items[item_id]
    unblocks = _unblocks(item_id, items, open_ids)
    task.log("plan", f"Ask about '{item['title']}': first in order (nothing blocking it"
                     + (f", it comes before {len(unblocks)} other item(s))" if unblocks else ")"),
             item_id=item_id, candidates=candidates)

    # ACT: prepare the one question (nothing is written anywhere).
    question, claims = _compose_question(record["first_name"], care["eligibility"]["day"],
                                         record["city"], item, unblocks, greet)
    task.log("act", "Prepared one question for the employee", item_id=item_id)

    # VERIFY: every company statement in the question must come from the checklist data.
    problems = _verify(claims, item)
    if problems:
        task.errors.extend(problems)
        task.log("verify", "Verification failed; question not shown", problems=problems)
        task.message = "I couldn't prepare a checked question this time."
        task.finish("stopped", "verification failed")
        return
    task.log("verify", f"{len(claims)} company statement(s) checked against the source: all found")

    # STOP: one question per run, then wait for the employee.
    task.question, task.focus_item = question, item_id
    task.log("stop", "Waiting for the employee's answer")
    task.finish("waiting_for_user", "asked one question; waiting for the employee")

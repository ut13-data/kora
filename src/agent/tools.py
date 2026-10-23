"""
The agent's tools: the ONLY things it is able to do.

A tool is like an instrument handed to an assistant: a thermometer can read a
temperature, it cannot paint a wall. Each tool here has
    - one clear purpose,
    - checked inputs        (a bad employee ID is refused, not guessed),
    - a permission check    done HERE, in Python, every time it is called,
    - a predictable result  (ToolResult: ok + data, or not ok + error).

Stage 2 tools only READ. Stage 3 adds ONE write tool, record_checklist_progress,
which changes only the employee's own checklist progress and can be undone.
The loop can only reach a tool through call_tool(), which refuses tools that
aren't registered, tools above the permission level the caller allows, and
any write that the employee has not explicitly confirmed.
"""

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.intelligence.germany import ChecklistItem, GermanyCare, load_progress

EMPLOYEE_ID = re.compile(r"^EMP-\d{4}$")


@dataclass
class ToolContext:
    """Where the tools read from. Tests point these at temporary copies."""
    raw_dir: Path = Path("data/raw/data")
    intelligence_dir: Path = Path("data/intelligence")
    progress_path: Path = Path("data/context/checklist_progress.json")
    history_path: Path = Path("data/context/progress_history.jsonl")
    today: date | None = None   # None: the checklist's as_of date (the demo date, 2025-04-22)

    # Stage 4: everything else KORA stores about a person lives next to the progress file,
    # so a test that points progress_path at a temporary folder moves ALL of it there.
    @property
    def context_dir(self) -> Path:
        return self.progress_path.parent

    @property
    def memory_dir(self) -> Path:           # personal memory, one JSON file per employee
        return self.context_dir / "memory"

    @property
    def memory_history_path(self) -> Path:  # what changed in memory (never the content itself)
        return self.context_dir / "memory_history.jsonl"

    @property
    def prefs_path(self) -> Path:           # Phase 3's own input: preferred tone lives here
        return self.context_dir / "communication_preferences.json"

    @property
    def tasks_dir(self) -> Path:            # saved conversations, one folder per employee
        return self.context_dir / "tasks"

    def current_date(self) -> date:
        if self.today:
            return self.today
        saved = json.loads((self.intelligence_dir / "germany_checklists.json").read_text(encoding="utf-8"))
        return date.fromisoformat(saved["as_of"])


@dataclass
class ToolResult:
    ok: bool
    data: dict = field(default_factory=dict)
    error: str | None = None


class ToolRefused(Exception):
    """Input or permission check failed. The message is safe to show the employee."""


def _check_self(actor: str, employee_id: str) -> None:
    for value in (actor, employee_id):
        if not isinstance(value, str) or not EMPLOYEE_ID.match(value):
            raise ToolRefused(f"'{value}' is not a valid employee ID (expected EMP-0000)")
    if actor != employee_id:
        # Relocation status and checklist progress are private (person_visibility):
        # only the person themself may read them through the agent.
        raise ToolRefused("you can only open your own settling-in information")


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- tools


def read_employee_record(ctx: ToolContext, actor: str, employee_id: str) -> dict:
    """The employee's own company record: where they are based and the relocation flag."""
    _check_self(actor, employee_id)
    row = next((r for r in _read_csv(ctx.raw_dir / "employees.csv") if r["employee_id"] == employee_id), None)
    if row is None:
        raise ToolRefused(f"{employee_id} is not in the company records")
    loc = next((r for r in _read_csv(ctx.raw_dir / "locations.csv") if r["location_id"] == row["location_id"]), {})
    return {
        "employee_id": employee_id,
        "first_name": row["first_name"],
        "location_id": row["location_id"],
        "city": loc.get("city"),
        "country": loc.get("country"),
        "join_date": row["join_date"],
        "relocating_to_germany": (row["international_relocation_germany"] or "").strip().lower() == "yes",
        "source": f"employees.csv row {employee_id}; locations.csv row {row['location_id']}",
    }


def read_settling_in_plan(ctx: ToolContext, actor: str, employee_id: str) -> dict:
    """Eligibility (from Phase 3) and the checklist plan, recomputed with the CURRENT progress file.

    Reuses Phase 3's GermanyCare.plan(), so the order, pace and 'blocked by' rules are the
    same ones My week already shows. Completion comes only from checklist_progress.json.
    """
    _check_self(actor, employee_id)
    saved = json.loads((ctx.intelligence_dir / "germany_checklists.json").read_text(encoding="utf-8"))
    as_of = date.fromisoformat(saved["as_of"])
    entry = next((e for e in saved["employees"] if e["employee_id"] == employee_id), None)
    if entry is None:
        # Phase 3 leaves out exactly two groups: not based in Germany, or based there but not relocating.
        record = read_employee_record(ctx, actor, employee_id)
        status = "germany_based_not_relocating" if (record["country"] or "").lower() == "germany" else "not_applicable"
        return {"eligibility": {"status": status, "evidence": [record["source"]]}, "as_of": saved["as_of"]}

    eligibility = entry["eligibility"]
    if eligibility["status"] not in ("active", "pre_start"):
        return {"eligibility": eligibility, "as_of": saved["as_of"]}

    progress, source = load_progress(ctx.progress_path)
    items = {d["item_id"]: ChecklistItem(**d) for d in saved["items"]}
    care = GermanyCare(saved["program"], items, [d["item_id"] for d in saved["items"]],
                       saved["locations"], saved["gaps"], source, progress, as_of)
    city = read_employee_record(ctx, actor, employee_id)["city"]
    plan = care.plan(employee_id, eligibility["day"], city)
    return {"eligibility": eligibility, "as_of": saved["as_of"], "progress_source": source,
            "plan": plan, "items": {i: items[i].to_dict() for i in items}}


def read_my_progress(ctx: ToolContext, actor: str, employee_id: str) -> dict:
    """Exactly what is stored for this employee in checklist_progress.json (used to VERIFY a write)."""
    _check_self(actor, employee_id)
    progress, _ = load_progress(ctx.progress_path)
    mine = progress.get(employee_id, {})
    return {"completed": dict(mine.get("completed", {})), "not_applicable": list(mine.get("not_applicable", []))}


ACTIONS = ("complete", "not_applicable", "undo")
EARLIEST_BEFORE_JOIN = timedelta(days=365)   # things like housing can be done before day 1, not years before


def record_checklist_progress(ctx: ToolContext, actor: str, employee_id: str, item_id: str,
                              action: str, on_date: str | None = None) -> dict:
    """Change ONE item of the employee's own progress. Reversible: 'undo' removes the entry.

    Checks, all in Python: own record only; the item is on the checklist; the action is known;
    a 'complete' date is a real date, not in the future, not long before joining.
    Writes safely (temporary file, then swap) and appends one line to the history file.
    """
    _check_self(actor, employee_id)
    if action not in ACTIONS:
        raise ToolRefused(f"unknown action '{action}' (allowed: {', '.join(ACTIONS)})")
    saved = json.loads((ctx.intelligence_dir / "germany_checklists.json").read_text(encoding="utf-8"))
    items = {i["item_id"]: i for i in saved["items"] if i["kind"] != "reference"}
    if item_id not in items:
        raise ToolRefused(f"'{item_id}' is not an item on the settling-in checklist")

    if action == "complete":
        try:
            when = date.fromisoformat(on_date or "")
        except ValueError:
            raise ToolRefused(f"'{on_date}' is not a date (expected YYYY-MM-DD)") from None
        if when > ctx.current_date():
            raise ToolRefused("that date is in the future; an item can only be recorded once it's done")
        joined = date.fromisoformat(read_employee_record(ctx, actor, employee_id)["join_date"])
        if when < joined - EARLIEST_BEFORE_JOIN:
            raise ToolRefused("that date is more than a year before you joined; please check it")

    raw = json.loads(ctx.progress_path.read_text(encoding="utf-8")) if ctx.progress_path.exists() \
        else {"employees": {}}
    entry = raw.setdefault("employees", {}).setdefault(employee_id, {"completed": {}, "not_applicable": []})
    entry.setdefault("completed", {})
    entry.setdefault("not_applicable", [])
    before = {"completed": entry["completed"].get(item_id), "not_applicable": item_id in entry["not_applicable"]}

    entry["completed"].pop(item_id, None)
    entry["not_applicable"] = [i for i in entry["not_applicable"] if i != item_id]
    if action == "complete":
        entry["completed"][item_id] = on_date
    elif action == "not_applicable":
        entry["not_applicable"].append(item_id)
    after = {"completed": entry["completed"].get(item_id), "not_applicable": item_id in entry["not_applicable"]}

    _atomic_write(ctx.progress_path, json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    change = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": actor,
              "employee_id": employee_id, "item_id": item_id, "action": action,
              "before": before, "after": after}
    ctx.history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ctx.history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(change, ensure_ascii=False) + "\n")
    return change


def _atomic_write(path: Path, text: str) -> None:
    """Write to a temporary file next to PATH, then swap it in: a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_my_memory(ctx: ToolContext, actor: str, employee_id: str) -> dict:
    """What the employee has told KORA about themselves (Stage 4). Own record only."""
    from src.agent.memory import read_memory          # imported here to avoid a circular import
    _check_self(actor, employee_id)
    return read_memory(ctx, employee_id)


def update_my_memory(ctx: ToolContext, actor: str, employee_id: str, action: str, value: str | None = None) -> dict:
    """Add a note, set name or tone, or forget something. Own record only; checks live in memory.py."""
    from src.agent.memory import update_memory
    _check_self(actor, employee_id)
    return update_memory(ctx, employee_id, action, value)


# ---------------------------------------------------------------- registry

# name -> (function, permission level). Levels, lowest first:
#   read        look things up, change nothing
#   write_self  change the employee's own record in KORA (Stage 3+), reversible
#   external    anything outside KORA: email, calendar (much later, always with approval)
LEVELS = ("read", "write_self", "external")
TOOLS = {
    "read_employee_record": (read_employee_record, "read"),
    "read_settling_in_plan": (read_settling_in_plan, "read"),
    "read_my_progress": (read_my_progress, "read"),
    "record_checklist_progress": (record_checklist_progress, "write_self"),
    "read_my_memory": (read_my_memory, "read"),
    "update_my_memory": (update_my_memory, "write_self"),
}


def call_tool(name: str, ctx: ToolContext, allowed_level: str = "read", confirmed: bool = False,
              **args) -> ToolResult:
    """The single door to every tool. Refused: unknown tools, tools above ALLOWED_LEVEL,
    any non-read tool without CONFIRMED=True (the employee said yes), and bad inputs."""
    if name not in TOOLS:
        return ToolResult(False, error=f"unknown tool '{name}'")
    func, level = TOOLS[name]
    if LEVELS.index(level) > LEVELS.index(allowed_level):
        return ToolResult(False, error=f"tool '{name}' needs '{level}' permission; this stage allows '{allowed_level}'")
    if level != "read" and not confirmed:
        return ToolResult(False, error=f"tool '{name}' changes data and needs the employee's confirmation first")
    try:
        return ToolResult(True, data=func(ctx, **args))
    except ToolRefused as e:
        return ToolResult(False, error=str(e))
    except TypeError as e:                       # wrong or missing arguments
        return ToolResult(False, error=f"bad arguments for '{name}': {e}")
    except (OSError, ValueError, KeyError) as e:  # missing file, broken JSON, missing column
        return ToolResult(False, error=f"'{name}' failed: {type(e).__name__}: {e}")

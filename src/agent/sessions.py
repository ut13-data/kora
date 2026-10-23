"""
Stage 4: task history -- saving a conversation so it can continue another day.

After every reply, the ChatSession is written to
    data/context/tasks/<EMP>/<TASK-id>.json
as plain JSON you can open and read. Next time, KORA looks for the most recent
conversation that was left waiting on a question and offers to continue it
(or to start fresh, which closes the old one).

Safety points:
    - The employee ID becomes part of a folder name, so it is checked (EMP-0000) first:
      "../../" can never be used to read or write somewhere else.
    - Only the employee's own folder is ever read.
    - A saved, UNCONFIRMED proposal is never carried over: on continue, KORA asks again.
      Saying yes to something proposed days ago, without seeing it again, isn't real consent.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.agent.tools import EMPLOYEE_ID, ToolContext, _atomic_write

RESUMABLE = ("waiting_for_user", "paused")


def _folder(ctx: ToolContext, employee_id: str) -> Path:
    if not isinstance(employee_id, str) or not EMPLOYEE_ID.match(employee_id):
        raise ValueError(f"'{employee_id}' is not a valid employee ID")
    return ctx.tasks_dir / employee_id


def save(ctx: ToolContext, employee_id: str, snapshot: dict) -> Path:
    folder = _folder(ctx, employee_id)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = {**snapshot, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = folder / f"{snapshot['task']['task_id']}.json"
    _atomic_write(path, json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    return path


def latest_open(ctx: ToolContext, employee_id: str) -> dict | None:
    """The most recent conversation still waiting on the employee, or None."""
    folder = _folder(ctx, employee_id)
    if not folder.exists():
        return None
    found = []
    for path in folder.glob("TASK-*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue                                   # a broken file is skipped, never fatal
        if data.get("task", {}).get("status") in RESUMABLE and data.get("task", {}).get("focus_item"):
            found.append(data)
    return max(found, key=lambda d: d.get("updated", ""), default=None)


def close(ctx: ToolContext, employee_id: str, snapshot: dict, reason: str) -> None:
    """Mark an old conversation as finished, so it is not offered again."""
    task = {**snapshot["task"], "status": "stopped", "stop_reason": reason}
    save(ctx, employee_id, {**snapshot, "task": task, "phase": "ended"})

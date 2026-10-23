"""
Stage 4: personal memory -- what an employee has told KORA about THEMSELVES.

Four kinds of memory, never mixed:
    working memory      the conversation happening now            (ChatSession, in RAM)
    task history        where a conversation stopped               (sessions.py, data/context/tasks/)
    personal memory     what the person explicitly told KORA       (THIS file, data/context/memory/)
    company knowledge   documents and the graph                    (Phases 2-5, never written here)

What personal memory may hold (decision 1a: only what the person explicitly asks):
    preferred_name   "call me Jules"                       -> data/context/memory/<EMP>.json
    preferred_tone   "set my tone to direct"               -> data/context/communication_preferences.json
                     (Phase 3's own input, so My week and the tone guidance use it too: one source of truth)
    notes            "remember that I start German classes on Mondays"

Rules, all in Python:
    - nothing is inferred; every entry says when the person told KORA;
    - obviously sensitive things (health, money, religion, ID numbers, ...) are refused, not stored;
      this is a keyword check, so it catches the obvious cases, not every case;
    - at most MAX_NOTES short notes, so memory stays something a person can read in one go;
    - "forget" really forgets: the history file records THAT something changed, never WHAT was said.
"""

import json
import re
from datetime import datetime, timezone

from src.agent.tools import ToolContext, ToolRefused, _atomic_write

MAX_NOTES = 20
MAX_NOTE_CHARS = 200
NAME = re.compile(r"^[^\W\d_][^\W\d_ .'-]*(?:[ .'-][^\W\d_]+){0,3}$")       # letters, up to 4 parts
TONES = {"neutral": "neutral", "direct": "direct", "friendly": "relationship_oriented",
         "warm": "relationship_oriented", "relationship_oriented": "relationship_oriented"}
ACTIONS = ("add_note", "set_name", "set_tone", "forget")

SENSITIVE = re.compile(
    r"\b(?:health|ill(?:ness)?|sick|disease|diagnos\w*|medic\w*|medication|therap\w*|doctor|hospital|pregnan\w*|"
    r"disab\w*|mental|depress\w*|anxiety|religio\w*|church|mosque|temple|pray\w*|politic\w*|vote[sd]?|"
    r"sexual\w*|gay|lesbian|salary|income|debt|loan|credit|bank account|iban|password|passport|"
    r"visa status|criminal|arrest\w*)\b", re.I)
LONG_NUMBER = re.compile(r"\d[\d -]{7,}\d")                                  # IDs, account or card numbers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(ctx: ToolContext, employee_id: str):
    return ctx.memory_dir / f"{employee_id}.json"       # employee_id is already checked (EMP-0000)


def _load(ctx: ToolContext, employee_id: str) -> dict:
    path = _path(ctx, employee_id)
    if not path.exists():
        return {"employee_id": employee_id, "preferred_name": None, "notes": [], "next_note": 1}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prefs(ctx: ToolContext) -> dict:
    if not ctx.prefs_path.exists():
        return {"employees": {}}
    return json.loads(ctx.prefs_path.read_text(encoding="utf-8"))


def read_memory(ctx: ToolContext, employee_id: str) -> dict:
    mem = _load(ctx, employee_id)
    pref = _load_prefs(ctx).get("employees", {}).get(employee_id) or {}
    tone = None
    if pref.get("preferred_tone") and pref.get("set_by") == "self":
        tone = {"value": pref["preferred_tone"], "said_on": pref.get("updated")}
    return {"preferred_name": mem["preferred_name"], "preferred_tone": tone, "notes": mem["notes"]}


def _check_text(text: str, what: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        raise ToolRefused(f"the {what} is empty")
    if SENSITIVE.search(text) or LONG_NUMBER.search(text):
        raise ToolRefused("that sounds personal or sensitive, so I won't store it "
                          "(you can always talk to HR directly)")
    return text


def update_memory(ctx: ToolContext, employee_id: str, action: str, value: str | None) -> dict:
    if action not in ACTIONS:
        raise ToolRefused(f"unknown memory action '{action}' (allowed: {', '.join(ACTIONS)})")
    now = _now()

    if action == "set_tone":
        tone = TONES.get((value or "").strip().lower().replace(" ", "_").replace("-", "_"))
        if tone is None:
            raise ToolRefused("tone must be neutral, direct or friendly")
        prefs = _load_prefs(ctx)
        entry = prefs.setdefault("employees", {}).setdefault(employee_id, {})
        entry.update({"preferred_tone": tone, "set_by": "self", "updated": now[:10]})
        _atomic_write(ctx.prefs_path, json.dumps(prefs, indent=2, ensure_ascii=False) + "\n")
        return _log(ctx, employee_id, action, "tone", now, {"preferred_tone": tone})

    mem = _load(ctx, employee_id)
    if action == "set_name":
        name = _check_text(value, "name")
        if len(name) > 30 or not NAME.match(name):
            raise ToolRefused("a name can only have letters, spaces, apostrophes or hyphens (up to 30 characters)")
        mem["preferred_name"] = {"value": name, "said_on": now}
        target, result = "name", {"preferred_name": name}
    elif action == "add_note":
        note = _check_text(value, "note")
        if len(note) > MAX_NOTE_CHARS:
            raise ToolRefused(f"that's a bit long to remember; please keep it under {MAX_NOTE_CHARS} characters")
        if len(mem["notes"]) >= MAX_NOTES:
            raise ToolRefused(f"I can keep {MAX_NOTES} notes; please ask me to forget one first")
        note_id = f"M{mem['next_note']}"
        mem["next_note"] += 1
        mem["notes"].append({"id": note_id, "text": note, "said_on": now})
        target, result = note_id, {"note_id": note_id, "text": note}
    else:                                                    # forget
        target = (value or "").strip()
        if target.lower() in ("all", "everything"):
            prefs = _load_prefs(ctx)
            if prefs.get("employees", {}).pop(employee_id, None) is not None:
                _atomic_write(ctx.prefs_path, json.dumps(prefs, indent=2, ensure_ascii=False) + "\n")
            mem = {"employee_id": employee_id, "preferred_name": None, "notes": [], "next_note": mem["next_note"]}
            target, result = "all", {"forgotten": "all"}
        elif target.lower() == "name":
            if not mem["preferred_name"]:
                raise ToolRefused("there's no name saved")
            mem["preferred_name"], result = None, {"forgotten": "name"}
        elif target.lower() == "tone":
            prefs = _load_prefs(ctx)
            if not prefs.get("employees", {}).pop(employee_id, None):
                raise ToolRefused("there's no tone saved")
            _atomic_write(ctx.prefs_path, json.dumps(prefs, indent=2, ensure_ascii=False) + "\n")
            return _log(ctx, employee_id, action, "tone", now, {"forgotten": "tone"})
        else:
            target = target.upper()
            if not any(n["id"] == target for n in mem["notes"]):
                raise ToolRefused(f"there's no note {target}")
            mem["notes"] = [n for n in mem["notes"] if n["id"] != target]
            result = {"forgotten": target}

    ctx.memory_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(_path(ctx, employee_id), json.dumps(mem, indent=2, ensure_ascii=False) + "\n")
    return _log(ctx, employee_id, action, target, now, result)


def _log(ctx: ToolContext, employee_id: str, action: str, target: str, at: str, result: dict) -> dict:
    """One line per change: WHO changed WHICH entry and HOW. Never the words themselves."""
    ctx.memory_history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ctx.memory_history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": at, "by": employee_id, "action": action, "target": target}) + "\n")
    return {"action": action, "target": target, **result}


def describe(memory: dict) -> str:
    """'What do you remember about me?' -- everything stored, with when the person said it."""
    lines = []
    if memory["preferred_name"]:
        lines.append(f"- You asked me to call you {memory['preferred_name']['value']} "
                     f"(you told me on {memory['preferred_name']['said_on'][:10]}).")
    if memory["preferred_tone"]:
        lines.append(f"- You prefer a {memory['preferred_tone']['value'].replace('_', '-')} tone "
                     f"(set on {memory['preferred_tone']['said_on']}).")
    for n in memory["notes"]:
        lines.append(f"- {n['id']}: {n['text']} (you told me on {n['said_on'][:10]})")
    if not lines:
        return ("I don't have anything saved about you yet. You can say \"remember that ...\", "
                "\"call me ...\" or \"set my tone to direct\".")
    return ("Here's everything I remember about you:\n" + "\n".join(lines) +
            "\nSay \"forget M1\", \"forget my name\" or \"forget everything\" to remove anything. "
            "(Your checklist progress is kept separately.)")

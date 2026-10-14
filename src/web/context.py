"""
Phase 5 M5 - "My week": one employee's own context, from Phase 3's saved
employee_context.json, made safe for that employee to see.

Phase 3 computed each context with full access. Before showing it, anything
that points at a document the employee may not read is removed: a process,
policy, approval rule, reading-list entry or source naming that document --
by ID, or by NAME inside a sentence ("governs Invoice and Payment Process,
which is relevant to you" is dropped for someone who can't read that process).
Only the employee themself gets their context (their relocation status,
onboarding stage and checklist are private: see person_visibility).
"""

import json
from pathlib import Path

ID_KEYS = ("document_id", "process_id", "policy_id")


def load_context(intelligence_dir: Path) -> dict:
    path = Path(intelligence_dir) / "employee_context.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _readable(item, can_read, hidden_names) -> bool:
    if isinstance(item, dict):
        if not all(can_read(item[k]) for k in ID_KEYS if isinstance(item.get(k), str)):
            return False
        return not any(n in json.dumps(item, ensure_ascii=False) for n in hidden_names)
    if isinstance(item, str):
        if item.isupper() and "-" in item and " " not in item and not can_read(item):
            return False
        return not any(n in item for n in hidden_names)
    return True


def scrub(value, can_read, hidden_names=frozenset()):
    """Recursively drop list entries that name a hidden document (by ID or by name)."""
    if isinstance(value, dict):
        return {k: scrub(v, can_read, hidden_names) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, can_read, hidden_names) for v in value if _readable(v, can_read, hidden_names)]
    return value


def my_context(data: dict, employee_id: str, can_read, hidden_names=frozenset()) -> dict | None:
    """HIDDEN_NAMES: IDs and names of documents/processes/topics this employee may not see."""
    person = next((e for e in data.get("employees", []) if e.get("employee_id") == employee_id), None)
    if person is None:
        return None
    out = scrub(person, can_read, frozenset(hidden_names))
    out["as_of"] = data.get("as_of")
    out["as_of_note"] = data.get("as_of_note")
    return out

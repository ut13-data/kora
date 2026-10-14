"""
Phase 4 M7 - access control: a user only gets answers from documents they may read.

Rules live in the config (section "access"), one line per restricted document:

    "access": {"rules": {"PROC-FIN-003": ["dept:DEPT-FIN", "dept:DEPT-OPS"],
                         "DOC-DEPT-FIN": ["dept:DEPT-FIN", "manager"]}}

* A document with NO rule is open to everyone.
* A document matched by a rule is readable only by users in at least one of
  the listed groups. If several rules match (e.g. "PROC-FIN-*" and
  "PROC-FIN-003"), the user must pass EVERY one (deny wins).
* Patterns are document IDs, with * allowed ("PROC-FIN-*"). An archived
  version shares its document's rule when the pattern covers its ID.
* Groups: all | manager | dept:<id> | loc:<id> | role:<id> | emp:<id>.
  A user's groups come from the employees table (Phase 2): their department,
  location, role, their own ID, "all", and "manager" if anyone reports to them.

Where it is enforced (so nothing leaks -- not a quote, a citation, a note, a
"closest source", or a document ID in the log):
  1. Search never sees chunks of documents the user can't read.
  2. The router's plan is cleaned (candidate documents, conflict sides, reasons).
  3. Every answer line is checked against the user's documents before
     verification; a conflict loses ALL its sides if one side is hidden
     (half a conflict would hint at the hidden document).
  4. Notes naming a hidden document (by ID or title) are removed.
A hidden document is never mentioned: the user gets the same "not found" as
for something that isn't documented at all.

user=None means no filtering (backend tools, evaluation). The Phase 5 UI/API
must always pass a user.
"""

import fnmatch
import re
from dataclasses import dataclass, field

GROUP_PATTERN = re.compile(r"^(all|manager|(dept|loc|role|emp):[A-Za-z0-9_.-]+)$")


@dataclass(frozen=True)
class User:
    user_id: str
    name: str = ""
    groups: frozenset = field(default_factory=frozenset)

    def describe(self) -> str:
        return f"{self.name or self.user_id} ({', '.join(sorted(self.groups))})"


def users_from_tables(tables: dict) -> dict[str, User]:
    """employee_id -> User, from the Phase 2 employees table (if any)."""
    rows = tables.get("employees") or []
    managers = {r.get("manager_id") for r in rows if r.get("manager_id")}
    users = {}
    for r in rows:
        emp = r.get("employee_id")
        if not emp:
            continue
        groups = {"all", f"emp:{emp}"}
        for prefix, column in (("dept", "dept_id"), ("loc", "location_id"), ("role", "role_id")):
            if r.get(column):
                groups.add(f"{prefix}:{r[column]}")
        if emp in managers:
            groups.add("manager")
        name = " ".join(x for x in (r.get("first_name"), r.get("last_name")) if x)
        users[emp] = User(emp, f"{name}, {r.get('role_title', '')}".strip(", "), frozenset(groups))
    return users


class AccessPolicy:
    def __init__(self, rules: dict[str, list[str]] | None = None):
        self.rules = [(pattern, frozenset(groups)) for pattern, groups in (rules or {}).items()]

    @property
    def active(self) -> bool:
        return bool(self.rules)

    def can_read(self, user: User | None, document_id: str) -> bool:
        if user is None:
            return True
        for pattern, groups in self.rules:
            if fnmatch.fnmatchcase(document_id, pattern) and not (groups & user.groups):
                return False
        return True

    def allowed(self, user: User | None, document_ids) -> set[str] | None:
        """The documents USER may read, or None for 'no filtering'."""
        if user is None:
            return None
        return {d for d in document_ids if self.can_read(user, d)}

    def restricted_documents(self, document_ids) -> set[str]:
        """Documents covered by at least one rule (for reports / the UI)."""
        return {d for d in document_ids if any(fnmatch.fnmatchcase(d, p) for p, _ in self.rules)}

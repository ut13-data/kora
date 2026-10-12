"""
Name resolution shared by every Phase 3 module.

Nexora writes the same thing in several ways:
    "IT / Security"  (departments.csv)  vs  "IT/Security"  (documents.csv)
    "Technical Support, IT / Security"   (one header naming TWO departments)
    "HR Manager (2022)"                  (a role with a historical note)

These helpers turn such text into real registry IDs, or report it as
unresolved. They never pick a "closest" match: that is the job of
duplicates.py, which only SUGGESTS.
"""

import re
from collections import defaultdict

from src.metadata.discovery import _slug as phase2_slug
from src.metadata.registry import EntityRegistry
from src.metadata.relationships import _PLACEHOLDERS, _core_value, _resolve_roles

# Phase 2's placeholder words, plus one more found in Nexora headers.
PLACEHOLDERS = set(_PLACEHOLDERS) | {"cross-department"}


def text_key(text: str) -> str:
    """Compare names on letters and digits only.
    'IT / Security', 'IT/Security' and 'it security' all become 'itsecurity'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def slug(text: str) -> str:
    """Same slug Phase 2 uses for topic IDs, so IDs stay consistent across phases."""
    return phase2_slug(text)


def unique(items) -> list:
    """Remove duplicates, keep first-seen order (sorted() would lose meaning)."""
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def version_key(version: str) -> str:
    """'2.0', '2' and '2.00' compare equal (same rule as Phase 2)."""
    try:
        return f"{float(version):g}"
    except (TypeError, ValueError):
        return (version or "").strip().lower()


def is_placeholder(raw: str) -> bool:
    return _core_value(raw).strip().lower() in PLACEHOLDERS


class DepartmentResolver:
    """Resolve free-text department values to DEPT-* IDs.

    Departments are only resolved from header FIELDS (Department:, Owner),
    never from running text: Phase 2 showed that "Sales" or "Operations"
    in a sentence is usually just an everyday word.
    """

    def __init__(self, registry: EntityRegistry):
        self._by_key: dict[str, str] = {}
        self.canonical: dict[str, str] = {}   # dept_id -> name from departments.csv
        for dept_id, rows in registry.by_id.get("departments", {}).items():
            name = (rows[0].get("name") or "").strip()
            self.canonical[dept_id] = name
            self._by_key[text_key(name)] = dept_id
        # dept_id -> spellings actually seen in documents (for aliases)
        self.variants_seen: dict[str, set[str]] = defaultdict(set)

    def _hit(self, text: str) -> str | None:
        dept_id = self._by_key.get(text_key(text))
        if dept_id and text.strip() != self.canonical[dept_id]:
            self.variants_seen[dept_id].add(text.strip())
        return dept_id

    def resolve(self, raw: str) -> tuple[list[str], list[str]]:
        """Return (department_ids, pieces_not_found).

        Order of attempts, mirroring Phase 2's role resolution:
          1. the whole value             'IT / Security'
          2. comma-separated parts       'Technical Support, IT / Security'
          3. slash-separated pieces, longest run first, so 'IT / Security'
             survives as ONE department inside 'Technical Support/IT/Security'
        """
        if not raw or raw.strip().lower() == "unknown" or is_placeholder(raw):
            return [], []

        whole = self._hit(raw)
        if whole:
            return [whole], []

        found: list[str] = []
        missing: list[str] = []
        for part in (p.strip() for p in raw.split(",")):
            if not part:
                continue
            hit = self._hit(part)
            if hit:
                found.append(hit)
                continue
            pieces = [p.strip() for p in part.split("/") if p.strip()]
            part_found: list[str] = []
            part_missing: list[str] = []
            i = 0
            while i < len(pieces):
                for j in range(len(pieces), i, -1):          # longest run first
                    hit = self._hit(" / ".join(pieces[i:j]))
                    if hit:
                        part_found.append(hit)
                        i = j
                        break
                else:
                    part_missing.append(pieces[i])
                    i += 1
            found.extend(part_found)
            # Nothing in this part matched: report it ONCE, as written
            # ('Executive / Corporate'), not as invented pieces ('Executive', 'Corporate').
            missing.extend(part_missing if part_found else [part])
        return unique(found), missing


def resolve_roles(raw: str, registry: EntityRegistry) -> tuple[list[str], list[str]]:
    """Phase 2's role resolution (exact title, documented alias, joint owners),
    with placeholders and empty values handled first."""
    if not raw or raw.strip().lower() == "unknown" or is_placeholder(raw):
        return [], []
    return _resolve_roles(raw, registry)

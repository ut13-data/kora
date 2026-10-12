"""
Entity normalization: find things that MAY be the same, and say why.

Every finding is a PotentialDuplicate. Nothing is merged, ever.

Method                 What it means                                        Confidence  Status
documented_alias       a Nexora document says two names mean the same      0.95        documented
documented_distinct    a Nexora document says one word means two things    0.95        documented
documented_related     overlapping terms, "not always interchangeable"     0.95        documented
version_pair           same name, different status (history, not a copy)   1.0         documented
shared_header_id       two files claim one Document ID in their headers    1.0         unreviewed
acronym                'CSM' looks like 'Customer Success Manager'         0.5         unreviewed
unresolved_reference   a name with no entity, and possible matches          0.5         unreviewed
name_similarity        near-identical names of the same type               0.5         unreviewed

"documented" means a source already settles it; "unreviewed" means a
person should look. Low-confidence methods only SUGGEST.
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.models import (
    CONF_DOCUMENTED, CONF_SIMILARITY, CONF_STRUCTURED, CUSTOMER, DEPARTMENT,
    DOCUMENT, EMPLOYEE, LOCATION, PROCESS, PRODUCT, ROLE, TERMINOLOGY,
    PotentialDuplicate, entity_key,
)

SIMILARITY_THRESHOLD = 0.85     # below this, names like 'Hiring Manager'/'HR Manager' (0.83) pair up
MAX_CANDIDATES = 3              # more possible matches than this = too vague to suggest

_METHOD_ORDER = ["documented_alias", "documented_distinct", "documented_related", "version_pair",
                 "shared_header_id", "acronym", "unresolved_reference", "name_similarity"]

# Tokens that mark deliberately different roles: 'Support Engineer L1' vs 'L2'.
_LEVEL_TOKEN = re.compile(r"^(l\d+|\d+|v\d+(\.\d+)?)$", re.I)
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")


def _tokens(name: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9&]+", name)


def _base_name(name: str) -> str:
    """'Customer Onboarding (archived)' -> 'Customer Onboarding'"""
    return _PARENTHETICAL.sub("", name).strip()


def is_acronym_of(short: str, long: str) -> bool:
    """True if `short` is `long` with one or more word runs replaced by initials.

    'CSM' ~ 'Customer Success Manager'      (whole name)
    'CS Manager' ~ 'Customer Success Manager' (part of the name)
    Only all-capital tokens of 2-4 letters count as acronyms.
    """
    a, b = _tokens(short), _tokens(long)
    i = j = 0
    used_acronym = False
    while i < len(a) and j < len(b):
        if a[i].lower() == b[j].lower():
            i, j = i + 1, j + 1
            continue
        n = len(a[i])
        if a[i].isupper() and 2 <= n <= 4 and j + n <= len(b) \
                and "".join(w[0] for w in b[j:j + n]).upper() == a[i]:
            i, j, used_acronym = i + 1, j + n, True
            continue
        return False
    return used_acronym and i == len(a) and j == len(b)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _only_levels_differ(a: str, b: str) -> bool:
    diff = set(t.lower() for t in _tokens(a)) ^ set(t.lower() for t in _tokens(b))
    return bool(diff) and all(_LEVEL_TOKEN.match(t) for t in diff)


# ---------------- Documented (a source settles it) ----------------

def _documented(index: EntityIndex) -> list[PotentialDuplicate]:
    found: list[PotentialDuplicate] = []

    # Role and product aliases, each with the evidence that documents it.
    for entity_type in (ROLE, PRODUCT):
        for entity in index.of_type(entity_type):
            for alias, evidence in entity.attributes.get("alias_evidence", {}).items():
                found.append(PotentialDuplicate(
                    "", entity.key, f"alias:{alias}", "documented_alias",
                    f"'{alias}' and '{entity.name}' name the same {entity_type}.",
                    CONF_DOCUMENTED, list(evidence), "documented"))

    # Terminology notes from the glossary.
    role_words = {e.name.lower() for e in index.of_type(ROLE)} | {
        a.lower() for e in index.of_type(ROLE) for a in e.aliases}
    for note in index.glossary_notes:
        term_key = entity_key(TERMINOLOGY, f"TERM-{note['note_id']}")
        evidence = [f"glossary {note['note_id']}: '{note['matched_phrase']}'"] if note["matched_phrase"] else []
        terms = note["terms"] or [note["note_id"]]   # guard: a note with no quoted term

        if note["relation"] == "equivalent":
            if all(t.lower() in role_words for t in terms):
                continue   # already reported as a role alias above
            found.append(PotentialDuplicate(
                "", term_key, "term:" + " | ".join(terms), "documented_alias",
                f"{note['note_id']} says {', '.join(repr(t) for t in terms)} refer to the same thing.",
                CONF_DOCUMENTED, evidence, "documented"))

        elif note["relation"] == "overloaded":
            # If the note names the processes involved, pair them explicitly.
            procs = [i for i in note["referenced_ids"] if index.get(PROCESS, i)]
            if len(procs) >= 2:
                for k in range(len(procs) - 1):
                    a, b = index.get(PROCESS, procs[k]), index.get(PROCESS, procs[k + 1])
                    found.append(PotentialDuplicate(
                        "", a.key, b.key, "documented_distinct",
                        f"'{a.name}' and '{b.name}' share the word {terms[0]!r}, but "
                        f"{note['note_id']} says it means two different things. NOT duplicates.",
                        CONF_DOCUMENTED, evidence, "documented"))
            else:
                found.append(PotentialDuplicate(
                    "", term_key, "term:" + " | ".join(terms), "documented_distinct",
                    f"{note['note_id']} says {terms[0]!r} is used for different things.",
                    CONF_DOCUMENTED, evidence, "documented"))

        elif note["relation"] == "not_interchangeable":
            found.append(PotentialDuplicate(
                "", term_key, "term:" + " | ".join(terms), "documented_related",
                f"{note['note_id']}: {', '.join(repr(t) for t in terms)} overlap but are not always "
                f"interchangeable. Resolve by context, not by merging.",
                CONF_DOCUMENTED, evidence, "documented"))
    return found


def _version_pairs(index: EntityIndex) -> list[PotentialDuplicate]:
    """Same base name, different status: two versions of one thing."""
    found: list[PotentialDuplicate] = []
    for entity_type in (PROCESS, DOCUMENT):
        groups: dict[str, list] = defaultdict(list)
        for e in index.of_type(entity_type):
            groups[_base_name(e.name).lower()].append(e)
        for members in groups.values():
            if len(members) < 2:
                continue
            statuses = {(m.attributes.get("status") or "").lower() for m in members}
            if len(statuses) < 2:
                continue   # same name AND same status: left to name_similarity
            members.sort(key=lambda m: (m.attributes.get("status") or "") != "Active")
            current = members[0]
            for older in members[1:]:
                found.append(PotentialDuplicate(
                    "", current.key, older.key, "version_pair",
                    f"Same name. {current.entity_id} is {current.attributes.get('status')} "
                    f"(v{current.attributes.get('version')}); {older.entity_id} is "
                    f"{older.attributes.get('status')} (v{older.attributes.get('version')}). "
                    f"Two versions of one {entity_type}, both kept as history.",
                    CONF_STRUCTURED,
                    [f"status column / header of {current.entity_id} and {older.entity_id}"],
                    "documented"))
    return found


def _shared_header_ids(index: EntityIndex) -> list[PotentialDuplicate]:
    """Two files whose headers claim the same Document ID."""
    by_header: dict[str, list] = defaultdict(list)
    for doc in index.of_type(DOCUMENT):
        header_id = doc.attributes.get("header_document_id")
        if header_id:
            by_header[header_id].append(doc)
    found = []
    for header_id, docs in sorted(by_header.items()):
        if len(docs) < 2:
            continue
        docs.sort(key=lambda d: d.entity_id != header_id)   # the one that "owns" the ID first
        owner = docs[0]
        for other in docs[1:]:
            found.append(PotentialDuplicate(
                "", owner.key, other.key, "shared_header_id",
                f"Both {owner.attributes['source_filename']} and {other.attributes['source_filename']} "
                f"say 'Document ID: {header_id}' in their headers; documents.csv lists them as "
                f"{owner.entity_id} and {other.entity_id}. Fix the header in the source data.",
                CONF_STRUCTURED,
                [f"{owner.attributes['source_filename']}: header", f"{other.attributes['source_filename']}: header",
                 "documents.csv: doc_id"],
                "unreviewed"))
    return found


# ---------------- Suggestions (a person should review) ----------------

_PHASE2_UNRESOLVED = [
    (re.compile(r"Related Roles value '(.+?)' not found"), ROLE),
    (re.compile(r"owner_role '(.+?)' not found"), ROLE),
    (re.compile(r"Related Processes value '(.+?)' not found"), PROCESS),
]


def _candidates(name: str, entities: list) -> list[tuple[object, str]]:
    """Possible matches for an unresolved name, each with the method that found it."""
    out = []
    name_tokens = {t.lower() for t in _tokens(name)}
    for e in entities:
        for label in [e.name, *e.aliases]:
            if is_acronym_of(name, label) or is_acronym_of(label, name):
                out.append((e, "acronym"))
                break
            label_tokens = {t.lower() for t in _tokens(label)}
            if name_tokens and name_tokens < label_tokens:
                out.append((e, "token_subset"))
                break
            if _similar(name, label) >= SIMILARITY_THRESHOLD:
                out.append((e, "spelling"))
                break
    return out


def _unresolved(index: EntityIndex, data: Phase2Data) -> tuple[list[PotentialDuplicate], list[dict]]:
    names: dict[str, set[str]] = defaultdict(set)
    for kind, values in index.unresolved_names.items():
        names[kind] |= values
    for note in data.unresolved_references:
        for pattern, kind in _PHASE2_UNRESOLVED:
            match = pattern.search(note)
            if match:
                names[kind].add(match.group(1).strip())

    found, unmatched = [], []
    for kind in sorted(names):
        for name in sorted(names[kind]):
            candidates = _candidates(name, index.of_type(kind))
            if not candidates or len(candidates) > MAX_CANDIDATES:
                unmatched.append({
                    "name": name, "expected_type": kind,
                    "reason": "no similar entity" if not candidates
                              else f"too vague: {len(candidates)} possible matches",
                })
                continue
            for entity, how in candidates:
                found.append(PotentialDuplicate(
                    "", entity.key, f"unresolved:{name}", "unresolved_reference",
                    f"'{name}' is used in the documents but no {kind} has that name; "
                    f"'{entity.name}' is a possible match ({how.replace('_', ' ')}).",
                    CONF_SIMILARITY, [f"unresolved {kind} name '{name}'"], "unreviewed"))
    return found, unmatched


def _name_similarity(index: EntityIndex, already: set[frozenset]) -> list[PotentialDuplicate]:
    found = []
    for entity_type in (ROLE, PROCESS, PRODUCT, DEPARTMENT, CUSTOMER, LOCATION):
        items = sorted(index.of_type(entity_type), key=lambda e: e.entity_id)
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if frozenset((a.key, b.key)) in already:
                    continue
                na, nb = _base_name(a.name), _base_name(b.name)
                if _only_levels_differ(na, nb):
                    continue
                if is_acronym_of(na, nb) or is_acronym_of(nb, na):
                    method, reason = "acronym", f"'{na}' looks like an abbreviation of '{nb}'."
                elif (score := _similar(na, nb)) >= SIMILARITY_THRESHOLD:
                    method, reason = "name_similarity", f"Names are {score:.0%} similar."
                else:
                    continue
                found.append(PotentialDuplicate("", a.key, b.key, method, reason, CONF_SIMILARITY,
                                                [f"{entity_type} names"], "unreviewed"))

    # Employees: the same full name could be two people, so it's only a flag.
    by_name: dict[str, list] = defaultdict(list)
    for e in index.of_type(EMPLOYEE):
        by_name[e.name.lower()].append(e)
    for same in by_name.values():
        for other in same[1:]:
            found.append(PotentialDuplicate(
                "", same[0].key, other.key, "name_similarity",
                "Same first and last name. Possibly two different people; check before merging.",
                CONF_SIMILARITY, ["employees.csv: first_name, last_name"], "unreviewed"))
    return found


# ---------------- Everything ----------------

def find_potential_duplicates(index: EntityIndex, data: Phase2Data) -> dict:
    documented = _documented(index) + _version_pairs(index) + _shared_header_ids(index)
    already = {frozenset((d.entity_a, d.entity_b)) for d in documented}
    unresolved, unmatched = _unresolved(index, data)
    similar = _name_similarity(index, already)

    every = documented + unresolved + similar
    every.sort(key=lambda d: (_METHOD_ORDER.index(d.method), d.entity_a, d.entity_b))
    for n, dup in enumerate(every, start=1):
        dup.duplicate_id = f"DUP-{n:03d}"   # stable as long as the inputs don't change

    return {
        "potential_duplicates": [d.to_dict() for d in every],
        "unmatched_names": unmatched,
    }

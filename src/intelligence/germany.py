"""
Germany new-joiner care: a friendly settling-in checklist, made personal.

This is NOT a legal advisory system. It is Nexora HR's checklist, organised so
KORA can behave like a knowledgeable colleague:
    "I'm new to Germany -- what should I take care of this month?"
    "What have I potentially forgotten?"     "What should I do next?"

Three kinds of statement are kept apart on every item, never blended:
    nexora_guidance           what Nexora's checklist itself advises
    external_requirements     sentences mentioning a legal/government requirement --
                              shown only with the documents' own caveat
    official_guidance_note    "check the current official guidance for your location
                              and circumstances", kept verbatim wherever the source says it
KORA never adds a deadline, rule or legal fact the documents don't contain.

Items come from the full checklist's sections. The two shorter documents
(first week, first 90 days) give the suggested PACE; each of their lines is
matched to an item by shared words in the item's heading. Lines that match
nothing become items of their own, so nothing is dropped.

Completion is never guessed: data/context/checklist_progress.json (empty by
default) is the only source. An item nobody reported is "not_reported", not "not done".
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.matchers import role_matcher
from src.intelligence.models import CHECKLIST, DOCUMENT, EMPLOYEE, LOCATION, ROLE
from src.intelligence.normalize import slug
from src.intelligence.text import blocks, plain, sentences

# Pace buckets used by the 90-day checklist ("By end of week 2:" ...), as day numbers.
STAGES = [("pre_start", -10**6, 0), ("week_1", 1, 7), ("week_2", 8, 14), ("month_1", 15, 30),
          ("month_2", 31, 60), ("month_3", 61, 90), ("settled", 91, 10**6)]
_PACE = [(re.compile(r"\bweek\s*one\b|\bweek\s*1\b|first[- ]week", re.I), 7),
         (re.compile(r"end of week 2\b", re.I), 14),
         (re.compile(r"end of month 1\b", re.I), 30),
         (re.compile(r"end of month 2\b", re.I), 60),
         (re.compile(r"end of month 3\b|90 days", re.I), 90)]
PROGRAM_HORIZON_DAYS = 90      # "First-90-Days Practical Checklist"

_OFFICIAL = re.compile(r"check (?:the )?current official guidance[^.]*", re.I)
_REQUIREMENT = re.compile(r"\b(?:required|mandatory|legal requirement|must)\b", re.I)
_DEPENDS_ON = re.compile(r"\b(?:after|once you have|as soon as you have|following)\s+(?:a\s+|your\s+|the\s+)?"
                         r"(?:German\s+)?([a-z][a-z ]{2,40}?)(?=[,.;)]|$| —| -)", re.I)
_REQUIRED_BY = re.compile(r"\(([^)]+)\)\s+require", re.I)
_ACTION = re.compile(r"\b(?:open|register|get|set up|arrange|find|confirm|order|begin|locate|say hello)\w*\b", re.I)
_LOCATION_QUALIFIER = re.compile(r"\(([A-Z][a-z]+)\)")
_HR = re.compile(r"\bHR\b")
_NEGATIVE = re.compile(r"\b(?:not able|cannot|can't|unable)\b", re.I)   # 'HR is not able to provide tax advice"
_RESOURCE = re.compile(r"\b[\w-]+\.md\b|\bDOC-[A-Z]+-\d+\b")
_EXTRA_STOP = {"common", "useful", "everyday", "finding", "rules", "setup", "people", "things", "few"}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z]{2,}", text.lower())
    return {w for w in words if w not in ENGLISH_STOP_WORDS and w not in _EXTRA_STOP}


@dataclass
class ChecklistItem:
    item_id: str
    title: str
    kind: str                         # action | milestone (a state to reach) | reference (information)
    source_document: str
    section: str
    chunk_ids: list[str]
    nexora_guidance: list[str] = field(default_factory=list)
    external_requirements: list[str] = field(default_factory=list)
    official_guidance_note: str | None = None
    nexora_contacts: list[dict] = field(default_factory=list)
    nexora_resources: list[str] = field(default_factory=list)  # Nexora documents it points to
    pace: list[dict] = field(default_factory=list)            # {due_by_day, statement, document_id}
    due_by_day: int | None = None
    depends_on: list[dict] = field(default_factory=list)      # {item_id, quote}
    commonly_forgotten: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def stage_for(day: int) -> str:
    return next(name for name, lo, hi in STAGES if lo <= day <= hi)


def day_number(join_date: str | None, as_of: date) -> int | None:
    try:
        joined = date.fromisoformat(join_date or "")
    except ValueError:
        return None
    return (as_of - joined).days + 1      # the join date itself is day 1


class _Builder:
    def __init__(self, data: Phase2Data, index: EntityIndex):
        self.data, self.index = data, index
        self.roles = role_matcher(index)
        self.docs = {d.entity_id: d for d in index.of_type(DOCUMENT)}
        self.checklists = sorted(index.of_type(CHECKLIST), key=lambda c: c.entity_id)
        self.items: dict[str, ChecklistItem] = {}
        self.notes: list[str] = []

    def _sections(self, doc_id: str):
        a = self.docs[doc_id].attributes
        return self.data.sections(a["header_document_id"], a["version"])

    def _full_checklist(self):
        """The checklist with the most sections is the full one; the others give pace."""
        return max(self.checklists, key=lambda c: len(self._sections(c.entity_id)), default=None)

    def _contacts(self, text: str) -> list[dict]:
        """Nexora people/teams named as a source of help. Sentences saying what Nexora
        does NOT do ('HR is not able to provide personal tax advice') are skipped."""
        found = []
        helpful = " ".join(s for s in sentences(text) if not _NEGATIVE.search(s))
        for m in self.roles.find(helpful):
            after = helpful[m.end:m.end + 20]
            where = _LOCATION_QUALIFIER.match(after.strip())
            role = self.index.get(ROLE, m.key.split(":", 1)[1])
            found.append({"role_id": role.entity_id, "name": role.name,
                          "location_qualifier": where.group(1) if where else None})
        if _HR.search(self.roles.mask(helpful)):     # bare 'HR', not part of a role title
            found.append({"role_id": None, "name": "HR (department)", "location_qualifier": None})
        unique = {(c["role_id"], c["name"], c["location_qualifier"]): c for c in found}
        return list(unique.values())

    def build_items(self) -> None:
        full = self._full_checklist()
        if full is None:
            self.notes.append("No checklist documents found")
            return
        forgotten_section = None
        for section in self._sections(full.entity_id):
            heading = section.name.split(" > ")[-1]
            if section.name.count(">") == 0:
                continue     # the document's own intro, not an item
            if re.search(r"forget", heading, re.I):
                forgotten_section = section
                continue
            text = plain(section.text)
            item = ChecklistItem(
                item_id=f"GER-{slug(re.sub(r'[(/].*', '', heading))}",
                title=heading, kind="action" if _ACTION.search(text) else "reference",
                source_document=full.entity_id, section=section.name, chunk_ids=section.chunk_ids)
            # Whole bullet text, bold part included: "**Check the current official guidance...** — ..."
            # is a sentence to keep, not a title to drop.
            statements = [x for b in blocks(section.text) if b.kind in ("bullet", "numbered", "paragraph")
                          for x in sentences(b.text)]
            for sentence in statements:
                official = _OFFICIAL.search(sentence)
                if official:
                    item.official_guidance_note = plain(official.group(0)).rstrip(".") + "."
                    rest = _OFFICIAL.sub("", sentence).strip(" —-*")
                    if rest:
                        item.nexora_guidance.append(rest)
                elif _REQUIREMENT.search(sentence):
                    item.external_requirements.append(sentence)
                else:
                    item.nexora_guidance.append(sentence)
            item.nexora_contacts = self._contacts(section.text)
            item.nexora_resources = sorted(set(_RESOURCE.findall(text)))
            self.items[item.item_id] = item
        self._dependencies()
        if forgotten_section:
            self._forgotten(forgotten_section)

    def _match(self, text: str) -> ChecklistItem | None:
        """The item sharing the most heading words with the text. Ties go to the item
        whose word appears first: 'Tax ID received ... after address registration'
        is about Tax, not Address."""
        words = _tokens(text)
        lowered = text.lower()

        def first_position(item):
            hits = [lowered.find(w) for w in words & _tokens(item.title)]
            return min(h for h in hits if h >= 0) if hits else 10**6

        scored = [(len(words & _tokens(i.title)), -first_position(i), i) for i in self.items.values()]
        best = max(scored, key=lambda s: (s[0], s[1]), default=(0, 0, None))
        return best[2] if best[0] > 0 else None

    def _dependencies(self) -> None:
        for item in list(self.items.values()):
            text = " ".join(item.nexora_guidance + item.external_requirements)
            for m in _DEPENDS_ON.finditer(text):
                target = self._match(m.group(1))
                if target and target.item_id != item.item_id:
                    item.depends_on.append({"item_id": target.item_id, "quote": m.group(0).strip()})
            for m in _REQUIRED_BY.finditer(text):
                for part in re.split(r",| and ", m.group(1)):
                    dependent = self._match(part)
                    if dependent and dependent.item_id != item.item_id:
                        dependent.depends_on.append({"item_id": item.item_id, "quote": m.group(0).strip()})
        for item in self.items.values():
            item.depends_on = list({d["item_id"]: d for d in item.depends_on}.values())

    def _forgotten(self, section) -> None:
        for block in blocks(section.text):
            if block.kind != "numbered":
                continue
            item = self._match(block.text)
            if item:
                item.commonly_forgotten = {"rank": block.number, "quote": plain(block.text),
                                           "note": plain(section.name.split(" > ")[-1])}

    def apply_pace(self) -> None:
        full = self._full_checklist()
        for checklist in self.checklists:
            if checklist is full:
                continue
            title_pace = next((d for p, d in _PACE if p.search(checklist.name)), None)
            current_pace = title_pace
            for section in self._sections(checklist.entity_id):
                for block in blocks(section.text):
                    if block.kind == "paragraph":     # '**By end of week 2:**' sets the pace below it
                        found = next((d for p, d in _PACE if p.search(block.text)), None)
                        if found:
                            current_pace = found
                        continue
                    if block.kind not in ("numbered", "bullet") or current_pace is None:
                        continue
                    statement = plain(block.text)
                    item = self._match(statement)
                    if item is None:
                        item = ChecklistItem(
                            item_id=f"GER-{slug(' '.join(statement.split()[:5]))}", title=statement.split(" — ")[0].rstrip("."),
                            kind="action" if _ACTION.search(statement) else "milestone",
                            source_document=checklist.entity_id, section=section.name, chunk_ids=section.chunk_ids,
                            nexora_guidance=[statement], nexora_contacts=self._contacts(statement),
                            nexora_resources=sorted(set(_RESOURCE.findall(statement))))
                        self.items[item.item_id] = item
                    official = _OFFICIAL.search(statement)
                    if official and not item.official_guidance_note:
                        item.official_guidance_note = plain(official.group(0)).rstrip(".") + "."
                    item.pace.append({"due_by_day": current_pace, "statement": statement,
                                      "document_id": checklist.entity_id})
        for item in self.items.values():
            if item.pace:
                item.due_by_day = min(p["due_by_day"] for p in item.pace)
                if item.kind == "reference":
                    item.kind = "action"     # something with a suggested pace is something to do


def _order(items: dict[str, ChecklistItem]) -> list[str]:
    """Dependency order first (a prerequisite before what needs it), then suggested pace."""
    remaining = dict(items)
    ordered: list[str] = []
    while remaining:
        ready = [i for i in remaining.values() if all(d["item_id"] in ordered or d["item_id"] not in remaining
                                                      for d in i.depends_on)]
        if not ready:                      # a cycle: fall back to pace order, and say so
            ready = list(remaining.values())
        ready.sort(key=lambda i: (i.due_by_day or 10**6, i.item_id))
        ordered.append(ready[0].item_id)
        del remaining[ready[0].item_id]
    return ordered


def load_progress(path: Path) -> tuple[dict, str]:
    """data/context/checklist_progress.json -- the ONLY source of completion."""
    if not path.exists():
        return {}, f"{Path(path).as_posix()} not found: nothing is marked complete"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("employees", {}), f"loaded {Path(path).as_posix()}"


@dataclass
class GermanyCare:
    program: dict
    items: dict[str, ChecklistItem]
    order: list[str]
    locations: list[dict]
    gaps: list[dict]
    progress_source: str
    progress: dict
    as_of: date

    # ---------- per employee ----------

    def eligibility(self, emp, index: EntityIndex) -> dict:
        a = emp.attributes
        loc = index.get(LOCATION, a.get("location_id") or "")
        in_germany = bool(loc) and (loc.attributes.get("country") or "").lower() == "germany"
        relocating = (a.get("international_relocation_germany") or "").lower() == "yes"
        day = day_number(a.get("join_date"), self.as_of)
        evidence = [f"employees.csv: location_id={a.get('location_id')}, "
                    f"international_relocation_germany={a.get('international_relocation_germany')}, "
                    f"join_date={a.get('join_date')}"]
        if not in_germany:
            status = "not_applicable"
        elif relocating and day is not None and 1 <= day <= PROGRAM_HORIZON_DAYS:
            status = "active"                      # relocating, inside the 90-day program window
        elif relocating and day is not None and day < 1:
            status = "pre_start"
        elif relocating:
            status = "past_program_window"         # still useful as reference, no pace
        else:
            status = "germany_based_not_relocating"
        return {"status": status, "in_germany": in_germany, "relocating_to_germany": relocating,
                "day": day, "stage": stage_for(day) if day is not None else None, "evidence": evidence,
                "audience_note": "The program is for international employees relocating to the German "
                                 "offices (DOC-CARE-001). Eligibility comes from employees.csv, never from "
                                 "a person's name or nationality."}

    def plan(self, emp_id: str, day: int | None, city: str | None) -> dict:
        reported = self.progress.get(emp_id, {})
        completed = reported.get("completed", {})
        not_applicable = set(reported.get("not_applicable", []))
        out = []
        for item_id in self.order:
            item = self.items[item_id]
            if item.kind == "reference" or item_id in not_applicable:
                continue
            status = "reported_complete" if item_id in completed else "not_reported"
            if item.due_by_day is None:
                timing = "no_suggested_pace"
            elif day is None or day < 1:
                timing = "upcoming"
            elif item.due_by_day < day:
                timing = "past_suggested_pace"
            elif item.due_by_day <= next(hi for name, lo, hi in STAGES if lo <= day <= hi):
                timing = "this_stage"
            else:
                timing = "upcoming"
            blocked_by = [d["item_id"] for d in item.depends_on
                          if d["item_id"] in self.items and d["item_id"] not in completed]
            local_contacts = [c for c in item.nexora_contacts
                              if not c["location_qualifier"] or c["location_qualifier"] == city]
            out.append({"item_id": item_id, "title": item.title, "kind": item.kind, "status": status,
                        "reported_on": completed.get(item_id), "timing": timing,
                        "due_by_day": item.due_by_day, "blocked_by": blocked_by,
                        "commonly_forgotten": bool(item.commonly_forgotten),
                        "official_guidance_note": item.official_guidance_note,
                        "contacts_here": local_contacts,
                        "contacts_elsewhere": [c for c in item.nexora_contacts if c not in local_contacts]})
        open_items = [p for p in out if p["status"] == "not_reported"]
        forgotten = [p for p in open_items if p["kind"] == "action" and
                     (p["timing"] == "past_suggested_pace" or (p["commonly_forgotten"] and p["timing"] != "upcoming"))]
        forgotten.sort(key=lambda p: (not p["commonly_forgotten"],
                                      self.items[p["item_id"]].commonly_forgotten["rank"] if p["commonly_forgotten"] else 0,
                                      p["due_by_day"] or 10**6))
        return {
            "progress_reported": bool(completed or not_applicable),
            "items": out,
            "next_steps": [p["item_id"] for p in open_items if p["kind"] == "action" and not p["blocked_by"]
                           and p["timing"] in ("past_suggested_pace", "this_stage")][:3]
                          or [p["item_id"] for p in open_items if p["kind"] == "action" and not p["blocked_by"]][:3],
            "possibly_forgotten": [p["item_id"] for p in forgotten],
            "reported_complete": [p["item_id"] for p in out if p["status"] == "reported_complete"],
            "tone": self.program.get("tone"),
            "caveat": self.program.get("not_advice"),
        }

    def to_dict(self, index: EntityIndex) -> dict:
        employees = []
        for emp in sorted(index.of_type(EMPLOYEE), key=lambda e: e.entity_id):
            elig = self.eligibility(emp, index)
            if elig["status"] in ("not_applicable", "germany_based_not_relocating"):
                continue
            entry = {"employee_id": emp.entity_id, "eligibility": elig}
            if elig["status"] in ("active", "pre_start"):
                loc = index.get(LOCATION, emp.attributes.get("location_id") or "")
                entry["plan"] = self.plan(emp.entity_id, elig["day"], loc.name if loc else None)
            employees.append(entry)
        return {
            "as_of": self.as_of.isoformat(),
            "progress_source": self.progress_source,
            "program": self.program,
            "stages": [{"stage": n, "from_day": max(lo, -1), "to_day": min(hi, 9999)} for n, lo, hi in STAGES],
            "items": [self.items[i].to_dict() for i in self.order],
            "general_steps": [{"item_id": i, "title": self.items[i].title, "due_by_day": self.items[i].due_by_day,
                               "official_guidance_note": self.items[i].official_guidance_note}
                              for i in self.order if self.items[i].kind == "action"],
            "locations": self.locations,
            "gaps": self.gaps,
            "employees": employees,
        }


def build_germany_care(data: Phase2Data, index: EntityIndex, as_of: date, progress_path: Path) -> GermanyCare:
    b = _Builder(data, index)
    b.build_items()
    b.apply_pace()
    items = b.items

    # Program framing, quoted from the overview document (the root the checklists point at).
    program: dict = {"documents": [c.entity_id for c in b.checklists]}
    overview = next((d for d in index.of_type(DOCUMENT)
                     if "care program" in (d.attributes.get("document_type") or "").lower()), None)
    if overview:
        program["overview_document"] = overview.entity_id
        for section in b._sections(overview.entity_id):
            heading = section.name.split(" > ")[-1].lower()
            text = plain(section.text)
            if heading == "philosophy":
                program["philosophy"] = text
            elif heading == "tone":
                program["tone"] = text
            elif "not" in heading:
                program["not_advice"] = text
            elif "example" in heading:
                program["example_interactions"] = [plain(x.text) for x in blocks(section.text) if x.kind == "bullet"]

    # Locations the checklist applies to: those whose country names the checklist.
    locations, gaps = [], []
    for loc in sorted(index.of_type(LOCATION), key=lambda e: e.entity_id):
        if (loc.attributes.get("country") or "").lower() != "germany":
            continue
        locations.append({"location_id": loc.entity_id, "city": loc.name, "checklists": program["documents"]})
        for item in items.values():
            named = [c for c in item.nexora_contacts if c["location_qualifier"]]
            if named and not any(c["location_qualifier"] == loc.name for c in named) and \
                    not [c for c in item.nexora_contacts if not c["location_qualifier"]]:
                gaps.append({
                    "gap_type": "checklist_contact_not_local", "severity": "low",
                    "item_id": item.item_id, "location_id": loc.entity_id,
                    "description": f"'{item.title}' names only {', '.join(c['name'] + ' (' + c['location_qualifier'] + ')' for c in named)}; "
                                   f"no {loc.name}-specific contact is documented.",
                    "source_documents": [item.source_document], "chunk_ids": item.chunk_ids})
    for item in items.values():
        if item.kind == "action" and not (item.nexora_contacts or item.nexora_resources or item.official_guidance_note):
            gaps.append({
                "gap_type": "checklist_item_without_internal_guidance", "severity": "low",
                "item_id": item.item_id, "location_id": None,
                "description": f"'{item.title}' is on the checklist, but it names no Nexora contact, no Nexora "
                               f"document and no official source to check.",
                "source_documents": [item.source_document], "chunk_ids": item.chunk_ids})

    progress, source = load_progress(progress_path)
    unknown = sorted(i for emp in progress.values() for i in emp.get("completed", {}) if i not in items)
    if unknown:
        source += f"; unknown item IDs ignored: {', '.join(unknown)}"
    return GermanyCare(program, items, _order(items), locations, gaps, source, progress, as_of)


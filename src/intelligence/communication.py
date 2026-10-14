"""
Culture and communication-tone intelligence: HOW something is said, kept apart from WHAT is said.

Nexora's own framework (DOC-COMM-001) defines four layers, and KORA keeps them separate:
    1 FACT                        never changes with the audience
    2 COMPANY PREFERENCE          Nexora policy (e.g. Internal Communication Guidelines)
    3 CULTURAL GUIDANCE           common workplace TENDENCIES in a context, a starting hypothesis
    4 INDIVIDUAL PREFERENCE       how a person actually prefers to communicate: always wins

So this module:
  * parses each tone profile into dimensions (directness, urgency, feedback, ...),
    each statement quoted with its source;
  * links profiles to office locations through Phase 2's location -> tone-guidance links;
  * checks every profile statement uses hedged language ("often", "may", "tends to"),
    the rule DOC-COMM-001 sets, so a profile can't turn into a stereotype unnoticed;
  * reads individual preferences ONLY from data/context/communication_preferences.json
    (empty by default) -- nothing about a person is ever inferred from location or name;
  * offers a presentation contract for a future UI: (fact, culture, tone) in, fact unchanged out.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.metadata.relationships import RelationType

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.models import DOCUMENT, EMPLOYEE, LOCATION
from src.intelligence.normalize import slug
from src.intelligence.text import blocks, durations, plain, sentences

# Canonical dimensions. A profile bullet is matched by its LABEL first ('**Urgency:**');
# unlabelled statements (e.g. the international baseline) by keywords.
DIMENSIONS = [
    ("directness", r"^directness", r"\b(?:direct|indirect|explicit|soften|hedg)"),
    ("formality", r"^formality", r"\b(?:formal|first names|titles)\b"),
    ("disagreement", r"^disagreement", r"\bdisagree"),
    ("feedback_style", r"^feedback", r"\bfeedback\b|\bcriticism\b"),
    ("urgency", r"^urgency", r"\burgen|\bdeadline|time-sensitiv"),
    ("meeting_style", r"^meetings?", r"\bmeetings?\b|\bagenda\b|\bpunctual"),
    ("message_conventions", r"^email|message", r"\bemail\b|\bchat\b|\bpreamble"),
    ("request_phrasing", None, r"\brequests?\b|\bthe ask\b|\bidioms?\b"),
    ("clarity", None, r"\bclear\b|\bclarity\b|\bexplicitly\b"),
    ("professional_boundaries", None, r"working hours|outside (?:your )?hours|personal time|work hours"),
]
_HEDGE = re.compile(r"\b(?:common|commonly|often|may|might|tend|tends|sometimes|generally|frequently|"
                    r"usually|can|some|relatively|fairly)\b", re.I)
_SOFT = re.compile(r"\bsoften|\bhedg|\brelational\b|\bpositive framing\b|\bpolite\b", re.I)
_DATE_OR_DAY = re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|\d{1,2}\s+[A-Z][a-z]+|"
                          r"\d{4}-\d{2}-\d{2})\b")
_HELPFUL = re.compile(r"\bhelp\b|\bappreciate\b|\bquestions?\b|\bunclear\b", re.I)
_WARMTH = re.compile(r"\bhelp\b|\bappreciate|\bnice work\b|\bgreat\b|\bthanks?\b", re.I)
_ABOUT_COMMUNICATION = re.compile(r"workplace|communication|tone|culture|etiquette", re.I)
_INVARIANCE = re.compile(r"rule that doesn't change", re.I)

TONES = {   # tone id -> how the framework names it; 'friendly' is the UI alias the spec uses
    "neutral": "Neutral",
    "direct": "More direct style",
    "relationship_oriented": "More relationship-oriented style",
}
TONE_ALIASES = {"friendly": "relationship_oriented", "warm": "relationship_oriented"}


@dataclass
class Statement:
    dimension: str
    text: str
    label: str | None
    hedged: bool
    source_document: str
    section: str
    chunk_ids: list[str]


@dataclass
class Profile:
    profile_id: str
    context: str
    document_id: str
    is_default: bool
    location_ids: list[str]
    statements: list[Statement] = field(default_factory=list)
    pitfalls: list[dict] = field(default_factory=list)
    appropriate_example: str | None = None
    individual_preference_reminder: str | None = None
    suggested_starting_tone: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dimensions"] = sorted({s.dimension for s in self.statements})
        return d


def _dimension(label: str | None, text: str) -> list[str]:
    if label:
        for name, label_pattern, _ in DIMENSIONS:
            if label_pattern and re.search(label_pattern, label.strip().lower()):
                return [name]
    return [name for name, _, keywords in DIMENSIONS if re.search(keywords, text, re.I)] or ["general"]


class _Comms:
    def __init__(self, data: Phase2Data, index: EntityIndex):
        self.data, self.index = data, index
        self.docs = {d.entity_id: d for d in index.of_type(DOCUMENT)}

    def _sections(self, doc_id: str):
        a = self.docs[doc_id].attributes
        return self.data.sections(a["header_document_id"], a["version"])

    def _chunks(self, doc_id: str, text: str, fallback: list[str]) -> list[str]:
        a = self.docs[doc_id].attributes
        return self.data.find_chunks(a["header_document_id"], a["version"], text) or fallback

    # ---------- profiles ----------

    def profiles(self) -> list[Profile]:
        location_links: dict[str, list[str]] = {}
        for rel in self.data.relationships:
            if rel.relation_type == RelationType.CULTURE_TONE_GUIDANCE:
                doc_id = self.index.document_for_relationship(rel) or rel.target
                location_links.setdefault(doc_id, []).append(rel.source)

        out = []
        for doc in sorted(self.docs.values(), key=lambda d: d.entity_id):
            if "cultural communication guidance" not in (doc.attributes.get("document_type") or "").lower():
                continue
            title = doc.name
            context = re.split(r"\s+[—–]\s+", title, maxsplit=1)[1] if re.search(r"\s[—–]\s", title) \
                else re.sub(r"\s*Communication Guidance$", "", title)
            sections = self._sections(doc.entity_id)
            intro = plain(sections[0].text) if sections else ""
            profile = Profile(
                profile_id=f"COMM-PROFILE-{slug(context.split('(')[0])}",
                context=context, document_id=doc.entity_id,
                is_default=bool(re.search(r"\bdefault\b", intro, re.I)),
                location_ids=sorted(set(location_links.get(doc.entity_id, []))))
            for section in sections:
                heading = section.name.split(" > ")[-1]
                for block in blocks(section.text):
                    if block.kind not in ("bullet", "numbered", "paragraph"):
                        continue
                    text = plain(block.body)
                    if not text:
                        continue
                    if re.search(r"land wrong", heading, re.I):
                        if block.title and "example" in block.title.lower():
                            profile.appropriate_example = text.strip('"“” ')
                        elif block.title:
                            profile.pitfalls.append({"label": block.title, "text": text,
                                                     "chunk_ids": self._chunks(doc.entity_id, text, section.chunk_ids)})
                        continue
                    if re.search(r"individual preference", heading, re.I):
                        profile.individual_preference_reminder = text
                        continue
                    if _INVARIANCE.search(heading):
                        continue    # a company-wide rule: kept in the company layer, not in a profile
                    if block.kind == "paragraph" and not block.title and block is not None and section is sections[0]:
                        continue    # the italic disclaimer line under the title
                    for dim in _dimension(block.title, text):
                        profile.statements.append(Statement(
                            dim, text, block.title, bool(_HEDGE.search(text)), doc.entity_id, section.name,
                            self._chunks(doc.entity_id, text, section.chunk_ids)))
            profile.suggested_starting_tone = self._starting_tone(profile)
            out.append(profile)
        self._supplementary(out)
        return out

    def _starting_tone(self, profile: Profile) -> dict:
        """Read from the profile's own 'appropriate example' message:
             warmth (help, appreciate, nice work)          -> relationship_oriented
             an explicit request with no softening words   -> direct
             the default profile, or no example            -> neutral"""
        example = profile.appropriate_example or ""
        if profile.is_default or not example:
            tone, markers = "neutral", []
        elif _WARMTH.search(example):
            tone, markers = "relationship_oriented", _WARMTH.findall(example)
        elif not _SOFT.search(example):
            tone, markers = "direct", ["no softening or warmth phrases"]
        else:
            tone, markers = "neutral", _SOFT.findall(example)
        return {"tone": tone, "basis": example or None, "markers": markers,
                "note": "A hypothesis from the profile's own example message, never a fact about a person. "
                        "An individual preference, once known, replaces it."}

    def _supplementary(self, profiles: list[Profile]) -> None:
        """Other documents can add context-specific tendencies: the Germany checklist's
        'Common German Workplace Practices' joins the Germany profile because the
        heading names the country (matched on the first five letters: German/Germany)."""
        for profile in profiles:
            stem = profile.context.split("(")[0].strip().lower()[:5]
            if len(stem) < 5:
                continue
            for doc in self.docs.values():
                if doc.entity_id == profile.document_id or \
                        "cultural communication guidance" in (doc.attributes.get("document_type") or "").lower():
                    continue
                for section in self._sections(doc.entity_id):
                    heading = section.name.split(" > ")[-1]
                    if not any(w.lower().startswith(stem) for w in heading.split()) \
                            or not _ABOUT_COMMUNICATION.search(heading):
                        continue
                    for block in blocks(section.text):
                        if block.kind != "bullet":
                            continue
                        text = plain(block.body)
                        for dim in _dimension(None, text):
                            profile.statements.append(Statement(
                                dim, text, None, bool(_HEDGE.search(text)), doc.entity_id, section.name,
                                self._chunks(doc.entity_id, text, section.chunk_ids)))

    # ---------- framework, company layer, tone options ----------

    def framework(self) -> dict:
        overview = next((d for d in self.docs.values()
                         if d.attributes.get("document_type", "").lower() == "guideline"
                         and "tone" in d.name.lower()), None)
        result = {"document_id": overview.entity_id if overview else None, "layers": [], "precedence": None,
                  "invariance_rule": None, "tone_examples": []}
        if not overview:
            return result
        for section in self._sections(overview.entity_id):
            heading = section.name.split(" > ")[-1].lower()
            for block in blocks(section.text):
                text = plain(block.body)
                if "framework" in heading and block.kind == "numbered" and block.title:
                    example = re.search(r"Example:\s*\"?([^\"]+)\"?", text)
                    result["layers"].append({
                        "rank": block.number, "layer": block.title.title(),
                        "definition": re.split(r"\s*\*?Example:", text)[0].strip(" *"),
                        "example": example.group(1).strip(" *\"") if example else None})
                elif "overrides" in text.lower() and not result["precedence"]:
                    result["precedence"] = next(s for s in sentences(text) if "overrides" in s.lower())
                elif "same instruction" in heading and block.title:
                    tone_id = next((t for t, label in TONES.items() if block.title.lower().startswith(label.lower())), None)
                    hint = re.search(r"\(([^)]+)\)", block.title)
                    result["tone_examples"].append({"tone": tone_id, "label": block.title,
                                                    "context_hint": hint.group(1) if hint else None,
                                                    "example": text.strip('"“” ')})
                elif "same instruction" in heading and "identical" in text.lower():
                    result["invariance_rule"] = text
        return result

    def company_layer(self) -> list[dict]:
        """Layer 2: Nexora's own communication policy (documents of type Guideline that
        are registered processes, e.g. PROC-HR-003), plus the 'one rule that doesn't change'."""
        out = []
        for doc in self.docs.values():
            is_policy = self.index.get("process", doc.entity_id) and \
                "guideline" in (doc.attributes.get("document_type") or "").lower()
            for section in self._sections(doc.entity_id):
                heading = section.name.split(" > ")[-1]
                if not (is_policy or re.search(r"rule that doesn't change", heading, re.I)):
                    continue
                for block in blocks(section.text):
                    if block.kind in ("bullet", "paragraph") and plain(block.body):
                        text = plain(block.body)
                        out.append({"statement": f"{block.title}: {text}" if block.title else text,
                                    "section": section.name, "source_document": doc.entity_id,
                                    "chunk_ids": self._chunks(doc.entity_id, text, section.chunk_ids)})
        return out


def load_preferences(path: Path) -> tuple[dict, str]:
    if not path.exists():
        return {}, f"{Path(path).as_posix()} not found: no individual preferences known"
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("employees", {}), f"loaded {Path(path).as_posix()}"


@dataclass
class CommunicationIntel:
    framework: dict
    company_layer: list[dict]
    profiles: list[Profile]
    preferences: dict
    preferences_source: str

    def profile_for_location(self, location_id: str | None) -> Profile | None:
        return next((p for p in self.profiles if location_id in p.location_ids), None)

    def default_profile(self) -> Profile | None:
        return next((p for p in self.profiles if p.is_default), None)

    def culture_ids(self) -> list[str]:
        return [p.profile_id for p in self.profiles]

    def for_employee(self, emp_id: str, location_id: str | None) -> dict:
        """Layer 4 if known; otherwise the location's profile, clearly marked as a hypothesis."""
        profile = self.profile_for_location(location_id)
        pref = self.preferences.get(emp_id)
        return {
            "individual_preference": pref or "unknown",
            "individual_preference_source": "data/context/communication_preferences.json" if pref else None,
            "location_profile": profile.profile_id if profile else None,
            "effective_tone": (pref or {}).get("preferred_tone") or
                              (profile.suggested_starting_tone["tone"] if profile else "neutral"),
            "basis": "individual preference (reported)" if pref and pref.get("preferred_tone") else
                     "location profile: a starting hypothesis about workplace tendencies, not about this person",
            "reminder": profile.individual_preference_reminder if profile else None,
        }

    def present(self, fact: str, culture: str | None = None, tone: str | None = None) -> dict:
        """The contract a future UI calls with (fact, culture, tone).

        The fact is returned UNCHANGED. Tone adds at most one documented closing
        phrase; everything else is guidance the writer can act on. No new facts,
        dates or deadlines are ever generated.
        """
        tone = TONE_ALIASES.get((tone or "neutral").lower(), (tone or "neutral").lower())
        if tone not in TONES:
            raise ValueError(f"Unknown tone '{tone}'. Options: {', '.join(list(TONES) + list(TONE_ALIASES))}")
        profile = next((p for p in self.profiles if p.profile_id == culture), None) or self.default_profile()
        guidance = [asdict(s) for s in (profile.statements if profile else [])
                    if s.dimension in ("urgency", "directness", "clarity")]
        checks = []
        has_deadline = bool(_DATE_OR_DAY.search(fact) or durations(fact))
        checks.append({"check": "explicit_deadline_in_fact", "result": has_deadline,
                       "why": "Every profile recommends stating the deadline explicitly rather than relying on tone."})
        if tone == "direct" and not re.search(r"\d", fact):
            checks.append({"check": "direct_style_date", "result": False,
                           "why": "The direct-style example in DOC-COMM-001 names the date ('Friday, 18 September'). "
                                  "Add the date to the fact itself if you know it; KORA will not invent one."})
        closing = None
        if tone == "relationship_oriented":
            candidates = []
            for p in ([profile] if profile else []) + [q for q in self.profiles if q is not profile]:
                if p and p.appropriate_example:
                    candidates += [(s, p.document_id) for s in sentences(p.appropriate_example)
                                   if _HELPFUL.search(s) and not durations(s) and not _DATE_OR_DAY.search(s)]
            for ex in self.framework.get("tone_examples", []):
                if ex["tone"] == "relationship_oriented":
                    candidates += [(s, self.framework["document_id"]) for s in sentences(ex["example"])
                                   if _HELPFUL.search(s) and not _DATE_OR_DAY.search(s)]
            closing = min(candidates, key=lambda c: len(c[0].split()), default=None)
        rendered = fact if not closing else f"{fact.rstrip()} {closing[0]}"
        return {
            "fact": fact, "fact_unchanged": rendered.startswith(fact),
            "culture": profile.profile_id if profile else None, "tone": tone,
            "rendered": rendered,
            "added_phrase": {"text": closing[0], "source_document": closing[1]} if closing else None,
            "checks": checks, "guidance": guidance,
            "layer_note": "Culture and tone change phrasing only. If the reader's individual preference is "
                          "known, it overrides the culture profile.",
        }

    def to_dict(self, index: EntityIndex) -> dict:
        coverage = {name: sorted({p.profile_id for p in self.profiles for s in p.statements if s.dimension == name})
                    for name, _, _ in DIMENSIONS}
        # Only profiles describing people in a PLACE must hedge; the default profile is advice to the writer.
        unhedged = [{"profile_id": p.profile_id, "dimension": s.dimension, "text": s.text,
                     "source_document": s.source_document}
                    for p in self.profiles if p.location_ids for s in p.statements if not s.hedged]
        locations = []
        for loc in sorted(index.of_type(LOCATION), key=lambda e: e.entity_id):
            p = self.profile_for_location(loc.entity_id)
            locations.append({"location_id": loc.entity_id, "city": loc.name,
                              "profile_id": p.profile_id if p else None})
        return {
            "framework": self.framework,
            "company_layer": self.company_layer,
            "profiles": [p.to_dict() for p in self.profiles],
            "locations": locations,
            "tone_options": [{"tone": t, "label": label} for t, label in TONES.items()],
            "tone_aliases": TONE_ALIASES,
            "culture_options": self.culture_ids(),
            "dimension_coverage": coverage,
            "hedging_check": {"rule": "Profiles must describe tendencies with hedged language (DOC-COMM-001).",
                              "profiles_checked": [p.profile_id for p in self.profiles if p.location_ids],
                              "unhedged_statements": unhedged},
            "individual_preferences": {"source": self.preferences_source, "count": len(self.preferences)},
            "presentation_contract": {
                "input": {"fact": "text, never modified", "culture": self.culture_ids(),
                          "tone": list(TONES) + list(TONE_ALIASES)},
                "output": ["fact", "rendered", "added_phrase", "checks", "guidance"],
                "example": self.present("Please complete the security training by Friday.",
                                        self.default_profile().profile_id if self.default_profile() else None,
                                        "friendly"),
            },
            "employees_with_individual_preference": sorted(e for e in self.preferences
                                                           if index.get(EMPLOYEE, e)),
        }


def build_communication(data: Phase2Data, index: EntityIndex, preferences_path: Path) -> CommunicationIntel:
    c = _Comms(data, index)
    prefs, source = load_preferences(preferences_path)
    return CommunicationIntel(c.framework(), c.company_layer(), c.profiles(), prefs, source)

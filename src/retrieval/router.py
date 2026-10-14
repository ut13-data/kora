"""
Phase 4 - Router: decide WHERE to look before searching.

    question -> topic / entity / conflict -> candidate documents -> chunks

The router never searches. It reads the question and returns a RoutePlan:
what it recognised, what kind of question this is, which documents to search
first, and whether archived documents are allowed. Every decision carries a
plain-English reason, so a wrong answer can be traced to "the router guessed
wrong" or "the search did".

What it knows comes from two places:
  * Phase 3 output (entities.json, topics.json, conflicts.json,
    outdated_knowledge.json) -- whatever the company's documents contain.
  * The company config (config/kora_config.json) -- company-specific
    programs (relocation, tone profiles) and every threshold. Without a
    config those features are off and the router still works.

Generic rules (any company):
  * Topics are the catalogue. A topic matches when its name, its anchor
    entity's name or an alias appears in the question as whole words. A
    single generic word from a department topic ("sales") is WEAK and only
    used when nothing more specific matched.
  * No topic name matched -> Phase 3 per-document keywords (2+ shared).
  * Conflicts Phase 3 detected are checked on EVERY question: with the word
    "conflict" (or similar) 2 shared topic words are enough; without it, 3
    shared words AND a confident conflict (fact-level, not org-chart guesses).
  * Archived documents (decision C): excluded unless the question asks about
    history, a matched conflict involves them, or the question names an
    archived document's ID. When excluded, older versions are noted.
  * "Which documents describe X?" is its own intent: the answer lists
    documents (from the topic catalogue) instead of quoting sentences.
  * Direct documents (named by ID, or a country's tone profile) are always
    searched, like a named document -- the router chose them on purpose.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.retrieval.config import KoraConfig, load_config
from src.retrieval.keyword import extract_id_tokens

# ---------------- generic intents (any company) ----------------

GENERIC_INTENTS: dict[str, list[str]] = {
    "conflict": [r"\bconflict", r"\bcontradict", r"\binconsisten", r"\bdisagree"],
    "gap": [r"\bmissing\b", r"\bgaps?\b", r"\bnot documented\b", r"\bundocumented\b",
            r"\bnot (?:specified|defined|covered)\b"],
    # History PHRASES, not single everyday words: "a deprecated product version"
    # or "change the go-live date" are not questions about document history
    # (bug found in M5: they opened archived documents and showed unrelated
    # step changes).
    "version": [r"\barchived?\b", r"\bsuperseded\b", r"\boutdated\b", r"\bhistor(?:y|ical)\b",
                r"\bpreviously\b", r"\bused to\b", r"\bchanged\b", r"\bchanges? (?:from|since|between)\b",
                r"\b(?:old|older|previous|earlier|prior|archived|first|last) (?:versions?|documents?|"
                r"polic(?:y|ies)|rules?|process(?:es)?|guidance)\b",
                r"\bversions? history\b", r"\bv\d+(?:\.\d+)?\b", r"\bin the past\b"],
    "approval": [r"\bapprov", r"\bsign(?:s|ed)? off\b", r"\bauthori[sz]"],
    "escalation": [r"\bescalat"],
    "ownership": [r"\bwho owns\b", r"\bowners?\b", r"\bresponsible\b", r"\baccountable\b"],
    "document_listing": [r"\b(?:which|what|list|show)\b(?:\s+\w+){0,3}?\s+(?:documents?|docs|sources|files)\b"],
}

# Conflict types shown WITHOUT the word "conflict" in the question: statements
# of fact (numbers, counts). Org-chart readings and document-identifier issues
# are metadata/structure findings -- only shown when asked about conflicts.
FACT_CONFLICT_TYPES = {"numeric_fact", "count_claim"}

# Intents whose evidence legitimately includes archived documents.
VERSION_INTENTS = {"version"}

# Small generic aliases for country names in locations.csv-style data.
_COUNTRY_ALIASES = {"usa": "united states", "us": "united states", "uk": "united kingdom",
                    "great britain": "united kingdom"}

# A trailing word like "Process" is dropped to make a second, shorter key:
# "Access Control Policy" also matches "access control".
_SUFFIXES = {"process", "processes", "policy", "policies", "procedure", "procedures",
             "guidelines", "services"}

# Words that say nothing about WHICH conflict is meant: all English stop
# words (bug: "new", "their", "first" -- 3 filler words -- matched the
# security-training conflict for a Germany first-week question) plus these.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _EN_STOP  # noqa: E402

_CONFLICT_STOPWORDS = set(_EN_STOP) | {
    "a", "an", "the", "of", "for", "about", "are", "is", "there", "any", "what", "which",
    "who", "how", "do", "does", "in", "on", "to", "and", "or", "with", "between", "still",
    "instruction", "conflict", "conflicting", "contradict", "contradictory", "document",
    "inconsistent", "inconsistency", "different", "rule", "say", "says", "disagree",
    "nexora", "have", "has", "many", "much", "this", "that", "it", "be", "can", "should",
}


def normalise(text: str) -> str:
    """Lowercase, '&' -> 'and', every other non-letter/digit -> space."""
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _phrase_regex(key: str) -> re.Pattern:
    """Whole-word match of a normalised phrase, allowing a plural 's'."""
    return re.compile(r"\b" + re.escape(key) + r"s?\b")


def _is_acronym(alias: str) -> bool:
    return alias.isupper() and len(alias) <= 5 and " " not in alias


def _content_words(text: str) -> set[str]:
    """Normalised words minus stopwords, then a plural 's' removed. The
    stopword check comes FIRST (bug: "does" became "doe" and counted as a
    content word). Document IDs are removed: they already route the question
    to their document and must not match Phase 3's identifier conflicts."""
    for token in extract_id_tokens(text):
        text = text.replace(token, " ")
    out = set()
    for w in normalise(text).split():
        if w in _CONFLICT_STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w not in _CONFLICT_STOPWORDS:
            out.add(w)
    return out


def _successors(outdated: dict) -> dict[str, list[str]]:
    """archived document_id -> active document_ids that replaced it."""
    return {old["document_id"]: [new["document_id"] for new in old.get("superseded_by", [])]
            for old in outdated.get("superseded_documents", [])}


# ---------------- plan ----------------

@dataclass
class MatchedEntity:
    key: str             # Phase 3 entity key, e.g. "topic:TOPIC-CUSTOMER-ONBOARDING"
    entity_type: str     # topic, role, location, document
    entity_id: str
    name: str
    matched_text: str    # what in the question matched
    strength: str        # "strong" or "weak" (topics); "info" for roles/locations


@dataclass
class RoutePlan:
    question: str
    intents: list[str]
    entities: list[MatchedEntity]
    candidate_documents: list[str]
    include_archived: bool
    reasons: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)          # Phase 3 conflict IDs matched
    pinned_chunks: list[str] = field(default_factory=list)      # both sides of matched conflicts
    direct_documents: list[str] = field(default_factory=list)   # always searched (named IDs, profiles)
    tone_profiles: list[str] = field(default_factory=list)      # country tone profiles chosen
    countries: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)             # topic IDs used for candidates

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _TopicKey:
    regex: re.Pattern
    text: str
    topic: dict
    strength: str


# ---------------- router ----------------

class Router:
    def __init__(self, entities: dict, topics: dict, conflicts: dict | None = None,
                 outdated: dict | None = None, config: KoraConfig | None = None):
        self.config = config or KoraConfig()
        self._entities = {e["key"]: e for e in entities.get("entities", [])}
        self._topics = {t["topic_id"]: t for t in topics.get("topics", [])}
        self._documents = {e["entity_id"]: e for e in entities.get("entities", [])
                           if e["entity_type"] == "document"}
        self._topic_keys = self._build_topic_keys()
        self._role_keys = self._build_keys_for("role")
        self._location_keys = self._build_keys_for("location")
        self._doc_keywords = self._build_document_keywords(topics.get("document_keywords", {}))
        self._conflicts = (conflicts or {}).get("conflicts", [])
        self._successors = _successors(outdated or {})
        self.config.check_references(set(self._documents), set(self._topics))
        self._intents = {name: [re.compile(p) for p in pats] for name, pats in GENERIC_INTENTS.items()}
        for name, pats in self.config.extra_intent_patterns.items():
            self._intents.setdefault(name, []).extend(re.compile(p) for p in pats)

    # ---- construction ----

    @classmethod
    def from_outputs(cls, outputs: dict, config: KoraConfig | None = None) -> "Router":
        """From IntelligenceResult.outputs (Phase 3 run in memory)."""
        return cls(outputs["entities"], outputs["topics"], outputs.get("conflicts"),
                   outputs.get("outdated_knowledge"), config)

    @classmethod
    def from_files(cls, intelligence_dir: Path = Path("data/intelligence"),
                   config: KoraConfig | None = None) -> "Router":
        """From the saved Phase 3 files -- no need to re-run Phase 3."""
        def read(name):
            path = intelligence_dir / f"{name}.json"
            if not path.exists():
                return {}
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return cls(read("entities"), read("topics"), read("conflicts"), read("outdated_knowledge"),
                   config if config is not None else load_config())

    def _build_topic_keys(self) -> list[_TopicKey]:
        keys: list[_TopicKey] = []
        for topic in self._topics.values():
            names = [topic["name"]]
            anchor = self._entities.get(topic.get("anchor_key", ""))
            if anchor:
                names.append(anchor["name"])
                names.extend(anchor.get("aliases", []))
            texts: set[str] = set()
            for name in names:
                base = normalise(name)
                if not base:
                    continue
                texts.add(base)
                words = base.split()
                if len(words) > 1 and words[-1] in _SUFFIXES:
                    texts.add(" ".join(words[:-1]))
                if "/" in name:                       # "IT / Security" -> "it", "security"
                    texts.update(normalise(part) for part in name.split("/") if normalise(part))
            for text in texts:
                weak = topic.get("level") == "domain" and len(text.split()) == 1
                keys.append(_TopicKey(_phrase_regex(text), text, topic, "weak" if weak else "strong"))
        return keys

    def _build_keys_for(self, entity_type: str) -> list[tuple]:
        """(regex, matched_text, entity, case_sensitive) for roles / locations."""
        out = []
        for e in self._entities.values():
            if e["entity_type"] != entity_type:
                continue
            for name in [e["name"], *e.get("aliases", [])]:
                if _is_acronym(name):                 # "AE", "CSM": exact case only
                    out.append((re.compile(r"\b" + re.escape(name) + r"s?\b"), name, e, True))
                elif normalise(name):
                    out.append((_phrase_regex(normalise(name)), normalise(name), e, False))
        return out

    def _build_document_keywords(self, document_keywords: dict) -> dict[str, list[tuple]]:
        """document_id -> [(regex, keyword)] from Phase 3's per-document
        keywords. Keywords too short or made only of digits are skipped."""
        out: dict[str, list[tuple]] = {}
        for doc_id, keywords in document_keywords.items():
            keys = []
            for kw in dict.fromkeys(normalise(k) for k in keywords):
                if len(kw) >= 3 and not kw.replace(" ", "").isdigit():
                    keys.append((_phrase_regex(kw), kw))
            out[doc_id] = keys
        return out

    @property
    def tone_profile_documents(self) -> set[str]:
        comm = self.config.communication
        return set(comm.profiles.values()) if comm else set()

    def is_archived_document(self, document_id: str) -> bool:
        doc = self._documents.get(document_id)
        return bool(doc) and str(doc["attributes"].get("status", "")).lower() == "archived"

    # ---- routing ----

    def route(self, question: str) -> RoutePlan:
        t = self.config.t
        norm = normalise(question)
        reasons: list[str] = []

        # 1. Generic intents
        intents = [name for name, regexes in self._intents.items() if any(r.search(norm) for r in regexes)]
        for intent in intents:
            reasons.append(f"intent {intent}: keyword in question")

        # 2. Entities: topics, roles, locations, named documents
        matched = self._match_topics(norm)
        for keys, etype in ((self._role_keys, "role"), (self._location_keys, "location")):
            found: dict[str, MatchedEntity] = {}
            for regex, text, entity, case_sensitive in keys:
                if regex.search(question if case_sensitive else norm) and entity["entity_id"] not in found:
                    found[entity["entity_id"]] = MatchedEntity(
                        entity["key"], etype, entity["entity_id"], entity["name"], text, "info")
            matched.extend(found.values())
        named_documents = []
        for token in sorted(extract_id_tokens(question)):
            doc = self._documents.get(token)
            if doc:
                named_documents.append(doc)
                matched.append(MatchedEntity(doc["key"], "document", doc["entity_id"], doc["name"], token, "strong"))

        countries = self._countries(question, norm, matched)

        # 3. Company programs from config (relocation, communication)
        reloc = self.config.relocation
        if reloc:
            hit = any(re.search(p, norm) for p in reloc.intent_patterns)
            by_country = [c for c in countries if c in reloc.countries]
            if hit or by_country:
                intents.append("relocation")
                reasons.append(f"intent relocation ({reloc.name}): " +
                               ("keyword in question" if hit else f"country {by_country[0]} named"))
        comm = self.config.communication
        if comm:
            strong = any(re.search(p, norm) for p in comm.intent_patterns)
            weak = any(re.search(p, norm) for p in comm.weak_intent_patterns)
            if strong or (weak and countries):
                intents.append("communication")
                reasons.append("intent communication: " + ("tone/phrasing words" if strong else
                                                          f"communication words + country {countries[0]}"))

        # 4. Candidate documents
        candidates: list[str] = []
        direct: list[str] = []
        topics_used: list[str] = []

        def add_docs(doc_ids, why):
            new = [d for d in dict.fromkeys(doc_ids) if d and d not in candidates]
            if new:
                candidates.extend(new)
                reasons.append(f"candidates +{len(new)} from {why}")

        conflict_ids, pinned = self._match_conflicts(question, "conflict" in intents, reasons)
        add_docs([s["document_id"] for c in self._conflicts if c["conflict_id"] in conflict_ids
                  for s in c["sides"] if s.get("document_id")], f"conflict(s) {', '.join(conflict_ids)}")

        tone_profiles: list[str] = []
        if "communication" in intents and comm:
            tone_profiles = [comm.profiles[c] for c in countries if c in comm.profiles]
            if tone_profiles:
                docs = tone_profiles + ([comm.overview_document] if comm.overview_document else [])
                add_docs(docs, f"communication question about {', '.join(countries)}: tone profile + overview")
                direct.extend(tone_profiles)
                if "relocation" in intents:
                    intents.remove("relocation")
                    reasons.append("relocation documents not added: this is a communication question")
            elif comm.topic_id:
                topic = self._topics[comm.topic_id]
                add_docs(topic.get("home_documents", []), f"intent communication -> {comm.topic_id}")
                topics_used.append(comm.topic_id)

        topic_matches = [m for m in matched if m.entity_type == "topic"]
        strong = [m for m in topic_matches if m.strength == "strong"]
        weak = [m for m in topic_matches if m.strength == "weak"]
        keyword_docs = [] if strong else self._match_keywords(norm, reasons)
        for m in strong:
            add_docs(self._topics[m.entity_id].get("home_documents", []),
                     f"topic {m.entity_id} (matched '{m.matched_text}', strong)")
            topics_used.append(m.entity_id)
        for doc_id, hits in keyword_docs:
            add_docs([doc_id], f"keywords {hits} of {doc_id}")
            for successor in self._successors.get(doc_id, []):
                add_docs([successor], f"{successor} replaced archived {doc_id}")
        specific = bool(strong or keyword_docs or conflict_ids or tone_profiles)
        if weak and not specific:
            reasons.append("no specific topic, keywords or conflict matched; using broad department topic(s)")
            for m in weak:
                add_docs(self._topics[m.entity_id].get("home_documents", []),
                         f"topic {m.entity_id} (matched '{m.matched_text}', weak)")
                topics_used.append(m.entity_id)
        elif weak:
            reasons.append(f"broad department word(s) {[m.matched_text for m in weak]} ignored: "
                           f"something more specific matched")

        # A version/history question that names nothing specific: search the
        # documents that ARE archived, and what replaced them (bug: "the
        # archived onboarding document" was routed to employee onboarding).
        archive_talk = re.search(r"\b(?:archived?|superseded|outdated|old(?:er)? (?:version|document|polic|rule))", norm)
        if "version" in intents and archive_talk and not (strong or conflict_ids or named_documents):
            for old_id, new_ids in self._successors.items():
                header = (self._documents.get(old_id, {}).get("attributes", {}).get("header_document_id")
                          or old_id)
                add_docs([old_id, header, *new_ids], f"version question: archived {old_id} and its replacement")

        if "relocation" in intents and reloc and reloc.topic_id:
            add_docs(self._topics[reloc.topic_id].get("home_documents", []),
                     f"intent relocation -> {reloc.topic_id}")
            topics_used.append(reloc.topic_id)

        for doc in named_documents:
            # An archived file can carry another document's ID in its header
            # (PROC-IMPL-001-V1's chunks are labelled PROC-IMPL-001).
            header_id = doc["attributes"].get("header_document_id") or doc["entity_id"]
            add_docs([doc["entity_id"], header_id], f"document ID {doc['entity_id']} named in question")
            direct.extend(d for d in (doc["entity_id"], header_id) if d not in direct)
        if not candidates:
            reasons.append("nothing recognised: search the whole knowledge base")

        # 5. Archived documents (decision C)
        archived_named = [d["entity_id"] for d in named_documents if self.is_archived_document(d["entity_id"])]
        archived_in_conflict = any(self.is_archived_document(s.get("document_id", ""))
                                   for c in self._conflicts if c["conflict_id"] in conflict_ids for s in c["sides"])
        if "version" in intents:
            include_archived = True
            reasons.append("archived documents included: question is about versions/history")
        elif archived_in_conflict:
            include_archived = True
            reasons.append("archived documents included: a matched conflict involves an archived document")
        elif archived_named:
            include_archived = True
            reasons.append(f"archived documents included: question names {', '.join(archived_named)}")
        else:
            include_archived = False
            reasons.append("archived documents excluded (current knowledge only); older versions will be noted")

        return RoutePlan(question, intents, matched, candidates, include_archived, reasons,
                         conflict_ids, pinned, direct, tone_profiles, countries, topics_used)

    # ---- pieces ----

    def _match_topics(self, norm: str) -> list[MatchedEntity]:
        seen: dict[str, MatchedEntity] = {}
        for key in self._topic_keys:
            if not key.regex.search(norm):
                continue
            tid = key.topic["topic_id"]
            current = seen.get(tid)
            # One match per topic: strong beats weak, longer text beats shorter.
            if current and (current.strength == "strong" or key.strength == "weak") \
                    and len(current.matched_text) >= len(key.text):
                continue
            seen[tid] = MatchedEntity(f"topic:{tid}", "topic", tid, key.topic["name"], key.text, key.strength)
        return list(seen.values())

    def _countries(self, question: str, norm: str, matched) -> list[str]:
        """Countries named in the question: config country words (upper-case
        ones like 'UK'/'US' only as written), or a recognised office
        location's country from Phase 3 data."""
        found: list[str] = []
        comm = self.config.communication
        for word, country in (comm.country_words.items() if comm else []):
            hay, pattern = (question, word) if word.isupper() else (norm, normalise(word))
            if re.search(r"\b" + re.escape(pattern) + r"\b", hay) and country not in found:
                found.append(country)
        for m in matched:
            if m.entity_type == "location":
                country = normalise(str(self._entities[m.key]["attributes"].get("country", "")))
                country = _COUNTRY_ALIASES.get(country, country)
                if country and country not in found:
                    found.append(country)
        return found

    def _match_conflicts(self, question: str, asked: bool, reasons: list[str]) -> tuple[list[str], list[str]]:
        """Phase 3 conflicts whose topic shares the most content words with the
        question. Asked about conflicts: CONFLICT_OVERLAP_WITH_INTENT words.
        Not asked: more shared words AND a confident conflict, so a normal
        question only shows a conflict when it is clearly about that fact."""
        t = self.config.t
        words = _content_words(question)
        need = t("conflict_overlap_with_intent") if asked else t("conflict_overlap_without_intent")
        scored = []
        for c in self._conflicts:
            if not asked and (float(c.get("confidence") or 0) < t("conflict_confidence_without_intent")
                              or c.get("conflict_type") not in FACT_CONFLICT_TYPES):
                continue
            # Match on the topic label AND the sides' own statements. The label
            # is Phase 3's top TF-IDF terms, which can differ between library
            # versions (seen: "awareness, new" vs "complete, mandatory"); the
            # statements are source text, identical everywhere.
            vocabulary = _content_words(" ".join([c.get("topic", "")] +
                                                 [side.get("statement", "") for side in c.get("sides", [])]))
            overlap = words & vocabulary
            # Short questions ("How many office locations?") can't reach 3
            # shared words: all of their 2+ content words matching is as strong.
            whole_question = not asked and len(words) >= 2 and overlap == words
            if len(overlap) >= need or whole_question:
                scored.append((len(overlap), c, sorted(overlap)))
        if not scored:
            if asked:
                reasons.append("conflict question, but no detected conflict matches its words")
            return [], []
        best = max(n for n, _, _ in scored)
        ids, pinned = [], []
        for n, c, overlap in scored:
            if n != best:
                continue
            ids.append(c["conflict_id"])
            for side in c["sides"]:
                pinned.extend(cid for cid in side.get("chunk_ids", []) if cid not in pinned)
            reasons.append(f"conflict {c['conflict_id']} matched on {overlap} "
                           f"(Phase 3 topic: '{c.get('topic')}', status {c.get('status')}"
                           f"{'' if asked else ', found without the word conflict'})")
        return ids, pinned

    def _match_keywords(self, norm: str, reasons: list[str]) -> list[tuple[str, list[str]]]:
        """Documents whose Phase 3 keywords appear in the question at least
        keyword_min_overlap times, most overlap first."""
        t = self.config.t
        hits = []
        for doc_id, keys in self._doc_keywords.items():
            found = [kw for regex, kw in keys if regex.search(norm)]
            if len(found) >= t("keyword_min_overlap"):
                hits.append((doc_id, found))
        hits.sort(key=lambda pair: (-len(pair[1]), pair[0]))
        limit = t("keyword_max_documents")
        if len(hits) > limit:
            reasons.append(f"{len(hits)} documents matched on keywords; kept the top {limit}")
        return hits[:limit]

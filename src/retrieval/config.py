"""
Phase 4 - company configuration (one JSON file, editable later from the UI).

KORA's code is company-independent. Everything that only makes sense for one
company -- a relocation program, country tone profiles, topic shortcuts --
and every tuning threshold lives in config/kora_config.json.

Rules (decided with UT):
  * No config, or a feature missing from it -> that feature is OFF. KORA
    still searches, cites, flags conflicts/gaps and abstains.
  * The file is VALIDATED when loaded: wrong types or out-of-range numbers
    raise ConfigError with every problem listed (a broken config must never
    half-work).
  * References to documents/topics that don't exist in the current index
    are not errors (the knowledge base changes): that part of the feature is
    switched off and a warning says why (see KoraConfig.check_references).
"""

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path("config/kora_config.json")


class ConfigError(ValueError):
    """The config file is malformed. The message lists every problem."""


# Every tunable number, its default and its allowed range. The UI can only
# set values inside these ranges.
THRESHOLDS = {
    "top_k":                        (5, 1, 20),
    "global_reserved_slots":        (2, 0, 10),     # whole-corpus slots the router can never take
    "routed_min_score":             (0.05, 0.0, 1.0),
    "max_lines":                    (3, 1, 10),     # answer length (structured facts are never cut)
    "max_lines_per_chunk":          (2, 1, 10),
    "sentence_min_score":           (0.10, 0.0, 1.0),
    "strong_evidence_score":        (0.25, 0.0, 1.0),
    "rule_min_score":               (0.15, 0.0, 1.0),
    "max_rules":                    (2, 0, 10),
    "min_quote_words":              (5, 1, 50),
    "unrouted_quote_ratio":         (0.5, 0.0, 1.0),
    "conflict_overlap_with_intent": (2, 1, 10),
    "conflict_overlap_without_intent": (3, 1, 10),
    "conflict_confidence_without_intent": (0.8, 0.0, 1.0),
    "keyword_min_overlap":          (2, 1, 10),
    "document_list_min_confidence": (0.9, 0.0, 1.0),  # "which documents describe X": the topic's own documents
    "quote_relative_floor":         (0.5, 0.0, 1.0),  # later quotes must score >= this share of the best quote
    # M5b semantic thresholds (cosine similarity of the embedding model). Set
    # from the calibration run on the real model -- see calibrate_semantic.py.
    "semantic_top_n":               (20, 1, 100),     # meaning-based hits that join the safety-net ranking
    "semantic_min_score":           (0.30, 0.0, 1.0), # a chunk can join the search by meaning from here
    "semantic_quote_min":           (0.58, 0.0, 1.0), # a sentence can be quoted by meaning from here (calibrated: above p75 wrong-document)
    "semantic_strong_evidence":     (0.59, 0.0, 1.0), # meaning match strong enough to accept paraphrases (calibrated: above every no-answer question)
    "semantic_word_min":            (0.54, 0.0, 1.0), # an unknown word must match a quoted word this well (calibrated: above unrelated pairs)
    "keyword_max_documents":        (4, 1, 20),
}
GAP_SEVERITIES = {"high", "medium", "low"}


@dataclass
class RelocationProgram:
    """A relocation / new-joiner program (Nexora: Germany)."""
    name: str
    intent_patterns: list[str]                 # regexes on the normalised question
    countries: list[str]                       # office countries that switch the program on
    topic_id: str | None = None                # Phase 3 topic whose documents are searched
    checklist_file: str | None = None          # Phase 3 output with timed checklist items
    disclaimer: bool = True                    # show the program's "not legal advice" text


@dataclass
class CommunicationGuidance:
    intent_patterns: list[str]                 # clearly about tone/phrasing: always a communication question
    weak_intent_patterns: list[str] = field(default_factory=list)   # e.g. "communicate": only with a country
    topic_id: str | None = None                # all communication documents
    overview_document: str | None = None       # framework document, always searched with a profile
    profiles: dict[str, str] = field(default_factory=dict)       # country -> tone-profile document
    country_words: dict[str, str] = field(default_factory=dict)  # word in a question -> country
    note: str = ""                             # shown with every communication answer


@dataclass
class KoraConfig:
    company: str = "unknown"
    thresholds: dict = field(default_factory=lambda: {k: v[0] for k, v in THRESHOLDS.items()})
    gap_severities: list[str] = field(default_factory=lambda: ["high", "medium"])
    extra_intent_patterns: dict[str, list[str]] = field(default_factory=dict)
    # Company word list: everyday phrase -> the words the documents use
    # ("trip" -> ["travel"]). Simple, visible, editable (Phase 5 UI).
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    # Access control (M7): document ID pattern -> groups allowed to read it.
    # Documents without a rule are open to everyone. See src/retrieval/access.py.
    access_rules: dict[str, list[str]] = field(default_factory=dict)
    relocation: RelocationProgram | None = None
    communication: CommunicationGuidance | None = None
    embeddings_enabled: bool = False           # off without config (no download surprises)
    embeddings_model: str = "all-MiniLM-L6-v2"
    embeddings_cache: str = "data/retrieval/embeddings_cache.npz"
    source: str = "defaults"                   # file path, or "defaults"
    warnings: list[str] = field(default_factory=list)

    def t(self, name: str):
        """A threshold value."""
        return self.thresholds[name]

    # ---- references ----

    def check_references(self, document_ids: set[str], topic_ids: set[str]) -> None:
        """Switch off what points at missing documents/topics, with a warning.
        Called once the index is loaded."""
        if self.relocation and self.relocation.topic_id and self.relocation.topic_id not in topic_ids:
            self.warnings.append(f"relocation: topic {self.relocation.topic_id} not found; program documents "
                                 f"won't be searched")
            self.relocation.topic_id = None
        comm = self.communication
        if comm:
            if comm.topic_id and comm.topic_id not in topic_ids:
                self.warnings.append(f"communication: topic {comm.topic_id} not found")
                comm.topic_id = None
            if comm.overview_document and comm.overview_document not in document_ids:
                self.warnings.append(f"communication: overview document {comm.overview_document} not found")
                comm.overview_document = None
            for country, doc in list(comm.profiles.items()):
                if doc not in document_ids:
                    self.warnings.append(f"communication: profile {doc} for '{country}' not found; "
                                         f"that country has no profile")
                    del comm.profiles[country]


# ---------------- loading + validation ----------------

def _check_patterns(patterns, where, problems):
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
        problems.append(f"{where}: must be a list of strings")
        return []
    good = []
    for p in patterns:
        try:
            re.compile(p)
            good.append(p)
        except re.error as e:
            problems.append(f"{where}: invalid regex {p!r} ({e})")
    return good


def parse_config(raw: dict, source: str = "dict") -> KoraConfig:
    """Validate a config dict and build a KoraConfig. Raises ConfigError
    listing every problem; never returns a half-valid config."""
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top level must be a JSON object")
    known = {"company", "thresholds", "gap_severities", "intent_patterns", "relocation", "communication",
             "embeddings", "synonyms", "access", "description"}
    for key in raw:
        if key not in known:
            problems.append(f"unknown key '{key}' (allowed: {', '.join(sorted(known))})")

    cfg = KoraConfig(source=source)
    cfg.company = str(raw.get("company", "unknown"))

    for name, value in (raw.get("thresholds") or {}).items():
        if name not in THRESHOLDS:
            problems.append(f"thresholds.{name}: unknown threshold")
            continue
        default, low, high = THRESHOLDS[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"thresholds.{name}: must be a number")
        elif isinstance(default, int) and not isinstance(value, int):
            problems.append(f"thresholds.{name}: must be a whole number")
        elif not low <= value <= high:
            problems.append(f"thresholds.{name}: {value} outside allowed range {low}..{high}")
        else:
            cfg.thresholds[name] = value
    if cfg.thresholds["global_reserved_slots"] >= cfg.thresholds["top_k"]:
        problems.append("thresholds: global_reserved_slots must be smaller than top_k")

    sev = raw.get("gap_severities", cfg.gap_severities)
    if not isinstance(sev, list) or not set(sev) <= GAP_SEVERITIES:
        problems.append(f"gap_severities: must be a list drawn from {sorted(GAP_SEVERITIES)}")
    else:
        cfg.gap_severities = list(sev)

    for intent, patterns in (raw.get("intent_patterns") or {}).items():
        cfg.extra_intent_patterns[intent] = _check_patterns(patterns, f"intent_patterns.{intent}", problems)

    syn = raw.get("synonyms", {})
    if not isinstance(syn, dict):
        problems.append("synonyms: must be an object of phrase -> list of words")
    else:
        for phrase, words in syn.items():
            key = " ".join(re.sub(r"[^a-z0-9]+", " ", str(phrase).lower()).split())
            if phrase.startswith("_"):                      # "_comment" style keys are allowed
                continue
            if not key:
                problems.append(f"synonyms: empty phrase {phrase!r}")
            elif not isinstance(words, list) or not words or not all(isinstance(w, str) and w.strip() for w in words):
                problems.append(f"synonyms.{phrase}: must be a non-empty list of words")
            else:
                cfg.synonyms[key] = [" ".join(re.sub(r"[^a-z0-9]+", " ", w.lower()).split()) for w in words]

    acc = raw.get("access")
    if acc is not None:
        from src.retrieval.access import GROUP_PATTERN
        if not isinstance(acc, dict) or not isinstance(acc.get("rules", {}), dict):
            problems.append("access: must be an object with 'rules': {document pattern: [groups]}")
        else:
            for key in acc:
                if key not in {"rules"} and not key.startswith("_"):
                    problems.append(f"access.{key}: unknown key (allowed: rules)")
            for pattern, groups in acc.get("rules", {}).items():
                if pattern.startswith("_"):
                    continue
                if not pattern.strip():
                    problems.append("access.rules: empty document pattern")
                elif not isinstance(groups, list) or not groups:
                    problems.append(f"access.rules.{pattern}: must be a non-empty list of groups")
                else:
                    bad = [g for g in groups if not isinstance(g, str) or not GROUP_PATTERN.match(g)]
                    if bad:
                        problems.append(f"access.rules.{pattern}: invalid group(s) {bad} "
                                        f"(use all, manager, dept:<id>, loc:<id>, role:<id>, emp:<id>)")
                    else:
                        cfg.access_rules[pattern] = list(groups)

    reloc = raw.get("relocation")
    if reloc is not None:
        if not isinstance(reloc, dict) or "name" not in reloc:
            problems.append("relocation: must be an object with at least 'name'")
        else:
            cfg.relocation = RelocationProgram(
                name=str(reloc["name"]),
                intent_patterns=_check_patterns(reloc.get("intent_patterns", []), "relocation.intent_patterns", problems),
                countries=[str(c).lower() for c in reloc.get("countries", [])],
                topic_id=reloc.get("topic_id"),
                checklist_file=reloc.get("checklist_file"),
                disclaimer=bool(reloc.get("disclaimer", True)),
            )

    emb = raw.get("embeddings")
    if emb is not None:
        if not isinstance(emb, dict):
            problems.append("embeddings: must be an object")
        else:
            for key in emb:
                if key not in {"enabled", "model", "cache_file"}:
                    problems.append(f"embeddings.{key}: unknown key (allowed: enabled, model, cache_file)")
            if not isinstance(emb.get("enabled", False), bool):
                problems.append("embeddings.enabled: must be true or false")
            else:
                cfg.embeddings_enabled = emb.get("enabled", False)
            if not isinstance(emb.get("model", cfg.embeddings_model), str) or not emb.get("model", "x"):
                problems.append("embeddings.model: must be a non-empty string")
            else:
                cfg.embeddings_model = emb.get("model", cfg.embeddings_model)
            cfg.embeddings_cache = str(emb.get("cache_file", cfg.embeddings_cache))

    comm = raw.get("communication")
    if comm is not None:
        if not isinstance(comm, dict):
            problems.append("communication: must be an object")
        else:
            profiles = comm.get("profiles", {})
            words = comm.get("country_words", {})
            if not isinstance(profiles, dict) or not isinstance(words, dict):
                problems.append("communication.profiles / country_words: must be objects")
                profiles, words = {}, {}
            cfg.communication = CommunicationGuidance(
                intent_patterns=_check_patterns(comm.get("intent_patterns", []), "communication.intent_patterns",
                                                problems),
                weak_intent_patterns=_check_patterns(comm.get("weak_intent_patterns", []),
                                                     "communication.weak_intent_patterns", problems),
                topic_id=comm.get("topic_id"),
                overview_document=comm.get("overview_document"),
                profiles={str(k).lower(): str(v) for k, v in profiles.items()},
                country_words={str(k): str(v).lower() for k, v in words.items()},
                note=str(comm.get("note", "")),
            )
            unknown_countries = set(cfg.communication.country_words.values()) - set(cfg.communication.profiles)
            for c in sorted(unknown_countries):
                cfg.warnings.append(f"communication: country '{c}' has words but no profile document")

    if problems:
        raise ConfigError(f"{source}: {len(problems)} problem(s):\n  - " + "\n  - ".join(problems))
    return cfg


def load_config(path: Path | None = DEFAULT_PATH) -> KoraConfig:
    """The company config, or safe defaults (all special features off) when
    there is no file. A file that exists but is invalid raises ConfigError."""
    if path is None or not Path(path).exists():
        cfg = KoraConfig()
        cfg.warnings.append(f"no config file at {path}: special features are off, default thresholds used")
        return cfg
    with open(path, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}: not valid JSON ({e})") from e
    return parse_config(raw, str(path))


def config_fingerprint(cfg: KoraConfig) -> str:
    """Stable text form of the config, used to detect a changed config."""
    data = deepcopy(cfg.__dict__)
    data.pop("warnings", None)
    data.pop("source", None)
    return json.dumps(data, sort_keys=True, default=lambda o: o.__dict__)

"""
Conflict detection: places where Nexora's sources disagree.

KORA never decides which side is right. It reports both sides, word for word,
plus the metadata signals a person (or a later reasoning step) would weigh:
    active vs archived  |  later effective date  |  newer version of the SAME document
    a source that declares the supersession itself  |  primary document vs FAQ/derived
and a status that only summarises those signals:
    superseded_per_source   a source itself says one side replaced the other
    signals_point_one_way   every available signal favours the same side
    open                    no signals, or they disagree

Detectors (each explainable on its own):
  1. numeric_fact       Two rule statements in different documents are TF-IDF-similar
                        (>= FACT_SIMILARITY) but state different numbers of the same kind.
                        ("within the first 10 working days" vs "within 30 days")
  2. approver           Two approval rules on the same subject name different approvers.
  3. count_claim        Text claims a count ("four office locations") that the tables contradict.
  4. org_statement      The org chart contradicts the organization document's own sentence.
  5. org_vs_employees   employees.csv reporting lines differ from the (simplified) org chart.
  6. identifier         A file header and documents.csv give different IDs.

Agreements found on the way are kept as `corroborations`: they show the
cross-checks that passed, which matters as much as what failed.
"""

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.matchers import role_matcher
from src.intelligence.models import (
    CONF_KEYWORD, CONF_PATTERN, CONF_SIMILARITY, CONF_STRUCTURED,
    DEPARTMENT, DOCUMENT, EMPLOYEE, LOCATION, PROCESS,
)
from src.intelligence.roles import RoleRecord
from src.intelligence.rules import Rule
from src.intelligence.text import amounts, blocks, durations, percents, plain, sentences, top_terms

FACT_SIMILARITY = 0.60   # tuned on Nexora: real conflict scored 0.88; different tiers of one rule 0.47-0.54
APPROACH_NOTE = "KORA reports both sides and the signals; it does not decide which statement is correct."

_ILLUSTRATIVE = re.compile(r"^\*?\s*example\b|^\s*e\.g\.", re.I)
_NARRATION = re.compile(
    r"\b(?:earlier|older|previous) (?:version|document|guidance)|\bsupersed\w*|\bno longer current\b"
    r"|\bsaid\b|\bstated that\b|\bwas (?:shortened|changed|replaced)\b|\breplaced\b", re.I)
_SUPERSEDE = re.compile(r"\bsupersed\w*|\bno longer current\b|\breplaced\b", re.I)
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split())}
_COUNT_CLAIM = re.compile(
    r"\b(?P<n>\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:office\s+)?(?P<noun>locations|departments)\b", re.I)
_PEOPLE_CLAIM = re.compile(
    r"(?P<approx>approximately|about|around|roughly|~)?\s*\(?(?P<n>\d+)(?:\s+people|-person)\b", re.I)
_APPROX_TOLERANCE = 0.10


# ---------------- records ----------------

@dataclass
class Side:
    document_id: str
    statement: str
    values: list[str]
    section: str
    chunk_ids: list[str]
    version: str | None = None
    status: str | None = None
    effective_date: str | None = None
    owner_role: str | None = None
    document_type: str | None = None


@dataclass
class Conflict:
    conflict_id: str
    conflict_type: str
    topic: str
    sides: list[Side]
    signals: dict
    status: str
    confidence: float
    detail: str
    method: str
    note: str = APPROACH_NOTE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["documents_involved"] = sorted({s.document_id for s in self.sides if s.document_id})
        d["document_versions"] = {s.document_id: s.version for s in self.sides if s.document_id}
        d["effective_dates"] = {s.document_id: s.effective_date for s in self.sides if s.document_id}
        d["conflicting_information"] = [{"document_id": s.document_id, "statement": s.statement,
                                         "values": s.values} for s in self.sides]
        return d


def _cid(*parts: str) -> str:
    return "CONF-" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8].upper()


class _Docs:
    def __init__(self, index: EntityIndex):
        self.index = index

    def side(self, doc_id: str, statement: str, values: list[str], section: str, chunk_ids: list[str]) -> Side:
        doc = self.index.get(DOCUMENT, doc_id)
        a = doc.attributes if doc else {}
        return Side(doc_id, statement, values, section, chunk_ids, a.get("version"), a.get("status"),
                    a.get("effective_date"), a.get("owner_role"), a.get("document_type"))

    def lineage(self, doc_id: str) -> str:
        """Documents in the same lineage can have their versions compared:
        the process they document (processes.csv), else their own ID."""
        for p in self.index.of_type(PROCESS):
            if p.attributes.get("document_id") == doc_id:
                return re.sub(r"(-V\d+|-ADDENDUM)$", "", p.entity_id)
        return doc_id


def _version_tuple(v: str | None) -> tuple:
    try:
        return tuple(int(x) for x in (v or "").split("."))
    except ValueError:
        return ()


def _signals(sides: list[Side], docs: _Docs, narration: list[dict]) -> tuple[dict, str]:
    """Collect metadata signals; summarise them into a status (never a verdict)."""
    signals: dict = {}
    by_doc = {s.document_id: s for s in sides if s.document_id}
    ids = sorted(by_doc)
    favour: list[str] = []

    statuses = {d: (by_doc[d].status or "").lower() for d in ids}
    if "active" in statuses.values() and "archived" in statuses.values():
        active = [d for d, s in statuses.items() if s == "active"]
        signals["active_vs_archived"] = {"active": active, "archived": [d for d, s in statuses.items() if s == "archived"]}
        favour += active

    dated = {d: by_doc[d].effective_date for d in ids if re.match(r"\d{4}-\d{2}-\d{2}$", by_doc[d].effective_date or "")}
    if len(set(dated.values())) > 1:
        latest = max(dated.values())
        signals["later_effective_date"] = {"latest": [d for d, v in dated.items() if v == latest], "dates": dated}
        favour += signals["later_effective_date"]["latest"]

    lineages = {d: docs.lineage(d) for d in ids}
    if len(ids) > 1 and len(set(lineages.values())) == 1:
        versions = {d: _version_tuple(by_doc[d].version) for d in ids}
        top = max(versions.values())
        signals["newer_version_same_lineage"] = {"newest": [d for d, v in versions.items() if v == top],
                                                 "versions": {d: by_doc[d].version for d in ids}}
        favour += signals["newer_version_same_lineage"]["newest"]
    elif len(ids) > 1:
        signals["versions_not_comparable"] = "different documents: version numbers belong to different lineages"

    secondary = [d for d in ids if any(w in (by_doc[d].document_type or "").lower() for w in ("faq", "scenario"))]
    if secondary:
        signals["secondary_sources"] = secondary

    declared = [n for n in narration if n["document_id"] in ids]
    if declared:
        signals["declared_by_source"] = declared

    if declared:
        status = "superseded_per_source"
    elif favour and len(set(favour)) == 1:
        status = "signals_point_one_way"
    else:
        status = "open"
    if favour:
        signals["signals_favour"] = dict(Counter(favour))
    return signals, status


# ---------------- 1. numeric facts ----------------

@dataclass
class _Fact:
    document_id: str
    section: str
    statement: str
    chunk_ids: list[str]
    values: dict[str, set[str]]
    narration: bool


def _facts(data: Phase2Data, index: EntityIndex) -> list[_Fact]:
    out = []
    for doc in sorted(index.of_type(DOCUMENT), key=lambda d: d.entity_id):
        a = doc.attributes
        if "scenario" in (a.get("document_type") or "").lower():
            continue     # scenarios are hypothetical examples, their numbers are inputs, not rules
        for section in data.sections(a["header_document_id"], a["version"]):
            for block in blocks(section.text):
                if block.kind in ("heading", "code"):
                    continue
                if block.kind == "table_row":
                    pieces = [" ".join(c.strip() for c in block.text.strip("|").split("|"))]
                else:
                    pieces = sentences(block.body)
                for piece in pieces:
                    text = plain(piece)
                    if _ILLUSTRATIVE.search(text) or set(text) <= set("-:| "):
                        continue
                    values = {
                        "duration": {f"{d['value']} {d['unit'].rstrip('s')}" for d in durations(text)},
                        "amount": set(amounts(text)),
                        "percent": set(percents(text)),
                    }
                    values = {k: v for k, v in values.items() if v}
                    if not values:
                        continue
                    statement = f"{block.title}: {text}" if block.title else text
                    out.append(_Fact(doc.entity_id, section.name, statement,
                                     data.find_chunks(a["header_document_id"], a["version"], text) or section.chunk_ids,
                                     values, bool(_NARRATION.search(text))))
    return out


def _shared_terms(vec, row_a, row_b, n=4) -> list[str]:
    prod = row_a.multiply(row_b).toarray().ravel()
    return top_terms(vec.get_feature_names_out(), prod, n, positive_only=True)


def _numeric_conflicts(facts: list[_Fact], docs: _Docs) -> tuple[list[Conflict], list[dict]]:
    compared = [f for f in facts if not f.narration]
    narration = [f for f in facts if f.narration]
    if len(compared) < 2:
        return [], []
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, token_pattern=r"(?u)\b[A-Za-z][A-Za-z-]{2,}\b")
    X = vec.fit_transform([f.statement for f in compared])
    sim = cosine_similarity(X)

    conflicts, corroborations = [], []
    for i in range(len(compared)):
        for j in range(i + 1, len(compared)):
            a, b = compared[i], compared[j]
            if a.document_id == b.document_id or sim[i, j] < FACT_SIMILARITY:
                continue
            kinds = set(a.values) & set(b.values)
            if not kinds:
                continue
            differing = {k for k in kinds if a.values[k].isdisjoint(b.values[k])}
            topic = ", ".join(_shared_terms(vec, X[i], X[j])) or "related rule"
            if not differing:
                corroborations.append({"type": "numeric_fact", "topic": topic, "similarity": round(float(sim[i, j]), 2),
                                       "documents": [a.document_id, b.document_id],
                                       "statements": [a.statement, b.statement]})
                continue
            # Either document narrating a supersession ("...supersedes that guidance...",
            # "...was superseded by PROC-HR-001...") is the source declaring the change itself.
            ids = {a.document_id, b.document_id}
            declared = [{"document_id": n.document_id, "quote": n.statement, "chunk_ids": n.chunk_ids}
                        for n in narration if n.document_id in ids and _SUPERSEDE.search(n.statement)]
            sides = [docs.side(f.document_id, f.statement, sorted(set().union(*[f.values[k] for k in differing])),
                               f.section, f.chunk_ids) for f in (a, b)]
            signals, status = _signals(sides, docs, declared)
            conflicts.append(Conflict(
                _cid("numeric", *sorted([a.statement, b.statement])), "numeric_fact", topic, sides, signals, status,
                round(float(sim[i, j]), 2),
                f"Similar statements (TF-IDF cosine {sim[i, j]:.2f}) state different {', '.join(sorted(differing))} values.",
                "tfidf_similarity + value comparison"))
    return conflicts, corroborations


# ---------------- 2. approvers ----------------

APPROVER_SIMILARITY = 0.40   # on the approval SUBJECT (role names and approval words removed); tuned on Nexora
_APPROVAL_VOCAB = {"approval", "approve", "approves", "approved", "authority", "final", "requires", "require",
                   "required", "pre-approval", "sign-off", "manager", "explicit", "owns"}
_RANGE = re.compile(
    r"(?P<q>above|over|more than|under|below|up to|less than)?\s*"
    r"(?P<a>€\s?\d[\d,.]*|\d+(?:\.\d+)?%?)(?:\s?(?:–|-|to)\s?(?P<b>€?\s?\d[\d,.]*%?))?", re.I)


def _number(text: str) -> float | None:
    digits = re.sub(r"[^\d.]", "", text.replace(",", "")).rstrip(".")
    try:
        return float(digits)
    except ValueError:
        return None


def _interval(rule: Rule) -> tuple[str, float, float] | None:
    """The amount or percent range a rule applies to: 'above €10,000' -> ('€', 10000, inf)."""
    for m in _RANGE.finditer(rule.statement):
        a = m.group("a")
        if not ("€" in a or "%" in a or (m.group("b") and ("€" in m.group("b") or "%" in m.group("b")))):
            continue
        unit = "€" if "€" in (a + (m.group("b") or "")) else "%"
        lo, hi = _number(a), _number(m.group("b")) if m.group("b") else None
        q = (m.group("q") or "").lower()
        if hi is not None:
            return unit, lo, hi
        if q in ("above", "over", "more than"):
            return unit, lo, float("inf")
        if q in ("under", "below", "up to", "less than"):
            return unit, 0.0, lo
        return unit, lo, lo
    return None


def _separate_tiers(a: Rule, b: Rule) -> bool:
    """True if the two rules cover ranges that don't overlap (touching ends allowed)."""
    ia, ib = _interval(a), _interval(b)
    if not ia or not ib or ia[0] != ib[0]:
        return False
    return ia[2] <= ib[1] or ib[2] <= ia[1]


def _approver_conflicts(rules: list[Rule], docs: _Docs, index: EntityIndex) -> tuple[list[Conflict], list[dict]]:
    """Compare approval rules on their SUBJECT: role names are masked and approval
    words dropped, so 'requires Sales Manager approval' and 'requires CFO approval'
    don't look alike just because both say 'requires ... approval'."""
    approvals = [r for r in rules if r.rule_type == "approval" and (r.role_ids or r.generic_roles)]
    if len(approvals) < 2:
        return [], []
    roles = role_matcher(index)

    def subject(rule: Rule) -> str:
        text = roles.mask(f"{rule.context} {rule.statement}").replace("_", " ").lower()
        return " ".join(w for w in re.findall(r"[a-z][a-z-]+", text) if w not in _APPROVAL_VOCAB)

    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    X = vec.fit_transform([subject(r) for r in approvals])
    sim = cosine_similarity(X)
    conflicts, corroborations = [], []
    for i in range(len(approvals)):
        for j in range(i + 1, len(approvals)):
            a, b = approvals[i], approvals[j]
            if a.document_id == b.document_id or sim[i, j] < APPROVER_SIMILARITY:
                continue
            who_a = set(a.role_ids) | {g.lower() for g in a.generic_roles}
            who_b = set(b.role_ids) | {g.lower() for g in b.generic_roles}
            topic = ", ".join(_shared_terms(vec, X[i], X[j])) or "approval"
            base = {"topic": topic, "similarity": round(float(sim[i, j]), 2),
                    "documents": [a.document_id, b.document_id], "statements": [a.statement, b.statement],
                    "rule_ids": [a.rule_id, b.rule_id]}
            if _separate_tiers(a, b):
                corroborations.append({"type": "approval_tiers", **base,
                                       "note": "same subject, different ranges: tiers of one rule, not a conflict"})
            elif who_a & who_b:
                corroborations.append({"type": "approver", **base, "approver": sorted(who_a & who_b),
                                       "note": "same approver; different names in the text are documented aliases"
                                               if a.historical != b.historical else "same approver"})
            else:
                sides = [docs.side(r.document_id, r.statement, r.role_ids + r.generic_roles, r.section, r.chunk_ids)
                         for r in (a, b)]
                signals, status = _signals(sides, docs, [])
                conflicts.append(Conflict(
                    _cid("approver", a.rule_id, b.rule_id), "approver", topic, sides, signals, status,
                    round(float(sim[i, j]), 2), "Approval rules on the same subject name different approvers.",
                    "subject similarity + approver comparison"))
    return conflicts, corroborations


# ---------------- 3. count claims ----------------

def _count_claims(data: Phase2Data, index: EntityIndex, docs: _Docs) -> tuple[list[Conflict], list[dict]]:
    truth = {
        "locations": (len(index.of_type(LOCATION)), "locations.csv rows"),
        "departments": (len(index.of_type(DEPARTMENT)), "departments.csv rows"),
    }
    employees = len(index.of_type(EMPLOYEE))
    claims: dict[str, list[tuple]] = defaultdict(list)
    corroborations: list[dict] = []

    for doc in sorted(index.of_type(DOCUMENT), key=lambda d: d.entity_id):
        a = doc.attributes
        dept_ids = a.get("department_ids", [])
        for section in data.sections(a["header_document_id"], a["version"]):
            for sentence in sentences(section.text):
                for m in _COUNT_CLAIM.finditer(sentence):
                    noun = m.group("noun").lower()
                    raw = m.group("n").lower()
                    n = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
                    claims[noun].append((doc.entity_id, section, sentence, n))
                for m in _PEOPLE_CLAIM.finditer(sentence):
                    n, approx = int(m.group("n")), bool(m.group("approx"))
                    is_company = "department description" not in (a.get("document_type") or "").lower()
                    if is_company:
                        expected, basis = employees, "employees.csv rows"
                    elif len(dept_ids) == 1:
                        dept = index.get(DEPARTMENT, dept_ids[0])
                        expected = int(dept.attributes.get("employees_in_employees_csv") or 0)
                        basis = f"employees.csv rows for {dept_ids[0]} (departments.csv approx_headcount: {dept.attributes.get('approx_headcount_stated')})"
                    else:
                        continue
                    ok = abs(n - expected) <= (expected * _APPROX_TOLERANCE if approx else 0)
                    record = {"type": "count_claim", "document": doc.entity_id, "statement": sentence,
                              "claimed": n, "approximate": approx, "actual": expected, "basis": basis}
                    if ok:
                        corroborations.append(record)
                    else:
                        claims[f"people:{doc.entity_id}"].append((doc.entity_id, section, sentence, n, expected, basis))

    conflicts = []
    for noun, found in sorted(claims.items()):
        if noun.startswith("people:"):
            for doc_id, section, sentence, n, expected, basis in found:
                side = docs.side(doc_id, sentence, [str(n)], section.name, section.chunk_ids)
                table = Side("", f"{basis}: {expected}", [str(expected)], "", [])
                conflicts.append(Conflict(_cid("people", doc_id, sentence), "count_claim", "headcount",
                                          [side, table], {}, "open", CONF_PATTERN,
                                          f"Text claims {n} people; {basis} gives {expected}.", "count_claim vs table"))
            continue
        actual, basis = truth[noun]
        wrong = [c for c in found if c[3] != actual]
        for c in found:
            if c[3] == actual:
                corroborations.append({"type": "count_claim", "document": c[0], "statement": c[2],
                                       "claimed": c[3], "actual": actual, "basis": basis})
        if not wrong:
            continue
        sides = [docs.side(doc_id, sentence, [str(n)], section.name, section.chunk_ids)
                 for doc_id, section, sentence, n in wrong]
        sides.append(Side("", f"{basis}: {actual}", [str(actual)], "", []))
        conflicts.append(Conflict(
            _cid("count", noun, *[s.statement for s in sides]), "count_claim", f"number of {noun}", sides,
            {"structured_source": basis, "documents_claiming": sorted({s.document_id for s in sides if s.document_id})},
            "open", CONF_PATTERN,
            f"{len(wrong)} statement(s) give a different number of {noun} than {basis} ({actual}).",
            "count_claim vs table"))
    return conflicts, corroborations


# ---------------- 4 + 5. organization ----------------

_HEAD_RULE = re.compile(r"department head[^.]*\breports? to\b[^.]*", re.I)


def _org_conflicts(data: Phase2Data, index: EntityIndex, roles: list[RoleRecord], org_chart: dict,
                   docs: _Docs) -> list[Conflict]:
    conflicts: list[Conflict] = []
    role_by_id = {r.role_id: r for r in roles}
    head_roles = {rid: d.entity_id for d in index.of_type(DEPARTMENT) for rid in d.attributes.get("head_role_ids", [])}

    # 4. "each led by a department head who reports to the CEO or ... CTO"
    for doc in index.of_type(DOCUMENT):
        if "organization" not in (doc.attributes.get("document_type") or "").lower():
            continue
        a = doc.attributes
        for section in data.sections(a["header_document_id"], a["version"]):
            for sentence in sentences(section.text):
                if not _HEAD_RULE.search(sentence):
                    continue
                allowed = set(re.findall(r"\b[A-Z]{3,4}\b", sentence))      # 'CEO', 'CTO'
                for edge in org_chart.get("edges", []):
                    if edge["source_document"] != doc.entity_id:
                        continue
                    for child in edge["child_role_ids"]:
                        if child not in head_roles or edge["parent"] in allowed:
                            continue
                        quote = edge["quote"]
                        sides = [docs.side(doc.entity_id, sentence, sorted(allowed), section.name, section.chunk_ids),
                                 docs.side(doc.entity_id, quote, [edge["parent"]], edge["section"], edge["chunk_ids"])]
                        conflicts.append(Conflict(
                            _cid("orghead", child, edge["parent"]), "org_statement",
                            f"reporting line of {role_by_id[child].name} (head of {head_roles[child]})",
                            sides, {"same_document": doc.entity_id}, "open", CONF_KEYWORD,
                            f"The text says department heads report to {' or '.join(sorted(allowed))}; the org chart "
                            f"places {role_by_id[child].name} under {edge['parent']}.", "statement vs org chart"))

    # 5. employees.csv manager roles vs org-chart parents (the chart is labelled 'simplified')
    chart_parents: dict[str, set[str]] = defaultdict(set)
    for edge in org_chart.get("edges", []):
        for child in edge["child_role_ids"]:
            chart_parents[child] |= set(edge["parent_role_ids"])
    pairs: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    for emp in index.of_type(EMPLOYEE):
        rid = emp.attributes.get("role_id")
        if rid not in chart_parents or not chart_parents[rid]:
            continue
        mgr = index.get(EMPLOYEE, emp.attributes.get("manager_id") or "")
        mgr_role = mgr.attributes.get("role_id") if mgr else None
        if mgr_role is None or mgr_role not in chart_parents[rid]:
            pairs[(rid, mgr_role)].append(emp.entity_id)
    for (rid, mgr_role), emps in sorted(pairs.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        expected = sorted(role_by_id[p].name for p in chart_parents[rid])
        actual = role_by_id[mgr_role].name if mgr_role else "no manager_id"
        sides = [Side("", f"org chart: {role_by_id[rid].name} reports to {' / '.join(expected)}", expected,
                      "Reporting Lines (simplified)", []),
                 Side("", f"employees.csv: {len(emps)} {role_by_id[rid].name}(s) with manager role '{actual}'",
                      [actual], "", [])]
        conflicts.append(Conflict(
            _cid("orgemp", rid, mgr_role or "none"), "org_vs_employees", f"reporting line of {role_by_id[rid].name}",
            sides, {"employees": sorted(emps), "chart_is_simplified": True}, "open", CONF_SIMILARITY,
            "employees.csv reporting differs from the org chart. The chart is labelled 'simplified', so this may "
            "be skip-level or local practice rather than an error.", "employees.csv vs org chart"))
    return conflicts


# ---------------- 6. identifiers ----------------

def _identifier_conflicts(duplicates: dict, docs: _Docs) -> list[Conflict]:
    out = []
    for dup in duplicates.get("potential_duplicates", []):
        if dup["method"] != "shared_header_id":
            continue
        a, b = (k.split(":", 1)[1] for k in (dup["entity_a"], dup["entity_b"]))
        sides = []
        for d in (a, b):
            attrs = docs.index.get(DOCUMENT, d).attributes
            sides.append(docs.side(d, f"{attrs['source_filename']}: header says Document ID {attrs['header_document_id']}; "
                                      f"documents.csv lists the file as {d}", [attrs["header_document_id"]], "header", []))
        out.append(Conflict(_cid("id", a, b), "identifier", "document identifier", sides, {},
                            "open", CONF_STRUCTURED, dup["reason"], "header vs documents.csv"))
    return out


# ---------------- everything ----------------

def detect_conflicts(data: Phase2Data, index: EntityIndex, rules: list[Rule], roles: list[RoleRecord],
                     org_chart: dict, duplicates: dict) -> dict:
    docs = _Docs(index)
    facts = _facts(data, index)
    numeric, corr_numeric = _numeric_conflicts(facts, docs)
    approver, corr_approver = _approver_conflicts(rules, docs, index)
    counts, corr_counts = _count_claims(data, index, docs)
    org = _org_conflicts(data, index, roles, org_chart, docs)
    ident = _identifier_conflicts(duplicates, docs)

    conflicts = numeric + approver + counts + org + ident
    order = ["numeric_fact", "approver", "count_claim", "org_statement", "identifier", "org_vs_employees"]
    conflicts.sort(key=lambda c: (order.index(c.conflict_type), -c.confidence, c.conflict_id))
    return {
        "approach": APPROACH_NOTE,
        "thresholds": {"fact_similarity": FACT_SIMILARITY, "approver_subject_similarity": APPROVER_SIMILARITY,
                       "approximate_count_tolerance": _APPROX_TOLERANCE},
        "fact_statements_compared": len([f for f in facts if not f.narration]),
        "conflicts": [c.to_dict() for c in conflicts],
        "corroborations": corr_numeric + corr_approver + corr_counts,
    }

"""
Phase 5 M6 - what the answer log says about KORA: where it helps, where it
struggles, and what to improve next.

    log = read_log(Path("data/logs/answers.jsonl"))
    report = summarise(log, channel="ui", days=30, min_askers=2)

Rules (each one is in DEFINITIONS, shown on the dashboard next to its number):

* Real data only. Nothing is invented or sampled; an empty log gives empty
  sections that say so.
* Every line is counted once. A malformed line (not JSON, no timestamp, an
  unknown status) is skipped and COUNTED as skipped, with the reason. An exact
  repeat of a line (same timestamp, question, user...) is counted once.
* A crash is not a "not found". Records with kind "error" are system errors,
  reported separately and never in the answer-status counts.
* Old records (before log schema 2) have no channel: they show as "unknown".
* Privacy. No number is ever broken down by person. A question's text is shown
  only when at least MIN_ASKERS different people asked it (questions can be
  personal: "my salary"); the rest are counted, not shown. Words that are names
  (people asked about colleagues) are counted, never listed, and a question
  containing a name is never shown, however many people asked it.
* No causes. The log shows what happened (e.g. refused questions cluster on a
  topic), not why; the dashboard never labels a correlation as a cause.
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATUSES = ("answered", "partial", "not_found", "conflict")       # chart/stack order
CHANNELS = ("ui", "api", "cli", "python", "unknown")

DEFINITIONS = {
    "answers": "Questions KORA replied to (log records of kind 'answer'), after skipping malformed lines and exact repeats.",
    "status_share": "Share of answers with each status. 'Not found' is an honest refusal (nothing documented), not a failure.",
    "errors": "Times KORA crashed instead of answering (kind 'error'). Never counted as an answer status.",
    "skipped": "Log lines that could not be read (not JSON, no timestamp, unknown status). Counted here, left out everywhere else.",
    "unknown_words": "Words in questions that no Nexora document uses. To review, not a gap list: a word may be a synonym "
                     "(add it to Company words), a typo, or something KORA should never answer.",
    "refused": "Questions answered 'not found', grouped by their text. Text is shown only when enough different people asked "
               "it, and never when it names a person. Some refusals are access rules working as intended, not missing knowledge.",
    "partial": "Questions answered 'partial', grouped by text, with the words no quoted source covered.",
    "weak_topics": "Topics the router matched, with the share of their answers that were partial or not found "
                   "(at least 3 answers). A high share points where to look; it does not say why. Access rules cause "
                   "refusals too: people who may not read a document get 'not found' for it.",
    "latency": "Total answer time per day (median and 95th percentile), from each answer's own timing.",
    "steps": "Average time per answer step. 'graph' appears once the knowledge graph is on.",
    "startup": "KORA's start-up time, and how many answers had to wait for an index rebuild first.",
    "usage": "Answers per channel and per kind of question. No breakdown by person.",
    "cited": "Documents cited most often in answers (company records counted together).",
}


@dataclass
class LogRead:
    records: list[dict] = field(default_factory=list)
    lines: int = 0
    skipped: Counter = field(default_factory=Counter)        # reason -> count
    repeats: int = 0
    missing_file: bool = False


def _parse_time(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def read_log(path: Path) -> LogRead:
    out = LogRead()
    path = Path(path)
    if not path.exists():
        out.missing_file = True
        return out
    seen = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            if not raw.strip():
                continue
            out.lines += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                out.skipped["not valid JSON"] += 1
                continue
            if not isinstance(rec, dict):
                out.skipped["not a JSON object"] += 1
                continue
            at = _parse_time(rec.get("at"))
            if at is None:
                out.skipped["no readable timestamp"] += 1
                continue
            kind = rec.get("kind", "answer")                     # schema 1 had no kind: all answers
            if kind not in ("answer", "error"):
                out.skipped[f"unknown kind '{kind}'"] += 1
                continue
            if kind == "answer" and rec.get("status") not in STATUSES:
                out.skipped["answer without a known status"] += 1
                continue
            key = json.dumps(rec, sort_keys=True, ensure_ascii=False)
            if key in seen:
                out.repeats += 1
                continue
            seen.add(key)
            channel = rec.get("channel") if rec.get("channel") in CHANNELS else "unknown"
            out.records.append({**rec, "_at": at, "_kind": kind, "_channel": channel})
    return out


# ---------------- helpers ----------------

def question_key(q: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(q).lower()).split())


def _pct(values, p):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(p / 100 * (len(values) - 1))))
    return round(values[k], 1)


def _name_words(rec) -> set[str]:
    """Unknown words KORA itself flagged as names in the question (a colleague, a city)."""
    return {w.get("word") for w in (rec.get("decision") or {}).get("unknown_words", [])
            if isinstance(w, dict) and w.get("blocked") == "a name in the question"}


def _grouped(records, min_askers):
    """Group answers by question text; text only when enough different people asked it."""
    groups = defaultdict(lambda: {"count": 0, "askers": set(), "text": None, "words": Counter(), "names": False})
    for r in records:
        g = groups[question_key(r.get("question", ""))]
        g["count"] += 1
        g["askers"].add(r.get("user") or "anonymous")
        g["text"] = g["text"] or r.get("question")
        names = _name_words(r)
        g["names"] = g["names"] or bool(names)
        g["words"].update(w for w in (r.get("uncovered_terms") or []) if w not in names)
    shown, hidden = [], 0
    for g in sorted(groups.values(), key=lambda g: (-g["count"], g["text"] or "")):
        if len(g["askers"]) >= min_askers and not g["names"]:
            shown.append({"question": g["text"], "times": g["count"], "people": len(g["askers"]),
                          "words": [w for w, _ in g["words"].most_common(5)]})
        else:
            hidden += 1
    return shown, hidden


# ---------------- the report ----------------

def summarise(log: LogRead, channel: str | None = None, days: int | None = None, now: datetime | None = None,
              min_askers: int = 2, can_read=lambda doc_id: True) -> dict:
    now = now or datetime.now(timezone.utc)
    recs = log.records
    if channel:
        recs = [r for r in recs if r["_channel"] == channel]
    if days:
        since = now - timedelta(days=days)
        recs = [r for r in recs if r["_at"] >= since]
    answers = [r for r in recs if r["_kind"] == "answer"]
    errors = [r for r in recs if r["_kind"] == "error"]
    n = len(answers)

    # A. status
    status = Counter(r["status"] for r in answers)
    per_day = defaultdict(Counter)
    for r in answers:
        per_day[r["_at"].date().isoformat()][r["status"]] += 1
    days_sorted = sorted(per_day)
    by_status = {s: status.get(s, 0) for s in STATUSES}

    # B. gaps
    unknown, names_hidden, word_people = Counter(), 0, defaultdict(set)
    for r in answers:
        names = _name_words(r)
        names_hidden += len(names)
        for w in (r.get("decision") or {}).get("unknown", []) or []:
            if w not in names:
                unknown[w] += 1
                word_people[w].add(r.get("user") or "anonymous")
    refused, refused_hidden = _grouped([r for r in answers if r["status"] == "not_found"], min_askers)
    partial, partial_hidden = _grouped([r for r in answers if r["status"] == "partial"], min_askers)
    topic_counts = defaultdict(Counter)
    for r in answers:
        for e in ((r.get("retrieval") or {}).get("entities") or []):
            if isinstance(e, str) and e.startswith("topic:"):       # "topic:TOPIC-X (strong)" -> "TOPIC-X"
                topic_counts[e.split(":", 1)[1].split(" ")[0]][r["status"]] += 1
    weak_topics = sorted(({"topic": t, "answers": sum(c.values()),
                           "weak_share": round((c["partial"] + c["not_found"]) / sum(c.values()), 2)}
                          for t, c in topic_counts.items() if sum(c.values()) >= 3),
                         key=lambda x: (-x["weak_share"], -x["answers"]))

    # C. performance
    lat_day = defaultdict(list)
    steps = defaultdict(list)
    for r in answers:
        t = r.get("timings") or {}
        if isinstance(t.get("total_ms"), (int, float)):
            lat_day[r["_at"].date().isoformat()].append(t["total_ms"])
        for k, v in t.items():
            if k != "total_ms" and isinstance(v, (int, float)):
                steps[k.removesuffix("_ms")].append(v)
    all_ms = [v for vs in lat_day.values() for v in vs]
    slow_cut = _pct(all_ms, 95)
    slowest = sorted((r for r in answers if isinstance((r.get("timings") or {}).get("total_ms"), (int, float))),
                     key=lambda r: -r["timings"]["total_ms"])[:8]
    asked_by, has_names = defaultdict(set), set()
    for r in answers:
        k = question_key(r.get("question", ""))
        asked_by[k].add(r.get("user") or "anonymous")
        if _name_words(r):
            has_names.add(k)
    showable = lambda q: len(asked_by[question_key(q)]) >= min_askers and question_key(q) not in has_names  # noqa: E731
    startups = [r.get("startup") or {} for r in answers]
    rebuilds = sum(1 for s in startups if "rebuild" in s)

    # E. usage
    def category(r):
        g = (r.get("decision") or {}).get("graph") or {}
        if g.get("used"):
            return "relationship (graph): " + g.get("kind", "?")
        intents = (r.get("retrieval") or {}).get("intents") or []
        return intents[0] if intents else "general"
    cited = Counter()
    records_cited = 0
    for r in answers:
        if r["status"] == "not_found":
            continue
        docs = set()
        for cid in r.get("citations") or []:
            if not isinstance(cid, str):
                continue
            if cid.startswith("record:"):
                records_cited += 1
            elif cid.startswith(("header:", "file:")):
                continue
            else:
                docs.add(cid.rsplit("_", 2)[0])
        cited.update(d for d in docs if can_read(d))

    return {
        "definitions": DEFINITIONS,
        "filters": {"channel": channel or "all", "days": days or "all", "min_askers": min_askers},
        "log": {"lines": log.lines, "skipped": dict(log.skipped), "skipped_total": sum(log.skipped.values()),
                "repeats": log.repeats, "missing_file": log.missing_file,
                "first": min((r["_at"] for r in recs), default=None), "last": max((r["_at"] for r in recs), default=None)},
        "answers": n,
        "status": {"counts": by_status, "share": {s: round(c / n, 3) if n else None for s, c in by_status.items()},
                   "per_day": [{"day": d, **{s: per_day[d].get(s, 0) for s in STATUSES}} for d in days_sorted]},
        "errors": {"count": len(errors),
                   "by_stage": dict(Counter(e.get("stage", "unknown") for e in errors)),
                   "by_type": dict(Counter(e.get("error_type", "unknown") for e in errors)),
                   "recent": [{"at": e["_at"], "stage": e.get("stage"), "type": e.get("error_type")}
                              for e in sorted(errors, key=lambda e: e["_at"], reverse=True)[:8]]},
        "gaps": {"unknown_words": [{"word": w, "times": c, "people": len(word_people[w])} for w, c in unknown.most_common(25)],
                 "name_words_hidden": names_hidden,
                 "refused": refused[:20], "refused_hidden": refused_hidden,
                 "partial": partial[:20], "partial_hidden": partial_hidden,
                 "weak_topics": weak_topics[:15]},
        "performance": {
            "per_day": [{"day": d, "p50": _pct(v, 50), "p95": _pct(v, 95), "answers": len(v)} for d, v in sorted(lat_day.items())],
            "overall": {"p50": _pct(all_ms, 50), "p95": _pct(all_ms, 95), "max": round(max(all_ms), 1) if all_ms else None},
            "steps": {k: round(statistics.mean(v), 2) for k, v in sorted(steps.items())},
            "slowest": [{"at": r["_at"], "ms": r["timings"]["total_ms"], "status": r["status"],
                         "question": r.get("question") if showable(r.get("question", "")) else None,
                         "steps": {k.removesuffix("_ms"): v for k, v in r["timings"].items() if k != "total_ms"}}
                        for r in slowest if slow_cut is not None],
            "startup_s": sorted({s.get("total_s") for s in startups if isinstance(s.get("total_s"), (int, float))}),
            "rebuilds": rebuilds},
        "usage": {"channels": dict(Counter(r["_channel"] for r in recs)),
                  "categories": dict(Counter(category(r) for r in answers).most_common()),
                  "cited_documents": [{"document_id": d, "answers": c} for d, c in cited.most_common(12)],
                  "company_record_citations": records_cited,
                  "active_days": len(days_sorted)},
    }

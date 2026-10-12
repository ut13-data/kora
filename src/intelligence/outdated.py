"""
Outdated knowledge: what is history, what replaced it, and what changed.

Nothing is deleted. An archived document stays in the graph (Phase 2 decision),
so KORA can answer both "what is the rule?" and "what was the rule before?".

For every archived document:
  superseded_by   found three ways, each kept as evidence:
                    its own text ("superseded by PROC-IMPL-001 v2.0")
                    a same-name pair with a different status (duplicates.py version_pair)
                    the process register (an archived process row next to the current one)
  signals         status, version, effective dates
  changes         step by step and approver by approver, when both are process documents
  terminology     names it uses that current documents have replaced ('Project Lead')

For ACTIVE documents, two staleness checks:
  legacy_terminology  an old name used as a live instruction (not inside quotes)
  stale_references    a sentence pointing at archived content without saying it's history
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.intelligence.entities import EntityIndex
from src.intelligence.loader import Phase2Data
from src.intelligence.matchers import role_matcher
from src.intelligence.models import DOCUMENT, PROCESS, ROLE
from src.intelligence.normalize import version_key
from src.intelligence.processes import ProcessRecord
from src.intelligence.rules import Rule
from src.intelligence.text import plain, sentences

_SUPERSEDED_BY = re.compile(
    r"superseded by\s+((?:PROC|DOC)-[A-Z0-9]+(?:-[A-Z0-9]+)*)(?:,?\s+(?:v|Version\s*)(\d+(?:\.\d+)*))?", re.I)
_HISTORY_WORDS = re.compile(r"\b(?:earlier|older|previous|archived|superseded|historical|no longer)\b", re.I)
_QUOTES = "\"“”'‘’"


@dataclass
class _Ctx:
    data: Phase2Data
    index: EntityIndex
    processes: dict[str, ProcessRecord]
    rules: list[Rule]


def _doc(ctx: _Ctx, doc_id: str) -> dict:
    d = ctx.index.get(DOCUMENT, doc_id)
    a = d.attributes if d else {}
    return {"document_id": doc_id, "title": d.name if d else None, "version": a.get("version"),
            "status": a.get("status"), "effective_date": a.get("effective_date"),
            "source_filename": a.get("source_filename")}


def _body(ctx: _Ctx, doc_id: str) -> str:
    d = ctx.index.get(DOCUMENT, doc_id)
    if not d:
        return ""
    return ctx.data.source_texts.get((d.attributes.get("source_filename") or "").lower(), "")


def _superseded_by(ctx: _Ctx, doc_id: str, duplicates: dict) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for sentence in sentences(_body(ctx, doc_id)):
        for m in _SUPERSEDED_BY.finditer(sentence):
            ref, version = m.group(1), m.group(2)
            target = ctx.index.doc_by_header.get((ref, version_key(version))) if version else None
            target = target or (ref if ctx.index.get(DOCUMENT, ref) else None)
            if target and target != doc_id:
                quote = sentence[max(0, m.start() - 60):m.end() + 40].strip()
                found.setdefault(target, []).append(f'its own text: "...{quote}..."')
    for dup in duplicates.get("potential_duplicates", []):
        if dup["method"] == "version_pair" and dup["entity_b"] == f"document:{doc_id}":
            found.setdefault(dup["entity_a"].split(":", 1)[1], []).append(f"same title, different status ({dup['duplicate_id']})")
    for p in ctx.index.of_type(PROCESS):
        if p.attributes.get("document_id") != doc_id or (p.attributes.get("status") or "").lower() != "archived":
            continue
        base = re.sub(r"(-V\d+|-ADDENDUM)$", "", p.entity_id)
        current = ctx.index.get(PROCESS, base)
        if current and current.attributes.get("document_id"):
            found.setdefault(current.attributes["document_id"], []).append(
                f"processes.csv: archived row {p.entity_id} sits next to current {base}")
    return found


def _step_changes(old: ProcessRecord, new: ProcessRecord, ctx: _Ctx) -> list[dict]:
    def names(ids):
        return [ctx.index.get(ROLE, r).name for r in ids if ctx.index.get(ROLE, r)]
    changes = []
    old_steps = {s["number"]: s for s in old.steps}
    new_steps = {s["number"]: s for s in new.steps}
    for n in sorted(set(old_steps) | set(new_steps)):
        a, b = old_steps.get(n), new_steps.get(n)
        before = (a["title"] + ": " if a and a["title"] else "") + (a["text"] if a else "")
        after = (b["title"] + ": " if b and b["title"] else "") + (b["text"] if b else "")
        roles_a, roles_b = set(a["role_ids"]) if a else set(), set(b["role_ids"]) if b else set()
        changes.append({
            "step": n,
            "before": before or None,
            "after": after or None,
            "text_similarity": round(SequenceMatcher(None, before.lower(), after.lower()).ratio(), 2),
            "roles_added": names(sorted(roles_b - roles_a)),
            "roles_removed": names(sorted(roles_a - roles_b)),
        })
    return changes


def _approver_changes(old_doc: str, new_doc: str, ctx: _Ctx) -> dict:
    def approvers(doc_id):
        return sorted({r for rule in ctx.rules if rule.rule_type == "approval" and rule.document_id == doc_id
                       for r in rule.role_ids})
    before, after = approvers(old_doc), approvers(new_doc)
    return {"before": before, "after": after, "unchanged": before == after,
            "note": "Compared by role ID, so a renamed title that is a documented alias "
                    "(e.g. 'Project Lead' = Implementation Manager) does not count as a change."}


def _alias_usages(ctx: _Ctx, doc_id: str) -> list[dict]:
    """Names in this document that are documented ALTERNATIVE names of a role
    (from a terminology note, not an acronym), with whether they appear in quotes."""
    matcher = role_matcher(ctx.index)
    out = []
    text = plain(_body(ctx, doc_id))
    for m in matcher.find(text):
        role = ctx.index.get(ROLE, m.key.split(":", 1)[1])
        term = m.term.rstrip("s") if m.term not in [role.name, *role.aliases] else m.term
        if term.lower() == role.name.lower() or term.isupper():
            continue
        evidence = role.attributes.get("alias_evidence", {}).get(term, [])
        if not any("TN-" in e for e in evidence):
            continue
        before = text[max(0, m.start - 2):m.start]
        quoted = any(q in before for q in _QUOTES)
        context = text[max(0, m.start - 80):m.end + 80]
        out.append({"term": term, "current_name": role.name, "role_id": role.entity_id, "quoted": quoted,
                    "context": context.strip(), "alias_evidence": evidence})
    return out


def detect_outdated(data: Phase2Data, index: EntityIndex, processes: list[ProcessRecord],
                    rules: list[Rule], duplicates: dict, conflicts: dict) -> dict:
    ctx = _Ctx(data, index, {p.document_id: p for p in processes if p.document_id}, rules)
    docs = sorted(index.of_type(DOCUMENT), key=lambda d: d.entity_id)

    superseded = []
    for d in docs:
        if (d.attributes.get("status") or "").lower() != "archived":
            continue
        replacements = _superseded_by(ctx, d.entity_id, duplicates)
        record = {**_doc(ctx, d.entity_id), "superseded_by": [], "changes": [], "approver_changes": None,
                  "terminology": _alias_usages(ctx, d.entity_id),
                  "conflicts": [c["conflict_id"] for c in conflicts.get("conflicts", [])
                                if d.entity_id in c["documents_involved"]],
                  "kept_because": "History stays: KORA can answer 'what was the rule before?' and explain changes."}
        for target, evidence in sorted(replacements.items()):
            t = _doc(ctx, target)
            signals = {"archived_vs_active": f"{d.attributes.get('status')} vs {t['status']}"}
            if re.match(r"\d{4}-", d.attributes.get("effective_date") or "") and re.match(r"\d{4}-", t["effective_date"] or ""):
                signals["effective_dates"] = f"{d.attributes.get('effective_date')} -> {t['effective_date']}"
            if ctx.processes.get(d.entity_id) and ctx.processes.get(target) and \
                    re.sub(r"(-V\d+|-ADDENDUM)$", "", ctx.processes[d.entity_id].process_id) == ctx.processes[target].process_id:
                signals["version"] = f"{d.attributes.get('version')} -> {t['version']} (same process)"
            record["superseded_by"].append({**t, "evidence": evidence, "signals": signals})
            old_p, new_p = ctx.processes.get(d.entity_id), ctx.processes.get(target)
            if old_p and new_p and old_p.steps and new_p.steps:
                record["changes"] = _step_changes(old_p, new_p, ctx)
                record["approver_changes"] = _approver_changes(d.entity_id, target, ctx)
        superseded.append(record)

    legacy, stale = [], []
    archived_ids = {d.entity_id for d in docs if (d.attributes.get("status") or "").lower() == "archived"}
    archived_files = {index.get(DOCUMENT, a).attributes.get("source_filename") for a in archived_ids}
    for d in docs:
        if d.entity_id in archived_ids:
            continue
        for usage in _alias_usages(ctx, d.entity_id):
            if not usage["quoted"]:
                legacy.append({"document_id": d.entity_id, **usage})
        for sentence in sentences(_body(ctx, d.entity_id)):
            pointed = [a for a in archived_ids if re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", sentence)]
            pointed += [f for f in archived_files if f and f in sentence]
            if pointed and not _HISTORY_WORDS.search(sentence):
                stale.append({"document_id": d.entity_id, "points_at": pointed, "sentence": sentence})

    history_notes = [{"process_id": p.process_id, "document_id": p.document_id, **note}
                     for p in processes for note in p.version_notes]

    return {
        "policy": "Archived documents are kept and linked to what replaced them; nothing is deleted.",
        "superseded_documents": superseded,
        "legacy_terminology_in_active_documents": legacy,
        "stale_references_in_active_documents": stale,
        "history_notes_in_active_documents": history_notes,
        "checks": {
            "archived_documents": len(superseded),
            "archived_without_known_replacement": [s["document_id"] for s in superseded if not s["superseded_by"]],
            "active_documents_checked": len(docs) - len(archived_ids),
        },
    }

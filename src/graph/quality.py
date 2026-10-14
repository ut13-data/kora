"""
Knowledge graph health: what is documented, what is only derived, and what
is missing. Feeds the Phase 5 dashboards; runnable on its own.

    python -m src.graph.quality              # writes reports/graph_quality.md
    python -m src.graph.quality --as EMP-0120   # as one user sees it (admins: their view)

Every number below has a written definition (DEFINITIONS), so a dashboard
tile can show exactly what it counts. All counts are computed on the graph
the caller passes in: a GraphView for a user gives that user's numbers, so
nothing restricted is counted, listed or named.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.graph.schema import DOCUMENTED, TIER_LABELS, TIERS, VALID_SHAPES
from src.graph.trace import Tracer

DEFINITIONS = {
    "links_by_tier": "Links counted by their strongest VERIFIED evidence (quote > header > record > "
                     "mention > derived). 'unverified' = no evidence passed its check.",
    "documented_share": "Share of links whose strongest verified evidence is a quote, a header line or a "
                        "company record: links an answer may state as fact.",
    "unverified_evidence": "Evidence items whose source no longer says what Phase 3 recorded (quote missing, "
                           "header line changed, table row gone).",
    "unknown_shapes": "Links whose (source type, relationship, target type) is not in src/graph/schema.py.",
    "dangling": "Links Phase 3 found in the documents but could not attach, because one end is not a known "
                "entity (from relationships.json).",
    "processes_without_owner": "Active processes with no role linked by 'owns' with documented evidence.",
    "processes_without_participants": "Active processes with no role linked by 'participates_in' or 'performs'.",
    "roles_without_department": "Roles with no 'in_department' link.",
    "isolated": "Entities with no visible link at all.",
    "open_conflicts": "Phase 3 conflicts whose status is not resolved (KORA never resolves them itself). "
                      "A conflict is shown only if the user may read every document involved.",
    "superseded_documents": "Archived documents and what replaced them (Phase 3 outdated_knowledge.json).",
    "gaps": "Phase 3 knowledge gaps by severity, only those whose source documents the user may read.",
}


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def quality_report(graph, tracer: Tracer, intelligence_dir: Path = Path("data/intelligence"),
                   dangling: list[dict] | None = None, can_read=lambda doc_id: True) -> dict:
    rels = graph.relationships
    nodes = {k: graph.node(k) for k in graph.index.entities}
    nodes = {k: e for k, e in nodes.items() if e is not None}

    tiers, unverified, unknown_shapes = Counter(), [], []
    by_rel_tier: dict[str, Counter] = {}
    best: dict[str, str | None] = {}
    for r in rels:
        t = tracer.tier(r, r.evidence)
        best[r.relationship_id] = t
        tiers[t or "unverified"] += 1
        by_rel_tier.setdefault(r.relationship, Counter())[t or "unverified"] += 1
        for p in tracer.trace(r):
            if not p.verified and p.evidence in r.evidence:
                unverified.append({"relationship_id": r.relationship_id, "link": f"{r.source_key} "
                                   f"--{r.relationship}--> {r.target_key}", "method": p.evidence.method,
                                   "source_file": p.evidence.source_file, "reason": p.reason})
        if (r.source_type, r.relationship, r.target_type) not in VALID_SHAPES:
            unknown_shapes.append(f"{r.source_key} --{r.relationship}--> {r.target_key}")

    linked = Counter()
    for r in rels:
        linked[r.source_key] += 1
        linked[r.target_key] += 1

    def has(key, relationship, direction_in=True, documented=False, source_type=None):
        for r in rels:
            end, other = (r.target_key, r.source_key) if direction_in else (r.source_key, r.target_key)
            if end == key and r.relationship == relationship and \
                    (source_type is None or other.startswith(source_type + ":")) and \
                    (not documented or best[r.relationship_id] in DOCUMENTED):
                return True
        return False

    active_processes = [e for k, e in nodes.items() if e.entity_type == "process"
                        and (e.attributes.get("status") or "").lower() != "archived"]
    no_owner = sorted(e.entity_id for e in active_processes
                      if not has(f"process:{e.entity_id}", "owns", documented=True, source_type="role"))
    no_participants = sorted(e.entity_id for e in active_processes
                             if not (has(f"process:{e.entity_id}", "participates_in")
                                     or has(f"process:{e.entity_id}", "performs")))
    roles_no_dept = sorted(e.entity_id for k, e in nodes.items() if e.entity_type == "role"
                           and not has(k, "in_department", direction_in=False))
    isolated = sorted(k for k in nodes if not linked[k])

    conflicts = _read(intelligence_dir / "conflicts.json").get("conflicts", [])
    conflicts = [c for c in conflicts if all(can_read(d) for d in c.get("documents_involved", []))]
    open_conflicts = [{"conflict_id": c["conflict_id"], "type": c["conflict_type"], "status": c["status"],
                       "documents": c.get("documents_involved", []), "detail": c.get("detail", "")}
                      for c in conflicts if c.get("status") != "resolved"]
    superseded = [{"document_id": d["document_id"], "title": d.get("title"), "status": d.get("status"),
                   "superseded_by": [s["document_id"] for s in d.get("superseded_by", []) if can_read(s["document_id"])]}
                  for d in _read(intelligence_dir / "outdated_knowledge.json").get("superseded_documents", [])
                  if can_read(d["document_id"])]
    gaps = [g for g in _read(intelligence_dir / "knowledge_gaps.json").get("gaps", [])
            if all(can_read(d) for d in g.get("source_documents") or [])]

    total = len(rels)
    documented = sum(tiers[t] for t in DOCUMENTED)
    return {
        "definitions": DEFINITIONS,
        "nodes": len(nodes),
        "nodes_by_type": dict(sorted(Counter(e.entity_type for e in nodes.values()).items())),
        "links": total,
        "links_by_tier": {t: tiers.get(t, 0) for t in (*TIERS, "unverified")},
        "links_by_relationship_and_tier": {k: dict(v) for k, v in sorted(by_rel_tier.items())},
        "documented_share": round(documented / total, 3) if total else None,
        "unverified_evidence": unverified,
        "unknown_shapes": unknown_shapes,
        "dangling": list(dangling or []),
        "processes_without_owner": no_owner,
        "processes_without_participants": no_participants,
        "roles_without_department": roles_no_dept,
        "isolated": isolated,
        "open_conflicts": open_conflicts,
        "superseded_documents": superseded,
        "gaps_by_severity": dict(sorted(Counter(g["severity"] for g in gaps).items())),
    }


def render(report: dict, who: str) -> str:
    t = report["links_by_tier"]
    out = ["# KORA knowledge graph quality", "",
           f"_Generated by `python -m src.graph.quality` for: {who}. Do not edit by hand._", "",
           f"**{report['nodes']} entities, {report['links']} links. "
           f"{report['documented_share']:.0%} of links are documented** (quote, header or company record): "
           "those are the only links an answer may state as fact.", "",
           "## Links by how KORA knows them", "", "| Tier | Meaning | Links |", "|---|---|---|"]
    for tier in TIERS:
        out.append(f"| {tier} | {TIER_LABELS[tier]} | {t[tier]} |")
    out.append(f"| unverified | no evidence passed its check | {t['unverified']} |")
    out += ["", "## Per relationship", "", "| Relationship | " + " | ".join(TIERS) + " | unverified |",
            "|---|" + "---|" * (len(TIERS) + 1)]
    for rel, counts in report["links_by_relationship_and_tier"].items():
        out.append(f"| {rel} | " + " | ".join(str(counts.get(x, 0)) for x in (*TIERS, "unverified")) + " |")

    def section(title, key, items, fmt=str):
        lines = ["", f"## {title} ({len(items)})", "", f"_{DEFINITIONS[key]}_", ""]
        return lines + ([f"- {fmt(i)}" for i in items] if items else ["None."])

    out += section("Unverified evidence", "unverified_evidence", report["unverified_evidence"],
                   lambda u: f"{u['link']} ({u['source_file']}, {u['method']}): {u['reason']}")
    out += section("Unknown relationship shapes", "unknown_shapes", report["unknown_shapes"])
    out += section("Dangling links", "dangling", report["dangling"], lambda d: json.dumps(d, ensure_ascii=False))
    out += section("Active processes without a documented owner", "processes_without_owner",
                   report["processes_without_owner"])
    out += section("Active processes without participants", "processes_without_participants",
                   report["processes_without_participants"])
    out += section("Roles without a department", "roles_without_department", report["roles_without_department"])
    out += section("Entities with no links", "isolated", report["isolated"])
    out += section("Open conflicts", "open_conflicts", report["open_conflicts"],
                   lambda c: f"{c['conflict_id']} [{c['type']}, {c['status']}] {', '.join(c['documents'])}: "
                             f"{c['detail']}")
    out += section("Superseded documents", "superseded_documents", report["superseded_documents"],
                   lambda d: f"{d['document_id']} ({d['status']}) -> replaced by "
                             f"{', '.join(d['superseded_by']) or 'not stated'}")
    out += ["", "## Knowledge gaps by severity", "", f"_{DEFINITIONS['gaps']}_", "",
            ", ".join(f"{k}: {v}" for k, v in report["gaps_by_severity"].items()) or "None.", ""]
    return "\n".join(out)


def main() -> int:
    from src.graph.load import load_graph
    from src.graph.view import GraphView
    from src.retrieval.access import AccessPolicy, users_from_tables
    from src.retrieval.config import load_config

    parser = argparse.ArgumentParser(description="Knowledge graph quality report.")
    parser.add_argument("--as", dest="as_user", metavar="EMPLOYEE_ID", help="as this employee sees it")
    parser.add_argument("--out", type=Path, default=Path("reports/graph_quality.md"))
    args = parser.parse_args()

    loaded = load_graph()
    tracer = Tracer.from_dirs(loaded.graph)
    config = load_config()
    graph, can_read, who = loaded.graph, (lambda d: True), "full access (backend)"
    if args.as_user:
        tables = {p.stem: json.loads(p.read_text(encoding="utf-8"))["rows"]
                  for p in Path("data/processed/tables").glob("*.json")}
        user = users_from_tables(tables).get(args.as_user)
        if user is None:
            print(f"Unknown employee ID '{args.as_user}'.")
            return 2
        graph = GraphView.for_user(loaded.graph, user, config)
        policy = AccessPolicy(config.access_rules)
        can_read, who = (lambda d: policy.can_read(user, d)), user.describe()
    report = quality_report(graph, tracer, dangling=loaded.dangling if not args.as_user else [],
                            can_read=can_read)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(report, who), encoding="utf-8")
    t = report["links_by_tier"]
    print(f"{report['nodes']} entities, {report['links']} links, {report['documented_share']:.0%} documented")
    print("  " + ", ".join(f"{k} {v}" for k, v in t.items()))
    print(f"  unverified evidence {len(report['unverified_evidence'])}, unknown shapes "
          f"{len(report['unknown_shapes'])}, dangling {len(report['dangling'])}")
    print(f"  processes without documented owner: {report['processes_without_owner'] or 'none'}")
    print(f"  processes without participants: {report['processes_without_participants'] or 'none'}")
    print(f"  open conflicts {len(report['open_conflicts'])}, superseded documents "
          f"{len(report['superseded_documents'])}, gaps {report['gaps_by_severity']}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

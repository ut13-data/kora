"""
Explore KORA's knowledge graph from the terminal, with the evidence for every link.

    python -m src.explain EMP-0077                         # everything linked to one entity
    python -m src.explain PROC-IMPL-001 --rel approves     # only one relationship
    python -m src.explain EMP-0077 PROD-NEXFLOW            # how two entities connect
    python -m src.explain PROC-IT-001                      # an ID used by several kinds: pick one
    python -m src.explain PROC-FIN-002 --as EMP-0040       # as one employee sees it (access + privacy)

IDs can be bare ('EMP-0077') or typed ('employee:EMP-0077').

Phase 5: reads the SAVED graph (data/intelligence) instead of rebuilding it,
and shows, per link, how KORA knows it (quote / header / record / mention /
derived) after checking the evidence against its source again.
"""

import argparse
import json
import sys
from pathlib import Path

from src.graph.load import load_graph
from src.graph.schema import TIER_LABELS
from src.graph.trace import Tracer
from src.graph.view import GraphView
from src.intelligence.graph import BOTH, IN, OUT
from src.retrieval.access import users_from_tables
from src.retrieval.config import load_config


def _resolve(graph, text: str) -> str | None:
    if ":" in text:
        return text if graph.node(text) else None
    matches = [k for k in graph.index.entities if k.split(":", 1)[1] == text and graph.node(k)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        print(f"'{text}' is used by several kinds of entity: {', '.join(sorted(matches))}. Use one of these.")
    else:
        print(f"No entity with ID '{text}'.")      # same message whether missing or hidden
    return None


def _explain(graph, tracer, rel) -> str:
    tier = tracer.tier(rel, rel.evidence)
    return f"{graph.explain(rel)}\n  => {TIER_LABELS.get(tier, 'UNVERIFIED: no evidence passed its check')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain entities and links in KORA's knowledge graph.")
    parser.add_argument("entity")
    parser.add_argument("other", nargs="?", help="Second entity: show how the two connect")
    parser.add_argument("--rel", help="Only this relationship name (e.g. approves, has_role)")
    parser.add_argument("--dir", choices=[OUT, IN, BOTH], default=BOTH)
    parser.add_argument("--current-only", action="store_true", help="Leave out archived knowledge")
    parser.add_argument("--as", dest="as_user", metavar="EMPLOYEE_ID",
                        help="as this employee sees it; default: full access")
    args = parser.parse_args()

    loaded = load_graph()
    tracer = Tracer.from_dirs(loaded.graph)
    graph = loaded.graph
    if args.as_user:
        tables = {p.stem: json.loads(p.read_text(encoding="utf-8"))["rows"]
                  for p in Path("data/processed/tables").glob("*.json")}
        user = users_from_tables(tables).get(args.as_user)
        if user is None:
            print(f"Unknown employee ID '{args.as_user}'.")
            sys.exit(2)
        graph = GraphView.for_user(loaded.graph, user, load_config())
        print(f"As {user.describe()}\n")

    start = _resolve(graph, args.entity)
    if start is None:
        sys.exit(1)

    if args.other:
        end = _resolve(graph, args.other)
        if end is None:
            sys.exit(1)
        hops = graph.path(start, end, current_only=args.current_only)
        if hops is None:
            print(f"No connection within 4 steps between {graph.label(start)} and {graph.label(end)}.")
            return
        print(graph.explain_path(hops, start))
        print()
        for hop in hops:
            print(_explain(graph, tracer, hop.relationship))
        return

    hops = graph.hops(start, args.rel, args.dir, args.current_only)
    print(f"{graph.label(start)}: {len(hops)} link(s)\n")
    for hop in sorted(hops, key=lambda h: (h.relationship.relationship, h.node)):
        print(_explain(graph, tracer, hop.relationship))
        print()


if __name__ == "__main__":
    main()

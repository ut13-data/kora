"""
Explore KORA's knowledge graph from the terminal, with the evidence for every link.

    python -m src.explain EMP-0077                         # everything linked to one entity
    python -m src.explain PROC-IMPL-001 --rel approves     # only one relationship
    python -m src.explain EMP-0077 PROD-NEXFLOW            # how two entities connect
    python -m src.explain PROC-IT-001                      # an ID used by several kinds: pick one

IDs can be bare ('EMP-0077') or typed ('employee:EMP-0077').
"""

import argparse
import sys

from src.build_intelligence import IntelligenceSettings, run_intelligence
from src.intelligence.graph import BOTH, IN, OUT


def _resolve(result, text: str) -> str | None:
    if ":" in text:
        return text if text in result.entities.entities else None
    matches = [k for k, e in result.entities.entities.items() if e.entity_id == text]
    if len(matches) == 1:
        return matches[0]
    if matches:
        print(f"'{text}' is used by several kinds of entity: {', '.join(sorted(matches))}. Use one of these.")
    else:
        print(f"No entity with ID '{text}'.")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain entities and links in KORA's knowledge graph.")
    parser.add_argument("entity")
    parser.add_argument("other", nargs="?", help="Second entity: show how the two connect")
    parser.add_argument("--rel", help="Only this relationship name (e.g. approves, has_role)")
    parser.add_argument("--dir", choices=[OUT, IN, BOTH], default=BOTH)
    parser.add_argument("--current-only", action="store_true", help="Leave out archived knowledge")
    args = parser.parse_args()

    _, result = run_intelligence(IntelligenceSettings(save_output=False))
    graph = result.graph

    start = _resolve(result, args.entity)
    if start is None:
        sys.exit(1)

    if args.other:
        end = _resolve(result, args.other)
        if end is None:
            sys.exit(1)
        hops = graph.path(start, end, current_only=args.current_only)
        if hops is None:
            print(f"No connection within 4 steps between {graph.label(start)} and {graph.label(end)}.")
            return
        print(graph.explain_path(hops, start))
        print()
        for hop in hops:
            print(graph.explain(hop.relationship))
        return

    hops = graph.hops(start, args.rel, args.dir, args.current_only)
    print(f"{graph.label(start)}: {len(hops)} link(s)\n")
    for hop in sorted(hops, key=lambda h: (h.relationship.relationship, h.node)):
        print(graph.explain(hop.relationship))
        print()


if __name__ == "__main__":
    main()

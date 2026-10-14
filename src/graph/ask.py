"""
Try the graph's relationship answers on their own (Phase 5 M2), before they
are joined to the normal answer flow in M3.

    python -m src.graph.ask "Which teams are involved in customer onboarding?"
    python -m src.graph.ask "Who owns the procurement process?" --as EMP-0040
    python -m src.graph.ask --dev          # every graph development question: what the graph says

"Not a relationship question" means the graph stays silent; in M3 those go to
the Phase 4 answer flow unchanged.
"""

import argparse
import json
import sys
from pathlib import Path

from src.graph.load import load_graph
from src.graph.query import GraphQueries
from src.graph.trace import Tracer
from src.graph.view import GraphView
from src.retrieval.access import users_from_tables
from src.retrieval.config import load_config

DEV_SET = Path("data/evaluation/graph_dev_questions.json")


def open_queries(intelligence_dir=Path("data/intelligence"), processed_dir=Path("data/processed"),
                 chunks_dir=Path("data/chunks"), raw_dir=Path("data/raw"), config=None):
    """Load once; returns (base graph, tracer, users, config, make(user) -> GraphQueries)."""
    loaded = load_graph(intelligence_dir)
    tracer = Tracer.from_dirs(loaded.graph, processed_dir, chunks_dir, raw_dir)
    config = config or load_config()
    tables = {p.stem: json.loads(p.read_text(encoding="utf-8"))["rows"]
              for p in (processed_dir / "tables").glob("*.json")}
    users = users_from_tables(tables)

    def make(user):
        return GraphQueries(GraphView.for_user(loaded.graph, user, config), tracer, intelligence_dir)
    return loaded.graph, tracer, users, config, make


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the knowledge graph a relationship question.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--as", dest="as_user", metavar="EMPLOYEE_ID")
    parser.add_argument("--dev", action="store_true", help="run the graph development questions")
    args = parser.parse_args()

    _, _, users, _, make = open_queries()

    if args.dev:
        questions = json.loads(DEV_SET.read_text(encoding="utf-8"))["questions"]
        names_hit = names_total = status_ok = 0
        for q in questions:
            user = users.get(q["as_user"]) if q.get("as_user") else None
            a = make(user).answer(q["question"])
            status = a.status if a else "(not a graph question -> Phase 4)"
            text = (a.text() if a else "").lower()
            hit = sum(n.lower() in text for n in q["expected_names"])
            names_hit += hit
            names_total += len(q["expected_names"])
            ok = a is not None and a.status in q["expected_status"]
            status_ok += ok
            leaks = [f for f in q["forbidden_names"] if a and f.lower() in text]
            print(f"{q['id']} [{q['graph_kind']}] {'OK ' if ok else '-- '} {status:<34} names "
                  f"{hit}/{len(q['expected_names'])}{'  LEAK ' + str(leaks) if leaks else ''}  | {q['question']}")
        print(f"\nGraph alone: status right {status_ok}/{len(questions)}, expected names "
              f"{names_hit}/{names_total}. (Questions the graph leaves to Phase 4 count as not right here.)")
        return 0

    if not args.question:
        parser.print_help()
        return 2
    user = None
    if args.as_user:
        user = users.get(args.as_user)
        if user is None:
            print(f"Unknown employee ID '{args.as_user}'.")
            return 2
        print(f"As {user.describe()}\n")
    q = make(user)
    question = " ".join(args.question)
    a = q.answer(question)
    if a is None:
        print(f"Q: {question}\nNot a relationship question the graph recognises (intent: {q.intent(question)}, "
              f"entities: {[m.key for m in q.resolver.find(question)] or 'none'}). Phase 4 answers it.")
        return 0
    print(a.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())

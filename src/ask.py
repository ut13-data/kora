"""
Ask KORA from the terminal.

    python -m src.ask "Who approves purchases above 10000 euros?"
    python -m src.ask "How soon is a business trip paid back?" --why
    python -m src.ask                       # interactive: KORA loads ONCE, then ask many questions
    python -m src.ask "What are the payment terms for vendor invoices?" --as EMP-0032
                                            # as a given employee: only documents they may read

--why   shows every score and threshold behind the status, and which rule
        decided ("'boss' never appears in the documents -> not found; add it
        to the company word list if it means something they do say").
--as    asks as an employee: only documents that employee may read are used
        (config section "access"); without it there is no filtering.

Every answer ends with a timing line: this answer's steps in ms, and KORA's
start-up per step in seconds. The index is checked before every question
and rebuilt if data/raw changed (src.index).
"""

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask KORA a question.")
    parser.add_argument("question", nargs="*", help="the question (leave out for interactive mode)")
    parser.add_argument("--why", action="store_true", help="show why KORA chose this status")
    parser.add_argument("--top-k", type=int, default=None, help="chunks to retrieve (default from config)")
    parser.add_argument("--as", dest="as_user", default=None, metavar="EMPLOYEE_ID",
                        help="ask as this employee (access control); default: no filtering")
    args = parser.parse_args()

    from src.index import Kora                 # imported here so --help is instant

    t0 = time.perf_counter()
    print("Loading KORA...", flush=True)
    kora = Kora.open(channel="cli")
    print(f"Ready in {time.perf_counter() - t0:.1f} s.", flush=True)
    for event in kora.events:
        print(f"  ! {event}")
    user = None
    if args.as_user:
        user = kora.user(args.as_user)
        if user is None:
            print(f"Unknown employee ID '{args.as_user}'.")
            return 2
        print(f"Asking as {user.describe()}")
    else:
        print("Asking with full access (no user given).")

    def ask(q: str) -> None:
        answer = kora.ask(q, top_k=args.top_k, user=user)
        print()
        print(answer.render(why=args.why))
        print()

    if args.question:
        ask(" ".join(args.question))
        return
    print("Type a question (empty line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ask(q)


if __name__ == "__main__":
    sys.exit(main())

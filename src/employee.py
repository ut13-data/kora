"""
Show what KORA knows is relevant to one employee, in plain text.

    python -m src.employee EMP-0085
    python -m src.employee EMP-0085 --as-of 2025-05-20
    python -m src.employee --hypothetical          # the spec's "new Data Analyst, Munich, week 2"
"""

import argparse
import sys
from datetime import date

from src.build_intelligence import DEMO_AS_OF, IntelligenceSettings, run_intelligence


def _line(title: str) -> None:
    print(f"\n== {title}")


def show(e: dict, result) -> None:
    who = e["name"] if e["hypothetical"] else f"{e['name']} ({e['employee_id']})"
    print(f"{who}: {e['role']['name']}, {e['department']['name']}, {e['location']['city']}")
    print(f"Joined {e['join_date']} -> day {e['day']} ({e['joining_stage']}) as of {result.employee_context['as_of']}")
    if e["manager"]:
        print(f"Manager: {e['manager']['name']} ({e['manager']['employee_id']}, {e['manager']['role']})")

    _line("Processes relevant to you")
    for p in e["relevant_processes"][:6]:
        print(f"  {p['process_id']:16} {p['name']:35} {'; '.join(p['why'])}")

    _line("Approvals")
    for x in e["approvals"]["you_may_need_approval_from"]:
        people = ", ".join(f"{p['name']} ({p['employee_id']})" for a in x["approvers"] for p in a["people"])
        print(f"  you may need: {x['statement']}  -> {people}  [{x['document_id']}]")
    for x in e["approvals"]["your_manager_approves"][:4]:
        mgr = x["your_manager"]["name"] if x["your_manager"] else "your manager (not in employees.csv)"
        print(f"  {mgr} approves: {x['statement']}  [{x['document_id']}]")
    for x in e["approvals"]["you_approve"]:
        print(f"  you approve: {x['statement']}  [{x['document_id']}]")

    if e["onboarding"]:
        _line("Onboarding timeline")
        for s in e["onboarding"]["steps"]:
            print(f"  {s['date'] or '-':10}  {s['title']:20} {s['status']}")
        if e["onboarding"]["note"]:
            print(f"  note: {e['onboarding']['note']}")

    g = e["germany_care"]
    _line(f"Germany settling-in: {g['eligibility']['status']}")
    if "plan" in g:
        items = {i["item_id"]: i for i in g["plan"]["items"]}
        if not g["plan"]["progress_reported"]:
            print("  Nothing reported as done yet (data/context/checklist_progress.json).")
        print("  Next:", ", ".join(items[i]["title"] for i in g["plan"]["next_steps"]))
        print("  Worth checking:", ", ".join(items[i]["title"] for i in g["plan"]["possibly_forgotten"][:4]))
        for i in g["plan"]["next_steps"]:
            if items[i]["official_guidance_note"]:
                print(f"  ({items[i]['title']}: {items[i]['official_guidance_note']})")

    c = e["communication"]
    _line("Communication")
    print(f"  Effective tone: {c['effective_tone']}  ({c['basis']})")
    if c["reminder"]:
        print(f"  Reminder: {c['reminder']}")

    _line("Read first")
    for r in e["reading_list"][:8]:
        print(f"  {r['document_id']:18} {r['why']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show KORA's context for one employee.")
    parser.add_argument("employee_id", nargs="?")
    parser.add_argument("--as-of", type=date.fromisoformat, default=DEMO_AS_OF)
    parser.add_argument("--hypothetical", action="store_true", help="Show the hypothetical example profile")
    args = parser.parse_args()
    if not args.employee_id and not args.hypothetical:
        parser.error("give an employee ID (e.g. EMP-0085) or --hypothetical")

    _, result = run_intelligence(IntelligenceSettings(as_of=args.as_of, save_output=False))
    if args.hypothetical:
        for e in result.employee_context["hypothetical_examples"]:
            show(e, result)
        return
    e = result.employee(args.employee_id)
    if e is None:
        print(f"No employee {args.employee_id} in employees.csv")
        sys.exit(1)
    show(e, result)


if __name__ == "__main__":
    main()

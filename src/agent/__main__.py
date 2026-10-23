"""
Run the Stage 2 agent in the terminal and watch every step of the loop.

    python -m src.agent EMP-0085 "help me get settled"
    python -m src.agent EMP-0040 "help me with my Germany checklist"
    python -m src.agent EMP-0085 "what's the weather"            # not this agent's job
    python -m src.agent EMP-0085 "help me get settled" --max-steps 2
    python -m src.agent EMP-0085 "help me get settled" --json    # the whole notebook
    python -m src.agent EMP-0085 "help me get settled" --chat    # Stage 3: talk to it; it can record
                                                                 # what you confirm (type "stop" to end)
"""

import argparse
import json
import sys

from src.agent.loop import MAX_STEPS, run


def chat(employee_id: str, goal: str, show_steps: bool) -> int:
    from src.agent.chat import ChatSession

    session = ChatSession(goal, employee_id)
    if session.today:
        print(f"(For KORA, today is {session.today.isoformat()}: the demo date, so 'yesterday' means "
              f"the day before that. Type 'stop' to end.)\n")
    print("KORA: " + session.opening().replace("\n", "\n      ") + "\n")
    while not session.finished:
        try:
            text = input("You:  ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        print("\nKORA: " + session.reply(text).replace("\n", "\n      ") + "\n")
    task = session.task
    print(f"[{task.task_id}] status: {task.status} ({task.stop_reason})")
    if show_steps:
        for s in task.steps:
            tool = f"  [tool: {s.tool}]" if s.tool else ""
            print(f"  {s.number:>2}. {s.stage.upper():<8} {s.what}{tool}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="KORA Stage 2 agent (read-only).")
    parser.add_argument("employee_id", help="who is asking, e.g. EMP-0085")
    parser.add_argument("goal", help="what they want help with")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--json", action="store_true", help="print the whole task as JSON")
    parser.add_argument("--chat", action="store_true", help="Stage 3: answer KORA's questions in the terminal")
    parser.add_argument("--steps", action="store_true", help="with --chat: print the notebook at the end")
    args = parser.parse_args()
    if args.chat:
        return chat(args.employee_id, args.goal, args.steps)

    task = run(args.goal, args.employee_id, max_steps=args.max_steps)
    if args.json:
        print(json.dumps(task.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"{task.task_id}  goal: \"{task.goal}\"  as {task.actor}\n")
    for s in task.steps:
        tool = f"  [tool: {s.tool}]" if s.tool else ""
        print(f"  {s.number}. {s.stage.upper():<7} {s.what}{tool}")
    print(f"\nStatus: {task.status}  ({task.stop_reason})")
    if task.question:
        print("\nKORA asks:\n" + "\n".join("  " + line for line in task.question.splitlines()))
    if task.message:
        print("\nKORA says:\n  " + task.message)
    return 0


if __name__ == "__main__":
    sys.exit(main())

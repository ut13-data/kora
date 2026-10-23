"""
Phase 6 - KORA's agent layer.

Stage 2: one small, read-only agent for the Germany settling-in goal.

    python -m src.agent EMP-0085 "help me get settled"

Files:
    task.py    the agent's notebook: the goal, what it did, why it stopped
    tools.py   the only things the agent may do (read-only in Stage 2), each with
               input checks and a permission check done in Python
    loop.py    the loop itself: Goal -> State -> Plan -> Act -> Verify -> Stop
"""

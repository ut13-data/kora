"""
The agent's notebook for one task.

Everything the agent did is written down here, step by step, so a person can
read exactly what happened and why. Nothing is hidden in the agent's "head".

Status of a task:
    running            still working through the loop
    waiting_for_user   asked the employee a question and stopped to wait
    done               nothing left to do
    stopped            stopped on purpose (not eligible, goal not supported,
                       a tool failed, or the step limit was reached)
"""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

STATUSES = ("running", "waiting_for_user", "done", "stopped")


@dataclass
class Step:
    number: int
    stage: str          # goal | state | plan | act | verify | stop
    what: str           # one plain sentence: what happened
    by: str = "python"  # who decided: "python" today; later also "model" (Stage 5)
    tool: str | None = None
    data: dict = field(default_factory=dict)


@dataclass
class AgentTask:
    goal: str
    actor: str                     # the signed-in employee the agent works for
    max_steps: int = 8
    task_id: str = field(default_factory=lambda: "TASK-" + uuid.uuid4().hex[:8])
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    status: str = "running"
    steps: list[Step] = field(default_factory=list)
    question: str | None = None    # the one question waiting for the employee
    focus_item: str | None = None  # the checklist item that question is about
    message: str | None = None     # what KORA says when it stops without a question
    stop_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    def log(self, stage: str, what: str, *, tool: str | None = None, by: str = "python", **data) -> None:
        """Write one line in the notebook. Raises StepLimit when the budget is used up."""
        if len(self.steps) >= self.max_steps:
            raise StepLimit(self.max_steps)
        self.steps.append(Step(len(self.steps) + 1, stage, what, by, tool, data))

    def finish(self, status: str, reason: str) -> None:
        assert status in STATUSES and status != "running"
        self.status, self.stop_reason = status, reason

    def resume(self) -> None:
        """The employee answered: the task is running again (the notebook keeps every earlier step)."""
        self.status, self.stop_reason, self.question, self.message = "running", None, None, None

    def to_dict(self) -> dict:
        return asdict(self)


class StepLimit(Exception):
    """The agent tried to take more steps than it is allowed. Always stops the task."""

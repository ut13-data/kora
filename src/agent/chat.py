"""
Stage 3: the agent loop across a conversation, with its first WRITE action.

Each employee reply goes round the loop once more:

    OBSERVE   what did the employee say?          ("yes, yesterday")
    DECIDE    what does it mean? (plain rules)     yes + date 2025-04-21
    PROPOSE   say exactly what would be written    "Shall I record ... on 2025-04-21?"
    -- wait for "yes" --
    ACT       call the write tool, ONLY now, with confirmed=True
    VERIFY    read the file back: is it really there?
    CONTINUE  ask about the next item, or stop

The four states of an action are kept apart in the notebook:
    proposed   KORA asked "Shall I record ...?"
    attempted  the employee said yes, the tool was called
    succeeded  the tool reported no error
    verified   the record was read back and matches

Limits (so it never nags or runs on): at most MAX_ITEMS items per conversation,
MAX_TURNS replies, MAX_UNCLEAR unclear replies in a row. "stop" ends it any time.
"""

import re
from dataclasses import dataclass, field

from src.agent.dates import parse_when
from src.agent.loop import _tool, ask_next, start
from src.agent.task import StepLimit
from src.agent.tools import ToolContext, call_tool

MAX_ITEMS = 3        # questions about at most 3 checklist items per conversation (calm, not nagging)
MAX_TURNS = 20
MAX_UNCLEAR = 2
CHAT_MAX_STEPS = 80

_STOP = re.compile(r"^\s*(?:stop|quit|exit|bye|goodbye)\b|\benough\b|\bthat'?s all\b", re.I)
_UNDO = re.compile(r"\bundo\b|\btake (?:that|it) back\b", re.I)
_NA = re.compile(r"\bnot applicable\b|\bn/a\b|\bdoes(?:n'?t| not) apply\b|\bdon'?t need\b|\bdo not need\b|\birrelevant\b", re.I)
_NO = re.compile(r"\b(?:no|nope|nah|not yet|not done|not really|haven'?t|have not|didn'?t|did not|pending|"
                 r"still (?:working|waiting)|later|cancel|never ?mind)\b", re.I)
_YES = re.compile(r"\b(?:yes|yeah|yep|yup|ya|y|sure|ok|okay|correct|right|done|did|have|completed?|sorted|finished|"
                  r"already|go ahead|confirm)\b", re.I)


def classify(text: str) -> str:
    """stop | undo | not_applicable | no | yes | unclear -- checked in this order on purpose
    ("not applicable" contains "no"; "I have not" contains "have")."""
    for name, pattern in (("stop", _STOP), ("undo", _UNDO), ("not_applicable", _NA), ("no", _NO), ("yes", _YES)):
        if pattern.search(text or ""):
            return name
    return "unclear"


@dataclass
class ChatSession:
    goal: str
    actor: str
    ctx: ToolContext = field(default_factory=ToolContext)
    max_items: int = MAX_ITEMS

    def __post_init__(self):
        self.task, self.record, self.care = start(self.goal, self.actor, self.ctx, max_steps=CHAT_MAX_STEPS)
        self.today = self.ctx.current_date() if self.care else None
        self.phase = "asked" if self.task.status == "waiting_for_user" else "ended"
        self.pending: dict | None = None        # the proposed change, waiting for "yes"
        self.skipped: set[str] = set()          # "not yet" in this conversation: don't ask again today
        self.items_asked = 1 if self.phase == "asked" else 0
        self.turns = 0
        self.unclear = 0
        self.last_change: dict | None = None    # for "undo"

    @property
    def finished(self) -> bool:
        return self.phase == "ended"

    def opening(self) -> str:
        return self.task.question or self.task.message or ""

    # ------------------------------------------------------------ one reply

    def reply(self, text: str) -> str:
        if self.finished:
            return "This conversation has ended. Start a new one any time."
        self.turns += 1
        try:
            self.task.resume()
            self.task.log("observe", f"Employee replied: \"{text[:200]}\"", by="employee")
            if self.turns > MAX_TURNS:
                return self._end("stopped", "turn limit reached",
                                 "We've covered a lot. Let's pause here; your progress so far is saved.")
            handler = {"asked": self._on_asked, "ask_date": self._on_date, "confirm": self._on_confirm}[self.phase]
            return handler(text)
        except StepLimit:
            self.phase = "ended"
            self.task.finish("stopped", "step limit reached; stopped safely")
            return "Let's pause here. Anything already recorded is saved."

    # ------------------------------------------------------------ phases

    def _on_asked(self, text: str) -> str:
        intent = classify(text)
        item = self._item()
        if intent == "unclear" and parse_when(text, self.today)[0] in ("date", "future"):
            intent = "yes"                                    # "yesterday" on its own means "yes, yesterday"
        self._decide(f"Understood the reply as '{intent}'", intent=intent)
        if intent == "stop":
            return self._end("done", "employee stopped", "Okay, let's stop here. Come back any time.")
        if intent == "undo":
            return self._propose_undo()
        if intent == "not_applicable":
            return self._propose("not_applicable", item, None,
                                 f"Shall I mark {item['title']} as not applicable to you? (yes/no) You can undo this later.")
        if intent == "no":
            self.skipped.add(item["item_id"])
            return self._next(f"No problem, I've left {item['title']} open. Nothing was recorded.")
        if intent == "yes":
            return self._handle_date(text, item, first_time=True)
        return self._unclear("Sorry, I didn't catch that. You can answer yes, not yet, or not applicable.")

    def _on_date(self, text: str) -> str:
        intent = classify(text)
        if intent == "stop":
            return self._end("done", "employee stopped", "Okay, let's stop here. Nothing more was recorded.")
        if intent == "no" and parse_when(text, self.today)[0] == "none":
            self.skipped.add(self.pending["item_id"])
            return self._next("Okay, I won't record it. Nothing was changed.")
        return self._handle_date(text, self._item(), first_time=False)

    def _handle_date(self, text: str, item: dict, first_time: bool) -> str:
        kind, when = parse_when(text, self.today)
        self._decide(f"Date in the reply: {kind}" + (f" ({when.isoformat()})" if when else ""), date_kind=kind)
        if kind == "date":
            return self._propose("complete", item, when.isoformat(),
                                 f"Shall I record {item['title']} as done on {when.isoformat()}? (yes/no)")
        if kind == "future":
            self.skipped.add(item["item_id"])
            return self._next(f"That sounds like it's still coming up, so I've left {item['title']} open "
                              f"for now. I can only record it once it's done.")
        self.pending = {"item_id": item["item_id"]}
        self.phase = "ask_date"
        if kind == "none" and first_time:
            self.unclear = 0
            return ("Great! When did you do it? You can say today, yesterday, last Monday, "
                    "or a date like 2025-04-14.")
        return self._unclear("Sorry, I couldn't tell which date you mean. Could you give it like 2025-04-14 "
                             "or 14 April, or say today or yesterday?")

    def _on_confirm(self, text: str) -> str:
        intent = classify(text)
        if intent == "stop":
            return self._end("done", "employee stopped", "Okay, nothing was recorded. Come back any time.")
        if intent == "no":
            self._decide("Proposal declined: nothing written", proposal=self.pending)
            item_id = self.pending["item_id"]
            self.pending = None
            if self.task.focus_item == item_id:
                self.skipped.add(item_id)
            return self._next("Okay, I won't record it. Nothing was changed.")
        if intent != "yes":
            return self._unclear("Just to be safe, please answer yes or no.")
        return self._write()

    # ------------------------------------------------------------ the write

    def _propose(self, action: str, item: dict, on_date: str | None, question: str) -> str:
        self.pending = {"item_id": item["item_id"], "action": action, "on_date": on_date}
        self.phase, self.unclear = "confirm", 0
        self.task.log("propose", f"Proposed: {action} {item['item_id']}" + (f" on {on_date}" if on_date else ""),
                      **self.pending)
        return question

    def _propose_undo(self) -> str:
        if not self.last_change:
            return "There's nothing from this conversation to undo."
        c = self.last_change
        title = self.care["items"][c["item_id"]]["title"]
        self.pending = {"item_id": c["item_id"], "action": "undo", "on_date": None}
        self.phase, self.unclear = "confirm", 0
        self.task.log("propose", f"Proposed: undo {c['item_id']}", **self.pending)
        return f"Shall I undo the change to {title}? (yes/no)"

    def _write(self) -> str:
        p, self.pending = self.pending, None
        title = self.care["items"][p["item_id"]]["title"]
        self.task.log("act", f"Attempted: {p['action']} {p['item_id']} (employee confirmed)",
                      tool="record_checklist_progress", **p)
        result = call_tool("record_checklist_progress", self.ctx, allowed_level="write_self", confirmed=True,
                           actor=self.actor, employee_id=self.actor, item_id=p["item_id"],
                           action=p["action"], on_date=p["on_date"])
        if not result.ok:                                   # refused or failed: say so, never retry
            self.task.errors.append(result.error)
            self.task.log("act", f"Not written: {result.error}", tool="record_checklist_progress")
            self.skipped.add(p["item_id"])
            return self._next(f"I couldn't record that: {result.error}. Nothing was changed.")
        self.task.log("act", "Succeeded: the tool reported no error", tool="record_checklist_progress",
                      before=result.data["before"], after=result.data["after"])

        # VERIFY: read the record back, don't just trust the tool's answer.
        mine = _tool(self.task, self.ctx, "read_my_progress", actor=self.actor, employee_id=self.actor)
        expected = result.data["after"]
        if mine is None or mine["completed"].get(p["item_id"]) != expected["completed"] or \
                (p["item_id"] in mine["not_applicable"]) != expected["not_applicable"]:
            self.task.log("verify", "Verification FAILED: the stored record does not match", expected=expected)
            return self._end("stopped", "write could not be verified",
                             "Something went wrong: I can't confirm the change was saved. Please let HR know; "
                             "I've stopped so nothing else changes.")
        self.task.log("verify", "Verified: the stored record matches the change")
        self.last_change = result.data

        said = {"complete": f"Recorded and checked: {title}, done on {p['on_date']}.",
                "not_applicable": f"Recorded and checked: {title} is marked as not applicable to you.",
                "undo": f"Undone and checked: {title} is open again."}[p["action"]]
        if p["action"] == "undo":
            self.skipped.add(p["item_id"])           # don't ask straight back about what was just undone
        # Refresh the plan: the record changed, so what comes next may change too (Observe -> Adapt).
        self.care = _tool(self.task, self.ctx, "read_settling_in_plan", actor=self.actor, employee_id=self.actor)
        if self.care is None:
            return self._end("stopped", "plan could not be re-read", said + " I couldn't load the next step, so let's stop here.")
        return self._next(said + " (Say \"undo\" if that's not right.)")

    # ------------------------------------------------------------ helpers

    def _item(self) -> dict:
        item_id = (self.pending or {}).get("item_id") or self.task.focus_item
        return self.care["items"][item_id]

    def _decide(self, what: str, **data) -> None:
        self.task.log("decide", what, **data)

    def _next(self, prefix: str) -> str:
        self.phase, self.unclear, self.pending = "asked", 0, None
        if self.items_asked >= self.max_items:
            return self._end("done", f"asked about {self.max_items} items; enough for one conversation",
                             prefix + " That's enough for today. Your progress is saved, and we can carry on any time.")
        ask_next(self.task, self.record, self.care, exclude=self.skipped, greet=False)
        if self.task.question:
            self.items_asked += 1
            return prefix + "\n\n" + self.task.question
        self.phase = "ended"
        return prefix + " " + (self.task.message or "")

    def _unclear(self, ask_again: str) -> str:
        self.unclear += 1
        if self.unclear > MAX_UNCLEAR:
            return self._end("stopped", "too many unclear replies",
                             "Let's leave it here for now, so I don't record anything wrong. "
                             "Come back any time and we'll pick it up.")
        return ask_again

    def _end(self, status: str, reason: str, text: str) -> str:
        self.phase, self.pending = "ended", None
        self.task.question, self.task.message = None, text
        self.task.finish(status, reason)
        return text

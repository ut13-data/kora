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

Stage 4 adds memory:
    - the conversation is saved after every reply (sessions.py); next time KORA offers
      "continue where we left off, or start fresh?";
    - "stop" now PAUSES (it can be continued), it doesn't throw the conversation away;
    - while a checklist question is open, the employee can also say
          "what do you remember about me?"   "remember that ..."   "call me ..."
          "set my tone to direct"             "forget M1 / my name / everything"
      Every memory change is proposed first and written only after "yes", like progress.
"""

import re
from dataclasses import dataclass, field

from src.agent import sessions
from src.agent.dates import parse_when
from src.agent.loop import _tool, ask_next, start
from src.agent.memory import TONES, _check_text, describe
from src.agent.task import AgentTask, StepLimit
from src.agent.tools import ToolContext, ToolRefused, call_tool

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


# Stage 4: memory requests. Checked before the checklist reading of a reply.
_MEM_LIST = re.compile(r"\bwhat (?:do|did) you (?:remember|know) about me\b|\bshow (?:me )?my memory\b|"
                       r"\bwhat have you (?:saved|stored|remembered)\b", re.I)
_MEM_NAME = re.compile(r"^\s*(?:please\s+)?call me\s+(.+?)[.!]*\s*$", re.I)
_MEM_TONE = re.compile(r"\b(?:set my tone to|i prefer (?:a |an )?)\s*(neutral|direct|friendly|warm)\b", re.I)
_MEM_ADD = re.compile(r"^\s*(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s*$", re.I)
_MEM_FORGET = re.compile(r"^\s*(?:please\s+)?forget\s+(.+?)[.!]*\s*$", re.I)
_CONTINUE = re.compile(r"\b(?:continue|resume|carry on|go on|left off|yes|yeah|sure|ok|okay)\b", re.I)
_FRESH = re.compile(r"\b(?:fresh|new|start over|restart|scratch|no)\b", re.I)


@dataclass
class ChatSession:
    goal: str
    actor: str
    ctx: ToolContext = field(default_factory=ToolContext)
    max_items: int = MAX_ITEMS

    offer_resume: bool = True

    def __post_init__(self):
        self.pending: dict | None = None        # the proposed change, waiting for "yes"
        self.skipped: set[str] = set()          # "not yet" in this conversation: don't ask again today
        self.turns = 0
        self.unclear = 0
        self.last_change: dict | None = None    # for "undo"
        self.recorded: list[str] = []           # titles recorded in this conversation (for "welcome back")
        self.return_phase: str | None = None    # where to go back to after a memory request
        self.previous: dict | None = None       # an unfinished earlier conversation, if any

        self.task, self.record, self.care = start(self.goal, self.actor, self.ctx, max_steps=CHAT_MAX_STEPS)
        self.today = self.ctx.current_date() if self.care else None
        self.phase = "asked" if self.task.status == "waiting_for_user" else "ended"
        self.items_asked = 1 if self.phase == "asked" else 0
        self.last_question = self.task.question

        if self.phase == "asked" and self.offer_resume:
            self.previous = sessions.latest_open(self.ctx, self.actor)
        if self.previous:
            self.phase = "resume_choice"
            self.task.log("plan", f"Found an unfinished conversation from {self.previous['updated'][:10]}; "
                                  f"offering to continue it or start fresh",
                          previous_task=self.previous["task"]["task_id"])
        elif self.phase == "asked":
            self._save()

    @property
    def finished(self) -> bool:
        return self.phase == "ended"

    def opening(self) -> str:
        if self.phase == "resume_choice":
            return self._resume_prompt()
        return self.task.question or self.task.message or ""

    def _resume_prompt(self, short: bool = False) -> str:
        if short:
            return "Shall we continue from where we left off, or start fresh? (continue / fresh)"
        prev = self.previous
        focus = self.care["items"].get(prev["task"]["focus_item"], {}).get("title", "a checklist item")
        lines = [f"Welcome back, {self.record['first_name']}! Last time ({prev['updated'][:10]})"
                 + (f" you recorded {', '.join(prev['recorded'])}, and" if prev.get("recorded") else "")
                 + f" we stopped at {focus}."]
        if prev["task"]["steps"] and prev["task"]["steps"][-1]["stage"] == "propose":
            lines.append("(Nothing you hadn't confirmed was saved.)")
        lines.append("Shall we continue from there, or start something fresh? (continue / fresh)")
        return "\n".join(lines)

    # ------------------------------------------------------------ one reply

    def reply(self, text: str) -> str:
        if self.finished:
            return "This conversation has ended. Start a new one any time."
        self.turns += 1
        try:
            self.task.resume()
            self.task.log("observe", f"Employee replied: \"{text[:200]}\"", by="employee")
            if self.turns > MAX_TURNS:
                return self._end("paused", "turn limit reached",
                                 "We've covered a lot. Let's pause here; your progress so far is saved.")
            if self.phase in ("asked", "resume_choice"):
                answer = self._memory_request(text)
                if answer is not None:
                    return answer
            handler = {"asked": self._on_asked, "ask_date": self._on_date, "confirm": self._on_confirm,
                       "resume_choice": self._on_resume_choice}[self.phase]
            return handler(text)
        except StepLimit:
            self.phase = "ended"
            self.task.finish("stopped", "step limit reached; stopped safely")
            return "Let's pause here. Anything already recorded is saved."
        finally:
            if self.phase not in ("ended", "resume_choice") and self.task.status == "running":
                # Mid-conversation (asking for a date, or waiting for "yes"): the task is waiting on the
                # employee. Saying so means a closed window can still be continued next time.
                self.task.finish("waiting_for_user", "waiting for the employee's reply")
            if self.phase != "resume_choice":
                self._save()

    # ------------------------------------------------------------ phases

    def _on_asked(self, text: str) -> str:
        intent = classify(text)
        item = self._item()
        if intent == "unclear" and parse_when(text, self.today)[0] in ("date", "future"):
            intent = "yes"                                    # "yesterday" on its own means "yes, yesterday"
        self._decide(f"Understood the reply as '{intent}'", intent=intent)
        if intent == "stop":
            return self._pause()
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
            return self._pause()
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
            return self._pause()
        if self.pending.get("kind") == "memory":
            if intent == "no":
                self._decide("Memory proposal declined: nothing saved", proposal=self.pending)
                return self._back("Okay, I won't save that.")
            if intent != "yes":
                return self._unclear("Just to be safe, please answer yes or no.")
            return self._write_memory()
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
        if p["action"] == "undo":
            self.recorded = [t for t in self.recorded if t != title]
        else:
            self.recorded.append(title)

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
            self.last_question = self.task.question
            return prefix + "\n\n" + self.task.question
        self.phase = "ended"
        return prefix + " " + (self.task.message or "")

    def _unclear(self, ask_again: str) -> str:
        self.unclear += 1
        if self.unclear > MAX_UNCLEAR:
            return self._end("paused", "too many unclear replies",
                             "Let's leave it here for now, so I don't record anything wrong. "
                             "Come back any time and we'll pick it up.")
        return ask_again

    def _end(self, status: str, reason: str, text: str) -> str:
        self.phase, self.pending = "ended", None
        self.task.question, self.task.message = None, text
        self.task.finish(status, reason)
        return text

    def _pause(self) -> str:
        return self._end("paused", "employee stopped",
                         "Okay, let's stop here. I'll remember where we were, so we can pick it up next time.")

    def _save(self) -> None:
        """Task history: write this conversation down after every reply (Stage 4)."""
        snapshot = {"goal": self.goal, "actor": self.actor, "phase": self.phase,
                    "skipped": sorted(self.skipped), "recorded": self.recorded,
                    "last_change": self.last_change, "task": self.task.to_dict()}
        try:
            sessions.save(self.ctx, self.actor, snapshot)
        except (OSError, ValueError) as e:     # saving history must never break the conversation
            self.task.errors.append(f"could not save the conversation: {e}")

    # ------------------------------------------------------------ continue or fresh (Stage 4)

    def _on_resume_choice(self, text: str) -> str:
        if _STOP.search(text):
            self.phase = "ended"
            self.task.finish("stopped", "employee left before choosing")
            return "Okay. Your earlier conversation is still saved for next time."
        if _FRESH.search(text):
            sessions.close(self.ctx, self.actor, self.previous, "employee started fresh")
            self._decide("Employee chose to start fresh; the earlier conversation is closed")
            self.previous, self.phase = None, "asked"
            return ("Okay, starting fresh. (Anything you recorded before is still saved.)\n\n" + self.last_question)
        if _CONTINUE.search(text):
            return self._continue_previous()
        return self._unclear(self._resume_prompt(short=True))

    def _continue_previous(self) -> str:
        prev, self.previous = self.previous, None
        self.task = AgentTask.from_dict(prev["task"])
        self.task.max_steps = len(self.task.steps) + CHAT_MAX_STEPS     # a new day: a new step budget
        self.task.resume()
        self.skipped, self.recorded = set(prev.get("skipped", [])), list(prev.get("recorded", []))
        self.last_change, self.items_asked, self.turns, self.unclear = prev.get("last_change"), 0, 1, 0
        self.task.log("plan", "Continued this conversation on a new visit; an unconfirmed proposal is never "
                              "carried over, and the plan is re-read because things may have changed")
        # OBSERVE again: progress may have changed since (another chat, HR, an undo).
        self.care = _tool(self.task, self.ctx, "read_settling_in_plan", actor=self.actor, employee_id=self.actor)
        if self.care is None:
            return self._end("stopped", "plan could not be re-read", "I couldn't load your checklist, so let's stop here.")
        return self._next("Great, let's carry on.")

    # ------------------------------------------------------------ personal memory (Stage 4)

    def _memory_request(self, text: str) -> str | None:
        """None: not a memory request, read the reply as a checklist answer instead."""
        if _MEM_LIST.search(text):
            self._decide("Memory request: show what is stored")
            mem = _tool(self.task, self.ctx, "read_my_memory", actor=self.actor, employee_id=self.actor)
            if mem is None:                       # a read failure stops the task in _tool; say so plainly
                return self.task.message
            return self._back(describe(mem), keep_phase=True)
        if m := _MEM_NAME.match(text):
            return self._propose_memory("set_name", m[1].strip(), f"Shall I call you {m[1].strip()} from now on? (yes/no)")
        if m := _MEM_TONE.search(text):
            tone = m[1].lower()
            return self._propose_memory("set_tone", tone, f"Shall I save that you prefer a {tone} tone? (yes/no) "
                                                          f"It also changes the tone guidance in My week.")
        if m := _MEM_FORGET.match(text):
            return self._propose_forget(m[1].strip())
        if m := _MEM_ADD.match(text):
            note = m[1].strip().rstrip(".!")
            return self._propose_memory("add_note", note, f"Shall I remember this: \"{note}\"? (yes/no)")
        return None

    def _propose_memory(self, action: str, value: str, question: str) -> str:
        if action in ("add_note", "set_name"):
            try:
                _check_text(value, "note")            # a friendly early check; the tool checks again
            except ToolRefused as e:
                self._decide(f"Memory request refused before proposing: {e}")
                return self._back(f"Sorry, {e}.", keep_phase=True)
        self.return_phase = self.phase
        self.pending = {"kind": "memory", "action": action, "value": value}
        self.phase, self.unclear = "confirm", 0
        self.task.log("propose", f"Proposed: memory {action}", action=action)    # the words stay out of the log
        return question

    def _propose_forget(self, what: str) -> str:
        w = what.lower()
        if w in ("everything", "all", "everything about me", "all of it"):
            return self._propose_memory("forget", "all", "Shall I forget everything you've told me about yourself "
                                        "(name, tone and notes)? Your checklist progress stays. (yes/no)")
        if w in ("my name", "name"):
            return self._propose_memory("forget", "name", "Shall I forget the name you asked me to use? (yes/no)")
        if w in ("my tone", "tone"):
            return self._propose_memory("forget", "tone", "Shall I forget your tone preference? (yes/no)")
        mem = _tool(self.task, self.ctx, "read_my_memory", actor=self.actor, employee_id=self.actor)
        if mem is None:
            return self.task.message
        notes = mem["notes"]
        hits = [n for n in notes if n["id"].lower() == w] or [n for n in notes if w in n["text"].lower()]
        if len(hits) == 1:
            n = hits[0]
            return self._propose_memory("forget", n["id"], f"Shall I forget {n['id']}: \"{n['text']}\"? (yes/no)")
        return self._back("I'm not sure which one you mean. " + describe(mem), keep_phase=True)

    def _write_memory(self) -> str:
        p, self.pending = self.pending, None
        self.task.log("act", f"Attempted: memory {p['action']} (employee confirmed)", tool="update_my_memory")
        result = call_tool("update_my_memory", self.ctx, allowed_level="write_self", confirmed=True,
                           actor=self.actor, employee_id=self.actor, action=p["action"], value=p["value"])
        if not result.ok:
            self.task.errors.append(result.error)
            self.task.log("act", f"Not saved: {result.error}", tool="update_my_memory")
            return self._back(f"I couldn't save that: {result.error}. Nothing was changed.")
        self.task.log("act", "Succeeded: the tool reported no error", tool="update_my_memory")

        mem = _tool(self.task, self.ctx, "read_my_memory", actor=self.actor, employee_id=self.actor)
        if mem is None or not _memory_matches(p, result.data, mem):
            self.task.log("verify", "Verification FAILED: memory does not show the change")
            return self._end("stopped", "memory change could not be verified",
                             "Something went wrong: I can't confirm that change was saved, so I've stopped.")
        self.task.log("verify", "Verified: memory shows the change")

        if p["action"] == "set_name":
            self.record["first_name"] = p["value"]
        elif p["action"] == "forget" and p["value"] in ("name", "all"):
            self.record["first_name"] = self.record.get("real_first_name", self.record["first_name"])
        said = {"add_note": f"Got it, I'll remember that ({result.data.get('note_id')}).",
                "set_name": f"Nice to meet you, {p['value']}! I'll call you that from now on.",
                "set_tone": f"Saved: you prefer a {p['value']} tone.",
                "forget": "Done, I've forgotten that."}[p["action"]]
        return self._back(said)

    def _back(self, said: str, keep_phase: bool = False) -> str:
        """After a memory request, go back to exactly where the conversation was."""
        if not keep_phase:
            self.phase, self.return_phase = self.return_phase or "asked", None
        self.unclear = 0
        if self.phase == "resume_choice":
            return said + "\n\n" + self._resume_prompt(short=True)
        item = self.care["items"].get(self.task.focus_item) if self.care else None
        return said + ("\n\nBack to where we were: have you sorted out " + item["title"] + " yet?" if item else "")


def _memory_matches(p: dict, result: dict, mem: dict) -> bool:
    """VERIFY a memory change by reading memory back."""
    if p["action"] == "set_name":
        return (mem["preferred_name"] or {}).get("value") == p["value"]
    if p["action"] == "set_tone":
        return (mem["preferred_tone"] or {}).get("value") == TONES[p["value"]]
    if p["action"] == "add_note":
        return any(n["id"] == result["note_id"] for n in mem["notes"])
    target = p["value"]
    if target == "all":
        return not (mem["preferred_name"] or mem["preferred_tone"] or mem["notes"])
    if target == "name":
        return mem["preferred_name"] is None
    if target == "tone":
        return mem["preferred_tone"] is None
    return all(n["id"] != target for n in mem["notes"])

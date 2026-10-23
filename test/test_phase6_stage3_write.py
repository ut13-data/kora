"""Phase 6 Stage 3: dates, the write tool, confirmation, verification and the conversation."""

import json
import shutil
from datetime import date

import pytest

from src.agent import chat as chat_module
from src.agent import tools
from src.agent.chat import ChatSession, classify
from src.agent.dates import parse_when
from src.agent.tools import ToolContext, call_tool

REAL = ToolContext()
TODAY = date(2025, 4, 22)          # the demo date (a Tuesday)


@pytest.fixture
def ctx(tmp_path):
    progress = tmp_path / "checklist_progress.json"
    shutil.copy(REAL.progress_path, progress)
    return ToolContext(REAL.raw_dir, REAL.intelligence_dir, progress, tmp_path / "history.jsonl")


def _stored(ctx, emp="EMP-0085"):
    return json.loads(ctx.progress_path.read_text(encoding="utf-8"))["employees"].get(emp, {})


def _history(ctx):
    if not ctx.history_path.exists():
        return []
    return [json.loads(line) for line in ctx.history_path.read_text(encoding="utf-8").splitlines()]


def _talk(ctx, *replies, emp="EMP-0085"):
    s = ChatSession("help me get settled", emp, ctx)
    out = [s.opening()] + [s.reply(r) for r in replies]
    return s, out


# ---------------------------------------------------------------- dates


@pytest.mark.parametrize("text, expected", [
    ("yesterday", ("date", date(2025, 4, 21))),
    ("yes, yesterday", ("date", date(2025, 4, 21))),
    ("day before yesterday", ("date", date(2025, 4, 20))),
    ("today", ("date", TODAY)),
    ("last Monday", ("date", date(2025, 4, 21))),
    ("on friday", ("date", date(2025, 4, 18))),
    ("on tuesday", ("date", date(2025, 4, 15))),          # never "today": the most recent one BEFORE today
    ("3 days ago", ("date", date(2025, 4, 19))),
    ("a week ago", ("date", date(2025, 4, 15))),
    ("14 April", ("date", date(2025, 4, 14))),
    ("April 14th, 2025", ("date", date(2025, 4, 14))),
    ("14.04.2025", ("date", date(2025, 4, 14))),
    ("2025-04-14", ("date", date(2025, 4, 14))),
    ("on the 14th", ("date", date(2025, 4, 14))),
    ("tomorrow", ("future", date(2025, 4, 23))),
    ("30 April", ("future", date(2025, 4, 30))),
    ("next monday", ("future", None)),
    ("last week", ("unclear", None)),
    ("recently", ("unclear", None)),
    ("04/05/2025", ("unclear", None)),                     # 4 May or 5 April? ask, never guess
    ("2025-02-30", ("unclear", None)),
    ("the 25th", ("unclear", None)),
    ("yes", ("none", None)),
])
def test_parse_when(text, expected):
    assert parse_when(text, TODAY) == expected


@pytest.mark.parametrize("text, intent", [
    ("yes", "yes"), ("yeah done", "yes"), ("not yet", "no"), ("I have not", "no"), ("nope", "no"),
    ("I'll do it later", "no"), ("not applicable", "not_applicable"), ("doesn't apply to me", "not_applicable"),
    ("stop", "stop"), ("that's all", "stop"), ("undo", "undo"), ("purple", "unclear"),
])
def test_classify(text, intent):
    assert classify(text) == intent


# ---------------------------------------------------------------- the write tool on its own


def test_write_needs_confirmation(ctx):
    r = call_tool("record_checklist_progress", ctx, allowed_level="write_self",
                  actor="EMP-0085", employee_id="EMP-0085", item_id="GER-HOUSING",
                  action="complete", on_date="2025-04-20")
    assert not r.ok and "confirmation" in r.error
    assert _stored(ctx) == {}


def test_write_refused_at_read_level_even_if_confirmed(ctx):
    r = call_tool("record_checklist_progress", ctx, allowed_level="read", confirmed=True,
                  actor="EMP-0085", employee_id="EMP-0085", item_id="GER-HOUSING",
                  action="complete", on_date="2025-04-20")
    assert not r.ok and "needs 'write_self'" in r.error


def _write(ctx, **kw):
    args = dict(actor="EMP-0085", employee_id="EMP-0085", item_id="GER-HOUSING",
                action="complete", on_date="2025-04-20")
    args.update(kw)
    return call_tool("record_checklist_progress", ctx, allowed_level="write_self", confirmed=True, **args)


def test_write_records_keeps_notes_and_logs_history(ctx):
    r = _write(ctx)
    assert r.ok and r.data["after"]["completed"] == "2025-04-20"
    raw = json.loads(ctx.progress_path.read_text(encoding="utf-8"))
    assert "_note" in raw and "_example" in raw                     # the file's own notes survive
    assert raw["employees"]["EMP-0085"]["completed"] == {"GER-HOUSING": "2025-04-20"}
    h = _history(ctx)
    assert len(h) == 1 and h[0]["before"]["completed"] is None and h[0]["by"] == "EMP-0085"


@pytest.mark.parametrize("kw, error", [
    ({"employee_id": "EMP-0040"}, "your own"),                       # someone else's record
    ({"item_id": "GER-NOT-AN-ITEM"}, "not an item"),
    ({"action": "delete"}, "unknown action"),
    ({"on_date": "2025-04-23"}, "future"),                           # one day after the demo date
    ({"on_date": "last week"}, "not a date"),
    ({"on_date": "2020-01-01"}, "more than a year before"),
])
def test_write_validation(ctx, kw, error):
    r = _write(ctx, **kw)
    assert not r.ok and error in r.error
    assert _stored(ctx) == {} and _history(ctx) == []


def test_not_applicable_and_undo(ctx):
    assert _write(ctx, action="not_applicable").ok
    assert _stored(ctx)["not_applicable"] == ["GER-HOUSING"]
    assert _write(ctx, action="undo").ok
    assert _stored(ctx) == {"completed": {}, "not_applicable": []}
    assert [h["action"] for h in _history(ctx)] == ["not_applicable", "undo"]


def test_only_one_write_tool_and_nothing_external():
    levels = [level for _, level in tools.TOOLS.values()]
    assert levels.count("write_self") == 1 and "external" not in levels


# ---------------------------------------------------------------- conversations


def test_yes_then_date_then_confirm_records_and_verifies(ctx):
    s, out = _talk(ctx, "yes", "yesterday", "yes")
    assert "When did you do it?" in out[1]
    assert "Shall I record Housing as done on 2025-04-21?" in out[2]
    assert "Recorded and checked: Housing, done on 2025-04-21" in out[3]
    assert "Address Registration" in out[3]                         # the plan adapted: next unblocked item
    assert _stored(ctx)["completed"] == {"GER-HOUSING": "2025-04-21"}
    whats = [st.what for st in s.task.steps]
    order = [next(i for i, w in enumerate(whats) if w.startswith(p))
             for p in ("Proposed:", "Attempted:", "Succeeded:", "Verified:")]
    assert order == sorted(order)                                   # proposed -> attempted -> succeeded -> verified


def test_date_alone_counts_as_yes(ctx):
    _, out = _talk(ctx, "last monday")
    assert "Shall I record Housing as done on 2025-04-21?" in out[1]


def test_nothing_written_until_the_employee_confirms(ctx):
    _talk(ctx, "yes", "yesterday")                                  # proposed, never confirmed
    assert _stored(ctx) == {} and _history(ctx) == []


def test_declining_the_proposal_writes_nothing(ctx):
    _, out = _talk(ctx, "yes", "yesterday", "no")
    assert "Nothing was changed" in out[3]
    assert _stored(ctx) == {}


def test_tomorrow_is_not_done(ctx):
    _, out = _talk(ctx, "yes, tomorrow")
    assert "still coming up" in out[1] and "Have you sorted out" in out[1]
    assert _stored(ctx) == {}


def test_unclear_date_is_asked_politely_then_stops(ctx):
    s, out = _talk(ctx, "yes", "last week", "recently", "sometime")
    assert "couldn't tell which date" in out[2]
    assert s.finished and "so I don't record anything wrong" in out[4]
    assert _stored(ctx) == {}


def test_not_yet_moves_on_and_does_not_ask_again(ctx):
    s, out = _talk(ctx, "not yet")
    assert "left Housing open" in out[1] and "Have you sorted out Housing" not in out[1]
    assert "GER-HOUSING" in s.skipped and _stored(ctx) == {}


def test_not_applicable_needs_confirmation(ctx):
    _, out = _talk(ctx, "doesn't apply to me", "yes")
    assert "not applicable to you? (yes/no)" in out[1]
    assert _stored(ctx)["not_applicable"] == ["GER-HOUSING"]


def test_undo_last_change(ctx):
    _, out = _talk(ctx, "yes", "today", "yes", "undo", "yes")
    assert "Shall I undo the change to Housing?" in out[4]
    assert "Undone and checked" in out[5]
    assert _stored(ctx)["completed"] == {}
    assert [h["action"] for h in _history(ctx)] == ["complete", "undo"]


def test_stop_any_time(ctx):
    s, out = _talk(ctx, "yes", "stop")
    assert s.finished and s.task.status == "done" and _stored(ctx) == {}


def test_at_most_three_items_per_conversation(ctx):
    s, out = _talk(ctx, "not yet", "not yet", "not yet")
    assert s.finished and "enough for today" in out[3]
    assert s.task.question is None


def test_tool_refusal_is_reported_and_not_retried(ctx, monkeypatch):
    monkeypatch.setattr(ChatSession, "_handle_date", lambda self, text, item, first_time:
                        self._propose("complete", item, "2025-04-23", "Shall I record it? (yes/no)"))
    s, out = _talk(ctx, "yes", "yes")                               # a future date slips past the chat
    assert "I couldn't record that" in out[2] and "future" in out[2]
    attempts = [st for st in s.task.steps if st.what.startswith("Attempted")]
    assert len(attempts) == 1 and _stored(ctx) == {}


def test_failed_verification_stops_everything(ctx, monkeypatch):
    real = chat_module._tool

    def stale(task, c, name, **args):
        data = real(task, c, name, **args)
        return {"completed": {}, "not_applicable": []} if name == "read_my_progress" else data

    monkeypatch.setattr(chat_module, "_tool", stale)
    s, out = _talk(ctx, "yes", "today", "yes")
    assert s.finished and s.task.status == "stopped" and "can't confirm the change was saved" in out[3]


def test_not_eligible_employee_cannot_start_a_conversation(ctx):
    s, out = _talk(ctx, emp="EMP-0040")
    assert s.finished and "doesn't apply" in out[0]
    assert s.reply("yes").startswith("This conversation has ended")


def test_two_employees_never_touch_each_others_progress(ctx):
    _talk(ctx, "yes", "today", "yes", emp="EMP-0085")
    _talk(ctx, "yes", "yesterday", "yes", emp="EMP-0023")
    raw = json.loads(ctx.progress_path.read_text(encoding="utf-8"))["employees"]
    assert raw["EMP-0085"]["completed"] == {"GER-HOUSING": "2025-04-22"}
    assert raw["EMP-0023"]["completed"] == {"GER-HOUSING": "2025-04-21"}

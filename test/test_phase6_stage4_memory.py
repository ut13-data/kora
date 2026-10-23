"""Phase 6 Stage 4: personal memory and task history (continue / start fresh)."""

import json
import shutil

import pytest

from src.agent import sessions
from src.agent.chat import ChatSession
from src.agent.memory import describe
from src.agent.tools import ToolContext, call_tool

REAL = ToolContext()


@pytest.fixture
def ctx(tmp_path):
    """Everything KORA stores about people goes to a temporary folder (all paths follow progress_path)."""
    progress = tmp_path / "checklist_progress.json"
    shutil.copy(REAL.progress_path, progress)
    shutil.copy(REAL.prefs_path, tmp_path / "communication_preferences.json")
    return ToolContext(REAL.raw_dir, REAL.intelligence_dir, progress, tmp_path / "history.jsonl")


def _mem(ctx, emp="EMP-0085"):
    return call_tool("read_my_memory", ctx, actor=emp, employee_id=emp).data


def _update(ctx, action, value=None, emp="EMP-0085", confirmed=True):
    return call_tool("update_my_memory", ctx, allowed_level="write_self", confirmed=confirmed,
                     actor=emp, employee_id=emp, action=action, value=value)


def _chat(ctx, *replies, emp="EMP-0085"):
    s = ChatSession("help me get settled", emp, ctx)
    return s, [s.opening()] + [s.reply(r) for r in replies]


# ---------------------------------------------------------------- the memory tool


def test_memory_starts_empty_and_lives_in_the_context_folder(ctx):
    assert _mem(ctx) == {"preferred_name": None, "preferred_tone": None, "notes": []}
    assert ctx.memory_dir.parent == ctx.progress_path.parent          # never the real data folder in tests


def test_memory_write_needs_confirmation(ctx):
    r = _update(ctx, "add_note", "I like tea", confirmed=False)
    assert not r.ok and "confirmation" in r.error and _mem(ctx)["notes"] == []


def test_note_name_tone_and_forget(ctx):
    assert _update(ctx, "add_note", "I start German classes on Mondays").ok
    assert _update(ctx, "set_name", "Jules").ok
    assert _update(ctx, "set_tone", "friendly").ok
    m = _mem(ctx)
    assert m["notes"][0]["id"] == "M1" and m["preferred_name"]["value"] == "Jules"
    assert m["preferred_tone"]["value"] == "relationship_oriented"     # Phase 3's own tone ID
    assert _update(ctx, "forget", "M1").ok and _mem(ctx)["notes"] == []
    assert _update(ctx, "forget", "all").ok
    assert _mem(ctx) == {"preferred_name": None, "preferred_tone": None, "notes": []}


def test_tone_is_written_where_phase3_reads_it(ctx):
    _update(ctx, "set_tone", "direct")
    prefs = json.loads(ctx.prefs_path.read_text(encoding="utf-8"))
    assert prefs["employees"]["EMP-0085"] == {"preferred_tone": "direct", "set_by": "self",
                                              "updated": prefs["employees"]["EMP-0085"]["updated"]}
    assert "_note" in prefs                                            # the file's own notes survive


@pytest.mark.parametrize("note", [
    "I have a doctor appointment for my anxiety", "my salary is 70k", "my IBAN is DE89 3704 0044 0532 0130 00",
    "I go to the mosque on Fridays", "my password is hunter2", "passport number 1234567890",
])
def test_sensitive_notes_are_refused(ctx, note):
    r = _update(ctx, "add_note", note)
    assert not r.ok and "sensitive" in r.error and _mem(ctx)["notes"] == []


@pytest.mark.parametrize("action, value, error", [
    ("set_name", "J0hn!!", "letters"), ("set_name", "x" * 40, "letters"), ("set_tone", "angry", "neutral, direct"),
    ("add_note", "", "empty"), ("add_note", "word " * 60, "long"), ("forget", "M9", "no note"),
    ("forget", "name", "no name"), ("explode", None, "unknown memory action"),
])
def test_memory_validation(ctx, action, value, error):
    r = _update(ctx, action, value)
    assert not r.ok and error in r.error


def test_note_limit(ctx):
    for i in range(20):
        assert _update(ctx, "add_note", f"note number {i}").ok
    r = _update(ctx, "add_note", "one too many")
    assert not r.ok and "forget one first" in r.error


def test_memory_is_private_per_employee(ctx):
    _update(ctx, "add_note", "I like tea")
    r = call_tool("read_my_memory", ctx, actor="EMP-0040", employee_id="EMP-0085")
    assert not r.ok and "your own" in r.error
    r = call_tool("update_my_memory", ctx, allowed_level="write_self", confirmed=True,
                  actor="EMP-0040", employee_id="EMP-0085", action="forget", value="all")
    assert not r.ok and _mem(ctx)["notes"]
    assert _mem(ctx, "EMP-0023")["notes"] == []


def test_history_records_changes_but_never_the_words(ctx):
    _update(ctx, "add_note", "I start German classes on Mondays")
    _update(ctx, "set_name", "Jules")
    _update(ctx, "forget", "M1")
    history = ctx.memory_history_path.read_text(encoding="utf-8")
    assert [json.loads(line)["action"] for line in history.splitlines()] == ["add_note", "set_name", "forget"]
    assert "German" not in history and "Jules" not in history          # forgetting really forgets


def test_describe_says_when_each_thing_was_said(ctx):
    _update(ctx, "add_note", "I like tea")
    text = describe(_mem(ctx))
    assert "M1: I like tea (you told me on" in text and "forget" in text
    assert "don't have anything saved" in describe({"preferred_name": None, "preferred_tone": None, "notes": []})


# ---------------------------------------------------------------- memory in the conversation


def test_call_me_changes_how_kora_greets(ctx):
    s, out = _chat(ctx, "call me Jules", "yes")
    assert "Shall I call you Jules from now on?" in out[1]
    assert "Nice to meet you, Jules!" in out[2] and "Back to where we were: have you sorted out Housing yet?" in out[2]
    s2 = ChatSession("help me get settled", "EMP-0085", ctx, offer_resume=False)
    assert s2.opening().startswith("Hi Jules!")


def test_remember_needs_yes_and_nothing_is_saved_on_no(ctx):
    _, out = _chat(ctx, "remember that I like tea", "no")
    assert "Shall I remember this" in out[1] and "won't save" in out[2]
    assert _mem(ctx)["notes"] == []


def test_sensitive_request_is_refused_before_proposing(ctx):
    _, out = _chat(ctx, "remember that my salary is 70k")
    assert "sensitive" in out[1] and "Shall I" not in out[1]


def test_what_do_you_remember(ctx):
    _update(ctx, "add_note", "I like tea")
    _, out = _chat(ctx, "what do you remember about me?")
    assert "M1: I like tea" in out[1] and "have you sorted out Housing yet?" in out[1]


def test_forget_by_words_and_everything(ctx):
    _update(ctx, "add_note", "I like tea")
    _update(ctx, "add_note", "I cycle to work")
    _, out = _chat(ctx, "forget cycle", "yes", "forget everything", "yes")
    assert 'forget M2: "I cycle to work"' in out[1]
    assert "Checklist progress stays" in out[3] or "checklist progress stays" in out[3]
    assert _mem(ctx)["notes"] == []


def test_memory_answer_does_not_touch_checklist_progress(ctx):
    before = ctx.progress_path.read_text(encoding="utf-8")
    _chat(ctx, "remember that I like tea", "yes", "call me Jules", "yes")
    assert ctx.progress_path.read_text(encoding="utf-8") == before


def test_broken_memory_file_does_not_stop_kora(ctx):
    ctx.memory_dir.mkdir(parents=True)
    (ctx.memory_dir / "EMP-0085.json").write_text("{ not json", encoding="utf-8")
    s, out = _chat(ctx)
    assert "Have you sorted out Housing yet?" in out[0]
    assert any("continuing without personal memory" in st.what for st in s.task.steps)


# ---------------------------------------------------------------- task history: continue or fresh


def test_conversation_is_saved_after_every_reply(ctx):
    s, _ = _chat(ctx, "not yet")
    saved = json.loads((ctx.tasks_dir / "EMP-0085" / f"{s.task.task_id}.json").read_text(encoding="utf-8"))
    # Address Registration is blocked by Housing, so after "not yet" KORA moves to Public Transport
    assert saved["skipped"] == ["GER-HOUSING"] and saved["task"]["focus_item"] == "GER-PUBLIC-TRANSPORT"


def test_stop_pauses_and_next_visit_offers_to_continue(ctx):
    _chat(ctx, "yes", "yesterday", "yes", "stop")                    # Housing recorded, stopped at Address Reg.
    s, out = _chat(ctx)
    assert s.phase == "resume_choice"
    assert "Welcome back, Julia!" in out[0] and "you recorded Housing" in out[0]
    assert "stopped at Address Registration (Anmeldung)" in out[0] and "(continue / fresh)" in out[0]


def test_continue_picks_up_where_it_stopped(ctx):
    first, _ = _chat(ctx, "not yet", "stop")                         # skipped Housing, stopped at Public Transport
    s, out = _chat(ctx, "continue")
    assert s.task.task_id == first.task.task_id                      # the SAME task, notebook kept
    assert "Have you sorted out Public Transport yet?" in out[1]
    assert "GER-HOUSING" in s.skipped
    assert any("Continued this conversation" in st.what for st in s.task.steps)


def test_fresh_closes_the_old_conversation(ctx):
    first, _ = _chat(ctx, "not yet", "stop")
    s, out = _chat(ctx, "fresh")
    assert "starting fresh" in out[1] and "Have you sorted out Housing yet?" in out[1]
    assert s.task.task_id != first.task.task_id
    s3, _ = _chat(ctx)
    assert s3.phase == "asked" or s3.previous["task"]["task_id"] == s.task.task_id   # the old one is never offered


def test_a_saved_unconfirmed_proposal_is_never_carried_over(ctx):
    _chat(ctx, "yes", "yesterday")                                    # proposal made, window closed (no "yes")
    s, out = _chat(ctx)
    assert "Nothing you hadn't confirmed was saved" in out[0]
    out2 = s.reply("continue")
    assert "Have you sorted out Housing yet?" in out2 and s.pending is None
    assert json.loads(ctx.progress_path.read_text(encoding="utf-8"))["employees"] == {}


def test_continue_re_reads_the_plan(ctx):
    _chat(ctx, "stop")                                                # paused at Housing
    call_tool("record_checklist_progress", ctx, allowed_level="write_self", confirmed=True,   # done elsewhere
              actor="EMP-0085", employee_id="EMP-0085", item_id="GER-HOUSING", action="complete", on_date="2025-04-20")
    s, out = _chat(ctx, "continue")
    assert "Housing yet" not in out[1] and "Address Registration" in out[1]


def test_finished_conversation_is_not_offered(ctx):
    _chat(ctx, "not yet", "not yet", "not yet")                      # 3 items: done for today
    s, out = _chat(ctx)
    assert s.phase == "asked" and "Welcome back" not in out[0]


def test_memory_commands_work_while_choosing(ctx):
    _chat(ctx, "stop")
    s, out = _chat(ctx, "what do you remember about me?", "continue")
    assert "don't have anything saved" in out[1] and "(continue / fresh)" in out[1]
    assert "Have you sorted out Housing yet?" in out[2]


def test_conversations_are_private(ctx):
    _chat(ctx, "stop", emp="EMP-0085")
    s, out = _chat(ctx, emp="EMP-0023")
    assert "Welcome back" not in out[0]
    with pytest.raises(ValueError):
        sessions.latest_open(ctx, "../EMP-0085")                     # a path trick is refused


def test_not_eligible_employee_saves_nothing(ctx):
    _chat(ctx, emp="EMP-0040")
    assert not (ctx.tasks_dir / "EMP-0040").exists()

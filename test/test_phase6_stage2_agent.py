"""Phase 6 Stage 2: the read-only Germany settling-in agent (src/agent/)."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.agent import loop, tools
from src.agent.loop import run
from src.agent.tools import ToolContext, call_tool

REAL = ToolContext()


@pytest.fixture
def ctx(tmp_path):
    """Real company data, but a PRIVATE copy of the progress file, so tests can't touch yours."""
    progress = tmp_path / "checklist_progress.json"
    shutil.copy(REAL.progress_path, progress)
    return ToolContext(REAL.raw_dir, REAL.intelligence_dir, progress)


def _report(ctx: ToolContext, employee_id: str, completed: dict) -> None:
    raw = json.loads(ctx.progress_path.read_text(encoding="utf-8"))
    raw["employees"][employee_id] = {"completed": completed, "not_applicable": []}
    ctx.progress_path.write_text(json.dumps(raw), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- the normal path


def test_relocating_employee_gets_one_question_about_housing(ctx):
    task = run("help me get settled", "EMP-0085", ctx)
    assert task.status == "waiting_for_user"
    assert "Have you sorted out Housing yet?" in task.question
    assert task.question.count("?") == 1                       # ONE question per run
    assert [s.stage for s in task.steps] == ["goal", "state", "state", "state", "plan", "plan", "act", "verify", "stop"]


def test_pace_is_a_company_suggestion_with_its_source_never_a_legal_deadline(ctx):
    _report(ctx, "EMP-0085", {"GER-HOUSING": "2025-04-12"})
    q = run("help me get settled", "EMP-0085", ctx).question
    assert "Address Registration" in q
    assert "DOC-CARE-003" in q and "not a legal deadline" in q
    assert "Check the current official guidance" in q          # the document's own caveat, verbatim


def test_reported_progress_changes_the_next_question(ctx):
    _report(ctx, "EMP-0085", {"GER-HOUSING": "2025-04-12", "GER-ADDRESS-REGISTRATION": "2025-04-14"})
    task = run("help me get settled", "EMP-0085", ctx)
    assert "Housing yet" not in task.question and "Address Registration (Anmeldung) yet" not in task.question
    assert task.steps[4].data["reported_complete"] == ["GER-HOUSING", "GER-ADDRESS-REGISTRATION"]


def test_dependency_never_completes_an_item_by_itself(ctx):
    # Bank Account reported, Housing NOT reported: KORA must still ask about Housing.
    _report(ctx, "EMP-0085", {"GER-BANK-ACCOUNT": "2025-04-15"})
    task = run("help me get settled", "EMP-0085", ctx)
    assert task.steps[4].data["reported_complete"] == ["GER-BANK-ACCOUNT"]
    assert "Have you sorted out Housing yet?" in task.question


def test_everything_reported_means_done(ctx):
    items = json.loads((REAL.intelligence_dir / "germany_checklists.json").read_text(encoding="utf-8"))["items"]
    _report(ctx, "EMP-0085", {i["item_id"]: "2025-04-20" for i in items})
    task = run("help me get settled", "EMP-0085", ctx)
    assert task.status == "done" and task.question is None


def test_agent_writes_nothing(ctx):
    before = _digest(ctx.progress_path)
    real_before = _digest(REAL.progress_path)
    for emp in ("EMP-0085", "EMP-0040", "EMP-0031"):
        run("help me get settled", emp, ctx)
    assert _digest(ctx.progress_path) == before
    assert _digest(REAL.progress_path) == real_before


# ---------------------------------------------------------------- eligibility


def test_not_relocating_employee_is_told_politely_why(ctx):
    task = run("help me with my Germany checklist", "EMP-0040", ctx)
    assert task.status == "stopped" and task.question is None
    assert "Bengaluru, India" in task.message and "doesn't apply" in task.message
    assert "HR can update your record" in task.message


def test_past_program_window(ctx):
    task = run("relocation checklist", "EMP-0031", ctx)
    assert task.status == "stopped" and "90 days" in task.message


# ---------------------------------------------------------------- stopping and limits


def test_unsupported_goal_stops_without_reading_anything(ctx):
    task = run("what's the weather", "EMP-0085", ctx)
    assert task.status == "stopped" and len(task.steps) == 1
    assert not any(s.tool for s in task.steps)


def test_step_limit_stops_safely(ctx):
    task = run("help me get settled", "EMP-0085", ctx, max_steps=3)
    assert task.status == "stopped" and "step limit" in task.stop_reason
    assert len(task.steps) == 3 and task.question is None


def test_failed_tool_stops_and_is_not_retried(tmp_path):
    broken = ToolContext(REAL.raw_dir, tmp_path / "no_such_dir", tmp_path / "p.json")
    task = run("help me get settled", "EMP-0085", broken)
    assert task.status == "stopped" and "not retried" in task.stop_reason
    assert sum(1 for s in task.steps if s.tool == "read_settling_in_plan") == 1
    assert len(task.errors) == 1


def test_failed_verification_hides_the_question(ctx, monkeypatch):
    original = loop._compose_question

    def tampered(*args):
        text, claims = original(*args)
        claims.append({"text": "You must register within 3 days by law.", "field": "official_guidance_note"})
        return text, claims

    monkeypatch.setattr(loop, "_compose_question", tampered)
    task = run("help me get settled", "EMP-0085", ctx)
    assert task.status == "stopped" and task.question is None and "verification failed" in task.stop_reason


# ---------------------------------------------------------------- tools: inputs and permissions


def test_tool_refuses_someone_elses_record(ctx):
    r = call_tool("read_settling_in_plan", ctx, actor="EMP-0040", employee_id="EMP-0085")
    assert not r.ok and "your own" in r.error


@pytest.mark.parametrize("bad", ["EMP-85", "emp-0085", "EMP-0085; rm -rf /", "", None])
def test_tool_refuses_bad_employee_ids(ctx, bad):
    r = call_tool("read_employee_record", ctx, actor=bad, employee_id=bad)
    assert not r.ok and "not a valid employee ID" in r.error


def test_unknown_employee(ctx):
    r = call_tool("read_employee_record", ctx, actor="EMP-9999", employee_id="EMP-9999")
    assert not r.ok and "not in the company records" in r.error


def test_unknown_tool_and_bad_arguments_are_refused(ctx):
    assert "unknown tool" in call_tool("delete_everything", ctx).error
    assert "bad arguments" in call_tool("read_employee_record", ctx, actor="EMP-0085").error


def test_more_powerful_tool_is_refused_at_read_level(ctx, monkeypatch):
    monkeypatch.setitem(tools.TOOLS, "pretend_write", (lambda c, **a: {"written": True}, "write_self"))
    r = call_tool("pretend_write", ctx, allowed_level="read")
    assert not r.ok and "needs 'write_self' permission" in r.error


def test_the_loop_itself_only_reads():
    # Stage 3 added one write tool, but the Stage 2 loop calls tools with allowed_level="read" only.
    import inspect
    from src.agent import loop as loop_module
    assert 'allowed_level="read"' in inspect.getsource(loop_module._tool)
    assert "write_self" not in inspect.getsource(loop_module)
    assert "confirmed=True" not in inspect.getsource(loop_module)

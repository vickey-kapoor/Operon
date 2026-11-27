"""KAN-207: OPERON_FRESH_LOOK in the loop's execute attempt, with no screen and no server."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from PIL import Image, ImageDraw

from operon.agent.loop import AgentLoop
from operon.models.capture import CaptureFrame
from operon.models.common import FailureCategory
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction, PolicyDecision


def _save(path: Path, popup: bool = False) -> str:
    img = Image.new("RGB", (400, 300), (40, 40, 40))
    if popup:
        ImageDraw.Draw(img).rectangle((150, 100, 260, 200), fill=(240, 240, 240))
    img.save(path)
    return str(path)


def _frame(path: str, left: int = 0, top: int = 0) -> CaptureFrame:
    return CaptureFrame(artifact_path=path, width=400, height=300, monitor_left=left, monitor_top=top)


def _setup(tmp_path, monkeypatch, *, flag: bool, action: AgentAction, peek, left: int = 0):
    monkeypatch.setenv("OPERON_FRESH_LOOK", "on" if flag else "off")
    before = _save(tmp_path / "before.png")
    execute = AsyncMock(return_value=ExecutedAction(action=action, success=True, detail="ok"))
    executor = SimpleNamespace(execute=execute)
    if peek is not None:
        executor.peek_frame = peek
    hints = Mock()
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=SimpleNamespace(add_advisory_hints=hints),
        run_store=Mock(),
        policy_service=Mock(),
        executor=executor,
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )
    loop.executor_adapter = SimpleNamespace(execute=execute)
    loop._persist_execution_trace = lambda run_id, step, ex: ex  # type: ignore[method-assign]
    ctx = SimpleNamespace(
        run_id="r1",
        step_index=1,
        record=SimpleNamespace(run_id="r1"),
        state=SimpleNamespace(),
        frame=_frame(before, left),
        record_stage=lambda name, t0: 0.0,
        after_artifact_path=str(tmp_path / "after.png"),
        decision=PolicyDecision(action=action, rationale="r", confidence=0.9, active_subgoal="s"),
        executed_action=None,
        after_captured=False,
    )
    return loop, ctx, execute, hints


def _peek_returning(tmp_path, popup: bool):
    path = _save(tmp_path / "peek.png", popup=popup)
    return AsyncMock(return_value=_frame(path)), path


CLICK = AgentAction(action_type=ActionType.CLICK, x=200, y=150)


async def test_flag_off_never_peeks_and_runs(tmp_path, monkeypatch) -> None:
    peek, _ = _peek_returning(tmp_path, popup=True)
    loop, ctx, execute, _ = _setup(tmp_path, monkeypatch, flag=False, action=CLICK, peek=peek)
    await loop._execute_attempt(ctx)
    peek.assert_not_awaited()
    execute.assert_awaited_once()
    assert ctx.executed_action.success


async def test_same_frame_runs_the_action_and_deletes_the_peek(tmp_path, monkeypatch) -> None:
    peek, path = _peek_returning(tmp_path, popup=False)
    loop, ctx, execute, hints = _setup(tmp_path, monkeypatch, flag=True, action=CLICK, peek=peek)
    await loop._execute_attempt(ctx)
    peek.assert_awaited_once()
    execute.assert_awaited_once()
    hints.assert_not_called()
    assert not Path(path).exists()


async def test_changed_spot_skips_the_action_and_tells_the_model(tmp_path, monkeypatch) -> None:
    peek, _ = _peek_returning(tmp_path, popup=True)
    loop, ctx, execute, hints = _setup(tmp_path, monkeypatch, flag=True, action=CLICK, peek=peek)
    await loop._execute_attempt(ctx)
    execute.assert_not_awaited()
    done = ctx.executed_action
    assert done.success is False
    assert done.failure_category is FailureCategory.STALE_TARGET_BEFORE_ACTION
    assert done.detail.startswith("screen_changed:")
    assert done.artifact_path is None
    assert ctx.after_captured is False
    assert hints.call_args.kwargs["source"] == "fresh_look"
    assert "look again" in hints.call_args.args[0][0]


async def test_monitor_origin_is_subtracted(tmp_path, monkeypatch) -> None:
    peek, _ = _peek_returning(tmp_path, popup=True)
    action = AgentAction(action_type=ActionType.CLICK, x=1400, y=150)
    loop, ctx, execute, _ = _setup(tmp_path, monkeypatch, flag=True, action=action, peek=peek, left=1200)
    await loop._execute_attempt(ctx)
    execute.assert_not_awaited()


async def test_peek_returning_none_runs_the_action(tmp_path, monkeypatch) -> None:
    loop, ctx, execute, _ = _setup(tmp_path, monkeypatch, flag=True, action=CLICK, peek=AsyncMock(return_value=None))
    await loop._execute_attempt(ctx)
    execute.assert_awaited_once()
    assert ctx.executed_action.success


async def test_peek_error_is_counted_and_the_action_runs(tmp_path, monkeypatch) -> None:
    peek = AsyncMock(side_effect=RuntimeError("boom"))
    loop, ctx, execute, _ = _setup(tmp_path, monkeypatch, flag=True, action=CLICK, peek=peek)
    await loop._execute_attempt(ctx)
    execute.assert_awaited_once()
    assert ctx.executed_action.success
    assert loop._recovered_errors["r1"]


async def test_an_approved_action_runs_without_a_peek(tmp_path, monkeypatch) -> None:
    """A person approved this action at a pause (e.g. coordinate thrash): the fresh look must not skip it again."""
    peek, _ = _peek_returning(tmp_path, popup=True)
    loop, ctx, execute, _ = _setup(tmp_path, monkeypatch, flag=True, action=CLICK, peek=peek)
    loop._approved_gate_actions["r1"] = (loop._gate_fingerprint(CLICK), frozenset({"coordinate_thrash"}))
    await loop._execute_attempt(ctx)
    peek.assert_not_awaited()
    execute.assert_awaited_once()


async def test_non_pointer_action_does_not_peek(tmp_path, monkeypatch) -> None:
    peek, _ = _peek_returning(tmp_path, popup=True)
    wait = AgentAction(action_type=ActionType.WAIT, wait_ms=10)
    loop, ctx, execute, _ = _setup(tmp_path, monkeypatch, flag=True, action=wait, peek=peek)
    await loop._execute_attempt(ctx)
    peek.assert_not_awaited()
    execute.assert_awaited_once()


async def test_executor_without_peek_runs(tmp_path, monkeypatch) -> None:
    loop, ctx, execute, _ = _setup(tmp_path, monkeypatch, flag=True, action=CLICK, peek=None)
    await loop._execute_attempt(ctx)
    execute.assert_awaited_once()

"""Clicks on a secondary monitor land on that monitor (G10), batch children too (G11)."""

from __future__ import annotations

import pytest

from operon.agent.loop import AgentLoop
from operon.models.capture import CaptureFrame
from operon.models.common import StepRequest
from operon.models.policy import ActionType, AgentAction
from tests.agent.test_trust_gate_resume import _loop


def _frame(left: int, top: int) -> CaptureFrame:
    return CaptureFrame(artifact_path="f.png", width=10, height=10, mime_type="image/png", monitor_left=left, monitor_top=top)


async def test_a_click_on_a_secondary_monitor_gets_its_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    click = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    loop, execute, _ = _loop(monkeypatch, [click])
    loop.capture_service.capture.return_value = _frame(1920, 0)

    await loop.step_run(StepRequest(run_id="run-1"))

    executed = execute.call_args_list[0].args[0]
    assert (executed.x, executed.y) == (1930, 20)


async def test_the_primary_monitor_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    click = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    loop, execute, _ = _loop(monkeypatch, [click])
    loop.capture_service.capture.return_value = _frame(0, 0)

    await loop.step_run(StepRequest(run_id="run-1"))

    executed = execute.call_args_list[0].args[0]
    assert (executed.x, executed.y) == (10, 20)


def test_batch_children_get_the_monitor_origin() -> None:
    from operon.models.perception import ScreenPerception

    perception = ScreenPerception(
        summary="s", page_hint="unknown", visible_elements=[], capture_artifact_path="f.png", monitor_origin=(-1280, 100)
    )
    batch = AgentAction(
        action_type=ActionType.BATCH,
        actions=[
            AgentAction(action_type=ActionType.CLICK, x=5, y=6),
            AgentAction(action_type=ActionType.DRAG, x=1, y=2, x_end=3, y_end=4),
            AgentAction(action_type=ActionType.BATCH, actions=[AgentAction(action_type=ActionType.CLICK, x=7, y=8)]),
        ],
    )

    moved = AgentLoop._apply_monitor_origin(batch, perception)

    first, drag, nested = moved.actions
    assert (first.x, first.y) == (-1275, 106)
    assert (drag.x, drag.y, drag.x_end, drag.y_end) == (-1279, 102, -1277, 104)
    assert (nested.actions[0].x, nested.actions[0].y) == (-1273, 108)

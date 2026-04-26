"""Tests for multi-monitor coordinate offset.

Architecture:
  AgentLoop._apply_monitor_origin  — translates monitor-local coords to
      virtual-desktop coords before the action reaches the executor.
  DesktopExecutor._coord_in_monitor_bounds  — guards that the final virtual
      coord actually falls on a known monitor before calling pyautogui.

The two layers are tested independently:
  - _apply_monitor_origin: 4 monitor layout tests + drag x_end/y_end coverage
  - bounds guard: out-of-bounds test through the executor's click path
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.models.perception import PageHint, ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, AgentAction


# ── shared helpers ────────────────────────────────────────────────────────────

def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / f"{name}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_executor(artifact_dir: Path | None = None):
    with patch("src.executor.desktop._set_dpi_awareness"):
        from src.executor.desktop import DesktopExecutor
        return DesktopExecutor(artifact_dir=artifact_dir or _local_test_dir("monitor-offset"))


def _make_perception(origin: tuple[int, int], local_x: int, local_y: int) -> ScreenPerception:
    """ScreenPerception with one button at monitor-local (local_x, local_y)."""
    element = UIElement(
        element_id="btn1",
        element_type=UIElementType.BUTTON,
        text="Submit",
        x=local_x,
        y=local_y,
        width=80,
        height=30,
        is_interactable=True,
    )
    return ScreenPerception(
        summary="test screen",
        page_hint=PageHint.UNKNOWN,
        visible_elements=[element],
        capture_artifact_path="test.png",
        monitor_origin=origin,
    )


def _mss_ctx(monitors: list[dict]) -> MagicMock:
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.monitors = monitors
    return ctx


def _apply_origin(action: AgentAction, perception: ScreenPerception) -> AgentAction:
    """Call the loop's static transform without instantiating AgentLoop."""
    from src.agent.loop import AgentLoop
    return AgentLoop._apply_monitor_origin(action, perception)


# ── _apply_monitor_origin: 4 monitor layouts ─────────────────────────────────

def test_primary_only_origin_is_noop() -> None:
    """Origin (0, 0): action coords are returned unchanged."""
    perception = _make_perception((0, 0), 300, 400)
    action = AgentAction(action_type=ActionType.CLICK, x=300, y=400)

    result = _apply_origin(action, perception)

    assert result.x == 300
    assert result.y == 400


def test_secondary_right_positive_x_offset() -> None:
    """Origin (1920, 0): x shifts by +1920, y unchanged."""
    local_x, local_y = 100, 200
    perception = _make_perception((1920, 0), local_x, local_y)
    action = AgentAction(action_type=ActionType.CLICK, x=local_x, y=local_y)

    result = _apply_origin(action, perception)

    assert result.x == local_x + 1920  # 2020
    assert result.y == local_y + 0     # 200


def test_secondary_left_negative_x_offset() -> None:
    """Origin (-1920, 0): x shifts by -1920 — sign must be preserved."""
    local_x, local_y = 500, 300
    perception = _make_perception((-1920, 0), local_x, local_y)
    action = AgentAction(action_type=ActionType.CLICK, x=local_x, y=local_y)

    result = _apply_origin(action, perception)

    assert result.x == local_x + (-1920)  # -1420
    assert result.y == local_y + 0        # 300


def test_secondary_above_negative_y_offset() -> None:
    """Origin (0, -1080): y shifts by -1080 — sign must be preserved."""
    local_x, local_y = 200, 150
    perception = _make_perception((0, -1080), local_x, local_y)
    action = AgentAction(action_type=ActionType.CLICK, x=local_x, y=local_y)

    result = _apply_origin(action, perception)

    assert result.x == local_x + 0        # 200
    assert result.y == local_y + (-1080)  # -930


def test_drag_x_end_y_end_also_offset() -> None:
    """Drag end-point (x_end, y_end) must receive the same offset as the start."""
    perception = _make_perception((1920, 0), 0, 0)
    action = AgentAction(action_type=ActionType.DRAG, x=100, y=200, x_end=300, y_end=400)

    result = _apply_origin(action, perception)

    assert result.x == 100 + 1920    # 2020
    assert result.y == 200 + 0       # 200
    assert result.x_end == 300 + 1920  # 2220
    assert result.y_end == 400 + 0     # 400


# ── executor bounds guard ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_out_of_bounds_click_skipped_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Executor skips the click and logs WARNING when the virtual coord is outside all monitors.

    The action arrives with virtual-desktop coords (loop already applied the origin).
    We pass (2020, 100) but mock mss to return only the primary monitor (0..1920 x 0..1080),
    making (2020, 100) out-of-bounds.
    """
    executor = _make_executor()
    # Virtual coords — as if the loop applied origin (1920, 0) to local (100, 100)
    action = AgentAction(action_type=ActionType.CLICK, x=2020, y=100)

    monitors_primary_only = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},  # monitors[0]: virtual combined
        {"left": 0, "top": 0, "width": 1920, "height": 1080},  # monitors[1]: primary only
    ]

    with (
        patch("src.executor.desktop.pyautogui.click") as mock_click,
        patch("src.executor.desktop.mss.mss", return_value=_mss_ctx(monitors_primary_only)),
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
        caplog.at_level(logging.WARNING, logger="src.executor.desktop"),
    ):
        result = await executor.execute(action)

    assert result.success is False
    mock_click.assert_not_called()
    assert "bounds" in caplog.text.lower() or "outside" in caplog.text.lower()
    assert "2020" in caplog.text  # virtual x appears in the log
    assert "100" in caplog.text   # virtual y appears in the log

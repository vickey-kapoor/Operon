"""KAN-92: OPERON_FOCUS_GUARD skips an action when another process took focus."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from operon.models.common import FailureCategory
from operon.models.policy import ActionType, AgentAction
from tests.executor.test_desktop import _make_executor

MOD = "operon.executor.desktop"
OLD = (100, 1, "Notepad")


def _guarded(monkeypatch, *, on: bool = True):
    monkeypatch.setenv("OPERON_FOCUS_GUARD", "on" if on else "off")
    executor = _make_executor()
    executor._focus_ref = OLD
    return executor


def _windows(fg: int | None, owners: dict[int, tuple[int | None, str]]):
    return (
        patch(f"{MOD}._foreground_window", return_value=fg),
        patch(f"{MOD}._window_owner", side_effect=lambda h: owners.get(h, (None, ""))),
    )


async def _run_click(executor, fg, owners):
    action = AgentAction(action_type=ActionType.CLICK, x=500, y=300)
    fg_p, own_p = _windows(fg, owners)
    with (
        fg_p,
        own_p,
        patch(f"{MOD}.pyautogui.click") as click,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="a.png"),
        patch.object(executor, "_region_has_content", return_value=(True, 100.0)) as region,
    ):
        result = await executor.execute(action)
    return result, click, region


@pytest.mark.asyncio
async def test_flag_off_does_not_check(monkeypatch) -> None:
    executor = _guarded(monkeypatch, on=False)
    result, click, _ = await _run_click(executor, 200, {200: (2, "Other")})
    assert result.success is True
    click.assert_called_once()


@pytest.mark.asyncio
async def test_same_window_executes_and_servo_runs(monkeypatch) -> None:
    executor = _guarded(monkeypatch)
    result, click, region = await _run_click(executor, 100, {100: (1, "Notepad")})
    assert result.success is True
    click.assert_called_once()
    region.assert_called()


@pytest.mark.asyncio
async def test_other_process_stole_focus_blocks(monkeypatch) -> None:
    executor = _guarded(monkeypatch)
    result, click, region = await _run_click(executor, 200, {200: (2, "Update Ready")})
    assert result.success is False
    assert result.failure_category is FailureCategory.FOCUS_STOLEN
    assert "focus_stolen: Update Ready took focus from Notepad" in result.detail
    click.assert_not_called()
    region.assert_not_called()


@pytest.mark.asyncio
async def test_same_process_dialog_executes(monkeypatch) -> None:
    executor = _guarded(monkeypatch)
    result, click, _ = await _run_click(executor, 300, {300: (1, "Save As")})
    assert result.success is True
    click.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        AgentAction(action_type=ActionType.TYPE, text="hi"),
        AgentAction(action_type=ActionType.PRESS_KEY, key="enter"),
        AgentAction(action_type=ActionType.HOTKEY, key="ctrl+s"),
        AgentAction(action_type=ActionType.SCROLL, x=10, y=10, scroll_amount=3),
        AgentAction(action_type=ActionType.DRAG, x=1, y=1, x_end=5, y_end=5),
    ],
)
async def test_non_click_actions_are_guarded(monkeypatch, action) -> None:
    executor = _guarded(monkeypatch)
    fg_p, own_p = _windows(200, {200: (2, "Popup")})
    with fg_p, own_p:
        result = await executor.execute(action)
    assert result.success is False
    assert result.failure_category is FailureCategory.FOCUS_STOLEN


@pytest.mark.asyncio
async def test_unguarded_action_still_runs(monkeypatch) -> None:
    executor = _guarded(monkeypatch)
    fg_p, own_p = _windows(200, {200: (2, "Popup")})
    with fg_p, own_p:
        result = await executor.execute(AgentAction(action_type=ActionType.STOP))
    assert result.success is True


def test_record_focus_remembers_foreground(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_FOCUS_GUARD", "on")
    executor = _make_executor()
    fg_p, own_p = _windows(100, {100: (1, "Notepad")})
    with fg_p, own_p:
        executor._record_focus()
    assert executor._focus_ref == OLD


def test_linux_executor_never_guards() -> None:
    from operon.executor.linux_desktop import LinuxDesktopExecutor

    assert LinuxDesktopExecutor._focus_guard_platform is False


def _burst_patches(executor):
    from types import SimpleNamespace

    frame = SimpleNamespace(artifact_path="f.png")
    return patch.object(executor, "_capture_burst_sync", return_value=(frame, 0.0))


@pytest.mark.asyncio
async def test_capture_arms_guard(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_FOCUS_GUARD", "on")
    executor = _make_executor()
    fg_p, own_p = _windows(100, {100: (1, "Notepad")})
    with fg_p, own_p, _burst_patches(executor):
        await executor.capture()
    assert executor._focus_ref == OLD


@pytest.mark.asyncio
async def test_internal_post_action_capture_does_not_arm(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_FOCUS_GUARD", "on")
    executor = _make_executor()
    fg_p, own_p = _windows(100, {100: (1, "Notepad")})
    with fg_p, own_p, _burst_patches(executor):
        await executor._capture_after()
    assert executor._focus_ref is None


@pytest.mark.asyncio
async def test_batch_only_first_child_is_checked(monkeypatch) -> None:
    executor = _guarded(monkeypatch)
    batch = AgentAction(
        action_type=ActionType.BATCH,
        actions=[
            AgentAction(action_type=ActionType.PRESS_KEY, key="enter"),
            AgentAction(action_type=ActionType.PRESS_KEY, key="enter"),
        ],
    )
    # Child 1 sees the recorded window; another pid's window appears before child 2.
    fg_p, own_p = _windows(100, {100: (1, "Notepad"), 200: (2, "New App")})
    with fg_p as fg, own_p, patch(f"{MOD}.pyautogui.press"), patch.object(
        executor, "_capture_after", new_callable=AsyncMock, return_value="a.png"
    ):
        fg.side_effect = [100, 200, 200, 200]
        result = await executor.execute(batch)
    assert result.success is True


@pytest.mark.asyncio
async def test_batch_first_child_blocked_carries_category(monkeypatch) -> None:
    executor = _guarded(monkeypatch)
    batch = AgentAction(
        action_type=ActionType.BATCH,
        actions=[
            AgentAction(action_type=ActionType.TYPE, text="x"),
            AgentAction(action_type=ActionType.PRESS_KEY, key="enter"),
        ],
    )
    fg_p, own_p = _windows(200, {200: (2, "Popup")})
    with fg_p, own_p:
        result = await executor.execute(batch)
    assert result.success is False
    assert result.batch_failed_index == 0
    assert result.failure_category is FailureCategory.FOCUS_STOLEN


@pytest.mark.asyncio
async def test_flag_off_never_reads_windows(monkeypatch) -> None:
    executor = _guarded(monkeypatch, on=False)
    fg_p, own_p = _windows(200, {200: (2, "Other")})
    with fg_p, own_p as owner, patch(f"{MOD}.pyautogui.press"), patch.object(
        executor, "_capture_after", new_callable=AsyncMock, return_value="a.png"
    ):
        await executor.execute(AgentAction(action_type=ActionType.PRESS_KEY, key="enter"))
        executor._record_focus()
    owner.assert_not_called()


@pytest.mark.asyncio
async def test_peek_frame_does_not_rerecord_focus(monkeypatch) -> None:
    """KAN-207: a fresh look must not hide a focus steal that happened while the model decided."""
    executor = _guarded(monkeypatch)
    with (
        patch.object(executor, "_capture_burst_sync", return_value=("frame", 0.0)),
        patch.object(executor, "_record_focus") as record,
    ):
        assert await executor.peek_frame() == "frame"
        record.assert_not_called()
        assert executor._internal_capture is False
        await executor.capture()
        record.assert_called_once()

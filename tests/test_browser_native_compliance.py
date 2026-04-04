from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.executor.browser_native import NativeBrowserExecutor
from src.models.policy import ActionType, AgentAction


@pytest.mark.asyncio
async def test_browser_native_type_respects_clear_and_enter_flags() -> None:
    executor = NativeBrowserExecutor(artifact_dir=Path(".test-artifacts/native-compliance"), headless=True)
    keyboard = type(
        "Keyboard",
        (),
        {
            "press": AsyncMock(),
            "type": AsyncMock(),
        },
    )()
    mouse = type("Mouse", (), {"click": AsyncMock()})()
    page = type(
        "Page",
        (),
        {
            "mouse": mouse,
            "keyboard": keyboard,
            "wait_for_load_state": AsyncMock(),
        },
    )()
    executor._current_page = AsyncMock(return_value=page)
    executor._capture_after = AsyncMock(return_value="after.png")

    action = AgentAction(
        action_type=ActionType.TYPE,
        text="hello",
        x=100,
        y=200,
        clear_before_typing=True,
        press_enter=True,
    )

    result = await executor.execute(action)

    assert result.success is True
    mouse.click.assert_awaited_once_with(100, 200)
    keyboard.press.assert_any_await("Control+A")
    keyboard.press.assert_any_await("Backspace")
    keyboard.type.assert_awaited_once_with("hello")
    keyboard.press.assert_any_await("Enter")


@pytest.mark.asyncio
async def test_browser_native_executes_batch_actions_in_order() -> None:
    executor = NativeBrowserExecutor(artifact_dir=Path(".test-artifacts/native-batch"), headless=True)
    executor.execute = AsyncMock(
        side_effect=[
            type("Result", (), {"success": True, "artifact_path": "a.png", "detail": "ok", "execution_trace": None, "failure_category": None, "failure_stage": None})(),
            type("Result", (), {"success": True, "artifact_path": "b.png", "detail": "ok", "execution_trace": None, "failure_category": None, "failure_stage": None})(),
        ]
    )
    batch = AgentAction(
        action_type=ActionType.BATCH,
        actions=[
            AgentAction(action_type=ActionType.NAVIGATE, url="https://example.com"),
            AgentAction(action_type=ActionType.WAIT, wait_ms=5000),
        ],
    )

    result = await NativeBrowserExecutor._execute_batch(executor, batch)

    assert result.success is True
    assert result.artifact_path == "b.png"

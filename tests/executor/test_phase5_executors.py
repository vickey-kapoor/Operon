"""Tests for the thin real executor adapters."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from operon.executor.browser_adapter import BrowserExecutor
from operon.executor.desktop_adapter import DesktopExecutor
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction


@pytest.mark.asyncio
async def test_browser_executor_delegates_to_legacy_executor() -> None:
    result = ExecutedAction(
        action=AgentAction(action_type=ActionType.NAVIGATE, url="https://example.com"),
        success=True,
        detail="ok",
    )
    legacy = SimpleNamespace(execute=AsyncMock(return_value=result))
    executor = BrowserExecutor(legacy_executor=legacy)

    action = AgentAction(action_type=ActionType.NAVIGATE, url="https://example.com")
    executed = await executor.execute(action)

    legacy.execute.assert_awaited_once_with(action)
    assert executed is result


@pytest.mark.asyncio
async def test_desktop_executor_delegates_to_legacy_executor() -> None:
    result = ExecutedAction(
        action=AgentAction(action_type=ActionType.LAUNCH_APP, text="Notepad"),
        success=True,
        detail="ok",
    )
    legacy = SimpleNamespace(execute=AsyncMock(return_value=result))
    executor = DesktopExecutor(legacy_executor=legacy)

    action = AgentAction(action_type=ActionType.LAUNCH_APP, text="Notepad")
    executed = await executor.execute(action)

    legacy.execute.assert_awaited_once_with(action)
    assert executed is result


@pytest.mark.asyncio
async def test_executor_adapters_delegate_capture() -> None:
    legacy = SimpleNamespace(capture=AsyncMock(return_value="frame"))

    assert await BrowserExecutor(legacy_executor=legacy).capture() == "frame"
    assert await DesktopExecutor(legacy_executor=legacy).capture() == "frame"

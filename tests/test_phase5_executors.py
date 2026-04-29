"""Phase 5 tests for thin real executor adapters."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core.contracts.planner import ActionType as ContractActionType
from src.core.contracts.planner import PlannerAction
from src.executors.browser_executor import BrowserExecutor
from src.executors.desktop_executor import DesktopExecutor
from src.models.execution import ExecutedAction
from src.models.policy import ActionType, AgentAction


@pytest.mark.asyncio
async def test_browser_executor_delegates_to_legacy_executor() -> None:
    result = ExecutedAction(
        action=AgentAction(action_type=ActionType.WAIT, wait_ms=1),
        success=True,
        detail="ok",
    )
    legacy = SimpleNamespace(execute=AsyncMock(return_value=result))
    executor = BrowserExecutor(legacy_executor=legacy)

    action = executor._to_legacy_action(
        PlannerAction(action_type=ContractActionType.NAVIGATE, url="https://example.com")
    )
    executed = await executor.execute(action)

    assert action.action_type is ActionType.NAVIGATE
    assert executed is result


@pytest.mark.asyncio
async def test_desktop_executor_delegates_to_legacy_executor() -> None:
    result = ExecutedAction(
        action=AgentAction(action_type=ActionType.WAIT, wait_ms=1),
        success=True,
        detail="ok",
    )
    legacy = SimpleNamespace(execute=AsyncMock(return_value=result))
    executor = DesktopExecutor(legacy_executor=legacy)

    action = executor._to_legacy_action(
        PlannerAction(action_type=ContractActionType.LAUNCH_APP, app_name="Notepad")
    )
    executed = await executor.execute(action)

    assert action.action_type is ActionType.LAUNCH_APP
    assert action.text == "Notepad"
    assert executed is result


def test_browser_executor_translates_supported_contract_actions() -> None:
    action = BrowserExecutor._to_legacy_action(
        PlannerAction(action_type=ContractActionType.PRESS_HOTKEY, hotkey=["ctrl", "l"])
    )
    assert action.action_type is ActionType.HOTKEY
    assert action.key == "ctrl+l"


def test_desktop_executor_translates_supported_contract_actions() -> None:
    action = DesktopExecutor._to_legacy_action(
        PlannerAction(action_type=ContractActionType.WAIT, wait_ms=750)
    )
    assert action.action_type is ActionType.WAIT
    assert action.wait_ms == 750

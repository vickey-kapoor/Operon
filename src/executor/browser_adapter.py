"""Thin browser executor adapter for unified orchestration."""

from __future__ import annotations

from src.core.contracts.actor import ExecutorChoice
from src.core.contracts.planner import ActionType as ContractActionType
from src.core.contracts.planner import PlannerAction
from src.models.execution import ExecutedAction
from src.models.policy import ActionType, AgentAction


class BrowserExecutor:
    """Thin wrapper around the existing Playwright-backed browser executor."""

    executor = ExecutorChoice.BROWSER

    def __init__(self, legacy_executor=None, **legacy_kwargs) -> None:
        if legacy_executor is None:
            from src.executor.browser_native import NativeBrowserExecutor

            legacy_executor = NativeBrowserExecutor(**legacy_kwargs)
        self.legacy_executor = legacy_executor

    async def execute(self, action: AgentAction) -> ExecutedAction:
        return await self.legacy_executor.execute(action)

    async def capture(self):
        return await self.legacy_executor.capture()

    async def execute_contract_action(self, action: PlannerAction) -> ExecutedAction:
        """Execute one unified contract action through the legacy browser executor."""

        return await self.execute(self._to_legacy_action(action))

    @staticmethod
    def _to_legacy_action(action: PlannerAction) -> AgentAction:
        if action.action_type is ContractActionType.NAVIGATE:
            return AgentAction(action_type=ActionType.NAVIGATE, url=action.url)
        if action.action_type is ContractActionType.CLICK:
            return AgentAction(action_type=ActionType.CLICK, target_element_id=action.target_id)
        if action.action_type is ContractActionType.TYPE_TEXT:
            return AgentAction(
                action_type=ActionType.TYPE,
                target_element_id=action.target_id,
                text=action.text,
            )
        if action.action_type is ContractActionType.PRESS_HOTKEY:
            return AgentAction(action_type=ActionType.HOTKEY, key="+".join(action.hotkey))
        if action.action_type is ContractActionType.WAIT:
            return AgentAction(action_type=ActionType.WAIT, wait_ms=action.wait_ms)
        if action.action_type is ContractActionType.UPLOAD_FILE_NATIVE:
            return AgentAction(
                action_type=ActionType.UPLOAD_FILE_NATIVE,
                target_element_id=action.target_id,
                text=action.picker_title,
            )
        raise ValueError(f"Browser executor does not support contract action {action.action_type.value!r}")

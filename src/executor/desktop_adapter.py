"""Thin desktop executor adapter for unified orchestration."""

from __future__ import annotations

from src.core.contracts.actor import ExecutorChoice
from src.core.contracts.planner import ActionType as ContractActionType
from src.core.contracts.planner import PlannerAction
from src.models.execution import ExecutedAction
from src.models.policy import ActionType, AgentAction


class DesktopExecutor:
    """Thin wrapper around the existing pyautogui-backed desktop executor."""

    executor = ExecutorChoice.DESKTOP

    def __init__(self, legacy_executor=None, **legacy_kwargs) -> None:
        if legacy_executor is None:
            from src.executor.desktop import DesktopExecutor as LegacyDesktopExecutor

            legacy_executor = LegacyDesktopExecutor(**legacy_kwargs)
        self.legacy_executor = legacy_executor

    async def execute(self, action: AgentAction) -> ExecutedAction:
        return await self.legacy_executor.execute(action)

    async def capture(self):
        return await self.legacy_executor.capture()

    async def execute_contract_action(self, action: PlannerAction) -> ExecutedAction:
        """Execute one unified contract action through the legacy desktop executor."""

        return await self.execute(self._to_legacy_action(action))

    @staticmethod
    def _to_legacy_action(action: PlannerAction) -> AgentAction:
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
        if action.action_type is ContractActionType.LAUNCH_APP:
            return AgentAction(action_type=ActionType.LAUNCH_APP, text=action.app_name)
        if action.action_type is ContractActionType.WAIT:
            return AgentAction(action_type=ActionType.WAIT, wait_ms=action.wait_ms)
        raise ValueError(f"Desktop executor does not support contract action {action.action_type.value!r}")

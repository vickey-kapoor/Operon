"""Thin desktop executor adapter that wraps the pyautogui-backed executor."""

from __future__ import annotations

from operon.models.execution import ExecutedAction
from operon.models.policy import AgentAction


class DesktopExecutor:
    """Thin wrapper around the existing pyautogui-backed desktop executor."""

    def __init__(self, legacy_executor=None, **legacy_kwargs) -> None:
        if legacy_executor is None:
            from operon.executor.desktop import DesktopExecutor as LegacyDesktopExecutor

            legacy_executor = LegacyDesktopExecutor(**legacy_kwargs)
        self.legacy_executor = legacy_executor

    async def execute(self, action: AgentAction) -> ExecutedAction:
        return await self.legacy_executor.execute(action)

    async def capture(self):
        return await self.legacy_executor.capture()

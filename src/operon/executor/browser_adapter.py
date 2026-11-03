"""Thin browser executor adapter that wraps the Playwright-backed executor."""

from __future__ import annotations

from operon.models.execution import ExecutedAction
from operon.models.policy import AgentAction


class BrowserExecutor:
    """Thin wrapper around the existing Playwright-backed browser executor."""

    def __init__(self, legacy_executor=None, **legacy_kwargs) -> None:
        if legacy_executor is None:
            from operon.executor.browser_native import NativeBrowserExecutor

            legacy_executor = NativeBrowserExecutor(**legacy_kwargs)
        self.legacy_executor = legacy_executor

    @property
    def batch_child_check(self):
        return getattr(self.legacy_executor, "batch_child_check", None)

    @batch_child_check.setter
    def batch_child_check(self, check) -> None:
        if hasattr(self.legacy_executor, "batch_child_check"):
            self.legacy_executor.batch_child_check = check

    async def execute(self, action: AgentAction) -> ExecutedAction:
        return await self.legacy_executor.execute(action)

    async def capture(self):
        return await self.legacy_executor.capture()

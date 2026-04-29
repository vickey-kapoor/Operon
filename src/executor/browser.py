"""Executor interface for all automation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.capture import CaptureFrame
from src.models.execution import ExecutedAction
from src.models.policy import AgentAction


class Executor(ABC):
    """Abstract base class for all executors (desktop, etc.)."""

    @abstractmethod
    async def capture(self) -> CaptureFrame:
        """Capture the current screen state."""

    @abstractmethod
    async def execute(self, action: AgentAction) -> ExecutedAction:
        """Execute a typed action."""

    async def context_reset(self) -> None:
        """Dismiss open modals/dropdowns, clear focus traps, scroll to top.

        Concrete executors should override this. The default is a no-op so that
        executors that cannot perform UI resets do not break the recovery ladder.
        """

    async def session_reset(self, start_url: str | None = None) -> None:
        """Navigate back to the task start URL (browser) or re-baseline window
        focus (desktop). Concrete executors should override this.
        """


# Backwards-compatible alias
BrowserExecutor = Executor

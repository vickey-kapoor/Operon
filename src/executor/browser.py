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


# Backwards-compatible alias
BrowserExecutor = Executor

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.capture import CaptureFrame
from src.models.logs import ModelDebugArtifacts
from src.models.perception import ScreenPerception
from src.models.policy import PolicyDecision
from src.models.state import AgentState


class AgentBackend(ABC):
    @abstractmethod
    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        ...

    @abstractmethod
    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        ...

    @abstractmethod
    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        ...

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        """Reset hints to a known state. Optional hook — test use only."""
        return None

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        """Append advisory hints without discarding existing ones. Optional hook."""
        return None

    def clear_advisory_hints(self) -> None:
        """Drop any queued advisory hints. Optional hook."""
        return None

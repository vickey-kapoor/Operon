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

    def set_advisory_hints(self, hints: list[str]) -> None:
        """Replace all advisory hints. Optional hook — override to implement."""
        return None

    def add_advisory_hints(self, hints: list[str]) -> None:
        """Append advisory hints without discarding existing ones. Optional hook."""
        return None

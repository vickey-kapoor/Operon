from __future__ import annotations

from abc import ABC, abstractmethod

from operon.models.capture import CaptureFrame
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import PolicyDecision
from operon.models.state import AgentState


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

    def clear_advisory_hints(self, run_id: str = "") -> None:
        """Drop any queued advisory hints. Optional hook."""
        return None

    def supports_resampling(self, run_id: str = "") -> bool:
        """Whether a second ``choose_action`` for the same step yields a fresh
        decision. Best-of-N sampling needs it; backends that only replay a
        decision cached by ``perceive`` return False."""
        return True

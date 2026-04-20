from __future__ import annotations

from src.agent.backend import AgentBackend
from src.models.capture import CaptureFrame
from src.models.logs import ModelDebugArtifacts
from src.models.perception import ScreenPerception
from src.models.policy import PolicyDecision
from src.models.state import AgentState


class BackendCompatibilityError(RuntimeError):
    """Raised when a backend cannot serve the current request contract."""


class FallbackBackend(AgentBackend):
    def __init__(self, *, primary: AgentBackend, secondary: AgentBackend) -> None:
        self.primary = primary
        self.secondary = secondary
        self._active_backends: dict[str, AgentBackend] = {}
        self._latest_backend: AgentBackend = primary

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        backend = self._backend_for_run(state.run_id)
        try:
            perception = await backend.perceive(screenshot, state)
        except BackendCompatibilityError:
            if backend is self.secondary:
                raise
            self._active_backends[state.run_id] = self.secondary
            backend = self.secondary
            perception = await backend.perceive(screenshot, state)
        self._clear_inactive_hints(active=backend)
        self._latest_backend = backend
        return perception

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        backend = self._backend_for_run(state.run_id)
        decision = await backend.choose_action(state, perception)
        self._clear_inactive_hints(active=backend)
        self._latest_backend = backend
        return decision

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._latest_backend.latest_debug_artifacts()

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        """Reset hints to a known state on both backends. Test use only."""
        self.primary._reset_advisory_hints_for_test(hints)
        self.secondary._reset_advisory_hints_for_test(hints)

    def add_advisory_hints(self, hints: list[str], source: str = "") -> None:
        """Append hints on both backends without discarding existing ones."""
        self.primary.add_advisory_hints(hints, source=source)
        self.secondary.add_advisory_hints(hints, source=source)

    def _backend_for_run(self, run_id: str) -> AgentBackend:
        return self._active_backends.get(run_id, self.primary)

    def _clear_inactive_hints(self, *, active: AgentBackend) -> None:
        inactive = self.secondary if active is self.primary else self.primary
        inactive.clear_advisory_hints()

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
        self._latest_backend = backend
        return perception

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        backend = self._backend_for_run(state.run_id)
        self._latest_backend = backend
        return await backend.choose_action(state, perception)

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._latest_backend.latest_debug_artifacts()

    def set_advisory_hints(self, hints: list[str]) -> None:
        self.primary.set_advisory_hints(hints)
        self.secondary.set_advisory_hints(hints)

    def _backend_for_run(self, run_id: str) -> AgentBackend:
        return self._active_backends.get(run_id, self.primary)

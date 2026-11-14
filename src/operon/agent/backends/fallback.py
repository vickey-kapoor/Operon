from __future__ import annotations

import logging

from operon.agent.backends.base import AgentBackend
from operon.models.capture import CaptureFrame
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import PolicyDecision
from operon.models.state import AgentState

logger = logging.getLogger(__name__)


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
        except BackendCompatibilityError as exc:
            if backend is self.secondary:
                raise
            logger.warning("Switching run %s to the fallback backend for the rest of the run: %s", state.run_id, exc)
            self._active_backends[state.run_id] = self.secondary
            backend = self.secondary
            perception = await backend.perceive(screenshot, state)
        self._clear_inactive_hints(active=backend, run_id=state.run_id)
        self._latest_backend = backend
        return perception

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        backend = self._backend_for_run(state.run_id)
        decision = await backend.choose_action(state, perception)
        self._clear_inactive_hints(active=backend, run_id=state.run_id)
        self._latest_backend = backend
        return decision

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._latest_backend.latest_debug_artifacts()

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        """Reset hints to a known state on both backends. Test use only."""
        self.primary._reset_advisory_hints_for_test(hints)
        self.secondary._reset_advisory_hints_for_test(hints)

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        """Append hints on both backends without discarding existing ones."""
        self.primary.add_advisory_hints(hints, source=source, run_id=run_id)
        self.secondary.add_advisory_hints(hints, source=source, run_id=run_id)

    def release_run(self, run_id: str) -> None:
        # Either side may be a service without the hook (the JSON backend), and one
        # side failing must not keep the run pinned here.
        try:
            for backend in (self.primary, self.secondary):
                release = getattr(backend, "release_run", None)
                if release is not None:
                    release(run_id)
        finally:
            self._active_backends.pop(run_id, None)

    def note_not_executed(self, run_id: str, reason: str) -> None:
        self._backend_for_run(run_id).note_not_executed(run_id, reason)

    def has_held_actions(self, run_id: str) -> bool:
        return self._backend_for_run(run_id).has_held_actions(run_id)

    def note_replaced(self, run_id: str, action: str) -> None:
        self._backend_for_run(run_id).note_replaced(run_id, action)

    def note_rejected(self, run_id: str, reason: str) -> None:
        self._backend_for_run(run_id).note_rejected(run_id, reason)

    def note_attempt_failed(self, run_id: str, detail: str) -> None:
        self._backend_for_run(run_id).note_attempt_failed(run_id, detail)

    def set_perception_only(self, value: bool) -> None:
        for backend in (self.primary, self.secondary):
            setter = getattr(backend, "set_perception_only", None)
            if setter is not None:
                setter(value)

    def discard_held_actions(self, run_id: str, reason: str) -> None:
        self._backend_for_run(run_id).discard_held_actions(run_id, reason)

    def supports_resampling(self, run_id: str = "") -> bool:
        return self._backend_for_run(run_id).supports_resampling(run_id)

    def _backend_for_run(self, run_id: str) -> AgentBackend:
        return self._active_backends.get(run_id, self.primary)

    def _clear_inactive_hints(self, *, active: AgentBackend, run_id: str) -> None:
        inactive = self.secondary if active is self.primary else self.primary
        inactive.clear_advisory_hints(run_id=run_id)

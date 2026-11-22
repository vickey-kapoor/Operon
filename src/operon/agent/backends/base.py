from __future__ import annotations

from abc import ABC, abstractmethod

from operon.models.capture import CaptureFrame
from operon.models.logs import ModelDebugArtifacts
from operon.models.perception import ScreenPerception
from operon.models.policy import PolicyDecision
from operon.models.state import AgentState


class BackendCompatibilityError(RuntimeError):
    """Raised when a backend cannot serve the current request contract."""


class ModelBlockedError(BackendCompatibilityError):
    """The model API blocked the request (not the screen content's fault we can fix), even after one retry."""

    def __init__(self, block_reason: str) -> None:
        super().__init__(f"Gemini blocked the request ({block_reason})")
        self.block_reason = block_reason


def snapshot_hints(hints: dict[str, list[tuple[str, str]]], run_id: str) -> dict[str, list[tuple[str, str]] | None]:
    """What is queued for this run and in the shared ``""`` bucket, before a step pops it."""
    return {key: hints.get(key) for key in (run_id, "")}


def restore_hints(hints: dict[str, list[tuple[str, str]]], saved: dict[str, list[tuple[str, str]] | None]) -> None:
    """Put back hints a failed model call popped, in the bucket they came from, ahead of newer ones."""
    for key, before in saved.items():
        if before and hints.get(key) is not before:
            hints.setdefault(key, [])[:0] = before


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

    def note_not_executed(self, run_id: str, reason: str) -> None:
        """The last decision was stopped by a gate before it ran. Optional hook for
        backends that answer the model's calls with their outcome."""
        return None

    # Per-run caches the backends keep, keyed by run_id.
    _PER_RUN_CACHES = ("_conversations", "_last_frames", "_advisory_hints", "_runs", "_hints", "_cached_decisions")

    def release_run(self, run_id: str) -> None:
        """The run finished: drop everything kept for it (conversation, last frame,
        queued hints). Backends with other per-run state extend this."""
        for name in self._PER_RUN_CACHES:
            cache = getattr(self, name, None)
            if isinstance(cache, dict):
                cache.pop(run_id, None)

    def has_held_actions(self, run_id: str) -> bool:
        """Whether the run is paused on calls the model asked the user to confirm."""
        return False

    def discard_held_actions(self, run_id: str, reason: str) -> None:
        """The user resumed a confirmation pause with a correction instead of an
        approval. Optional hook for backends that hold the paused calls to run them
        on resume: drop them, unrun."""
        return None

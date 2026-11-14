from __future__ import annotations

import pytest

from operon.agent.backends.fallback import BackendCompatibilityError, FallbackBackend
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState


class _FailingBackend:
    def __init__(self) -> None:
        self.hints: list[str] = []

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        raise BackendCompatibilityError("primary unavailable")

    async def choose_action(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        raise AssertionError("choose_action should not be called on failing backend")

    def latest_debug_artifacts(self):
        return None

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        self.hints = hints

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        self.hints.extend(hints)

    def clear_advisory_hints(self, run_id: str = "") -> None:
        self.hints = []


class _RunScopedBackend:
    def __init__(self) -> None:
        self.hints: list[str] = []
        self.scoped_hints: dict[str, list[str]] = {}
        self.calls: list[str] = []

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        self.calls.append(state.run_id)
        if state.run_id == "run-1":
            raise BackendCompatibilityError("primary unavailable for this run")
        return ScreenPerception(
            summary=f"Primary visible for {state.run_id}",
            page_hint="unknown",
            visible_elements=[],
            capture_artifact_path=screenshot.artifact_path,
            confidence=0.9,
        )

    async def choose_action(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.WAIT, wait_ms=1000),
            rationale=f"primary:{state.run_id}",
            confidence=0.8,
            active_subgoal="retry",
        )

    def latest_debug_artifacts(self):
        return None

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        self.hints = hints

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        self.hints.extend(hints)
        self.scoped_hints.setdefault(run_id, []).extend(hints)

    def clear_advisory_hints(self, run_id: str = "") -> None:
        if run_id:
            self.scoped_hints.pop(run_id, None)
            self.hints = [hint for scoped in self.scoped_hints.values() for hint in scoped]
            return
        self.scoped_hints = {}
        self.hints = []


class _WorkingBackend:
    def __init__(self) -> None:
        self.hints: list[str] = []
        self.scoped_hints: dict[str, list[str]] = {}
        self.perception = ScreenPerception(
            summary="Browser visible",
            page_hint="unknown",
            visible_elements=[],
            capture_artifact_path="runs/run-1/step_1/before.png",
            confidence=0.9,
        )
        self.decision = PolicyDecision(
            action=AgentAction(action_type=ActionType.WAIT, wait_ms=1000),
            rationale="fallback",
            confidence=0.8,
            active_subgoal="retry",
        )

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        return self.perception

    async def choose_action(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        return self.decision

    def latest_debug_artifacts(self):
        return None

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        self.hints = hints

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        self.hints.extend(hints)
        self.scoped_hints.setdefault(run_id, []).extend(hints)

    def clear_advisory_hints(self, run_id: str = "") -> None:
        if run_id:
            self.scoped_hints.pop(run_id, None)
            self.hints = [hint for scoped in self.scoped_hints.values() for hint in scoped]
            return
        self.scoped_hints = {}
        self.hints = []


@pytest.mark.asyncio
async def test_fallback_backend_switches_on_compatibility_error() -> None:
    backend = FallbackBackend(primary=_FailingBackend(), secondary=_WorkingBackend())
    frame = CaptureFrame(
        artifact_path="runs/run-1/step_1/before.png",
        width=1280,
        height=720,
        mime_type="image/png",
    )
    state = AgentState(run_id="run-1", intent="Browse", status=RunStatus.PENDING)

    perception = await backend.perceive(frame, state)
    decision = await backend.choose_action(state, perception)

    assert perception.summary == "Browser visible"
    assert decision.action.action_type is ActionType.WAIT


@pytest.mark.asyncio
async def test_fallback_backend_scopes_switch_to_the_current_run_only() -> None:
    primary = _RunScopedBackend()
    secondary = _WorkingBackend()
    backend = FallbackBackend(primary=primary, secondary=secondary)
    frame = CaptureFrame(
        artifact_path="runs/run-1/step_1/before.png",
        width=1280,
        height=720,
        mime_type="image/png",
    )
    run_one = AgentState(run_id="run-1", intent="Browse", status=RunStatus.PENDING)
    run_two = AgentState(run_id="run-2", intent="Browse", status=RunStatus.PENDING)

    perception_one = await backend.perceive(frame, run_one)
    decision_one = await backend.choose_action(run_one, perception_one)
    perception_two = await backend.perceive(frame, run_two)
    decision_two = await backend.choose_action(run_two, perception_two)

    assert perception_one.summary == "Browser visible"
    assert decision_one.rationale == "fallback"
    assert perception_two.summary == "Primary visible for run-2"
    assert decision_two.rationale == "primary:run-2"


def test_fallback_backend_forwards_hints_to_active_backend() -> None:
    secondary = _WorkingBackend()
    primary = _FailingBackend()
    backend = FallbackBackend(primary=primary, secondary=secondary)

    backend._reset_advisory_hints_for_test(["avoid repeated clicks"])

    assert primary.hints == ["avoid repeated clicks"]
    assert secondary.hints == ["avoid repeated clicks"]


def test_fallback_backend_forwards_run_id_to_both_backends() -> None:
    primary = _RunScopedBackend()
    secondary = _WorkingBackend()
    backend = FallbackBackend(primary=primary, secondary=secondary)

    backend.add_advisory_hints(["run-scoped hint"], source="memory", run_id="run-2")

    assert primary.scoped_hints == {"run-2": ["run-scoped hint"]}
    assert secondary.scoped_hints == {"run-2": ["run-scoped hint"]}


@pytest.mark.asyncio
async def test_fallback_backend_clears_inactive_hints_after_primary_success() -> None:
    primary = _RunScopedBackend()
    secondary = _WorkingBackend()
    backend = FallbackBackend(primary=primary, secondary=secondary)
    frame = CaptureFrame(
        artifact_path="runs/run-2/step_1/before.png",
        width=1280,
        height=720,
        mime_type="image/png",
    )
    state = AgentState(run_id="run-2", intent="Browse", status=RunStatus.PENDING)

    backend.add_advisory_hints(["fresh hint"], source="memory", run_id="run-2")
    await backend.perceive(frame, state)

    assert primary.hints == ["fresh hint"]
    assert secondary.hints == []


@pytest.mark.asyncio
async def test_fallback_backend_clears_only_inactive_hints_for_current_run() -> None:
    primary = _RunScopedBackend()
    secondary = _WorkingBackend()
    backend = FallbackBackend(primary=primary, secondary=secondary)
    frame = CaptureFrame(
        artifact_path="runs/run-2/step_1/before.png",
        width=1280,
        height=720,
        mime_type="image/png",
    )
    state = AgentState(run_id="run-2", intent="Browse", status=RunStatus.PENDING)

    backend.add_advisory_hints(["run-1 hint"], source="memory", run_id="run-1")
    backend.add_advisory_hints(["run-2 hint"], source="memory", run_id="run-2")
    await backend.perceive(frame, state)

    assert secondary.scoped_hints == {"run-1": ["run-1 hint"]}


class _HookBackend:
    """Records the optional hooks the fallback forwards to the run's active backend."""

    def __init__(self, name: str, *, held: bool = False, resample: bool = True, fail: bool = False) -> None:
        self.name = name
        self.held = held
        self.resample = resample
        self.fail = fail
        self.calls: list[tuple] = []

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        if self.fail:
            raise BackendCompatibilityError(f"{self.name} unavailable")
        return ScreenPerception(
            summary=self.name, page_hint="unknown", visible_elements=[],
            capture_artifact_path=screenshot.artifact_path, confidence=0.9,
        )

    async def choose_action(self, state, perception):
        raise AssertionError("not used")

    def latest_debug_artifacts(self):
        return f"artifacts:{self.name}"

    def clear_advisory_hints(self, run_id: str = "") -> None:
        pass

    def note_not_executed(self, run_id: str, reason: str) -> None:
        self.calls.append(("note_not_executed", run_id, reason))

    def has_held_actions(self, run_id: str) -> bool:
        return self.held

    def discard_held_actions(self, run_id: str, reason: str) -> None:
        self.calls.append(("discard_held_actions", run_id, reason))

    def supports_resampling(self, run_id: str = "") -> bool:
        return self.resample


def _frame() -> CaptureFrame:
    return CaptureFrame(artifact_path="frame.png", width=10, height=10, mime_type="image/png")


@pytest.mark.asyncio
async def test_fallback_backend_reraises_when_the_secondary_is_incompatible_too() -> None:
    secondary = _HookBackend("s")
    backend = FallbackBackend(primary=_HookBackend("p", fail=True), secondary=secondary)
    state = AgentState(run_id="run-1", intent="x", status=RunStatus.RUNNING)

    await backend.perceive(_frame(), state)  # switches the run to the secondary
    secondary.fail = True
    with pytest.raises(BackendCompatibilityError, match="s unavailable"):
        await backend.perceive(_frame(), state)


@pytest.mark.asyncio
async def test_fallback_backend_forwards_run_hooks_to_the_run_s_active_backend() -> None:
    primary = _HookBackend("p", fail=True, held=False, resample=True)
    secondary = _HookBackend("s", held=True, resample=False)
    backend = FallbackBackend(primary=primary, secondary=secondary)

    # Before the switch, hooks go to the primary.
    assert backend.has_held_actions("run-1") is False
    assert backend.supports_resampling("run-1") is True

    await backend.perceive(_frame(), AgentState(run_id="run-1", intent="x", status=RunStatus.RUNNING))
    assert backend.latest_debug_artifacts() == "artifacts:s"

    backend.note_not_executed("run-1", "gate")
    backend.discard_held_actions("run-1", "correction")
    assert backend.has_held_actions("run-1") is True
    assert backend.supports_resampling("run-1") is False
    assert secondary.calls == [("note_not_executed", "run-1", "gate"), ("discard_held_actions", "run-1", "correction")]
    assert primary.calls == []
    # Other runs still use the primary.
    assert backend.has_held_actions("run-2") is False

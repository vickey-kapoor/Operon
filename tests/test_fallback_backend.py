from __future__ import annotations

import pytest

from src.agent.fallback_backend import BackendCompatibilityError, FallbackBackend
from src.models.capture import CaptureFrame
from src.models.common import RunStatus
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.state import AgentState


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

    def add_advisory_hints(self, hints: list[str], source: str = "") -> None:
        self.hints.extend(hints)

    def clear_advisory_hints(self) -> None:
        self.hints = []


class _RunScopedBackend:
    def __init__(self) -> None:
        self.hints: list[str] = []
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

    def add_advisory_hints(self, hints: list[str], source: str = "") -> None:
        self.hints.extend(hints)

    def clear_advisory_hints(self) -> None:
        self.hints = []


class _WorkingBackend:
    def __init__(self) -> None:
        self.hints: list[str] = []
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

    def add_advisory_hints(self, hints: list[str], source: str = "") -> None:
        self.hints.extend(hints)

    def clear_advisory_hints(self) -> None:
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

    backend.add_advisory_hints(["fresh hint"], source="memory")
    await backend.perceive(frame, state)

    assert primary.hints == ["fresh hint"]
    assert secondary.hints == []

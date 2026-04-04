from __future__ import annotations

import pytest

from src.agent.loop import AgentLoop
from src.agent.verifier import DeterministicVerifierService
from src.clients.gemini import PlaceholderGeminiClient
from src.models.common import FailureCategory, RunStatus
from src.models.execution import ExecutedAction
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.state import AgentState
from src.models.verification import VerificationFailureType, VerificationStatus


@pytest.mark.asyncio
async def test_verifier_marks_passive_wait_as_uncertain_when_browser_state_is_still_loading() -> None:
    verifier = DeterministicVerifierService(gemini_client=PlaceholderGeminiClient())
    perception = ScreenPerception(
        summary="Page visible but no structured browser elements extracted.",
        page_hint="unknown",
        visible_elements=[],
        capture_artifact_path="runs/run-1/step_1/before.png",
        confidence=0.4,
    )
    state = AgentState(
        run_id="run-1",
        intent="Open the page and inspect what is visible.",
        status=RunStatus.RUNNING,
        step_count=1,
        observation_history=[perception],
    )
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)
    decision = PolicyDecision(
        action=action,
        rationale="Need more context",
        confidence=0.8,
        active_subgoal="gather more browser context",
    )
    executed = ExecutedAction(
        action=action,
        success=True,
        detail="Waited 1.0s",
    )

    result = await verifier.verify(state, decision, executed)

    assert result.status is VerificationStatus.UNCERTAIN
    assert result.failure_type is VerificationFailureType.UNCERTAIN_SCREEN_STATE
    assert result.failure_category is FailureCategory.UNCERTAIN_SCREEN_STATE


def test_agent_loop_identifies_browser_wait_as_needing_more_signal_when_video_shows_loading() -> None:
    state = AgentState(
        run_id="run-1",
        intent="Open the page and inspect what is visible.",
        status=RunStatus.RUNNING,
        observation_history=[
            ScreenPerception(
                summary="Page visible but no structured browser elements extracted.",
                page_hint="unknown",
                visible_elements=[],
                capture_artifact_path="runs/run-1/step_1/before.png",
                confidence=0.4,
            )
        ],
    )
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)

    assert AgentLoop._passive_wait_needs_more_signal(state, action, "Page is still loading with a spinner visible.") is True


def test_agent_loop_allows_wait_when_video_reveals_usable_page_content() -> None:
    state = AgentState(
        run_id="run-1",
        intent="Open example.com and read the headline.",
        status=RunStatus.RUNNING,
        observation_history=[
            ScreenPerception(
                summary="Page visible but no structured browser elements extracted.",
                page_hint="unknown",
                visible_elements=[],
                capture_artifact_path="runs/run-1/step_1/before.png",
                confidence=0.4,
            )
        ],
    )
    action = AgentAction(action_type=ActionType.WAIT, wait_ms=1000)

    assert AgentLoop._passive_wait_needs_more_signal(state, action, "Example Domain headline is now visible.") is False

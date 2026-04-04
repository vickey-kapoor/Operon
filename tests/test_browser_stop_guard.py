from __future__ import annotations

import pytest

from src.agent.verifier import DeterministicVerifierService
from src.clients.gemini import PlaceholderGeminiClient
from src.models.common import FailureCategory, RunStatus
from src.models.execution import ExecutedAction
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.state import AgentState
from src.models.verification import VerificationStatus


@pytest.mark.asyncio
async def test_verifier_rejects_suspicious_first_step_stop_for_unknown_browser_state() -> None:
    verifier = DeterministicVerifierService(gemini_client=PlaceholderGeminiClient())
    perception = ScreenPerception(
        summary="Browser page visible but no structured elements extracted.",
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
    action = AgentAction(action_type=ActionType.STOP)
    decision = PolicyDecision(
        action=action,
        rationale="done",
        confidence=0.7,
        active_subgoal="complete task",
    )
    executed = ExecutedAction(
        action=action,
        success=True,
        detail="Acknowledged stop",
    )

    result = await verifier.verify(state, decision, executed)

    assert result.status is VerificationStatus.FAILURE
    assert result.stop_condition_met is False
    assert result.failure_category is FailureCategory.EXPECTED_OUTCOME_NOT_MET

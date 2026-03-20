"""Focused tests for deterministic verifier behavior."""

import pytest

from src.agent.verifier import DeterministicVerifierService
from src.clients.gemini import PlaceholderGeminiClient
from src.models.execution import ExecutedAction
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.state import AgentState
from src.models.verification import VerificationFailureType, VerificationStatus


def _service() -> DeterministicVerifierService:
    return DeterministicVerifierService(gemini_client=PlaceholderGeminiClient())


@pytest.mark.asyncio
async def test_verifier_success_case() -> None:
    state = AgentState(run_id="run-1", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.9, active_subgoal="open compose")
    executed = ExecutedAction(action=action, success=True, detail="clicked compose")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.SUCCESS
    assert result.expected_outcome_met is True
    assert result.stop_condition_met is False


@pytest.mark.asyncio
async def test_verifier_expected_outcome_not_met() -> None:
    state = AgentState(run_id="run-2", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.TYPE, target_element_id="subject", text="Hello")
    decision = PolicyDecision(action=action, rationale="Fill subject.", confidence=0.8, active_subgoal="fill subject")
    executed = ExecutedAction(action=action, success=False, detail="typing failed")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.FAILURE
    assert result.expected_outcome_met is False
    assert result.failure_type is VerificationFailureType.ACTION_FAILED
    assert result.recovery_hint == "retry_same_step"


@pytest.mark.asyncio
async def test_verifier_stop_before_send_boundary() -> None:
    state = AgentState(run_id="run-3", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.STOP)
    decision = PolicyDecision(action=action, rationale="Stop before send.", confidence=1.0, active_subgoal="stop before send")
    executed = ExecutedAction(action=action, success=True, detail="stopped before send")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.SUCCESS
    assert result.expected_outcome_met is True
    assert result.stop_condition_met is True
    assert result.failure_type is VerificationFailureType.STOP_BOUNDARY_REACHED


@pytest.mark.asyncio
async def test_verifier_uncertain_case() -> None:
    state = AgentState(run_id="run-4", intent="Create draft", status="running")
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="compose")
    decision = PolicyDecision(action=action, rationale="Open compose.", confidence=0.2, active_subgoal="open compose")
    executed = ExecutedAction(action=action, success=True, detail="clicked compose")

    result = await _service().verify(state, decision, executed)

    assert result.status is VerificationStatus.UNCERTAIN
    assert result.expected_outcome_met is False
    assert result.stop_condition_met is False
    assert result.failure_type is VerificationFailureType.UNCERTAIN_SCREEN_STATE

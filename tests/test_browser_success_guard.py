from __future__ import annotations

import pytest

from operon.agent.verifier import DeterministicVerifierService
from operon.clients.gemini import PlaceholderGeminiClient
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.perception import ScreenPerception, UIElement, UIElementType
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState
from operon.models.verification import VerificationStatus


@pytest.mark.asyncio
async def test_verifier_does_not_treat_successfully_navigated_as_task_completion() -> None:
    verifier = DeterministicVerifierService(gemini_client=PlaceholderGeminiClient())
    learn_more_link = UIElement(
        element_id="learn-more",
        element_type=UIElementType.LINK,
        name="Learn more",
        x=229,
        y=257,
        width=80,
        height=20,
        is_interactable=True,
    )
    perception = ScreenPerception(
        summary='I have successfully navigated to example.com and can now click the "Learn more" link.',
        page_hint="unknown",
        visible_elements=[learn_more_link, learn_more_link, learn_more_link],
        capture_artifact_path="runs/run-1/step_2/before.png",
        confidence=0.7,
    )
    state = AgentState(
        run_id="run-1",
        intent="Open example.com and click the More information link.",
        status=RunStatus.RUNNING,
        step_count=2,
        observation_history=[perception],
    )
    action = AgentAction(action_type=ActionType.CLICK, x=229, y=257)
    decision = PolicyDecision(
        action=action,
        rationale='Click the "Learn more" link.',
        confidence=0.7,
        active_subgoal="click_at",
    )
    executed = ExecutedAction(
        action=action,
        success=True,
        detail="Executed click",
    )

    result = await verifier.verify(state, decision, executed)

    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_condition_met is False
    assert result.stop_reason is None

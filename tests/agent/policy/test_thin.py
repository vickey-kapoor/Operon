"""Thin harness profile: the model decides, the executor runs it once, success is checked at the end."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from operon.agent.loop import AgentLoop
from operon.agent.policy.thin import (
    FULL,
    THIN,
    FinalCheckVerifier,
    ThinPolicy,
    ThinRecovery,
    final_check_model,
    harness_profile,
)
from operon.clients.anthropic import AnthropicClientError
from operon.models.common import FailureCategory, RunStatus, StopReason
from operon.models.execution import ExecutedAction
from operon.models.perception import PageHint, ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.recovery import RecoveryStrategy
from operon.models.state import AgentState
from operon.models.usage import ModelUsage
from operon.models.verification import VerificationStatus


def _state(tmp_path: Path) -> AgentState:
    screen = tmp_path / "before.png"
    screen.write_bytes(b"png")
    perception = ScreenPerception(
        summary="Inbox", page_hint=PageHint.UNKNOWN, capture_artifact_path=str(screen), elements_reported=False,
    )
    return AgentState(
        run_id="run-thin", intent="Archive the newest email", status=RunStatus.RUNNING,
        observation_history=[perception],
    )


def _decision(action: AgentAction, rationale: str = "Clicking archive.") -> PolicyDecision:
    return PolicyDecision(action=action, rationale=rationale, confidence=0.7, active_subgoal="archive")


def _executed(action: AgentAction, *, success: bool = True, detail: str = "ok") -> ExecutedAction:
    return ExecutedAction(action=action, success=success, detail=detail)


@pytest.mark.parametrize(
    ("requested", "backend", "expected"),
    [
        (None, "claude_computer_use", THIN),
        ("thin", "claude_computer_use", THIN),
        ("FULL", "claude_computer_use", FULL),
        (None, "computer_use", FULL),
        ("thin", "computer_use", FULL),
        ("thin", "json", FULL),
    ],
)
def test_claude_runs_thin_by_default_and_other_backends_run_full(monkeypatch, requested, backend, expected) -> None:
    if requested is None:
        monkeypatch.delenv("OPERON_HARNESS_PROFILE", raising=False)
    else:
        monkeypatch.setenv("OPERON_HARNESS_PROFILE", requested)
    assert harness_profile(backend) == expected


@pytest.mark.asyncio
async def test_thin_policy_returns_the_backend_decision_unchanged(tmp_path) -> None:
    decision = _decision(AgentAction(action_type=ActionType.CLICK, x=10, y=20))
    delegate = Mock()
    delegate.choose_action = AsyncMock(return_value=decision)
    state = _state(tmp_path)

    assert await ThinPolicy(delegate).choose_action(state, state.observation_history[-1]) is decision
    assert ThinPolicy(delegate).supports_resampling() is False


@pytest.mark.asyncio
async def test_an_ordinary_step_is_not_judged_by_a_model(tmp_path) -> None:
    client = Mock(generate_judgement=AsyncMock())
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)

    result = await FinalCheckVerifier(client).verify(_state(tmp_path), _decision(action), _executed(action))

    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_condition_met is False
    client.generate_judgement.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_action_is_left_for_the_model_not_retried(tmp_path) -> None:
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    executed = _executed(action, success=False, detail="Click did nothing").model_copy(
        update={"failure_category": FailureCategory.EXECUTION_ERROR}
    )

    result = await FinalCheckVerifier(Mock()).verify(_state(tmp_path), _decision(action), executed)

    # UNCERTAIN, not FAILURE: the loop's in-step retry only runs on FAILURE.
    assert result.status is VerificationStatus.UNCERTAIN
    assert "Click did nothing" in result.reason


@pytest.mark.asyncio
async def test_done_is_confirmed_once_against_the_task(tmp_path) -> None:
    client = Mock(generate_judgement=AsyncMock(return_value='{"complete": true, "reason": "Inbox shows no email."}'))
    stop = AgentAction(action_type=ActionType.STOP)
    state = _state(tmp_path)

    result = await FinalCheckVerifier(client).verify(state, _decision(stop, "Archived it."), _executed(stop))

    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_condition_met is True
    assert result.stop_reason is StopReason.TASK_COMPLETED
    call = client.generate_judgement.await_args
    assert "Archive the newest email" in call.args[0] and "Archived it." in call.args[0]
    assert call.kwargs["image_data"] and call.kwargs["effort"] == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        AsyncMock(return_value='{"complete": false, "reason": "The email is still in the inbox."}'),
        AsyncMock(return_value="I think so?"),
    ],
)
async def test_an_unconfirmed_done_goes_back_to_the_model(tmp_path, reply) -> None:
    hints: list[tuple[str, str]] = []
    verifier = FinalCheckVerifier(Mock(generate_judgement=reply), on_rejected=lambda run, hint: hints.append((run, hint)))
    stop = AgentAction(action_type=ActionType.STOP)

    result = await verifier.verify(_state(tmp_path), _decision(stop, "Done."), _executed(stop))

    assert result.status is VerificationStatus.UNCERTAIN
    assert result.stop_condition_met is False
    assert len(hints) == 1 and hints[0][0] == "run-thin"
    assert "not done yet" in hints[0][1]


@pytest.mark.asyncio
async def test_thin_recovery_never_resets_the_screen(tmp_path) -> None:
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    failed = await FinalCheckVerifier(Mock()).verify(
        _state(tmp_path), _decision(action), _executed(action, success=False, detail="x")
    )

    recovery = await ThinRecovery().recover(_state(tmp_path), _decision(action), _executed(action, success=False), failed)

    assert recovery.strategy is RecoveryStrategy.ADVANCE


@pytest.mark.asyncio
async def test_thin_recovery_stops_on_a_confirmed_success(tmp_path) -> None:
    client = Mock(generate_judgement=AsyncMock(return_value='{"complete": true, "reason": "Done."}'))
    stop = AgentAction(action_type=ActionType.STOP)
    confirmed = await FinalCheckVerifier(client).verify(_state(tmp_path), _decision(stop), _executed(stop))

    recovery = await ThinRecovery().recover(_state(tmp_path), _decision(stop), _executed(stop), confirmed)

    assert recovery.strategy is RecoveryStrategy.STOP
    assert recovery.stop_reason is StopReason.TASK_COMPLETED


def _loop(profile: str) -> AgentLoop:
    return AgentLoop(
        capture_service=Mock(capture=AsyncMock()),
        perception_service=Mock(perceive=AsyncMock()),
        run_store=Mock(),
        policy_service=Mock(),
        executor=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
        harness_profile=profile,
    )


@pytest.mark.asyncio
async def test_thin_loop_runs_a_failed_action_once_without_a_second_look(tmp_path) -> None:
    loop = _loop(THIN)
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    failed = _executed(action, success=False, detail="target moved").model_copy(
        update={"failure_category": FailureCategory.CLICK_NO_EFFECT}
    )
    loop.executor_adapter = Mock(execute=AsyncMock(return_value=failed))
    loop._persist_execution_trace = lambda run_id, step, executed: executed
    state = _state(tmp_path)

    _, executed = await loop._execute_with_hardening(
        record=state, state=state, perception=state.observation_history[-1],
        decision_action=action, step_index=1, use_unified_executor=True,
    )

    assert executed is failed
    loop.executor_adapter.execute.assert_awaited_once()
    loop.capture_service.capture.assert_not_awaited()
    loop.perception_service.perceive.assert_not_awaited()


def test_the_profile_reaches_the_loop() -> None:
    assert _loop(THIN).thin_harness is True
    assert _loop(FULL).thin_harness is False


@pytest.mark.asyncio
async def test_a_check_that_cannot_run_takes_the_model_at_its_word(tmp_path) -> None:
    """A failing check used to reject every claim, so runs went in circles to the step limit."""
    hints: list[tuple[str, str]] = []
    client = Mock(generate_judgement=AsyncMock(side_effect=AnthropicClientError("Anthropic HTTP error 400")))
    verifier = FinalCheckVerifier(client, on_rejected=lambda run, hint: hints.append((run, hint)))
    stop = AgentAction(action_type=ActionType.STOP)

    result = await verifier.verify(_state(tmp_path), _decision(stop, "Done."), _executed(stop))

    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_reason is StopReason.TASK_COMPLETED
    assert "400" in result.critic_fallback_reason
    assert hints == []


@pytest.mark.asyncio
async def test_the_check_request_uses_effort_not_disabled_thinking() -> None:
    """Opus 4.7+ and Sonnet 5+ reject thinking.type=disabled with a 400."""
    from operon.clients.anthropic import AnthropicHttpClient

    client = AnthropicHttpClient(api_key="test", model="claude-opus-5-5")
    client._post_json = AsyncMock(return_value={"content": [{"type": "text", "text": '{"complete": true, "reason": "ok"}'}]})

    await client.generate_judgement("Is it done?", image_data="aGk=", media_type="image/jpeg", effort="low")

    payload = client._post_json.await_args.args[0]
    assert "thinking" not in payload
    assert payload["output_config"] == {"effort": "low"}
    assert payload["messages"][0]["content"][1]["source"]["media_type"] == "image/jpeg"


def test_the_completion_check_can_use_a_cheaper_model(monkeypatch) -> None:
    monkeypatch.delenv("OPERON_FINAL_CHECK_MODEL", raising=False)
    assert final_check_model("claude-opus-5-5") == "claude-opus-5-5"
    monkeypatch.setenv("OPERON_FINAL_CHECK_MODEL", "claude-sonnet-5-5")
    assert final_check_model("claude-opus-5-5") == "claude-sonnet-5-5"


@pytest.mark.asyncio
async def test_the_completion_check_records_its_usage(tmp_path) -> None:
    usage = ModelUsage(provider="anthropic", model="claude-opus-5-5", request_kind="image", input_tokens=900, output_tokens=40)
    client = Mock(
        generate_judgement=AsyncMock(return_value='{"complete": true, "reason": "Done."}'),
        latest_usage=Mock(return_value=usage),
    )
    stop = AgentAction(action_type=ActionType.STOP)
    verifier = FinalCheckVerifier(client)

    await verifier.verify(_state(tmp_path), _decision(stop, "Done."), _executed(stop))

    debug = verifier.latest_debug_artifacts()
    assert debug.usage == usage
    assert debug.usage_artifact_path.endswith("final_check_usage.json")

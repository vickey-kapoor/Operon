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


@pytest.mark.asyncio
async def test_the_check_also_sees_earlier_screens_so_work_the_last_screen_hides_still_counts(tmp_path) -> None:
    # Seen live: "change the password, then sign out" ends on a sign-in page that
    # cannot show the change; with only that screen the check rejected "done" 20 times.
    from PIL import Image

    def screen(name: str, shade: int) -> str:
        path = tmp_path / name
        Image.new("RGB", (64, 48), (shade, shade, shade)).save(path)
        return str(path)

    paths = [screen(f"step_{i}.png", 40 * i) for i in range(1, 6)]
    state = AgentState(
        run_id="run-thin", intent="Change my password, then sign out", status=RunStatus.RUNNING,
        observation_history=[
            ScreenPerception(summary=f"step {i}", page_hint=PageHint.UNKNOWN, capture_artifact_path=path,
                             elements_reported=False)
            for i, path in enumerate(paths, 1)
        ],
    )
    client = Mock(generate_judgement=AsyncMock(return_value='{"complete": true, "reason": "Changed, then signed out."}'))
    stop = AgentAction(action_type=ActionType.STOP)

    result = await FinalCheckVerifier(client).verify(state, _decision(stop, "Done."), _executed(stop))

    assert result.status is VerificationStatus.SUCCESS
    call = client.generate_judgement.await_args
    # The first screen and the last three before the final one, oldest first; the final
    # one is the main image.
    assert len(call.kwargs["earlier_images"]) == 4
    assert "first 4 images are earlier screens" in call.args[0]


@pytest.mark.asyncio
async def test_earlier_screens_go_before_the_final_one_in_the_request() -> None:
    from operon.clients.anthropic import AnthropicHttpClient

    client = AnthropicHttpClient(api_key="test", model="claude-opus-5-5")
    client._post_json = AsyncMock(return_value={"content": [{"type": "text", "text": '{"complete": true, "reason": "ok"}'}]})

    await client.generate_judgement("Is it done?", image_data="ZmluYWw=", media_type="image/jpeg",
                                    earlier_images=[("b25l", "image/png"), ("dHdv", "image/png")])

    content = client._post_json.await_args.args[0]["messages"][0]["content"]
    assert [block.get("source", {}).get("data") for block in content[1:]] == ["b25l", "dHdv", "ZmluYWw="]


def _screens_state(tmp_path: Path, shades: list[int], actions: list[ExecutedAction] | None = None) -> AgentState:
    from PIL import Image

    observations = []
    for i, shade in enumerate(shades, 1):
        path = tmp_path / f"step_{i}.png"
        Image.new("RGB", (64, 48), (shade, shade, shade)).save(path)
        observations.append(ScreenPerception(summary=f"step {i}", page_hint=PageHint.UNKNOWN,
                                             capture_artifact_path=str(path), elements_reported=False))
    return AgentState(run_id="run-thin", intent="Register for the conference", status=RunStatus.RUNNING,
                      observation_history=observations, action_history=actions or [])


@pytest.mark.asyncio
async def test_repeated_identical_screens_do_not_crowd_out_the_proof(tmp_path) -> None:
    # Seen live: each rejected claim added another copy of the confirmation page, until
    # the screen showing the filled-in form fell out of the window the check sees.
    state = _screens_state(tmp_path, [10, 60, 200, 200, 200, 200])
    client = Mock(generate_judgement=AsyncMock(return_value='{"complete": true, "reason": "ok"}'))
    stop = AgentAction(action_type=ActionType.STOP)

    await FinalCheckVerifier(client).verify(state, _decision(stop, "Done."), _executed(stop))

    call = client.generate_judgement.await_args
    assert len(call.kwargs["earlier_images"]) == 2
    assert "first 2 images are earlier screens" in call.args[0]


def test_the_first_screen_stays_in_view_however_long_the_run(tmp_path) -> None:
    # Seen live: "order the cheapest lamp" was rejected 3 of 3 times because the price
    # list Claude picked from was on the first screen, which had left the window.
    from operon.agent.policy.thin import _earlier_screens

    state = _screens_state(tmp_path, [10, 50, 90, 130, 170, 210, 250])
    paths = [obs.capture_artifact_path for obs in state.observation_history]

    assert _earlier_screens(state, paths[-1]) == [paths[0], paths[3], paths[4], paths[5]]


@pytest.mark.asyncio
async def test_the_check_sees_what_was_typed_even_when_no_screen_shows_it(tmp_path) -> None:
    # Seen live: Claude typed the coupon and clicked Apply in one turn, so no screenshot
    # ever showed the code in the field and the check kept saying it was never entered.
    batch = AgentAction(action_type=ActionType.BATCH, actions=[
        AgentAction(action_type=ActionType.TYPE, text="SAVE10", x=120, y=80),
        AgentAction(action_type=ActionType.CLICK, x=300, y=80),
    ])
    actions = [
        _executed(AgentAction(action_type=ActionType.NAVIGATE, url="http://localhost:8765/shop")),
        _executed(batch),
        _executed(AgentAction(action_type=ActionType.PRESS_KEY, key="Enter"), success=False, detail="no focus"),
    ]
    state = _screens_state(tmp_path, [10, 60], actions)
    client = Mock(generate_judgement=AsyncMock(return_value='{"complete": true, "reason": "ok"}'))
    stop = AgentAction(action_type=ActionType.STOP)

    await FinalCheckVerifier(client).verify(state, _decision(stop, "Done."), _executed(stop))

    prompt = client.generate_judgement.await_args.args[0]
    assert "1. navigate http://localhost:8765/shop (ok)" in prompt
    assert '2. type "SAVE10" at (120, 80); click at (300, 80) (ok)' in prompt
    assert "3. press_key key Enter (failed: no focus)" in prompt


@pytest.mark.asyncio
async def test_a_claim_rejected_twice_ends_the_run_as_unverified(tmp_path) -> None:
    hints: list[tuple[str, str]] = []
    client = Mock(generate_judgement=AsyncMock(return_value='{"complete": false, "reason": "No confirmation."}'))
    verifier = FinalCheckVerifier(client, on_rejected=lambda run, hint: hints.append((run, hint)))
    stop = AgentAction(action_type=ActionType.STOP)
    state = _state(tmp_path)

    first = await verifier.verify(state, _decision(stop, "Done."), _executed(stop))
    state.verification_history.append(first)
    second = await verifier.verify(state, _decision(stop, "Done, really."), _executed(stop))

    assert first.stop_condition_met is False and len(hints) == 1
    assert second.status is VerificationStatus.FAILURE
    assert second.stop_condition_met is True
    assert second.stop_reason is StopReason.COMPLETION_UNVERIFIED
    assert len(hints) == 1
    recovery = await ThinRecovery().recover(state, _decision(stop), _executed(stop), second)
    assert recovery.strategy is RecoveryStrategy.STOP and recovery.stop_reason is StopReason.COMPLETION_UNVERIFIED

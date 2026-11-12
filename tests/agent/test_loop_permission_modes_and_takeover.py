"""Permission modes (ask / risky / auto) and takeover in the loop (KAN-89)."""

from __future__ import annotations

import json

import pytest

from operon.agent import control
from operon.agent.policy.trust_gate import TrustGate
from operon.models.common import RunStatus, RunTaskRequest, StepRequest
from operon.models.policy import ActionType
from tests.fixtures.frontdoor import CLICK_SAVE, PURCHASE, build_loop


async def _start(loop, mode: str | None = None, max_steps: int = 10) -> str:
    response = await loop.start_run(RunTaskRequest(intent="Do the task", permission_mode=mode, max_steps=max_steps))
    return response.run_id


async def _step(loop, run_id: str):
    return await loop.step_run(StepRequest(run_id=run_id))


async def test_ask_mode_pauses_before_an_ordinary_action(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = await _start(loop, "ask")

    response = await _step(loop, run_id)

    assert response.status is RunStatus.WAITING_FOR_USER
    assert executor.executed == []
    assert loop.pending_gate(run_id) == "trust"
    state = await loop.run_store.get_run(run_id)
    assert state.pause_reason.startswith("Approval needed (ask mode): click")


async def test_ask_mode_runs_the_action_once_approved_and_counts_it(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = await _start(loop, "ask")

    await _step(loop, run_id)
    await loop.resume_run(run_id)
    await _step(loop, run_id)  # approved: runs
    third = await _step(loop, run_id)  # the next action asks again

    assert [a.action_type for a in executor.executed] == [ActionType.CLICK]
    assert third.status is RunStatus.WAITING_FOR_USER
    state = await loop.run_store.get_run(run_id)
    assert state.approval_count == 1


async def test_ask_mode_asks_even_with_the_gate_off(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE], gate=TrustGate(enabled=False))
    run_id = await _start(loop, "ask")

    assert (await _step(loop, run_id)).status is RunStatus.WAITING_FOR_USER
    assert executor.executed == []


async def test_auto_mode_runs_a_confirm_list_action_without_asking(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE])
    run_id = await _start(loop, "auto")

    response = await _step(loop, run_id)

    assert response.status is not RunStatus.WAITING_FOR_USER
    assert [a.text for a in executor.executed] == ["purchase now"]


async def test_auto_mode_never_relaxes_a_deny_rule(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE], gate=TrustGate(enabled=True, deny=("purchase",)))
    run_id = await _start(loop, "auto")

    assert (await _step(loop, run_id)).status is RunStatus.WAITING_FOR_USER
    assert executor.executed == []


async def test_risky_mode_is_the_gate_as_configured(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE, PURCHASE])
    run_id = await _start(loop, "risky")

    await _step(loop, run_id)  # an ordinary click runs
    second = await _step(loop, run_id)  # a confirm-list action waits

    assert [a.action_type for a in executor.executed] == [ActionType.CLICK]
    assert second.status is RunStatus.WAITING_FOR_USER


async def test_a_run_without_a_mode_uses_the_server_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPERON_PERMISSION_MODE", "auto")
    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE])
    run_id = await _start(loop, None)

    await _step(loop, run_id)

    assert [a.text for a in executor.executed] == ["purchase now"]


async def test_the_mode_is_stored_on_the_run(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = await _start(loop, "ask")

    stored = json.loads((tmp_path / run_id / "state.json").read_text(encoding="utf-8"))
    assert stored["permission_mode"] == "ask"


async def test_takeover_pauses_the_run_and_no_step_runs(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = await _start(loop, "auto")
    await _step(loop, run_id)

    state = await loop.take_over(run_id)
    captures_before = loop.capture_service.capture.await_count
    response = await _step(loop, run_id)

    assert state.in_takeover and state.takeover_count == 1
    assert response.status is RunStatus.WAITING_FOR_USER
    assert loop.capture_service.capture.await_count == captures_before  # no screenshot
    assert loop.policy_service.calls == 1  # no model call
    assert len(executor.executed) == 1


async def test_hand_back_resumes_and_tells_the_model_the_screen_changed(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = await _start(loop, "auto")
    await loop.take_over(run_id)

    response = await loop.hand_back(run_id, "I logged in")

    state = await loop.run_store.get_run(run_id)
    assert response.status is RunStatus.RUNNING
    assert not state.in_takeover and state.force_fresh_perception
    hint = control.consume_override_hint(run_id)
    assert "handed it back" in hint and "I logged in" in hint


async def test_hand_back_after_a_gate_pause_does_not_approve_the_action(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE])
    run_id = await _start(loop, "risky")
    await _step(loop, run_id)  # paused on the purchase
    await loop.take_over(run_id)
    await loop.hand_back(run_id)

    response = await _step(loop, run_id)

    assert executor.executed == []
    assert response.status is RunStatus.WAITING_FOR_USER


async def test_take_over_refuses_a_finished_run(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = await _start(loop)
    await loop.run_store.set_status(run_id, RunStatus.SUCCEEDED)

    with pytest.raises(ValueError):
        await loop.take_over(run_id)


async def test_hand_back_refuses_a_run_not_taken_over(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = await _start(loop)

    with pytest.raises(ValueError):
        await loop.hand_back(run_id)


async def test_takeover_waits_for_the_step_in_flight(monkeypatch, tmp_path) -> None:
    """Control passes only once the agent's current step has ended, so it never acts
    on a screen the person is already changing."""
    import asyncio

    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = await _start(loop, "auto")
    release = asyncio.Event()
    real_perceive = loop.perception_service.perceive

    async def slow_perceive(*args, **kwargs):
        await release.wait()
        return await real_perceive(*args, **kwargs)

    loop.perception_service.perceive = slow_perceive
    step = asyncio.create_task(_step(loop, run_id))
    await asyncio.sleep(0.05)  # the step is now waiting on the model
    takeover = asyncio.create_task(loop.take_over(run_id))
    await asyncio.sleep(0.05)

    assert not takeover.done()
    state = await loop.run_store.get_run(run_id)
    assert not state.in_takeover

    release.set()
    await step
    state = await takeover
    assert state.in_takeover and state.status is RunStatus.WAITING_FOR_USER


async def test_resume_brings_back_the_window_the_run_was_using(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    restored: list[str] = []

    async def restore_work_window(run_id: str) -> bool:
        restored.append(run_id)
        return True

    executor.restore_work_window = restore_work_window
    run_id = await _start(loop, "ask")
    await _step(loop, run_id)  # paused; answering it puts the task page in front

    await loop.resume_run(run_id)

    assert restored == [run_id]


async def test_a_retried_action_stays_in_the_step_log(monkeypatch, tmp_path) -> None:
    from operon.models.logs import StepLog
    from operon.models.verification import VerificationResult, VerificationStatus
    from operon.store.background_writer import bg_writer
    from operon.store.replay import load_run_replay

    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE, CLICK_SAVE])
    loop.verifier_service.verify.side_effect = [
        VerificationResult(
            status=VerificationStatus.FAILURE, expected_outcome_met=False, stop_condition_met=False, reason="not yet"
        ),
        VerificationResult(
            status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=False, reason="done"
        ),
    ]
    run_id = await _start(loop, "auto")

    await _step(loop, run_id)
    await bg_writer.flush()

    assert [a.action_type for a in executor.executed] == [ActionType.TYPE, ActionType.CLICK]
    (entry,) = [e for e in load_run_replay(run_id, root_dir=loop.run_store.root_dir) if isinstance(e, StepLog)]
    assert entry.executed_action.action.action_type is ActionType.CLICK
    assert [a.action.text for a in entry.earlier_attempts] == ["purchase now"]


async def test_a_retry_that_waits_for_approval_is_not_logged_as_a_retry(monkeypatch, tmp_path) -> None:
    from operon.models.logs import StepLog
    from operon.models.verification import VerificationResult, VerificationStatus
    from operon.store.background_writer import bg_writer
    from operon.store.replay import load_run_replay

    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE, PURCHASE])
    loop.verifier_service.verify.return_value = VerificationResult(
        status=VerificationStatus.FAILURE, expected_outcome_met=False, stop_condition_met=False, reason="not yet"
    )
    run_id = await _start(loop, "risky")

    response = await _step(loop, run_id)  # the click fails its check; the retry (purchase) needs approval
    await bg_writer.flush()

    assert response.status is RunStatus.WAITING_FOR_USER
    assert [a.action_type for a in executor.executed] == [ActionType.CLICK]
    (entry,) = [e for e in load_run_replay(run_id, root_dir=loop.run_store.root_dir) if isinstance(e, StepLog)]
    assert entry.executed_action.action.action_type is ActionType.CLICK
    assert entry.earlier_attempts == []

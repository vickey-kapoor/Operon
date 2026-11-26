"""KAN-204: a run stops, FAILED, after 3 replies in a row with words only and no action."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from operon.agent import control as run_control
from operon.agent.backends.model_adapter.backend import (
    GAVE_UP_PREFIX,
    ModelAdapterBackend,
)
from operon.agent.backends.model_adapter.registry import ModelProfile
from operon.agent.loop import AgentLoop
from operon.agent.policy.thin import (
    WORDS_ONLY_LIMIT,
    OwnDoneVerifier,
    ThinPolicy,
    ThinRecovery,
)
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import (
    FailureCategory,
    RunFailureCause,
    RunStatus,
    RunTaskRequest,
    StepRequest,
    StopReason,
)
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.recovery import RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import VerificationResult, VerificationStatus
from operon.store.run_store import FileBackedRunStore

from .backends.model_adapter.conftest import ScriptedChat, chat_reply, tool_call

PROFILE = ModelProfile(name="qwen3-vl-32b", adapter="openai_compat", model="Qwen/Qwen3-VL-32B-Instruct", coords="relative_1000")

TALK = "Which account should I use? I need to know before I go on."


def _talk(text: str = TALK) -> dict:
    return chat_reply(content=text)


def _click() -> dict:
    return chat_reply(tool_calls=[tool_call("click", x=500, y=100)])


# --- ThinRecovery -----------------------------------------------------------------------------


def _decision(words: str | None) -> PolicyDecision:
    return PolicyDecision(
        action=AgentAction(action_type=ActionType.WAIT, wait_ms=500),
        rationale="r",
        confidence=0.4,
        active_subgoal="s",
        words_only=words,
    )


def _continue() -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=False, reason="ok"
    )


async def _recover(words: str | None, streak: int):
    state = AgentState(run_id="r", intent="t", status=RunStatus.RUNNING, words_only_streak=streak)
    executed = ExecutedAction(action=AgentAction(action_type=ActionType.WAIT, wait_ms=500), success=True, detail="ok")
    return await ThinRecovery().recover(state, _decision(words), executed, _continue())


async def test_recovery_stops_at_the_limit() -> None:
    assert WORDS_ONLY_LIMIT == 3
    recovery = await _recover(TALK, 3)
    assert recovery.strategy is RecoveryStrategy.STOP and recovery.terminal and not recovery.recoverable
    assert recovery.stop_reason is StopReason.MODEL_ONLY_TALKED
    assert recovery.failure_category is FailureCategory.MODEL_ONLY_TALKED
    assert "3 times in a row without taking any action" in recovery.message
    assert f'"{TALK}"' in recovery.message


async def test_recovery_does_not_stop_below_the_limit() -> None:
    for streak in (0, 1, 2):
        assert (await _recover(TALK, streak)).strategy is RecoveryStrategy.ADVANCE


async def test_recovery_ignores_a_normal_decision_even_with_a_streak() -> None:
    assert (await _recover(None, 3)).strategy is RecoveryStrategy.ADVANCE


async def test_recovery_message_trims_long_words_and_names_empty_ones() -> None:
    long = (await _recover("y" * 3000, 3)).message
    assert len(long) < 500 and long.endswith('..."')
    assert '(no text)' in (await _recover("", 3)).message


# --- the real loop with a scripted model ($0) -------------------------------------------------


def _model_loop(tmp_path: Path, replies) -> tuple[AgentLoop, FileBackedRunStore]:
    run_store = FileBackedRunStore(root_dir=tmp_path / "runs")
    backend = ModelAdapterBackend(
        profile=PROFILE,
        client=ScriptedChat(replies),
        environment="browser",
        task_prompt_path=prompts_dir() / "model_adapter_task_prompt.txt",
    )

    async def capture(record) -> CaptureFrame:
        path = tmp_path / "frames" / f"{record.run_id}-{record.step_count + 1}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1280, 800), "white").save(path)
        return CaptureFrame(artifact_path=str(path), width=1280, height=800)

    async def execute(action: AgentAction) -> ExecutedAction:
        return ExecutedAction(action=action, success=True, detail="ok")

    executor = Mock()
    executor.execute = AsyncMock(side_effect=execute)
    loop = AgentLoop(
        capture_service=SimpleNamespace(capture=capture),
        perception_service=backend,
        run_store=run_store,
        policy_service=ThinPolicy(backend),
        executor=executor,
        verifier_service=OwnDoneVerifier(gave_up_prefix=GAVE_UP_PREFIX),
        recovery_manager=ThinRecovery(),
    )
    return loop, run_store


def _rows(run_store: FileBackedRunStore, run_id: str) -> list[dict]:
    text = (run_store.root_dir / run_id / "run.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


async def _start(loop: AgentLoop):
    return await loop.start_run(RunTaskRequest(intent="Open the Products page", max_steps=10))


async def test_three_word_only_replies_end_the_run_failed(tmp_path: Path) -> None:
    loop, run_store = _model_loop(tmp_path, [_talk()])  # the script repeats its last reply
    run = await _start(loop)

    statuses = [(await loop.step_run(StepRequest(run_id=run.run_id))).status for _ in range(3)]

    assert statuses == [RunStatus.RUNNING, RunStatus.RUNNING, RunStatus.FAILED]
    state = await run_store.get_run(run.run_id)
    assert state is not None
    assert state.status is RunStatus.FAILED and state.step_count == 3
    assert state.stop_reason is StopReason.MODEL_ONLY_TALKED
    assert state.failure_cause is RunFailureCause.MODEL_ONLY_TALKED
    assert TALK in (state.failure_detail or "")
    last = _rows(run_store, run.run_id)[-1]
    assert last["failure"]["category"] == FailureCategory.MODEL_ONLY_TALKED.value
    assert last["failure"]["terminal"] is True


async def test_a_reply_that_takes_an_action_breaks_the_streak(tmp_path: Path) -> None:
    loop, run_store = _model_loop(tmp_path, [_talk(), _talk(), _click(), _talk(), _talk()])
    run = await _start(loop)

    for _ in range(5):
        response = await loop.step_run(StepRequest(run_id=run.run_id))

    assert response.status is RunStatus.RUNNING
    state = await run_store.get_run(run.run_id)
    assert state is not None and state.words_only_streak == 2 and state.failure_cause is None


async def test_the_streak_is_saved_with_the_run(tmp_path: Path) -> None:
    loop, run_store = _model_loop(tmp_path, [_talk()])
    run = await _start(loop)
    await loop.step_run(StepRequest(run_id=run.run_id))
    await loop.step_run(StepRequest(run_id=run.run_id))

    saved = json.loads((run_store.root_dir / run.run_id / "state.json").read_text(encoding="utf-8"))
    assert saved["words_only_streak"] == 2


async def test_a_persons_answer_starts_the_count_over(tmp_path: Path) -> None:
    loop, run_store = _model_loop(tmp_path, [_talk()])
    run = await _start(loop)
    await loop.step_run(StepRequest(run_id=run.run_id))
    await loop.step_run(StepRequest(run_id=run.run_id))

    run_control.set_override_hint(run.run_id, "Use the work account.")
    response = await loop.step_run(StepRequest(run_id=run.run_id))

    # The model talked again, but this is its first word-only reply since the answer.
    assert response.status is RunStatus.RUNNING
    state = await run_store.get_run(run.run_id)
    assert state is not None and state.words_only_streak == 1


async def test_resuming_a_paused_run_starts_the_count_over(tmp_path: Path) -> None:
    loop, run_store = _model_loop(tmp_path, [_talk()])
    run = await _start(loop)
    await loop.step_run(StepRequest(run_id=run.run_id))
    await loop.step_run(StepRequest(run_id=run.run_id))
    record = await run_store.get_run(run.run_id)
    assert record is not None and record.words_only_streak == 2
    record.status = RunStatus.WAITING_FOR_USER

    await loop.resume_run(run.run_id, approve_gated_action=False)

    assert record.words_only_streak == 0


async def test_a_model_that_gives_up_is_still_a_give_up(tmp_path: Path) -> None:
    loop, run_store = _model_loop(tmp_path, [_talk(), _talk(), chat_reply(tool_calls=[tool_call("fail", reason="blocked")])])
    run = await _start(loop)
    for _ in range(3):
        await loop.step_run(StepRequest(run_id=run.run_id))

    state = await run_store.get_run(run.run_id)
    assert state is not None and state.failure_cause is RunFailureCause.MODEL_GAVE_UP


# --- old files still load ---------------------------------------------------------------------


def test_old_state_and_decision_files_load_without_the_new_fields() -> None:
    state = AgentState.model_validate({"run_id": "r", "intent": "t", "status": "running"})
    assert state.words_only_streak == 0
    decision = PolicyDecision.model_validate(
        {"action": {"action_type": "wait", "wait_ms": 500}, "rationale": "r", "confidence": 0.4, "active_subgoal": "s"}
    )
    assert decision.words_only is None
    assert RunFailureCause("model_only_talked") is RunFailureCause.MODEL_ONLY_TALKED
    assert StopReason("model_only_talked") is StopReason.MODEL_ONLY_TALKED
    assert FailureCategory("model_only_talked") is FailureCategory.MODEL_ONLY_TALKED


@pytest.mark.parametrize("bad", [-1])
def test_the_streak_cannot_be_negative(bad: int) -> None:
    with pytest.raises(ValueError):
        AgentState.model_validate({"run_id": "r", "intent": "t", "status": "running", "words_only_streak": bad})

"""KAN-175: every failed run carries one cause from a fixed list, with its text."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from PIL import Image

from operon.agent.backends.base import BackendCompatibilityError, ModelBlockedError
from operon.agent.backends.model_adapter.backend import (
    GAVE_UP_PREFIX,
    ModelAdapterBackend,
)
from operon.agent.backends.model_adapter.registry import ModelProfile
from operon.agent.backends.model_adapter.transports import ModelTransportError
from operon.agent.efficiency import run_efficiency
from operon.agent.failure_cause import cause_from_state, cause_of_error, note_failure
from operon.agent.loop import AgentLoop
from operon.agent.perception import PerceptionLowQualityError
from operon.agent.policy.thin import OwnDoneVerifier, ThinPolicy, ThinRecovery
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
from operon.models.policy import AgentAction
from operon.models.state import AgentState
from operon.store.replay import load_run_replay, render_run_replay
from operon.store.run_store import FileBackedRunStore

from .backends.model_adapter.conftest import ScriptedChat, chat_reply, tool_call

PROFILE = ModelProfile(name="qwen3-vl-32b", adapter="openai_compat", model="Qwen/Qwen3-VL-32B-Instruct", coords="relative_1000")


# --- the cause table -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "stop_reason", "expected"),
    [
        ("failed", "max_step_limit_reached", "max_steps"),
        ("failed", "completion_unverified", "model_gave_up"),
        ("failed", "model_blocked", "model_blocked"),
        ("failed", "pre_step_perception_failed", "perception_failed"),
        ("failed", "perception_low_quality", "perception_failed"),
        ("failed", "repeated_failure_loop", "other"),
        ("failed", None, "other"),
        ("cancelled", None, "cancelled"),
        ("waiting_for_user", "interrupted", "interrupted"),
        ("waiting_for_user", "waiting_for_user", "needs_user"),
        ("succeeded", "task_completed", None),
        ("running", None, None),
    ],
)
def test_cause_from_state_derives_old_runs(status, stop_reason, expected) -> None:
    assert cause_from_state(status, stop_reason) == expected


def test_a_stored_cause_wins() -> None:
    assert cause_from_state(RunStatus.FAILED, StopReason.MAX_STEP_LIMIT_REACHED, RunFailureCause.API_ERROR) == "api_error"


def _chained(outer: Exception, inner: Exception) -> Exception:
    try:
        try:
            raise inner
        except Exception as exc:
            raise outer from exc
    except Exception as exc:
        return exc


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ModelBlockedError("SAFETY"), RunFailureCause.MODEL_BLOCKED),
        (_chained(BackendCompatibilityError("HTTP 503"), ModelTransportError("HTTP 503")), RunFailureCause.API_ERROR),
        (httpx.ConnectError("refused"), RunFailureCause.API_ERROR),
        (PerceptionLowQualityError("blank"), RunFailureCause.PERCEPTION_FAILED),
        # A Claude refusal is a plain BackendCompatibilityError with no transport error behind it.
        (BackendCompatibilityError("Claude declined the turn."), RunFailureCause.PERCEPTION_FAILED),
        (KeyError("x"), RunFailureCause.RUN_CRASHED),
    ],
)
def test_cause_of_error(error, expected) -> None:
    assert cause_of_error(error) is expected


def test_note_failure_never_overwrites() -> None:
    state = AgentState(run_id="r", intent="t", status=RunStatus.RUNNING, stop_reason=StopReason.COMPLETION_UNVERIFIED)
    note_failure(state, detail="first " * 200)
    note_failure(state, cause=RunFailureCause.API_ERROR, detail="second")
    assert state.failure_cause is RunFailureCause.MODEL_GAVE_UP
    assert state.failure_detail is not None and state.failure_detail.startswith("first") and len(state.failure_detail) <= 500


# --- real loop, scripted model ($0) ------------------------------------------------------


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


async def test_gave_up_records_the_models_reason(tmp_path) -> None:
    loop, run_store = _model_loop(tmp_path, [chat_reply(tool_calls=[tool_call("fail", reason="a CAPTCHA blocks the page")])])
    run = await loop.start_run(RunTaskRequest(intent="Open the Products page", max_steps=5))

    response = await loop.step_run(StepRequest(run_id=run.run_id))

    assert response.status is RunStatus.FAILED
    state = await run_store.get_run(run.run_id)
    assert state is not None
    assert state.failure_cause is RunFailureCause.MODEL_GAVE_UP
    assert "CAPTCHA" in (state.failure_detail or "")
    step = _rows(run_store, run.run_id)[-1]
    assert step["failure"]["category"] == FailureCategory.MODEL_GAVE_UP.value
    assert step["failure"]["terminal"] is True


async def test_a_normal_done_has_no_cause(tmp_path) -> None:
    loop, run_store = _model_loop(tmp_path, [chat_reply(tool_calls=[tool_call("done", summary="opened it")])])
    run = await loop.start_run(RunTaskRequest(intent="Open the Products page", max_steps=5))

    response = await loop.step_run(StepRequest(run_id=run.run_id))

    assert response.status is RunStatus.SUCCEEDED
    state = await run_store.get_run(run.run_id)
    assert state is not None and state.failure_cause is None
    assert _rows(run_store, run.run_id)[-1]["failure"] is None


async def test_api_error_saves_the_error_text(tmp_path) -> None:
    loop, run_store = _model_loop(tmp_path, [ModelTransportError("model server HTTP 503: overloaded")])
    run = await loop.start_run(RunTaskRequest(intent="Open the Products page", max_steps=5))

    response = await loop.step_run(StepRequest(run_id=run.run_id))

    assert response.status is RunStatus.FAILED
    state = await run_store.get_run(run.run_id)
    assert state is not None
    assert state.failure_cause is RunFailureCause.API_ERROR
    assert "HTTP 503" in (state.failure_detail or "")
    assert _rows(run_store, run.run_id)[-1]["record_type"] == "pre_step_failure"


async def test_max_steps_writes_a_run_end_row(tmp_path) -> None:
    wait = tool_call("wait", seconds=1)
    loop, run_store = _model_loop(tmp_path, [chat_reply(tool_calls=[wait])])
    run = await loop.start_run(RunTaskRequest(intent="Open the Products page", max_steps=2))

    for _ in range(3):
        response = await loop.step_run(StepRequest(run_id=run.run_id))

    assert response.status is RunStatus.FAILED
    state = await run_store.get_run(run.run_id)
    assert state is not None
    assert state.stop_reason is StopReason.MAX_STEP_LIMIT_REACHED
    assert state.failure_cause is RunFailureCause.MAX_STEPS
    last = _rows(run_store, run.run_id)[-1]
    assert last["record_type"] == "run_end"
    assert last["failure_cause"] == "max_steps"
    assert last["failure"]["category"] == FailureCategory.MAX_STEP_LIMIT_REACHED.value
    assert last["step_index"] == 2
    # Readers: replay parses it, the efficiency score does not count it as a step.
    entries = load_run_replay(run.run_id, root_dir=run_store.root_dir)
    assert len(entries) == 3
    assert "Run ended: failed | cause=max_steps" in render_run_replay(run.run_id, root_dir=run_store.root_dir)
    assert run_efficiency(run_store.root_dir / run.run_id)["total_steps"] == 2


async def test_resume_clears_a_cause(tmp_path) -> None:
    loop, run_store = _model_loop(tmp_path, [chat_reply(tool_calls=[tool_call("wait", seconds=1)])])
    run = await loop.start_run(RunTaskRequest(intent="Open the Products page", max_steps=5))
    state = await run_store.get_run(run.run_id)
    assert state is not None
    state.failure_cause = RunFailureCause.NEEDS_USER
    state.failure_detail = "paused"
    await run_store.set_status(run.run_id, RunStatus.WAITING_FOR_USER)

    await loop.resume_run(run.run_id)

    state = await run_store.get_run(run.run_id)
    assert state is not None and state.failure_cause is None and state.failure_detail is None


def test_old_state_json_still_loads() -> None:
    legacy = {"run_id": "r", "intent": "t", "status": "failed", "stop_reason": "max_step_limit_reached"}
    state = AgentState.model_validate(legacy)
    assert state.failure_cause is None
    assert cause_from_state(state.status, state.stop_reason, state.failure_cause) == "max_steps"



# --- API layer: crashes, the auto-run step budget, orphaned runs ------------------------


async def test_a_crashed_step_records_the_cause_and_a_run_end_row(tmp_path, monkeypatch) -> None:
    from operon.api.runtime import events

    monkeypatch.setattr(events, "publish_run_completed", lambda *_: None)
    loop, run_store = _model_loop(tmp_path, [chat_reply(tool_calls=[tool_call("wait", seconds=1)])])
    run = await loop.start_run(RunTaskRequest(intent="Open the Products page", max_steps=5))

    response = await events.mark_run_failed(loop, run.run_id, RuntimeError("boom"))

    assert response is not None and response.status is RunStatus.FAILED
    state = await run_store.get_run(run.run_id)
    assert state is not None
    assert state.failure_cause is RunFailureCause.RUN_CRASHED
    assert state.failure_detail == "RuntimeError: boom"
    last = _rows(run_store, run.run_id)[-1]
    assert last["record_type"] == "run_end" and last["failure"]["category"] == FailureCategory.EXECUTION_ERROR.value


async def test_a_model_api_crash_mid_step_is_an_api_error(tmp_path, monkeypatch) -> None:
    from operon.api.runtime import events

    monkeypatch.setattr(events, "publish_run_completed", lambda *_: None)
    loop, run_store = _model_loop(tmp_path, [chat_reply(tool_calls=[tool_call("wait", seconds=1)])])
    run = await loop.start_run(RunTaskRequest(intent="Open the Products page", max_steps=5))
    crash = _chained(BackendCompatibilityError("model server HTTP 429"), ModelTransportError("model server HTTP 429"))

    await events.mark_run_failed(loop, run.run_id, crash)

    state = await run_store.get_run(run.run_id)
    assert state is not None and state.failure_cause is RunFailureCause.API_ERROR
    assert "HTTP 429" in (state.failure_detail or "")


async def test_the_auto_run_step_budget_records_max_steps(tmp_path, monkeypatch) -> None:
    from operon.api.runtime import tasks

    monkeypatch.setattr(tasks, "publish_run_completed", lambda *_: None)
    monkeypatch.setattr(tasks, "publish_step_outcome", AsyncMock())
    monkeypatch.setattr(tasks, "release_parallel_run", AsyncMock())
    loop, run_store = _model_loop(tmp_path, [chat_reply(tool_calls=[tool_call("wait", seconds=1)])])
    run = await loop.start_run(RunTaskRequest(intent="Open the Products page", max_steps=10))

    await tasks.auto_run_loop(loop, run.run_id, max_steps=1)

    state = await run_store.get_run(run.run_id)
    assert state is not None
    assert state.status is RunStatus.FAILED and state.failure_cause is RunFailureCause.MAX_STEPS
    assert _rows(run_store, run.run_id)[-1]["record_type"] == "run_end"


def test_an_orphaned_browser_run_is_interrupted(tmp_path, monkeypatch) -> None:
    from operon.api import observer

    monkeypatch.setattr(observer, "runs_root", lambda: tmp_path)
    run_dir = tmp_path / "run-x"
    run_dir.mkdir()
    state = AgentState(run_id="run-x", intent="Open the website", status=RunStatus.RUNNING, start_url="https://example.com")
    (run_dir / "state.json").write_text(state.model_dump_json(), encoding="utf-8")

    reconciled = observer.reconcile_orphaned_browser_run("run-x", has_live_session=False)

    assert reconciled is not None and reconciled.failure_cause is RunFailureCause.INTERRUPTED
    saved = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert saved["failure_cause"] == "interrupted"

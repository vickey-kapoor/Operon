"""run.jsonl is complete and true per step (KAN-177): paths exist, times and source are real."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.harness import run_scenario
from evals.scenarios import cu_click_then_done, trust_gate_approved
from operon.agent.loop import AgentLoop
from operon.models.logs import ModelDebugArtifacts, PreStepFailureLog, StepLog
from operon.models.usage import ModelUsage
from operon.store.background_writer import bg_writer

_NEW_FIELDS = {"started_at", "finished_at", "step_latency_ms", "model_latency_ms"}


def _run_log_lines(root: Path) -> list[dict]:
    asyncio.run(bg_writer.flush())
    logs = list((root / "runs").glob("*/run.jsonl"))
    assert len(logs) == 1
    return [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines() if line.strip()]


def _artifact_paths(entry: dict) -> list[str]:
    paths: list[str] = []
    for key, value in entry.items():
        if key.endswith("_artifact_path") and value:
            paths.append(value)
        if key.endswith("_debug") and isinstance(value, dict):
            paths.extend(v for k, v in value.items() if k.endswith("_artifact_path") and v)
    return paths


@pytest.mark.parametrize("scenario", [cu_click_then_done, trust_gate_approved])
def test_every_step_log_is_complete(tmp_path: Path, scenario) -> None:
    result = asyncio.run(run_scenario(scenario(), root=tmp_path))
    assert result.steps >= 1
    entries = _run_log_lines(tmp_path)
    assert entries

    for entry in entries:
        log = StepLog.model_validate(entry)
        for path in _artifact_paths(entry):
            assert Path(path).exists(), f"step {log.step_index}: {path} does not exist"
        # OwnDoneVerifier makes no model call, so no verification files are named.
        assert log.verification_debug is None
        assert log.started_at is not None and log.finished_at is not None
        assert log.started_at <= log.finished_at
        assert log.step_latency_ms is not None and log.model_latency_ms is not None
        assert 0 <= log.model_latency_ms <= log.step_latency_ms
        assert log.decision_source is not None
        assert log.decision_source.startswith("[LLM] ")
        assert log.decision_source != "[LLM] gemini"
        # The same line as written before these fields existed still loads.
        old = {k: v for k, v in entry.items() if k not in _NEW_FIELDS}
        assert StepLog.model_validate(old).started_at is None


def test_a_paused_step_logs_no_after_screenshot(tmp_path: Path) -> None:
    asyncio.run(run_scenario(trust_gate_approved(), root=tmp_path))
    logs = [StepLog.model_validate(entry) for entry in _run_log_lines(tmp_path)]
    paused = [log for log in logs if (log.executed_action.detail or "").startswith("Paused for user")]
    ran = [log for log in logs if log not in paused]
    assert paused and ran
    # Nothing ran before the pause, so there is no after screenshot to name.
    assert all(log.after_artifact_path is None for log in paused)
    assert all(log.after_artifact_path and Path(log.after_artifact_path).exists() for log in ran)
    # Readers fall back to the before screenshot.
    from operon.api.frontdoor import _screenshot
    from operon.store.replay import render_run_replay

    assert _screenshot(paused[0]) == paused[0].before_artifact_path
    run_id = paused[0].run_id
    assert "after: none" in render_run_replay(run_id, root_dir=tmp_path / "runs")


def test_a_failed_perceive_step_names_only_files_that_exist(tmp_path: Path, monkeypatch) -> None:
    from evals.harness import FakeComputerUseClient
    from operon.clients.gemini_computer_use import GeminiComputerUseError
    from operon.store.replay import render_run_replay

    async def fail(self, **_kwargs):
        raise GeminiComputerUseError("model unavailable")

    monkeypatch.setattr(FakeComputerUseClient, "generate_turn", fail)
    monkeypatch.setattr(FakeComputerUseClient, "run_step", fail)
    asyncio.run(run_scenario(cu_click_then_done(), root=tmp_path))
    entries = _run_log_lines(tmp_path)
    assert [entry.get("record_type") for entry in entries] == ["pre_step_failure"]
    failure = PreStepFailureLog.model_validate(entries[0])
    # The backend reported no files of its own, so none are named that were never written.
    assert failure.perception_debug is None
    state = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text(encoding="utf-8"))
    paths = state["artifact_paths"]
    assert failure.before_artifact_path in paths
    assert all(Path(path).is_file() for path in paths)
    names = {Path(path).name for path in paths}
    assert "computer_use_prompt.txt" in names  # what the backend did write before it failed
    assert not names & {"perception_prompt.txt", "perception_raw.txt", "perception_parsed.json"}
    assert "perception: none" in render_run_replay(failure.run_id, root_dir=tmp_path / "runs")
    from operon.api import observer

    monkeypatch.setattr(observer, "runs_root", lambda: tmp_path / "runs")
    snapshot = observer.load_run_snapshot(failure.run_id)
    assert snapshot["pre_step_failures"][0]["perception"]["usage"] is None
    assert observer.summarize_run_usage(failure.run_id).request_count == 0


def test_old_run_log_lines_still_load() -> None:
    debug = {
        "prompt_artifact_path": "runs/r/step_1/p.txt",
        "raw_response_artifact_path": "runs/r/step_1/raw.txt",
        "parsed_artifact_path": "runs/r/step_1/parsed.json",
    }
    old_failure = {
        "record_type": "pre_step_failure",
        "run_id": "r",
        "step_id": "step_1",
        "step_index": 1,
        "before_artifact_path": "runs/r/step_1/before.png",
        "perception_debug": debug,
        "failure": {
            "category": "pre_step_perception_failed",
            "stage": "perceive",
            "retry_count": 0,
            "terminal": True,
            "recoverable": False,
            "reason": "boom",
        },
        "error_message": "boom",
    }
    loaded = PreStepFailureLog.model_validate(old_failure)
    assert loaded.logged_at is None
    # Every field added to StepLog is optional, so a line written before it still loads.
    for name in ("started_at", "finished_at", "step_latency_ms", "model_latency_ms", "verification_debug"):
        assert not StepLog.model_fields[name].is_required()


def _loop_with_policy(policy) -> AgentLoop:
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy_service = policy
    return loop


def _decision(rule_name: str | None = None):
    return SimpleNamespace(rule_name=rule_name)


def _debug(usage: ModelUsage | None) -> ModelDebugArtifacts:
    return ModelDebugArtifacts(prompt_artifact_path="p", raw_response_artifact_path="r", parsed_artifact_path="j", usage=usage)


def test_decision_source_names_the_real_model() -> None:
    usage = ModelUsage(provider="anthropic", model="some-model", request_kind="policy")
    loop = _loop_with_policy(SimpleNamespace())
    assert loop._decision_source(_decision("hint"), _debug(usage)) == "[RULE] hint"
    assert loop._decision_source(_decision(), _debug(usage)) == "[LLM] anthropic:some-model"
    # Combined backends report usage on the perception call.
    assert loop._decision_source(_decision(), _debug(None), _debug(usage)) == "[LLM] anthropic:some-model"
    assert loop._decision_source(_decision(), _debug(None)) == "[LLM] unknown"
    labelled = _loop_with_policy(SimpleNamespace(delegate=SimpleNamespace(model_label="label-x")))
    assert labelled._decision_source(_decision(), _debug(None)) == "[LLM] label-x"


def test_usage_path_dropped_when_no_usage_was_written() -> None:
    service = SimpleNamespace(
        latest_debug_artifacts=lambda: ModelDebugArtifacts(
            prompt_artifact_path="p", raw_response_artifact_path="r", parsed_artifact_path="j", usage_artifact_path="u.json"
        )
    )
    assert AgentLoop._latest_debug_artifacts(service).usage_artifact_path is None
    assert AgentLoop._latest_debug_artifacts(SimpleNamespace()) is None


def test_a_stop_step_logs_no_after_screenshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for KAN-218: a stop captures no after screenshot, so its row must not name one."""
    from evals import harness

    real_execute = harness.FakeExecutor.execute

    async def execute(self, action):  # like the real executors, a stop takes no after screenshot
        done = await real_execute(self, action)
        return done.model_copy(update={"artifact_path": None}) if action.action_type.value == "stop" else done

    monkeypatch.setattr(harness.FakeExecutor, "execute", execute)
    asyncio.run(run_scenario(cu_click_then_done(), root=tmp_path))
    logs = [StepLog.model_validate(entry) for entry in _run_log_lines(tmp_path)]
    stops = [log for log in logs if log.policy_decision.action.action_type.value == "stop"]
    others = [log for log in logs if log not in stops]
    assert stops and others
    assert all(log.after_artifact_path is None for log in stops)
    # A normal step still names an after file that exists.
    assert all(log.after_artifact_path and Path(log.after_artifact_path).exists() for log in others)
    state = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text(encoding="utf-8"))
    assert not any(p.endswith(f"step_{stops[-1].step_index}/after.png") for p in state["artifact_paths"])

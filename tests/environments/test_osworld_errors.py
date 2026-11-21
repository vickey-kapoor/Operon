"""KAN-176: an erroring OSWorld task is recorded and the run goes on; results carry a status and stop reason."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from evals import osworld

RID = "abc12345-0000-0000-0000-000000000000"
TASK = {"domain": "os", "id": "t", "instruction": "i"}


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    monkeypatch.setattr(osworld.time, "sleep", lambda s: None)


def _env(evaluate=lambda: 1.0, reset=lambda task_config: None):
    return SimpleNamespace(vm_ip="v", server_port=5000, reset=reset, evaluate=evaluate, closed=False)


def _write_run(runs, state=None, cost=0.4):
    d = runs / RID
    (d / "step_1").mkdir(parents=True)
    (d / "step_1" / "policy_usage.json").write_text(json.dumps({"estimated_cost_usd": cost}))
    if state is not None:
        (d / "state.json").write_text(json.dumps(state))


def _ok_run(cmd, **kw):
    return SimpleNamespace(stdout=f"Run {RID} started: x", stderr="", returncode=0)


def test_state_fields_are_read_from_state_json(tmp_path):
    _write_run(tmp_path / "runs", {"status": "completed", "stop_reason": "model_done"})
    r = osworld.run_one(_env(), TASK, {}, max_steps=1, timeout_s=9, settle_s=0, runs_root=str(tmp_path / "runs"), run=_ok_run)
    assert (r["status"], r["run_status"], r["stop_reason"], r["error"]) == ("passed", "completed", "model_done", None)
    assert r["cost_usd"] == 0.4


def test_failed_status_when_score_is_zero():
    r = osworld.run_one(_env(lambda: 0.0), TASK, {}, max_steps=1, timeout_s=9, settle_s=0, run=_ok_run)
    assert r["status"] == "failed" and r["note"] == "operon exit 0" and r["run_status"] is None


def test_timeout_status(tmp_path):
    def timing_out(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1, output=f"Run {RID} started".encode())

    _write_run(tmp_path / "runs", {"status": "running", "stop_reason": None})
    r = osworld.run_one(_env(lambda: 0.0), TASK, {}, max_steps=1, timeout_s=1, settle_s=0,
                        runs_root=str(tmp_path / "runs"), run=timing_out)
    assert r["status"] == "timed_out" and r["note"] == "operon timed out after 1s" and r["run_status"] == "running"


def test_error_after_operon_ran_keeps_run_id_and_cost(tmp_path):
    def boom():
        raise ValueError("checker crashed")

    _write_run(tmp_path / "runs", {"status": "completed", "stop_reason": "model_done"})
    r = osworld.run_one(_env(boom), TASK, {}, max_steps=1, timeout_s=9, settle_s=0,
                        runs_root=str(tmp_path / "runs"), logs_dir=str(tmp_path / "logs"), run=_ok_run)
    assert r["status"] == "error" and r["score"] == 0.0 and not r["passed"]
    assert r["note"] == "evaluate error: ValueError: checker crashed" and r["error"] == "ValueError: checker crashed"
    assert r["run_id"] == RID and r["cost_usd"] == 0.4 and r["stop_reason"] == "model_done"
    assert r["log"] and (tmp_path / "logs" / "t.log").exists()


def test_reset_error_is_recorded_and_operon_never_runs():
    def bad_reset(task_config):
        raise RuntimeError("VM boot failed")

    def never(cmd, **kw):
        raise AssertionError("operon must not run")

    r = osworld.run_one(_env(reset=bad_reset), TASK, {}, max_steps=1, timeout_s=9, settle_s=0, run=never)
    assert r["status"] == "error" and r["note"].startswith("reset error: RuntimeError") and r["run_id"] is None
    assert r["cost_usd"] == 0.0


def test_error_text_is_truncated():
    def boom():
        raise RuntimeError("x" * 1000)

    r = osworld.run_one(_env(boom), TASK, {}, max_steps=1, timeout_s=9, settle_s=0, run=_ok_run)
    assert len(r["error"]) == 300


def test_run_state_is_tolerant(tmp_path):
    assert osworld.run_state(None, tmp_path) == (None, None)
    assert osworld.run_state("nope", tmp_path) == (None, None)
    assert osworld.run_state("nope", None) == (None, None)
    d = tmp_path / "r"
    d.mkdir()
    (d / "state.json").write_text("{not json")
    assert osworld.run_state("r", tmp_path) == (None, None)
    (d / "state.json").write_text("[1]")
    assert osworld.run_state("r", tmp_path) == (None, None)
    (d / "state.json").write_text('{"status": "failed"}')
    assert osworld.run_state("r", tmp_path) == ("failed", None)


def _main(tmp_path, monkeypatch, tasks, env, task_config=lambda repo, task: {}):
    (tmp_path / "repo").mkdir(exist_ok=True)
    monkeypatch.setattr(osworld, "load_subset", lambda: tasks)
    monkeypatch.setattr(osworld, "task_config", task_config)
    return osworld.main(["--osworld-repo", str(tmp_path / "repo"), "--out", str(tmp_path / "out"), "--max-cost-usd", "0"],
                        env_factory=lambda repo: env, run=_ok_run)


def _report(tmp_path):
    return json.loads(next((tmp_path / "out").glob("osworld-*.json")).read_text())


def _tasks(*ids):
    return [{"domain": "os", "id": i, "instruction": "i"} for i in ids]


def test_evaluate_error_on_middle_task_does_not_skip_the_rest(tmp_path, monkeypatch, capsys):
    calls = []

    def evaluate():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("evaluator blew up")
        return 1.0

    env = _env(evaluate)
    env.close = lambda: setattr(env, "closed", True)
    assert _main(tmp_path, monkeypatch, _tasks("a", "b", "c"), env) == 0
    rep = _report(tmp_path)
    assert [r["status"] for r in rep["results"]] == ["passed", "error", "passed"]
    assert "evaluator blew up" in rep["results"][1]["error"] and rep["summary"]["errors"] == 1
    out = capsys.readouterr().out
    assert "::warning::task b errored" in out and "ERROR os b" in out and "PASS os a" in out and env.closed


def test_missing_task_config_is_recorded(tmp_path, monkeypatch):
    def cfg(repo, task):
        if task["id"] == "a":
            raise FileNotFoundError("no config")
        return {}

    assert _main(tmp_path, monkeypatch, _tasks("a", "b"), _env(), task_config=cfg) == 0
    rep = _report(tmp_path)
    assert rep["results"][0]["status"] == "error" and "FileNotFoundError" in rep["results"][0]["error"]
    assert rep["results"][0]["run_id"] is None and rep["results"][0]["cost_usd"] == 0.0
    assert rep["results"][1]["status"] == "passed"


def test_reset_error_in_main_continues(tmp_path, monkeypatch):
    n = []

    def reset(task_config):
        n.append(1)
        if len(n) == 1:
            raise RuntimeError("boot")

    assert _main(tmp_path, monkeypatch, _tasks("a", "b"), _env(reset=reset)) == 0
    assert [r["status"] for r in _report(tmp_path)["results"]] == ["error", "passed"]


def test_keyboard_interrupt_still_stops_and_saves(tmp_path, monkeypatch):
    n = []

    def evaluate():
        n.append(1)
        if len(n) == 2:
            raise KeyboardInterrupt
        return 1.0

    env = _env(evaluate)
    env.close = lambda: setattr(env, "closed", True)
    with pytest.raises(KeyboardInterrupt):
        _main(tmp_path, monkeypatch, _tasks("a", "b", "c"), env)
    rep = _report(tmp_path)
    assert [r["id"] for r in rep["results"]] == ["a"] and env.closed


def test_job_minutes_values():
    assert osworld.job_minutes(25, 1500) == 905
    assert osworld.job_minutes(5, 1500) == 205
    assert osworld.job_minutes(1, 1500) == 65
    assert osworld.job_minutes(0, 1500) == 30
    assert osworld.job_minutes(1, 61) == 30 + 12  # rounds a task up to whole minutes


def test_job_minutes_cli(capsys):
    assert osworld.main(["--job-minutes", "--limit", "5"]) == 0
    assert capsys.readouterr().out.strip() == "205"
    assert osworld.main(["--job-minutes", "--limit", "0"]) == 0
    assert capsys.readouterr().out.strip() == "905"
    first = osworld.load_subset()[0]["id"][:8]
    assert osworld.main(["--job-minutes", "--only", first, "--limit", "5"]) == 0
    assert capsys.readouterr().out.strip() == "65"

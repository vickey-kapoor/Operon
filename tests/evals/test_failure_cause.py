"""KAN-175: bench failure causes, the by-step final screenshot, and the reports that carry them."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import httpx

from evals import live_bench, osworld
from evals.failure_cause import bench_cause, by_cause, final_screenshot
from evals.harness import run_scenario
from evals.live_tasks import TASKS


def _touch(path, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_final_screenshot_is_by_step_index_not_mtime_or_name(tmp_path):
    _touch(tmp_path / "step_2" / "after.png", 100)
    _touch(tmp_path / "step_2" / "zoom.png", 9_999_999_999)
    last = _touch(tmp_path / "step_10" / "before.png", 50)
    assert final_screenshot(tmp_path) == last
    after = _touch(tmp_path / "step_10" / "after.png", 10)
    assert final_screenshot(tmp_path) == after


def test_final_screenshot_falls_back_to_a_lower_step_and_none(tmp_path):
    assert final_screenshot(tmp_path) is None
    assert final_screenshot(tmp_path / "missing") is None
    (tmp_path / "step_3").mkdir()
    low = _touch(tmp_path / "step_1" / "before.png")
    assert final_screenshot(tmp_path) == low


def test_bench_cause_precedence():
    kw = {"run_status": "failed", "stop_reason": "max_step_limit_reached", "state_cause": None}
    assert bench_cause(passed=True, bench_error=True, **kw) is None
    assert bench_cause(passed=False, bench_error=True, timed_out=True, **kw) == "bench_error"
    assert bench_cause(passed=False, bench_error=False, timed_out=True, **kw) == "timeout"
    assert bench_cause(passed=False, bench_error=False, **{**kw, "stop_reason": "interrupted"}) == "interrupted"
    assert bench_cause(passed=False, bench_error=False, **kw) == "max_steps"
    assert bench_cause(passed=False, bench_error=False, **{**kw, "state_cause": "api_error"}) == "api_error"
    ok = {"run_status": "succeeded", "stop_reason": "task_completed", "state_cause": None}
    assert bench_cause(passed=False, bench_error=False, **ok) == "wrong_result"
    assert bench_cause(passed=False, bench_error=False, run_status="running", stop_reason=None) == "other"


def test_by_cause_counts_sorted():
    assert by_cause(["b", "a", None, "b", "a", "c"]) == {"a": 2, "b": 2, "c": 1}
    assert list(by_cause(["x", "y", "y"])) == ["y", "x"]


def _result(outcome, cause=None):
    return live_bench.TaskResult("t", "c", "r", "failed", outcome, "-", 1, 0, 5, 0, 0, "why", "-", failure_cause=cause)


def test_live_summary_and_report_group_failures_by_cause():
    results = [_result("failure", "max_steps"), _result("failure", "max_steps"), _result("success"), _result("error", "bench_error")]
    summary = live_bench.summarize(results)
    assert summary["by_cause"] == {"max_steps": 2, "bench_error": 1}
    report = live_bench.format_report(results, summary, None)
    assert "failures by cause: max_steps 2 · bench_error 1" in report
    assert "[max_steps] why" in report


def test_an_old_report_row_without_the_new_fields_loads():
    row = {k: v for k, v in asdict(_result("failure")).items() if k not in ("failure_cause", "error")}
    loaded = live_bench.TaskResult(**row)
    assert loaded.failure_cause is None and loaded.error is None


async def test_run_task_keeps_the_api_error_text_and_bench_error(monkeypatch, tmp_path):
    class Client:
        async def post(self, path, json=None):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(live_bench, "_run_dir", lambda run_id: tmp_path / run_id)

    async def judge(task, shot):
        raise AssertionError("no screenshot to judge")

    result = await live_bench.run_task(Client(), TASKS[0], judge, timeout_s=5, approve_pauses=0)
    assert result.failure_cause == "bench_error"
    assert result.error == "ConnectError: connection refused"


def test_osworld_run_one_labels_a_max_steps_run(tmp_path, monkeypatch):
    rid = "abc12345-0000-0000-0000-000000000000"
    (tmp_path / rid).mkdir()
    (tmp_path / rid / "state.json").write_text(json.dumps({"status": "failed", "stop_reason": "max_step_limit_reached"}))
    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    monkeypatch.setattr(osworld.time, "sleep", lambda s: None)
    env = SimpleNamespace(vm_ip="v", server_port=1, reset=lambda task_config: None, evaluate=lambda: 0.0)
    ran = SimpleNamespace(stdout=f"Run {rid} started", stderr="", returncode=0)
    task = {"domain": "os", "id": "t", "instruction": "i"}
    r = osworld.run_one(env, task, {}, max_steps=1, timeout_s=9, settle_s=0, runs_root=str(tmp_path), run=lambda *a, **k: ran)
    assert r["failure_cause"] == "max_steps"
    assert osworld.summarize([r])["by_cause"] == {"max_steps": 1}
    assert osworld.error_result(task, "n", "boom")["failure_cause"] == "bench_error"


async def test_a_runaway_scenario_records_max_steps(tmp_path):
    from evals.scenarios import cu_relaunch_runaway

    result = await run_scenario(cu_relaunch_runaway(), root=tmp_path)
    assert result.outcome == "incomplete" and result.steps == 12
    assert result.failure_cause == "max_steps"
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert json.loads((run_dir / "state.json").read_text())["failure_cause"] == "max_steps"
    rows = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines() if line.strip()]
    assert rows[-1]["record_type"] == "run_end"


def test_a_succeeded_run_the_judge_could_not_grade_is_a_bench_error() -> None:
    # The run said done but the judge gave no verdict: the grader never disagreed, so not wrong_result.
    assert (
        bench_cause(passed=False, bench_error=True, run_status="succeeded", stop_reason="task_completed", state_cause=None)
        == "bench_error"
    )


def test_the_bench_cause_table_matches_operons_own() -> None:
    # evals/failure_cause.py must not import operon (the OSWorld runner's venv has none), so it keeps
    # its own copy of the stop-reason table; it must agree with the loop's for every state.
    from evals.failure_cause import state_cause_value
    from operon.agent.failure_cause import cause_from_state
    from operon.models.common import RunStatus, StopReason

    for status in [*RunStatus, None]:
        for reason in [*StopReason, None]:
            assert state_cause_value(status, reason) == cause_from_state(status, reason), (status, reason)


def test_the_bench_helpers_import_without_operon() -> None:
    import subprocess
    import sys
    from pathlib import Path

    # Block every operon import, as on the OSWorld runner's own venv.
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'operon' or name.startswith('operon.'):\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import evals.osworld as o\n"
        "assert o.failure_cause(passed=False, bench_error=False, timed_out=False, run_status='succeeded',"
        " stop_reason='task_completed', state_cause=None) == 'wrong_result'\n"
        "assert o.failure_cause(passed=False, bench_error=False, timed_out=False, run_status='cancelled',"
        " stop_reason=None, state_cause=None) == 'cancelled'\n"
    )
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)

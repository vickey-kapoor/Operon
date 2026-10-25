"""Offline checks for the live benchmark: task list, grading and the run loop."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import httpx
import pytest

from tests.eval import live_bench
from tests.eval.desktop_tasks import DESKTOP_TASKS
from tests.eval.live_bench import (
    Verdict,
    classify,
    parse_verdict,
    run_task,
    save_final_screens,
    select_tasks,
    summarize,
)
from tests.eval.live_tasks import TASKS


def test_live_tasks_are_well_formed() -> None:
    names = [t.name for t in TASKS]
    assert len(names) == len(set(names))
    assert 30 <= len(TASKS) <= 50
    for task in TASKS:
        assert urlparse(task.start_url).scheme == "https", task.name
        assert task.check and task.intent, task.name
    assert len(select_tasks("smoke", None, None)) == 5


def test_desktop_tasks_are_well_formed_and_separate_from_the_browser_tasks() -> None:
    names = [t.name for t in DESKTOP_TASKS]
    assert len(names) == len(set(names))
    assert not set(names) & {t.name for t in TASKS}
    assert 8 <= len(DESKTOP_TASKS) <= 12
    for task in DESKTOP_TASKS:
        assert task.start_url is None, task.name
        assert task.check and task.intent, task.name
    assert select_tasks("full", None, None, "desktop") == DESKTOP_TASKS
    assert len(select_tasks("smoke", None, None, "desktop")) == 3


@pytest.mark.parametrize(
    ("status", "visible", "outcome"),
    [
        ("succeeded", True, "success"),
        ("succeeded", False, "false_success"),
        ("failed", True, "missed"),
        ("failed", False, "failure"),
        ("running", True, "timeout"),
        ("succeeded", None, "error"),
        ("error", None, "error"),
    ],
)
def test_classify_compares_the_claim_with_the_screen(status, visible, outcome) -> None:
    assert classify(status, visible) == outcome


def test_parse_verdict_reads_fenced_json_and_rejects_garbage() -> None:
    assert parse_verdict('```json\n{"goal_visible": true, "reason": "ok"}\n```') == Verdict(True, "ok")
    assert parse_verdict("not json").goal_visible is None
    assert parse_verdict('{"goal_visible": "yes"}').goal_visible is None


async def test_run_task_approves_a_pause_and_grades_from_the_final_screen(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run-1"
    (run_dir / "step_1").mkdir(parents=True)
    (run_dir / "step_1" / "before.png").write_bytes(b"png")
    (run_dir / "step_1" / "computer_use_raw.json").write_text("{}")
    (run_dir / "state.json").write_text(json.dumps({"stop_reason": "stop_after_success"}))
    monkeypatch.setattr(live_bench, "runs_dir", lambda: tmp_path)
    monkeypatch.setattr(live_bench.asyncio, "sleep", lambda _s: _noop())

    statuses = iter(["waiting_for_user", "succeeded"])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/run-task":
            return httpx.Response(202, json={"run_id": "run-1", "status": "running"})
        if request.url.path == "/resume":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"status": next(statuses), "step_count": 1})

    seen: list[str] = []

    async def judge(task, screenshot):
        seen.append(screenshot)
        return Verdict(False, "goal not on screen")

    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        result = await run_task(client, TASKS[0], judge, timeout_s=60, approve_pauses=2)

    assert "POST /resume" in calls
    assert result.pauses == 1
    assert result.outcome == "false_success"
    assert result.cu_steps == 1 and result.other_steps == 0
    assert seen == [str(run_dir / "step_1" / "before.png")]
    assert summarize([result])["false_success"] == 1


async def test_desktop_run_task_uses_the_desktop_routes_and_cleans_up(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run-2"
    (run_dir / "step_1").mkdir(parents=True)
    (run_dir / "step_1" / "before.png").write_bytes(b"png")
    monkeypatch.setattr(live_bench, "runs_dir", lambda: tmp_path)
    monkeypatch.setattr(live_bench.asyncio, "sleep", lambda _s: _noop())

    calls: list[str] = []
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "POST":
            bodies.append(json.loads(request.content))
        if request.url.path == "/desktop/run-task":
            return httpx.Response(202, json={"run_id": "run-2", "status": "running"})
        if request.url.path == "/desktop/cleanup":
            return httpx.Response(200, json={"run_id": "run-2", "closed_count": 1, "detail": "ok"})
        return httpx.Response(200, json={"status": "succeeded", "step_count": 1})

    async def judge(task, screenshot):
        return Verdict(True, "Notepad shows the text")

    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        result = await run_task(client, DESKTOP_TASKS[0], judge, timeout_s=60, approve_pauses=2, mode="desktop")

    # Desktop runs only advance when stepped, so the benchmark drives them step by step.
    assert calls == ["POST /desktop/run-task", "POST /desktop/step", "POST /desktop/cleanup"]
    assert bodies[1] == {"run_id": "run-2"}
    assert "start_url" not in bodies[0]
    assert result.outcome == "success"
    # The desktop loop has no fallback backend, so no step counts as a fallback.
    assert result.cu_steps == 0 and result.other_steps == 0


def test_save_final_screens_copies_the_graded_screenshots(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(live_bench, "runs_dir", lambda: tmp_path / "runs")
    shot = tmp_path / "before.png"
    shot.write_bytes(b"png")
    for run_id in ("run-1", "run-3"):
        (tmp_path / "runs" / run_id / "step_1").mkdir(parents=True)
        (tmp_path / "runs" / run_id / "step_1" / "before.png").write_bytes(b"png")
    graded = live_bench.TaskResult(
        "notepad_type", "text", "run-1", "succeeded", "success", "-", 1, 0, 5, 0, 0, "", str(shot)
    )
    missing = live_bench.TaskResult(
        "cmd_echo", "shell", None, "error", "error", "-", 0, 0, 1, 0, 0, "no screenshot", "-"
    )
    failed = live_bench.TaskResult(
        "explorer_windows", "files", "run-3", "failed", "failure", "-", 5, 0, 9, 0, 0, "Home", str(shot)
    )

    save_final_screens([graded, missing, failed], tmp_path / "screens")

    saved = sorted(p.name for p in (tmp_path / "screens").iterdir())
    # Only the failed run's folder is kept; a successful run keeps just its final screen.
    assert saved == ["explorer_windows", "explorer_windows.png", "notepad_type.png"]
    assert (tmp_path / "screens" / "explorer_windows" / "step_1" / "before.png").is_file()


async def _noop() -> None:
    return None


def _write_usage(step_dir, **usage) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / f"{usage.pop('kind')}_usage.json").write_text(json.dumps(usage), encoding="utf-8")


def test_run_usage_sums_every_model_call_in_the_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(live_bench, "runs_dir", lambda: tmp_path)
    run = tmp_path / "run-1"
    _write_usage(
        run / "step_1", kind="computer_use", model="claude-opus-5-5", input_tokens=10,
        output_tokens=200, cache_creation_input_tokens=3000, cache_read_input_tokens=0, estimated_cost_usd=0.02,
    )
    _write_usage(
        run / "step_2", kind="computer_use", model="claude-opus-5-5", input_tokens=10,
        output_tokens=100, cache_creation_input_tokens=1500, cache_read_input_tokens=3000, estimated_cost_usd=0.01,
    )
    _write_usage(run / "step_2", kind="final_check", model="claude-opus-5-5", input_tokens=1200, output_tokens=50, estimated_cost_usd=0.006)
    (run / "step_2" / "broken_usage.json").write_text("{not json", encoding="utf-8")

    usage = live_bench.run_usage("run-1")

    assert usage["cost_usd"] == 0.036
    assert usage["cache_write_tokens"] == 4500 and usage["cache_read_tokens"] == 3000
    assert usage["output_tokens"] == 350
    assert usage["by_model"]["claude-opus-5-5"]["requests"] == 3


def test_summary_reports_cost_per_task_and_per_success() -> None:
    def result(outcome: str, cost: float) -> live_bench.TaskResult:
        return live_bench.TaskResult(
            "t", "c", "r", "succeeded", outcome, "-", 2, 0, 10, 2, 0, "", "-",
            cost_usd=cost, judge_cost_usd=0.001,
            by_model={"claude-opus-5-5": {"requests": 2, "input_tokens": 0, "output_tokens": 0,
                                          "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": cost}},
        )

    summary = summarize([result("success", 0.03), result("failure", 0.05)])

    assert summary["cost_usd"] == 0.08
    assert summary["cost_per_task_usd"] == 0.04
    # The failed task's spend is charged to the one success.
    assert summary["cost_per_success_usd"] == 0.08
    assert summary["judge_cost_usd"] == 0.002
    assert summary["by_model"]["claude-opus-5-5"]["requests"] == 4
    assert "per success" in live_bench.format_report([result("success", 0.03)], summarize([result("success", 0.03)]), None)


def _costly(name: str, cost: float) -> live_bench.TaskResult:
    return live_bench.TaskResult(name, "c", "r", "succeeded", "success", "-", 1, 0, 5, 1, 0, "", "-", cost_usd=cost, judge_cost_usd=0.01)


def test_budget_stop_needs_the_cap_spent_with_tasks_left() -> None:
    spent = [_costly("a", 2.0), _costly("b", 3.0)]
    assert live_bench.budget_stop(spent, planned=5, max_cost_usd=5.0) == {
        "limit_usd": 5.0, "spent_usd": 5.02, "tasks_run": 2, "tasks_planned": 5,
    }
    assert live_bench.budget_stop(spent[:1], planned=5, max_cost_usd=5.0) is None
    assert live_bench.budget_stop(spent, planned=2, max_cost_usd=5.0) is None  # nothing left to skip
    assert live_bench.budget_stop(spent, planned=5, max_cost_usd=0) is None  # 0 turns the cap off


async def test_run_bench_stops_at_the_cost_cap_and_records_it(tmp_path, monkeypatch) -> None:
    calls = []

    async def fake_run_task(client, task, judge, **kwargs):
        calls.append(task.name)
        return _costly(task.name, 3.0)

    monkeypatch.setattr(live_bench, "run_task", fake_run_task)
    real_client = httpx.AsyncClient
    health = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    monkeypatch.setattr(live_bench.httpx, "AsyncClient", lambda **kw: real_client(transport=health, **kw))
    monkeypatch.delenv("MAX_RUN_COST_USD", raising=False)
    args = live_bench.build_parser(default_suite="smoke").parse_args(["--out-dir", str(tmp_path), "--max-cost-usd", "5"])

    assert await live_bench.run_bench(args, judge=None) == live_bench.BUDGET_EXIT_CODE

    assert len(calls) == 2 < len(select_tasks("smoke", None, None))
    (saved,) = tmp_path.glob("smoke-*.json")
    report = json.loads(saved.read_text())
    assert report["budget_stop"]["tasks_run"] == 2 and report["summary"]["tasks"] == 2
    # A capped run is not the baseline the next run compares against.
    assert live_bench._previous_report(tmp_path, "smoke") is None


def test_max_cost_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("MAX_RUN_COST_USD", "1.5")
    assert live_bench.build_parser().parse_args([]).max_cost_usd == 1.5
    monkeypatch.setenv("MAX_RUN_COST_USD", "")
    assert live_bench.build_parser().parse_args([]).max_cost_usd == live_bench.DEFAULT_MAX_COST_USD

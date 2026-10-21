"""Offline checks for the live benchmark: task list, grading and the run loop."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import httpx
import pytest

from tests.eval import live_bench
from tests.eval.live_bench import (
    Verdict,
    classify,
    parse_verdict,
    run_task,
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


async def _noop() -> None:
    return None

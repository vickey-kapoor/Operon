"""Run lifecycle across the auto-run loop, /step, and the WebSocket resume paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from operon.api import ws_stream
from operon.api.runtime import tasks
from operon.models.common import RunResponse, RunStatus


class _FakeStore:
    def __init__(self, root_dir, status: RunStatus = RunStatus.RUNNING, max_steps: int = 7) -> None:
        self.root_dir = root_dir
        self.state = SimpleNamespace(
            run_id="run-1",
            intent="do a thing",
            status=status,
            step_count=0,
            max_steps=max_steps,
            hitl_message="Please solve the captcha",
            stop_reason=None,
        )
        self.statuses: list[RunStatus] = []

    async def get_run(self, run_id: str):
        return self.state if run_id == self.state.run_id else None

    async def set_status(self, run_id: str, status: RunStatus):
        self.state.status = status
        self.statuses.append(status)
        return self.state


class _FakeLoop:
    def __init__(self, store: _FakeStore, outcomes: list) -> None:
        self.run_store = store
        self._outcomes = outcomes
        self.resumed: list[str] = []

    async def step_run(self, request) -> RunResponse:
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        self.run_store.state.step_count += 1
        self.run_store.state.status = outcome
        return RunResponse(run_id="run-1", status=outcome, intent="do a thing", step_count=self.run_store.state.step_count)

    async def resume_run(self, run_id: str) -> RunResponse:
        self.resumed.append(run_id)
        await self.run_store.set_status(run_id, RunStatus.RUNNING)
        return RunResponse(run_id=run_id, status=RunStatus.RUNNING, intent="do a thing", step_count=0)

    async def _cleanup_completed_run(self, run_id: str) -> None:
        return None


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    published: list[dict] = []
    monkeypatch.setattr(ws_stream, "publish_event", published.append)
    return published


async def test_auto_run_publishes_step_and_completion_events(tmp_path, events) -> None:
    loop = _FakeLoop(_FakeStore(tmp_path), [RunStatus.RUNNING, RunStatus.SUCCEEDED])
    ws_stream.set_active_run("run-1")

    await tasks.auto_run_loop(loop, "run-1", max_steps=10)

    assert [e["type"] for e in events] == ["step", "step", "run_completed"]
    assert events[-1]["status"] == "succeeded"
    assert ws_stream._active_run_id is None


async def test_auto_run_publishes_hitl_required_when_run_pauses(tmp_path, events) -> None:
    loop = _FakeLoop(_FakeStore(tmp_path), [RunStatus.WAITING_FOR_USER])

    await tasks.auto_run_loop(loop, "run-1", max_steps=10)

    hitl = [e for e in events if e["type"] == "hitl_required"]
    assert len(hitl) == 1
    assert hitl[0]["message"] == "Please solve the captcha"


async def test_auto_run_marks_run_failed_when_step_crashes(tmp_path, events) -> None:
    store = _FakeStore(tmp_path)
    loop = _FakeLoop(store, [RuntimeError("model exploded")])

    await tasks.auto_run_loop(loop, "run-1", max_steps=10)  # must not raise

    assert store.state.status is RunStatus.FAILED
    assert events[-1] == {"type": "run_completed", "run_id": "run-1", "status": "failed"}


@pytest.mark.parametrize("message", [{"type": "resume"}, {"type": "override", "hint": "click Next"}])
async def test_ws_resume_and_override_reschedule_auto_run(
    tmp_path, events, monkeypatch: pytest.MonkeyPatch, message: dict
) -> None:
    store = _FakeStore(tmp_path, status=RunStatus.WAITING_FOR_USER, max_steps=7)
    loop = _FakeLoop(store, [])
    scheduled: list[tuple[str, int]] = []
    monkeypatch.setattr(tasks, "schedule_auto_run", lambda _loop, run_id, max_steps: scheduled.append((run_id, max_steps)))
    monkeypatch.setattr(ws_stream, "_agent_loop", loop)
    monkeypatch.setattr(ws_stream, "_active_run_id", "run-1")
    monkeypatch.setattr(ws_stream, "_pending_override_hint", None)

    await ws_stream._handle_control(message)

    assert loop.resumed == ["run-1"]
    assert scheduled == [("run-1", 7)]  # the run's own limit, not a hardcoded 200
    assert "error" not in events[-1]


def test_http_resume_reschedules_with_run_max_steps(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from operon.api.routes import browser
    from operon.api.server import app

    loop = _FakeLoop(_FakeStore(tmp_path, status=RunStatus.WAITING_FOR_USER, max_steps=12), [])
    scheduled: list[tuple[str, int]] = []
    monkeypatch.setattr(browser, "get_agent_loop", lambda: loop)
    monkeypatch.setattr(tasks, "schedule_auto_run", lambda _loop, run_id, max_steps: scheduled.append((run_id, max_steps)))

    resp = TestClient(app).post("/resume", json={"run_id": "run-1"})

    assert resp.status_code == 200
    assert scheduled == [("run-1", 12)]

"""Checkpoint and resume (KAN-80, OPERON_CHECKPOINTS): a run a restart interrupted is
flagged at startup, never resumed by itself, and continues from the current screen when asked."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from operon.api import ws_stream
from operon.api.runtime import loops, tasks
from operon.api.server import _lifespan, app
from operon.models.common import RunResponse, RunStatus, StepRequest, StopReason
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from operon.models.state import AgentState
from operon.models.verification import VerificationResult, VerificationStatus
from operon.store.checkpoints import list_interrupted, mark_interrupted_runs
from operon.store.run_store import FileBackedRunStore
from tests.fixtures.frontdoor import CLICK_SAVE, PURCHASE, build_loop

STOP = AgentAction(action_type=ActionType.STOP, text="all done")


def _crash(root: Path, status: RunStatus, *, steps: int = 3, **fields) -> str:
    """A run as the dead process left it on disk: ``steps`` clicks done, then ``status``."""
    store = FileBackedRunStore(root)
    state = store.create_run("Save the form", max_steps=10)
    state.status = status
    state.step_count = steps
    state.action_history = [ExecutedAction(action=CLICK_SAVE, success=True, detail="ok") for _ in range(steps)]
    for name, value in fields.items():
        setattr(state, name, value)
    store._write_state_sync(state)
    return state.run_id


def _on_disk(root: Path, run_id: str) -> AgentState:
    return AgentState.model_validate_json((root / run_id / "state.json").read_text(encoding="utf-8"))


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8080")


@pytest.fixture
def scheduled(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(tasks, "schedule_auto_run", lambda loop, run_id, max_steps: calls.append(run_id))
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)
    return calls


def _use(monkeypatch: pytest.MonkeyPatch, loop) -> list[tuple[list[str], str]]:
    monkeypatch.setattr(loops, "get_agent_loop", lambda: loop)
    monkeypatch.setattr(loops, "desktop_loop_holds_run", lambda run_id: False)
    monkeypatch.setattr("operon.api.routes.browser.get_agent_loop", lambda: loop)
    hints: list[tuple[list[str], str]] = []
    loop.perception_service.add_advisory_hints = lambda texts, source="", run_id="": hints.append((texts, source))
    return hints


# -- the startup scan ---------------------------------------------------------


@pytest.mark.parametrize("status", [RunStatus.RUNNING, RunStatus.PENDING, RunStatus.WAITING_FOR_USER])
def test_the_scan_flags_in_flight_runs_as_interrupted(tmp_path: Path, status: RunStatus) -> None:
    run_id = _crash(tmp_path, status, in_takeover=True)

    assert mark_interrupted_runs(tmp_path) == [run_id]

    state = _on_disk(tmp_path, run_id)
    assert state.status is RunStatus.WAITING_FOR_USER  # every client already shows it as waiting
    assert state.stop_reason is StopReason.INTERRUPTED
    assert state.interrupted_from is status
    assert state.in_takeover is False
    assert state.step_count == 3
    assert [s.run_id for s in list_interrupted(tmp_path)] == [run_id]
    assert mark_interrupted_runs(tmp_path) == []  # scanning twice changes nothing


def test_the_scan_leaves_finished_runs_and_recent_runs_alone(tmp_path: Path) -> None:
    done = _crash(tmp_path, RunStatus.SUCCEEDED)
    live = _crash(tmp_path, RunStatus.RUNNING)

    assert mark_interrupted_runs(tmp_path, min_age_s=3600) == []  # saved a moment ago: maybe live
    assert mark_interrupted_runs(tmp_path, skip=frozenset({live})) == []
    assert _on_disk(tmp_path, done).status is RunStatus.SUCCEEDED
    assert _on_disk(tmp_path, live).status is RunStatus.RUNNING


async def _lifespan_once() -> None:
    async with _lifespan(app):
        pass


@pytest.mark.parametrize(("flag", "marked"), [("on", True), ("off", False)])
def test_startup_marks_runs_only_with_the_flag_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str, marked: bool
) -> None:
    monkeypatch.setenv("OPERON_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("OPERON_RUNS_RETAIN_DAYS", "-1")
    monkeypatch.setenv("OPERON_CHECKPOINTS", flag)
    run_id = _crash(tmp_path, RunStatus.RUNNING)

    asyncio.run(_lifespan_once())

    state = _on_disk(tmp_path, run_id)
    assert (state.stop_reason is StopReason.INTERRUPTED) is marked
    assert state.status is (RunStatus.WAITING_FOR_USER if marked else RunStatus.RUNNING)


# -- serialization and schema compatibility -----------------------------------


def test_status_and_stop_reason_round_trip(tmp_path: Path) -> None:
    run_id = _crash(tmp_path, RunStatus.RUNNING)
    mark_interrupted_runs(tmp_path)
    raw = json.loads((tmp_path / run_id / "state.json").read_text(encoding="utf-8"))

    assert raw["status"] == "waiting_for_user"  # no new status value for old clients to trip on
    assert raw["stop_reason"] == "interrupted" and raw["interrupted_from"] == "running"
    assert AgentState.model_validate(raw).model_dump(mode="json")["interrupted_from"] == "running"


def test_a_state_saved_before_the_field_existed_still_loads() -> None:
    old = {"run_id": "r", "intent": "x", "status": "running", "step_count": 2}

    state = AgentState.model_validate(old)

    assert state.interrupted_from is None and state.stop_reason is None
    assert RunResponse(run_id="r", status=RunStatus.WAITING_FOR_USER, intent="x", step_count=2).status.value == "waiting_for_user"


# -- resuming -----------------------------------------------------------------


async def test_resume_continues_from_the_saved_step_count_and_completes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheduled: list[str]
) -> None:
    run_id = _crash(tmp_path, RunStatus.RUNNING)
    mark_interrupted_runs(tmp_path)
    monkeypatch.setenv("OPERON_CHECKPOINTS", "on")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])  # a fresh loop and store: the restarted process
    hints = _use(monkeypatch, loop)
    loop.verifier_service.verify = AsyncMock(
        return_value=VerificationResult(
            status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=True,
            stop_reason=StopReason.TASK_COMPLETED, reason="done"
        )
    )

    async with _client() as client:
        response = await client.post(f"/run/{run_id}/resume-interrupted")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "running"
        assert scheduled == [run_id]
        state = await loop.run_store.get_run(run_id)
        assert state.stop_reason is None and state.interrupted_from is None

        await loop.step_run(StepRequest(run_id=run_id))

    state = await loop.run_store.get_run(run_id)
    assert state.status is RunStatus.SUCCEEDED
    assert state.step_count == 4  # continued from 3, not restarted
    resume_hints = [text for texts, source in hints if source == "resume" for text in texts]
    assert len(resume_hints) == 1
    assert "Save the form" in resume_hints[0] and "clicked (10,20)" in resume_hints[0]


async def test_the_step_budget_continues_from_the_saved_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = _crash(tmp_path, RunStatus.RUNNING, steps=2, max_steps=3)
    mark_interrupted_runs(tmp_path)
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)

    await loop.resume_interrupted(run_id)
    await tasks.auto_run_loop(loop, run_id, 3)  # one step left of three

    assert len(executor.executed) == 1
    assert (await loop.run_store.get_run(run_id)).stop_reason is StopReason.MAX_STEP_LIMIT_REACHED


async def test_a_run_waiting_for_approval_re_pauses_instead_of_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheduled: list[str]
) -> None:
    run_id = _crash(tmp_path, RunStatus.WAITING_FOR_USER, stop_reason=StopReason.WAITING_FOR_USER)
    mark_interrupted_runs(tmp_path)
    monkeypatch.setenv("OPERON_CHECKPOINTS", "on")
    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE, PURCHASE])
    _use(monkeypatch, loop)

    async with _client() as client:
        response = await client.post(f"/run/{run_id}/resume-interrupted")
        assert response.json()["status"] == "waiting_for_user"
        assert scheduled == []  # nothing steps it until the person approves
        assert executor.executed == []

        # Approved after the restart: the model re-proposes the purchase and the gate asks again.
        await loop.resume_run(run_id)
        again = await loop.step_run(StepRequest(run_id=run_id))

    assert again.status is RunStatus.WAITING_FOR_USER
    assert executor.executed == []


async def test_resume_refuses_a_run_that_was_not_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheduled: list[str]
) -> None:
    run_id = _crash(tmp_path, RunStatus.RUNNING)  # never scanned
    monkeypatch.setenv("OPERON_CHECKPOINTS", "on")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    _use(monkeypatch, loop)

    async with _client() as client:
        assert (await client.post(f"/run/{run_id}/resume-interrupted")).status_code == 404

    assert scheduled == []
    assert (await loop.run_store.get_run(run_id)).status is RunStatus.RUNNING


async def test_the_route_is_off_without_the_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = _crash(tmp_path, RunStatus.RUNNING)
    mark_interrupted_runs(tmp_path)
    monkeypatch.setenv("OPERON_CHECKPOINTS", "off")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    _use(monkeypatch, loop)

    async with _client() as client:
        response = await client.post(f"/run/{run_id}/resume-interrupted")

    assert response.status_code == 404 and "OPERON_CHECKPOINTS" in response.json()["detail"]


async def test_the_task_page_api_lists_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheduled: list[str]
) -> None:
    run_id = _crash(tmp_path, RunStatus.RUNNING)
    mark_interrupted_runs(tmp_path)
    monkeypatch.setenv("OPERON_CHECKPOINTS", "on")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    _use(monkeypatch, loop)

    async with _client() as client:
        listed = (await client.get("/app/api/interrupted")).json()["runs"]
        assert [r["run_id"] for r in listed] == [run_id] and listed[0]["interrupted_from"] == "running"
        resumed = await client.post(f"/app/api/runs/{run_id}/resume-interrupted")

    assert resumed.status_code == 200 and resumed.json()["step_count"] == 3
    assert scheduled == [run_id]


# -- the CLI ------------------------------------------------------------------


def test_cli_lists_interrupted_runs_and_says_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from operon import cli

    monkeypatch.setenv("OPERON_RUNS_ROOT", str(tmp_path))
    run_id = _crash(tmp_path, RunStatus.RUNNING)
    mark_interrupted_runs(tmp_path)
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    _use(monkeypatch, loop)

    monkeypatch.setenv("OPERON_CHECKPOINTS", "off")
    assert cli.main(["runs", "--interrupted"]) == 1
    assert "OPERON_CHECKPOINTS" in capsys.readouterr().err

    monkeypatch.setenv("OPERON_CHECKPOINTS", "on")
    assert cli.main(["runs", "--interrupted"]) == 0
    out = capsys.readouterr().out
    assert run_id in out and "operon resume" in out


async def test_drive_resumes_an_interrupted_run_instead_of_starting_one() -> None:
    from operon import cli

    calls: list[str] = []
    finished = {"run_id": "r1", "environment": "browser", "permission_mode": "risky", "step_count": 3,
                "done": True, "status": "succeeded", "recent_steps": [], "waiting": None, "in_takeover": False}

    class Client:
        async def resume_interrupted(self, run_id: str) -> dict:
            calls.append(run_id)
            return finished

        async def report(self, run_id: str) -> dict:
            return {"status": "succeeded"}

        def format_report(self, report: dict) -> str:
            return "report"

    lines: list[str] = []
    code, _ = await cli.drive(Client(), {}, ask=input, out=lines.append, resume_run_id="r1")

    assert code == 0 and calls == ["r1"]
    assert "resumed from step 3" in lines[0]

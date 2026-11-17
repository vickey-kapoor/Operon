"""Resume keeps a desktop run in the environment provider it started in (KAN-80): a run
begun with --env windows-sandbox must not silently continue on the live host."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from operon.agent import control as run_control
from operon.api import frontdoor, ws_stream
from operon.api.server import app
from operon.models.common import RunStatus
from operon.store.checkpoints import mark_interrupted_runs
from tests.agent.test_checkpoints import STOP, _crash
from tests.fixtures.frontdoor import build_loop


def _interrupted(
    tmp_path: Path, *, environment: str | None, provider: str | None, record: bool = True
) -> str:
    run_id = _crash(tmp_path, RunStatus.RUNNING, environment=environment)
    settings = tmp_path / run_id / "settings.json"
    data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    data.pop("OPERON_ENVIRONMENT", None)
    if record and provider is not None:
        data["OPERON_ENVIRONMENT"] = provider
    settings.write_text(json.dumps(data), encoding="utf-8")
    mark_interrupted_runs(tmp_path)
    return run_id


async def test_sandbox_run_is_not_resumed_on_the_live_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = _interrupted(tmp_path, environment="desktop", provider="windows-sandbox")
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, executor = build_loop(monkeypatch, tmp_path, [STOP])

    with pytest.raises(ValueError, match=r"windows-sandbox.*live"):
        await loop.resume_interrupted(run_id)
    with pytest.raises(ValueError, match="windows-sandbox"):
        await loop.resume_run(run_id)  # the other way in refuses too

    state = await loop.run_store.get_run(run_id)
    assert state.status is RunStatus.WAITING_FOR_USER and state.interrupted_from is RunStatus.RUNNING
    assert executor.executed == []


async def test_same_provider_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = _interrupted(tmp_path, environment="desktop", provider="windows-sandbox")
    monkeypatch.setenv("OPERON_ENVIRONMENT", "windows-sandbox")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])

    response = await loop.resume_interrupted(run_id)

    assert response.status is RunStatus.RUNNING


async def test_browser_run_is_unaffected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = _interrupted(tmp_path, environment="browser", provider="windows-sandbox")
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])

    assert (await loop.resume_interrupted(run_id)).status is RunStatus.RUNNING


async def test_desktop_run_without_a_recorded_provider_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = _interrupted(tmp_path, environment="desktop", provider=None, record=False)
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, executor = build_loop(monkeypatch, tmp_path, [STOP])

    with pytest.raises(ValueError, match="no recorded environment provider"):
        await loop.resume_interrupted(run_id)
    assert executor.executed == []


def _paused(tmp_path: Path, *, provider: str) -> str:
    """A desktop run a dead sandbox process left waiting for the person, never flagged interrupted."""
    run_id = _crash(tmp_path, RunStatus.WAITING_FOR_USER, environment="desktop")
    settings = tmp_path / run_id / "settings.json"
    data = json.loads(settings.read_text(encoding="utf-8"))
    data["OPERON_ENVIRONMENT"] = provider
    settings.write_text(json.dumps(data), encoding="utf-8")
    return run_id


async def test_default_live_to_live_resume_still_works(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPERON_ENVIRONMENT", raising=False)
    run_id = _interrupted(tmp_path, environment="desktop", provider="live")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])

    assert (await loop.resume_interrupted(run_id)).status is RunStatus.RUNNING


async def test_a_paused_sandbox_run_loaded_from_disk_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = _paused(tmp_path, provider="windows-sandbox")
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, executor = build_loop(monkeypatch, tmp_path, [STOP])

    for approve in (True, False):
        with pytest.raises(ValueError, match="windows-sandbox"):
            await loop.resume_run(run_id, approve_gated_action=approve)

    assert (await loop.run_store.get_run(run_id)).status is RunStatus.WAITING_FOR_USER
    assert executor.executed == []


async def test_a_run_this_process_started_resumes_without_the_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = _paused(tmp_path, provider="windows-sandbox")
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    loop._created_runs.add(run_id)

    assert (await loop.resume_run(run_id)).status is RunStatus.RUNNING


async def test_approve_and_decline_answer_409_not_500_and_leave_no_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = _paused(tmp_path, provider="windows-sandbox")
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, executor = build_loop(monkeypatch, tmp_path, [STOP])
    monkeypatch.setattr(frontdoor, "_loop_of", lambda rid: loop)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8080") as client:
        approved = await client.post(f"/app/api/runs/{run_id}/approve")
        declined = await client.post(f"/app/api/runs/{run_id}/decline", json={"note": "no"})

    assert approved.status_code == 409 and "windows-sandbox" in approved.text
    assert declined.status_code == 409
    assert run_control.consume_override_hint(run_id) is None
    assert executor.executed == []


def _running_sandbox_run(tmp_path: Path, **fields) -> str:
    run_id = _crash(tmp_path, RunStatus.RUNNING, environment="desktop", **fields)
    settings = tmp_path / run_id / "settings.json"
    data = json.loads(settings.read_text(encoding="utf-8"))
    data["OPERON_ENVIRONMENT"] = "windows-sandbox"
    settings.write_text(json.dumps(data), encoding="utf-8")
    return run_id


async def test_a_disk_loaded_sandbox_run_is_not_stepped_on_the_live_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = _running_sandbox_run(tmp_path)
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, executor = build_loop(monkeypatch, tmp_path, [STOP])
    monkeypatch.setattr("operon.api.routes.desktop.get_desktop_agent_loop", lambda: loop)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8080") as client:
        response = await client.post("/desktop/step", json={"run_id": run_id})

    assert 400 <= response.status_code < 500 and "windows-sandbox" in response.text
    assert executor.executed == []
    assert (await loop.run_store.get_run(run_id)).step_count == 3


async def test_a_refused_hand_back_changes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = _running_sandbox_run(tmp_path, in_takeover=True)
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    loop, executor = build_loop(monkeypatch, tmp_path, [STOP])
    monkeypatch.setattr(frontdoor, "_loop_of", lambda rid: loop)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8080") as client:
        response = await client.post(f"/app/api/runs/{run_id}/handback")
    with pytest.raises(ValueError, match="windows-sandbox"):
        await loop.hand_back(run_id)  # the loop refuses before touching anything, too

    assert response.status_code == 409
    assert run_control.consume_override_hint(run_id) is None
    assert (await loop.run_store.get_run(run_id)).in_takeover is True
    assert executor.executed == []

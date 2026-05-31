"""Tests for the new-task modal submission paths and related API endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from operon.models.common import RunResponse, RunStatus


@pytest.fixture
def client():
    from operon.api.server import app
    return TestClient(app)


def _run_response(run_id: str = "modal-run-1", status: RunStatus = RunStatus.PENDING) -> RunResponse:
    return RunResponse(run_id=run_id, status=status, intent="Navigate to example.com", step_count=0)


# ── Submit success ───────────────────────────────────────────────────────────


def test_submit_browser_task_success(client: TestClient) -> None:
    """POST /run-task with valid intent returns 202 with run_id."""
    with patch("operon.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.start_run = AsyncMock(return_value=_run_response())
        resp = client.post("/run-task", json={"intent": "Navigate to example.com"})

    assert resp.status_code == 202
    data = resp.json()
    assert data["run_id"] == "modal-run-1"
    assert data["status"] == "pending"


def test_submit_browser_task_safe_mode_skips_cdp_and_background_loop(client: TestClient) -> None:
    """Unit tests run in safe mode, so /run-task must not launch browsers or background loops."""
    with (
        patch("operon.api.routes.get_agent_loop") as mock_loop,
        patch("operon.api.routes._ensure_cdp_ready", new_callable=AsyncMock) as mock_cdp,
        patch("operon.api.routes.asyncio.create_task") as mock_create_task,
    ):
        mock_loop.return_value.start_run = AsyncMock(return_value=_run_response())
        resp = client.post("/run-task", json={"intent": "Navigate to example.com"})

    assert resp.status_code == 202
    mock_cdp.assert_not_awaited()
    mock_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_auto_run_deduplicates_active_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only one background auto-run task should exist for a run_id."""
    from operon.api import routes

    routes._auto_run_tasks.clear()
    release = asyncio.Event()

    async def fake_auto_run(_loop, _run_id: str, _max_steps: int) -> None:
        await release.wait()

    monkeypatch.setattr(routes, "_auto_run_loop", fake_auto_run)

    first = routes._schedule_auto_run(object(), "run-dedupe", 5)
    second = routes._schedule_auto_run(object(), "run-dedupe", 200)

    assert first is second
    assert routes._auto_run_tasks["run-dedupe"] is first

    await routes._cancel_auto_run("run-dedupe")
    assert "run-dedupe" not in routes._auto_run_tasks


@pytest.mark.asyncio
async def test_schedule_auto_run_removes_completed_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Completed auto-run tasks should not remain in the registry."""
    from operon.api import routes

    routes._auto_run_tasks.clear()

    async def fake_auto_run(_loop, _run_id: str, _max_steps: int) -> None:
        return None

    monkeypatch.setattr(routes, "_auto_run_loop", fake_auto_run)

    task = routes._schedule_auto_run(object(), "run-complete", 5)
    await task
    await asyncio.sleep(0)

    assert "run-complete" not in routes._auto_run_tasks


def test_submit_desktop_task_success(client: TestClient) -> None:
    """POST /desktop/run-task with valid intent returns 202 with run_id."""
    with patch("operon.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_loop.return_value.start_run = AsyncMock(return_value=_run_response("desktop-run-1"))
        resp = client.post("/desktop/run-task", json={"intent": "Open Calculator"})

    assert resp.status_code == 202
    data = resp.json()
    assert data["run_id"] == "desktop-run-1"


# ── Submit failure ───────────────────────────────────────────────────────────


def test_submit_missing_intent_returns_422(client: TestClient) -> None:
    """POST /run-task without intent returns 422 (validation error)."""
    resp = client.post("/run-task", json={})
    assert resp.status_code == 422


def test_submit_empty_intent_returns_422(client: TestClient) -> None:
    """POST /run-task with empty-string intent returns 422."""
    resp = client.post("/run-task", json={"intent": ""})
    assert resp.status_code == 422


def test_submit_task_server_error_propagates(client: TestClient) -> None:
    """When start_run raises an unhandled RuntimeError, the TestClient re-raises it."""
    with patch("operon.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.start_run = AsyncMock(side_effect=RuntimeError("storage failure"))
        with pytest.raises(RuntimeError, match="storage failure"):
            client.post("/run-task", json={"intent": "Do something"})


# ── 409 conflict ─────────────────────────────────────────────────────────────


def test_submit_task_conflict_returns_409(client: TestClient) -> None:
    """POST /run-task when a run is already active returns 409."""
    from fastapi import HTTPException

    with patch("operon.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.start_run = AsyncMock(
            side_effect=HTTPException(status_code=409, detail="A run is already in progress")
        )
        resp = client.post("/run-task", json={"intent": "Do something"})
    assert resp.status_code == 409


# ── Blocked stop-current-run flow ────────────────────────────────────────────


def test_stop_current_run_before_new_task(client: TestClient) -> None:
    """POST /run/{id}/stop cancels a running task so modal can unblock."""
    running_state_cls = __import__(
        "operon.models.state", fromlist=["AgentState"]
    ).AgentState
    running = running_state_cls(
        run_id="old-run-1", intent="old task", status=RunStatus.RUNNING
    )
    cancelled = running_state_cls(
        run_id="old-run-1", intent="old task", status=RunStatus.CANCELLED
    )

    with (
        patch("operon.api.routes.get_agent_loop") as mock_loop,
        patch("operon.api.routes._cancel_auto_run", new_callable=AsyncMock) as mock_cancel,
    ):
        store = mock_loop.return_value.run_store
        store.get_run = AsyncMock(return_value=running)
        store.set_status = AsyncMock(return_value=cancelled)
        mock_loop.return_value._cleanup_completed_run = AsyncMock()

        resp = client.post("/run/old-run-1/stop")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"
    store.set_status.assert_awaited_once_with("old-run-1", RunStatus.CANCELLED)
    mock_cancel.assert_awaited_once_with("old-run-1")
    mock_loop.return_value._cleanup_completed_run.assert_awaited_once_with("old-run-1")


def test_stop_run_by_path_cleanup_failure_still_returns_cancelled(client: TestClient) -> None:
    """Cleanup failure after cancellation should not block the stop response."""
    running_state_cls = __import__(
        "operon.models.state", fromlist=["AgentState"]
    ).AgentState
    running = running_state_cls(
        run_id="cleanup-fails", intent="old task", status=RunStatus.RUNNING
    )
    cancelled = running_state_cls(
        run_id="cleanup-fails", intent="old task", status=RunStatus.CANCELLED
    )

    with (
        patch("operon.api.routes.get_agent_loop") as mock_loop,
        patch("operon.api.routes._cancel_auto_run", new_callable=AsyncMock) as mock_cancel,
    ):
        store = mock_loop.return_value.run_store
        store.get_run = AsyncMock(return_value=running)
        store.set_status = AsyncMock(return_value=cancelled)
        mock_loop.return_value._cleanup_completed_run = AsyncMock(side_effect=RuntimeError("close failed"))

        resp = client.post("/run/cleanup-fails/stop")

    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    mock_cancel.assert_awaited_once_with("cleanup-fails")
    mock_loop.return_value._cleanup_completed_run.assert_awaited_once_with("cleanup-fails")


def test_stop_run_by_body_cleans_up_cancelled_run(client: TestClient) -> None:
    """POST /stop cancels a running task and releases browser resources."""
    running_state_cls = __import__(
        "operon.models.state", fromlist=["AgentState"]
    ).AgentState
    running = running_state_cls(
        run_id="body-stop-1", intent="old task", status=RunStatus.RUNNING
    )
    cancelled = running_state_cls(
        run_id="body-stop-1", intent="old task", status=RunStatus.CANCELLED
    )

    with (
        patch("operon.api.routes.get_agent_loop") as mock_loop,
        patch("operon.api.routes._cancel_auto_run", new_callable=AsyncMock) as mock_cancel,
    ):
        store = mock_loop.return_value.run_store
        store.get_run = AsyncMock(return_value=running)
        store.set_status = AsyncMock(return_value=cancelled)
        mock_loop.return_value._cleanup_completed_run = AsyncMock()

        resp = client.post("/stop", json={"run_id": "body-stop-1"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    store.set_status.assert_awaited_once_with("body-stop-1", RunStatus.CANCELLED)
    mock_cancel.assert_awaited_once_with("body-stop-1")
    mock_loop.return_value._cleanup_completed_run.assert_awaited_once_with("body-stop-1")


def test_stop_run_by_path_not_found_returns_404(client: TestClient) -> None:
    """POST /run/{id}/stop with unknown run_id returns 404."""
    with patch("operon.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.run_store.get_run = AsyncMock(return_value=None)
        resp = client.post("/run/no-such-run/stop")
    assert resp.status_code == 404


def test_stop_run_by_path_already_done_is_noop(client: TestClient) -> None:
    """POST /run/{id}/stop on a completed run preserves status and does not call set_status."""
    running_state_cls = __import__(
        "operon.models.state", fromlist=["AgentState"]
    ).AgentState
    done = running_state_cls(
        run_id="done-run", intent="done task", status=RunStatus.SUCCEEDED
    )

    with (
        patch("operon.api.routes.get_agent_loop") as mock_loop,
        patch("operon.api.routes._cancel_auto_run", new_callable=AsyncMock) as mock_cancel,
    ):
        store = mock_loop.return_value.run_store
        store.get_run = AsyncMock(return_value=done)
        store.set_status = AsyncMock()
        mock_loop.return_value._cleanup_completed_run = AsyncMock()

        resp = client.post("/run/done-run/stop")

    assert resp.status_code == 200
    assert resp.json()["status"] == "succeeded"
    store.set_status.assert_not_called()
    mock_cancel.assert_not_awaited()
    mock_loop.return_value._cleanup_completed_run.assert_not_awaited()


# ── Instruction validation (server-side) ─────────────────────────────────────


def test_run_task_accepts_long_intent(client: TestClient) -> None:
    """POST /run-task accepts intents up to a reasonable length."""
    long_intent = "Search for the best restaurants near downtown " * 5
    with patch("operon.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.start_run = AsyncMock(
            return_value=_run_response(run_id="long-run")
        )
        resp = client.post("/run-task", json={"intent": long_intent})
    assert resp.status_code == 202


# ── URL validation (server-side) ─────────────────────────────────────────────


def test_run_task_with_valid_start_url(client: TestClient) -> None:
    """POST /run-task with a valid https start_url succeeds."""
    with patch("operon.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.start_run = AsyncMock(return_value=_run_response())
        resp = client.post(
            "/run-task",
            json={"intent": "Navigate to example.com", "start_url": "https://example.com"},
        )
    assert resp.status_code == 202


def test_run_task_with_http_start_url(client: TestClient) -> None:
    """POST /run-task with http:// start_url is also accepted."""
    with patch("operon.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.start_run = AsyncMock(return_value=_run_response())
        resp = client.post(
            "/run-task",
            json={"intent": "Navigate to example.com", "start_url": "http://example.com"},
        )
    assert resp.status_code == 202



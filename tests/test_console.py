"""Integration tests for the task console UI and stop endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models.common import RunResponse, RunStatus
from src.models.state import AgentState


@pytest.fixture
def client():
    from src.api.server import app
    return TestClient(app)


def _state(run_id: str = "console-run-1", status: RunStatus = RunStatus.RUNNING) -> AgentState:
    return AgentState(run_id=run_id, intent="Open example.com", status=status)


def _response(run_id: str = "console-run-1", status: RunStatus = RunStatus.RUNNING) -> RunResponse:
    return RunResponse(run_id=run_id, status=status, intent="Open example.com", step_count=1)


# ── console page ────────────────────────────────────────────────────────────


def test_console_page_served(client: TestClient) -> None:
    """GET /console should return the task console HTML."""
    resp = client.get("/console")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Command Center" in resp.text


def test_console_has_run_button(client: TestClient) -> None:
    """Console HTML should contain a Run and a Stop button."""
    resp = client.get("/console")
    assert "btnRun" in resp.text
    assert "btnStop" in resp.text


def test_console_has_log_panel(client: TestClient) -> None:
    """Console HTML should include the live log panel."""
    resp = client.get("/console")
    assert "logBody" in resp.text or "Live Log" in resp.text


def test_root_page_has_links_to_console_and_dashboard(client: TestClient) -> None:
    """Landing page should link to both primary UI surfaces."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'href="/console"' in resp.text
    assert 'href="/dashboard"' in resp.text



# ── POST /stop — create + stop ───────────────────────────────────────────────


def test_stop_running_run(client: TestClient) -> None:
    """POST /stop on a running run should mark it cancelled."""
    running_state = _state(status=RunStatus.RUNNING)
    cancelled_state = _state(status=RunStatus.CANCELLED)

    with patch("src.api.routes.get_agent_loop") as mock_loop:
        store = mock_loop.return_value.run_store
        store.get_run = AsyncMock(return_value=running_state)
        store.set_status = AsyncMock(return_value=cancelled_state)

        resp = client.post("/stop", json={"run_id": "console-run-1"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "console-run-1"
    assert data["status"] == "cancelled"
    store.set_status.assert_awaited_once_with("console-run-1", RunStatus.CANCELLED)


def test_stop_pending_run(client: TestClient) -> None:
    """POST /stop on a pending run should also cancel it."""
    pending_state = _state(status=RunStatus.PENDING)
    cancelled_state = _state(status=RunStatus.CANCELLED)

    with patch("src.api.routes.get_agent_loop") as mock_loop:
        store = mock_loop.return_value.run_store
        store.get_run = AsyncMock(return_value=pending_state)
        store.set_status = AsyncMock(return_value=cancelled_state)

        resp = client.post("/stop", json={"run_id": "console-run-1"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_stop_nonexistent_run_returns_404(client: TestClient) -> None:
    """POST /stop with unknown run_id should return 404."""
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.run_store.get_run = AsyncMock(return_value=None)

        resp = client.post("/stop", json={"run_id": "does-not-exist"})

    assert resp.status_code == 404


def test_stop_already_succeeded_run_preserves_status(client: TestClient) -> None:
    """POST /stop on a succeeded run should not change the status."""
    done_state = _state(status=RunStatus.SUCCEEDED)

    with patch("src.api.routes.get_agent_loop") as mock_loop:
        store = mock_loop.return_value.run_store
        store.get_run = AsyncMock(return_value=done_state)
        store.set_status = AsyncMock()

        resp = client.post("/stop", json={"run_id": "console-run-1"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "succeeded"
    store.set_status.assert_not_called()


def test_stop_already_failed_run_preserves_status(client: TestClient) -> None:
    """POST /stop on a failed run should not change the status."""
    done_state = _state(status=RunStatus.FAILED)

    with patch("src.api.routes.get_agent_loop") as mock_loop:
        store = mock_loop.return_value.run_store
        store.get_run = AsyncMock(return_value=done_state)
        store.set_status = AsyncMock()

        resp = client.post("/stop", json={"run_id": "console-run-1"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"
    store.set_status.assert_not_called()


def test_stop_already_cancelled_run_is_idempotent(client: TestClient) -> None:
    """POST /stop on an already-cancelled run should be a no-op."""
    done_state = _state(status=RunStatus.CANCELLED)

    with patch("src.api.routes.get_agent_loop") as mock_loop:
        store = mock_loop.return_value.run_store
        store.get_run = AsyncMock(return_value=done_state)
        store.set_status = AsyncMock()

        resp = client.post("/stop", json={"run_id": "console-run-1"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    store.set_status.assert_not_called()


def test_stop_requires_run_id(client: TestClient) -> None:
    """POST /stop with empty body should return 422."""
    resp = client.post("/stop", json={})
    assert resp.status_code == 422


# ── step after cancel ────────────────────────────────────────────────────────


def test_step_on_cancelled_run_returns_immediately(client: TestClient) -> None:
    """POST /step on a cancelled run should return cancelled status without executing."""
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.step_run = AsyncMock(
            return_value=RunResponse(
                run_id="console-run-1",
                status=RunStatus.CANCELLED,
                intent="Open example.com",
                step_count=2,
            )
        )
        resp = client.post("/step", json={"run_id": "console-run-1"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


# ── RunStatus enum has cancelled ─────────────────────────────────────────────


def test_run_status_has_cancelled() -> None:
    """RunStatus enum should include the cancelled value."""
    assert RunStatus.CANCELLED == "cancelled"
    assert "cancelled" in [s.value for s in RunStatus]


# ── get run status ───────────────────────────────────────────────────────────


def test_get_run_returns_status(client: TestClient) -> None:
    """GET /run/{id} should return the current run status."""
    state = _state(status=RunStatus.RUNNING)

    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.run_store.get_run = AsyncMock(return_value=state)
        resp = client.get("/run/console-run-1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "console-run-1"
    assert data["status"] == "running"
    assert data["intent"] == "Open example.com"


def test_get_run_not_found(client: TestClient) -> None:
    """GET /run/{id} with unknown id should return 404."""
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.run_store.get_run = AsyncMock(return_value=None)
        resp = client.get("/run/no-such-run")
    assert resp.status_code == 404


# ── create run ───────────────────────────────────────────────────────────────


def test_create_run(client: TestClient) -> None:
    """POST /run-task should create a run and return run_id and status."""
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_loop.return_value.start_run = AsyncMock(
            return_value=RunResponse(
                run_id="new-run-1",
                status=RunStatus.PENDING,
                intent="Open Chrome",
                step_count=0,
            )
        )
        resp = client.post("/run-task", json={"intent": "Open Chrome"})

    assert resp.status_code == 202
    data = resp.json()
    assert data["run_id"] == "new-run-1"
    assert data["status"] == "pending"

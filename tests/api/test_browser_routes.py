"""Browser run-control routes (api.routes.browser): branches the other route tests miss."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from operon.agent.loop import StartUrlBlockedError
from operon.models.common import RunResponse, RunStatus

ROUTES = "operon.api.routes.browser"


@pytest.fixture
def client():
    from operon.api.server import app

    return TestClient(app)


def _response(status: RunStatus = RunStatus.RUNNING, run_id: str = "run-1") -> RunResponse:
    return RunResponse(run_id=run_id, status=status, intent="Find a thing", step_count=1)


def test_run_task_refuses_a_blocked_start_url(client: TestClient) -> None:
    with patch(f"{ROUTES}.get_agent_loop") as get_loop:
        get_loop.return_value.start_run = AsyncMock(side_effect=StartUrlBlockedError("example.org is not allowed"))
        resp = client.post("/run-task", json={"intent": "go", "start_url": "https://example.org"})

    assert resp.status_code == 400
    assert "not allowed" in resp.json()["detail"]


def test_run_task_outside_safe_mode_readies_cdp_and_schedules_the_auto_run(client: TestClient) -> None:
    with (
        patch(f"{ROUTES}.test_safe_mode_enabled", return_value=False),
        patch(f"{ROUTES}.ensure_cdp_ready", new=AsyncMock()) as ensure_cdp,
        patch(f"{ROUTES}.schedule_auto_run") as schedule,
        patch(f"{ROUTES}.get_agent_loop") as get_loop,
        patch(f"{ROUTES}._ws_stream") as ws,
    ):
        get_loop.return_value.start_run = AsyncMock(return_value=_response(RunStatus.PENDING))
        resp = client.post("/run-task", json={"intent": "go", "mode": "observable"})

    assert resp.status_code == 202
    ensure_cdp.assert_awaited_once_with(mode="observable")
    schedule.assert_called_once()
    assert schedule.call_args.args[1] == "run-1"
    ws.set_active_run.assert_called_once_with("run-1")
    assert ws.publish_event.call_args.args[0]["type"] == "run_started"


def test_run_task_in_safe_mode_does_not_auto_run(client: TestClient) -> None:
    with (
        patch(f"{ROUTES}.test_safe_mode_enabled", return_value=True),
        patch(f"{ROUTES}.ensure_cdp_ready", new=AsyncMock()) as ensure_cdp,
        patch(f"{ROUTES}.schedule_auto_run") as schedule,
        patch(f"{ROUTES}.get_agent_loop") as get_loop,
        patch(f"{ROUTES}._ws_stream"),
    ):
        get_loop.return_value.start_run = AsyncMock(return_value=_response(RunStatus.PENDING))
        assert client.post("/run-task", json={"intent": "go"}).status_code == 202

    ensure_cdp.assert_not_awaited()
    schedule.assert_not_called()


def test_step_publishes_the_outcome(client: TestClient) -> None:
    with (
        patch(f"{ROUTES}.get_agent_loop") as get_loop,
        patch(f"{ROUTES}.publish_step_outcome", new=AsyncMock()) as publish,
    ):
        get_loop.return_value.step_run = AsyncMock(return_value=_response())
        resp = client.post("/step", json={"run_id": "run-1"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    publish.assert_awaited_once()


def test_step_of_an_unknown_run_is_404(client: TestClient) -> None:
    with patch(f"{ROUTES}.get_agent_loop") as get_loop:
        get_loop.return_value.step_run = AsyncMock(side_effect=ValueError("Run 'nope' not found"))
        resp = client.post("/step", json={"run_id": "nope"})

    assert resp.status_code == 404


def test_a_crashing_step_marks_the_run_failed(client: TestClient) -> None:
    failed = _response(RunStatus.FAILED)
    with (
        patch(f"{ROUTES}.get_agent_loop") as get_loop,
        patch(f"{ROUTES}.mark_run_failed", new=AsyncMock(return_value=failed)) as mark_failed,
    ):
        get_loop.return_value.step_run = AsyncMock(side_effect=RuntimeError("model exploded"))
        resp = client.post("/step", json={"run_id": "run-1"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"
    mark_failed.assert_awaited_once()


def test_a_crashing_step_that_cannot_be_marked_failed_is_500(client: TestClient) -> None:
    with (
        patch(f"{ROUTES}.get_agent_loop") as get_loop,
        patch(f"{ROUTES}.mark_run_failed", new=AsyncMock(return_value=None)),
    ):
        get_loop.return_value.step_run = AsyncMock(side_effect=RuntimeError("model exploded"))
        resp = client.post("/step", json={"run_id": "run-1"})

    assert resp.status_code == 500
    assert resp.json()["detail"] == "model exploded"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Gap: /step maps every ValueError to 404, and pydantic's ValidationError is a ValueError, "
        "so a step that crashes on a model validation error returns 'not found' and the run is "
        "never marked failed. Reported, not fixed."
    ),
)
def test_a_step_crashing_on_a_validation_error_marks_the_run_failed(client: TestClient) -> None:
    class _Strict(BaseModel):
        x: int

    try:
        _Strict(x="not a number")
    except ValueError as exc:
        crash = exc

    failed = _response(RunStatus.FAILED)
    with (
        patch(f"{ROUTES}.get_agent_loop") as get_loop,
        patch(f"{ROUTES}.mark_run_failed", new=AsyncMock(return_value=failed)),
    ):
        get_loop.return_value.step_run = AsyncMock(side_effect=crash)
        resp = client.post("/step", json={"run_id": "run-1"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"


def test_resume_of_an_unknown_run_is_404(client: TestClient) -> None:
    with (
        patch(f"{ROUTES}.get_agent_loop"),
        patch(f"{ROUTES}.resume_and_schedule", new=AsyncMock(side_effect=ValueError("not found"))),
    ):
        assert client.post("/resume", json={"run_id": "nope"}).status_code == 404


def _loop_with_run(status: RunStatus) -> MagicMock:
    record = SimpleNamespace(run_id="run-1", status=status, intent="Find a thing", step_count=2)
    loop = MagicMock()
    loop.run_store.get_run = AsyncMock(return_value=record)
    loop.run_store.set_status = AsyncMock(
        side_effect=lambda run_id, new: SimpleNamespace(**{**vars(record), "status": new})
    )
    return loop


def test_stop_by_id_cancels_an_active_run(client: TestClient) -> None:
    loop = _loop_with_run(RunStatus.RUNNING)
    with (
        patch(f"{ROUTES}.get_agent_loop", return_value=loop),
        patch(f"{ROUTES}.cancel_auto_run", new=AsyncMock()) as cancel,
        patch(f"{ROUTES}.cleanup_cancelled_run", new=AsyncMock()) as cleanup,
    ):
        resp = client.post("/run/run-1/stop")

    assert resp.json() == {"run_id": "run-1", "status": "cancelled"}
    cancel.assert_awaited_once_with("run-1")
    cleanup.assert_awaited_once()


def test_stop_by_id_is_idempotent_on_a_finished_run(client: TestClient) -> None:
    loop = _loop_with_run(RunStatus.SUCCEEDED)
    with (
        patch(f"{ROUTES}.get_agent_loop", return_value=loop),
        patch(f"{ROUTES}.cancel_auto_run", new=AsyncMock()) as cancel,
        patch(f"{ROUTES}.cleanup_cancelled_run", new=AsyncMock()),
    ):
        resp = client.post("/run/run-1/stop")

    assert resp.json() == {"run_id": "run-1", "status": "succeeded"}
    loop.run_store.set_status.assert_not_awaited()
    cancel.assert_not_awaited()


def test_stop_of_an_unknown_run_is_404(client: TestClient) -> None:
    loop = MagicMock()
    loop.run_store.get_run = AsyncMock(return_value=None)
    with patch(f"{ROUTES}.get_agent_loop", return_value=loop):
        assert client.post("/run/nope/stop").status_code == 404


def test_pause_of_an_unknown_run_is_404(client: TestClient) -> None:
    with patch(f"{ROUTES}.get_agent_loop") as get_loop:
        get_loop.return_value.pause_run = AsyncMock(side_effect=ValueError("missing"))
        resp = client.post("/run/nope/pause")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Run not found"


def test_pause_reports_the_new_status(client: TestClient) -> None:
    with patch(f"{ROUTES}.get_agent_loop") as get_loop:
        get_loop.return_value.pause_run = AsyncMock(return_value=SimpleNamespace(status=RunStatus.WAITING_FOR_USER))
        resp = client.post("/run/run-1/pause")

    assert resp.json() == {"run_id": "run-1", "status": "waiting_for_user"}

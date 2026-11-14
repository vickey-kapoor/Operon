"""A bad control message is logged and skipped; the stream stays up."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from operon.api import ws_stream
from operon.core.runtime import current_runtime


def test_malformed_control_message_is_logged_and_connection_survives(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setattr(current_runtime().stream, "browser_manager", None)
    monkeypatch.setattr(current_runtime().stream, "subscribers", [])
    monkeypatch.setattr(current_runtime().control, "confidence_threshold", 0.0)
    app = FastAPI()
    app.include_router(ws_stream.router)

    with caplog.at_level(logging.WARNING, logger="operon.api.ws_stream"):
        with TestClient(app).websocket_connect("/ws/stream") as ws:
            ws.send_text("{not json")
            ws.send_json({"type": "set_confidence_threshold", "threshold": 0.25})
            assert ws.receive_json()["threshold"] == 0.25

    assert any("control message that failed" in r.getMessage() for r in caplog.records)


async def test_pause_message_pauses_the_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from operon.models.common import RunStatus

    events: list[dict] = []
    loop = SimpleNamespace(pause_run=AsyncMock(return_value=SimpleNamespace(status=RunStatus.WAITING_FOR_USER)))
    monkeypatch.setattr(current_runtime(), "browser_loop", loop)
    monkeypatch.setattr(current_runtime().stream, "active_run_id", "run-1")
    monkeypatch.setattr(ws_stream, "publish_event", events.append)

    await ws_stream._handle_control({"type": "pause"})

    loop.pause_run.assert_awaited_once_with("run-1")
    assert events == [{"type": "control_ack", "action": "pause", "status": "waiting_for_user"}]

"""Static viewer initialization and clean WebSocket disconnects, without Chromium."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from operon.agent import control
from operon.api import ws_stream


@pytest.fixture
def stream_client(monkeypatch):
    monkeypatch.setenv("API_KEYS", "local-test-key")
    monkeypatch.setattr(
        ws_stream, "_browser_manager", SimpleNamespace(latest_frame=b"current frame")
    )
    monkeypatch.setattr(control, "_confidence_threshold", 0.0)
    monkeypatch.setattr(ws_stream, "_subscribers", [])
    app = FastAPI()
    app.include_router(ws_stream.router)
    with TestClient(app) as client:
        yield client
    assert not ws_stream._subscribers


def test_static_frame_replayed_on_initial_connection_and_reconnect(stream_client):
    for _ in range(10):
        with stream_client.websocket_connect("/ws/stream?api_key=local-test-key") as ws:
            assert ws.receive_bytes() == b"current frame"


@pytest.mark.parametrize("manager", [None, SimpleNamespace(latest_frame=None)])
def test_no_stale_frame_after_stream_is_cleared(stream_client, monkeypatch, manager):
    with stream_client.websocket_connect("/ws/stream?api_key=local-test-key") as ws:
        assert ws.receive_bytes() == b"current frame"
    monkeypatch.setattr(ws_stream, "_browser_manager", manager)
    with stream_client.websocket_connect("/ws/stream?api_key=local-test-key") as ws:
        ws.send_json({"type": "set_confidence_threshold", "threshold": 0.5})
        assert ws.receive_json() == {
            "type": "control_ack",
            "action": "set_confidence_threshold",
            "threshold": 0.5,
        }


def test_cached_frame_still_requires_authentication(stream_client):
    with pytest.raises(WebSocketDisconnect):
        with stream_client.websocket_connect("/ws/stream?api_key=wrong-key"):
            pytest.fail("Unauthenticated viewer connected")


async def test_client_disconnect_does_not_send_another_close(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setattr(ws_stream, "_browser_manager", None)
    monkeypatch.setattr(ws_stream, "_subscribers", [])
    received = asyncio.Queue()
    received.put_nowait({"type": "websocket.connect"})
    received.put_nowait({"type": "websocket.disconnect", "code": 1000})
    sent = []

    async def send(message):
        # A real ASGI transport may already be closed after websocket.disconnect.
        if message["type"] == "websocket.close":
            raise OSError("Client already disconnected")
        sent.append(message)

    ws = WebSocket({"type": "websocket"}, receive=received.get, send=send)
    await asyncio.wait_for(ws_stream.stream(ws), timeout=1)
    assert [message["type"] for message in sent] == ["websocket.accept"]
    assert not ws_stream._subscribers

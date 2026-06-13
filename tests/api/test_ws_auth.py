"""WebSocket auth on /ws/stream: opt-in via API_KEYS, key via query param or subprotocol."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from operon.api.server import app


def _client() -> TestClient:
    return TestClient(app)


def test_ws_opens_when_no_keys_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    with _client().websocket_connect("/ws/stream") as ws:
        ws.close()  # auth disabled → connection opens


def test_ws_rejected_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/ws/stream"):
            pass


def test_ws_rejected_with_wrong_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/ws/stream?api_key=nope"):
            pass


def test_ws_opens_with_valid_query_param_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret,other")
    with _client().websocket_connect("/ws/stream?api_key=other") as ws:
        ws.close()


def test_ws_opens_with_valid_subprotocol_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    with _client().websocket_connect("/ws/stream", subprotocols=["x-api-key", "s3cret"]) as ws:
        ws.close()

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
        with _client().websocket_connect("/ws/stream", subprotocols=["x-api-key", "nope"]):
            pass


def test_ws_refuses_a_query_param_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key in the URL would land in access logs, so it is not accepted."""
    monkeypatch.setenv("API_KEYS", "s3cret,other")
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/ws/stream?api_key=other"):
            pass


def test_ws_opens_with_valid_subprotocol_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    with _client().websocket_connect("/ws/stream", subprotocols=["x-api-key", "s3cret"]) as ws:
        ws.close()


# ── Origin gate (always on, independent of API_KEYS) ─────────────────────────


def test_ws_rejects_cross_site_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/ws/stream", headers={"origin": "https://evil.example"}):
            pass


def test_ws_rejects_null_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/ws/stream", headers={"origin": "null"}):
            pass


def test_ws_rejects_bad_origin_even_with_valid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect(
            "/ws/stream", subprotocols=["x-api-key", "s3cret"], headers={"origin": "https://evil.example"}
        ):
            pass


def test_ws_allows_origin_listed_in_cors_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173, https://ui.example")
    with _client().websocket_connect("/ws/stream", headers={"origin": "http://localhost:5173"}) as ws:
        ws.close()


def test_ws_allows_same_origin_loopback_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    headers = {"origin": "http://127.0.0.1:8080", "host": "127.0.0.1:8080"}
    with _client().websocket_connect("/ws/stream", headers=headers) as ws:
        ws.close()


def test_ws_rejects_dns_rebound_same_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rebound name resolves to 127.0.0.1, so its Host matches its Origin — but
    # it is not a loopback name, so it must not pass as same-origin.
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    headers = {"origin": "http://rebind.example:8080", "host": "rebind.example:8080"}
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/ws/stream", headers=headers):
            pass
def test_ws_echoes_x_api_key_subprotocol(monkeypatch: pytest.MonkeyPatch) -> None:
    # Browsers drop a connection whose offered subprotocol is not echoed back.
    monkeypatch.setenv("API_KEYS", "s3cret")
    with _client().websocket_connect("/ws/stream", subprotocols=["x-api-key", "s3cret"]) as ws:
        assert ws.accepted_subprotocol == "x-api-key"
        ws.close()


def test_ws_echoes_subprotocol_even_when_auth_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    with _client().websocket_connect("/ws/stream", subprotocols=["x-api-key", "anything"]) as ws:
        assert ws.accepted_subprotocol == "x-api-key"
        ws.close()


def test_ws_selects_no_subprotocol_when_none_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    with _client().websocket_connect("/ws/stream") as ws:
        assert ws.accepted_subprotocol is None
        ws.close()

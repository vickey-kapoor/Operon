"""Short-lived, scoped tokens on /ws/stream (KAN-89)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from operon.api import stream_tokens
from operon.api.server import app


@pytest.fixture(autouse=True)
def _fresh_tokens():
    stream_tokens.revoke_all()
    yield
    stream_tokens.revoke_all()


def test_a_token_has_its_scope_until_it_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_STREAM_TOKEN_TTL_S", "60")
    token = stream_tokens.mint("view", now=1000.0)

    assert stream_tokens.check(token.token, now=1059.0) == "view"
    assert stream_tokens.check(token.token, now=1060.0) is None
    assert stream_tokens.check(token.token, now=1000.0) is None  # expired tokens are dropped


def test_unknown_and_empty_tokens_are_refused() -> None:
    stream_tokens.mint("control")
    assert stream_tokens.check("not-a-token") is None
    assert stream_tokens.check("") is None
    assert stream_tokens.check(None) is None


def test_bad_scope_is_refused() -> None:
    with pytest.raises(ValueError):
        stream_tokens.mint("admin")


def test_ttl_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_STREAM_TOKEN_TTL_S", "1")
    assert stream_tokens.token_ttl_s() == 10
    monkeypatch.setenv("OPERON_STREAM_TOKEN_TTL_S", "oops")
    assert stream_tokens.token_ttl_s() == 600


def test_the_table_does_not_grow_without_bound() -> None:
    for _ in range(stream_tokens._MAX_TOKENS + 50):
        stream_tokens.mint("view")
    assert len(stream_tokens._tokens) <= stream_tokens._MAX_TOKENS + 1


def test_stream_token_route_needs_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    client = TestClient(app)

    assert client.post("/app/api/stream-token").status_code == 401
    response = client.post("/app/api/stream-token", json={"scope": "view"}, headers={"X-API-Key": "s3cret"})
    assert response.status_code == 200
    assert response.json()["scope"] == "view"


def test_a_token_opens_the_stream_without_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    token = stream_tokens.mint("control").token

    with TestClient(app).websocket_connect("/ws/stream", subprotocols=["x-stream-token", token]) as ws:
        assert ws.accepted_subprotocol == "x-stream-token"
        ws.close()


def test_an_expired_token_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    token = stream_tokens.mint("control", now=0.0).token

    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/ws/stream", subprotocols=["x-stream-token", token]):
            pass


def test_a_bad_token_is_refused_even_with_auth_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("OPERON_ALLOW_UNAUTHENTICATED", "true")

    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/ws/stream", subprotocols=["x-stream-token", "forged"]):
            pass


def test_a_view_token_cannot_send_control_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.api import ws_stream

    monkeypatch.setenv("API_KEYS", "s3cret")
    handled: list[dict] = []

    async def record(msg: dict) -> None:
        handled.append(msg)

    monkeypatch.setattr(ws_stream, "_handle_control", record)
    token = stream_tokens.mint("view").token

    with TestClient(app).websocket_connect("/ws/stream", subprotocols=["x-stream-token", token]) as ws:
        ws.send_text('{"type": "resume"}')
        ack = ws.receive_json()
        ws.close()

    assert ack == {"type": "control_ack", "action": "resume", "error": "view-only stream"}
    assert handled == []


def test_a_control_token_can_send_control_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.api import ws_stream

    monkeypatch.setenv("API_KEYS", "s3cret")

    async def ack(msg: dict) -> None:
        ws_stream.publish_event({"type": "control_ack", "action": msg["type"]})

    monkeypatch.setattr(ws_stream, "_handle_control", ack)
    token = stream_tokens.mint("control").token

    with TestClient(app).websocket_connect("/ws/stream", subprotocols=["x-stream-token", token]) as ws:
        ws.send_text('{"type": "resume"}')
        reply = ws.receive_json()
        ws.close()

    assert reply == {"type": "control_ack", "action": "resume"}

"""Tests for WebPilot API endpoints."""
from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from src.api.server import app
from src.api.webpilot_models import WebPilotAction
import src.api.webpilot_routes as routes_module


def make_dummy_screenshot():
    # 1x1 PNG
    import struct
    import zlib

    def _png_1x1():
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        raw = b'\x00\xff\xff\xff'
        compressed = zlib.compress(raw)
        idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        return sig + ihdr + idat + iend

    return base64.b64encode(_png_1x1()).decode()


DUMMY_SCREENSHOT = make_dummy_screenshot()

CLICK_ACTION = WebPilotAction(
    action="click", x=100, y=200,
    narration="Clicking the button", action_label="Click button", is_irreversible=False
)
DONE_ACTION = WebPilotAction(
    action="done",
    narration="Task complete", action_label="Done", is_irreversible=False
)
IRREVERSIBLE_ACTION = WebPilotAction(
    action="confirm_required", x=100, y=200,
    narration="About to purchase", action_label="Buy now", is_irreversible=True
)


@pytest.fixture(autouse=True)
def clear_sessions():
    routes_module._sessions.clear()
    yield
    routes_module._sessions.clear()


@pytest.fixture
def mock_handler():
    handler = MagicMock()
    handler.get_next_action = AsyncMock(side_effect=[CLICK_ACTION, DONE_ACTION])
    handler.get_interruption_replan = AsyncMock(return_value=DONE_ACTION)
    original = routes_module._handler
    routes_module._handler = handler
    yield handler
    routes_module._handler = original


def test_create_and_delete_session():
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid

    r2 = client.delete(f"/webpilot/sessions/{sid}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "deleted"

    r3 = client.delete(f"/webpilot/sessions/{sid}")
    assert r3.status_code == 404


def test_session_not_found_ws(mock_handler):
    client = TestClient(app)
    with client.websocket_connect("/webpilot/ws/nonexistent-id") as ws:
        # Should close immediately
        pass  # connection closed with 4404


def test_ws_task_flow(mock_handler):
    """Send task + screenshot, expect click action then done."""
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Find flights", "screenshot": DUMMY_SCREENSHOT})

        msg1 = ws.receive_json()
        assert msg1["type"] == "thinking"

        msg2 = ws.receive_json()
        assert msg2["type"] == "action"
        assert msg2["action"] == "click"

        ws.send_json({"type": "screenshot", "screenshot": DUMMY_SCREENSHOT})

        msg3 = ws.receive_json()
        assert msg3["type"] == "thinking"

        msg4 = ws.receive_json()
        assert msg4["type"] == "done"


def test_ws_stop(mock_handler):
    """Stop mid-task."""
    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    # handler returns click then done, but we stop after click
    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Find flights", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        _ = ws.receive_json()  # action (click)

        ws.send_json({"type": "stop"})
        msg = ws.receive_json()
        assert msg["type"] == "stopped"


def test_ws_irreversible_confirm(mock_handler):
    """Irreversible action confirmed by user → proceeds."""
    mock_handler.get_next_action = AsyncMock(side_effect=[IRREVERSIBLE_ACTION, DONE_ACTION])

    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Buy ticket", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        msg = ws.receive_json()
        assert msg["type"] == "confirmation_required"

        ws.send_json({"type": "confirm", "confirmed": True})

        action_msg = ws.receive_json()
        assert action_msg["type"] == "action"

        ws.send_json({"type": "screenshot", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        done = ws.receive_json()
        assert done["type"] == "done"


def test_ws_irreversible_deny(mock_handler):
    """Irreversible action denied → stops."""
    mock_handler.get_next_action = AsyncMock(return_value=IRREVERSIBLE_ACTION)

    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Buy ticket", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        msg = ws.receive_json()
        assert msg["type"] == "confirmation_required"

        ws.send_json({"type": "confirm", "confirmed": False})

        stopped = ws.receive_json()
        assert stopped["type"] == "stopped"


def test_ws_interruption(mock_handler):
    """Interrupt replans with new instruction."""
    mock_handler.get_next_action = AsyncMock(return_value=CLICK_ACTION)
    mock_handler.get_interruption_replan = AsyncMock(return_value=DONE_ACTION)

    client = TestClient(app)
    r = client.post("/webpilot/sessions")
    sid = r.json()["session_id"]

    with client.websocket_connect(f"/webpilot/ws/{sid}") as ws:
        ws.send_json({"type": "task", "intent": "Find flights", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking
        _ = ws.receive_json()  # action (click)

        # Send interrupt instead of screenshot
        ws.send_json({"type": "interrupt", "instruction": "Actually cancel", "screenshot": DUMMY_SCREENSHOT})

        _ = ws.receive_json()  # thinking (replan)
        done = ws.receive_json()
        assert done["type"] == "done"

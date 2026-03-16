"""Integration tests that exercise the REAL Gemini API (not stubs).

These tests require a valid GOOGLE_API_KEY in .env or environment.
They are slower (network I/O) and should be run separately from the
main test suite.

Covers:
  1. WebPilotHandler: get_next_action with real Gemini
  2. TTS narration: get_narration_audio returns valid WAV bytes
  3. Full server round-trip WITHOUT stub (real Gemini driving actions)
  4. TTS endpoint on real server

Run:
    python -m pytest tests/test_live_integration.py -v
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import struct
import threading
import time
import zlib
from contextlib import contextmanager
from typing import Generator

import httpx
import pytest
import websockets

# Skip entire module if no API key
_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not _KEY:
    # Try loading from .env
    _env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as f:
            for line in f:
                if line.startswith("GOOGLE_API_KEY="):
                    _KEY = line.strip().split("=", 1)[1]
                    os.environ["GOOGLE_API_KEY"] = _KEY
                    break

pytestmark = pytest.mark.skipif(not _KEY, reason="GOOGLE_API_KEY not set")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_b64() -> str:
    """Minimal valid 1x1 white PNG as base64."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return base64.b64encode(sig + ihdr + idat + iend).decode()


BLANK = _blank_b64()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _real_server_ctx() -> Generator[tuple[str, str], None, None]:
    """Start a real server WITHOUT WEBPILOT_STUB — uses actual Gemini API."""
    import uvicorn

    from src.api.server import app

    # Ensure WEBPILOT_STUB is NOT set
    old_stub = os.environ.pop("WEBPILOT_STUB", None)

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30.0
    ready = False
    while time.monotonic() < deadline:
        time.sleep(0.15)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
    if not ready:
        server.should_exit = True
        raise RuntimeError("Real server did not start in time")

    try:
        yield f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if old_stub is not None:
            os.environ["WEBPILOT_STUB"] = old_stub


def _create_session(http_url: str) -> str:
    r = httpx.post(f"{http_url}/webpilot/sessions")
    r.raise_for_status()
    return r.json()["session_id"]


# ---------------------------------------------------------------------------
# 1. WebPilotHandler — direct unit test
# ---------------------------------------------------------------------------


async def test_handler_get_next_action():
    """WebPilotHandler: single generate_content call returns a valid WebPilotAction."""
    from src.agent.planner import ActionPlanner
    from src.agent.vision import GeminiVisionClient
    from src.agent.webpilot_handler import WebPilotHandler
    from src.api.server import app  # noqa: F401

    vision = GeminiVisionClient()
    planner = ActionPlanner(vision_client=vision)
    handler = WebPilotHandler(vision_client=vision, planner=planner)

    action = await handler.get_next_action(
        BLANK, "Click the login button", history=[]
    )
    assert action.action is not None
    assert action.narration is not None


# ---------------------------------------------------------------------------
# 2. TTS narration
# ---------------------------------------------------------------------------


async def test_tts_narration_returns_audio_bytes():
    """TTS: get_narration_audio returns non-empty bytes (WAV audio)."""
    from src.agent.planner import ActionPlanner
    from src.agent.vision import GeminiVisionClient
    from src.agent.webpilot_handler import WebPilotHandler
    from src.api.server import app  # noqa: F401

    vision = GeminiVisionClient()
    planner = ActionPlanner(vision_client=vision)
    handler = WebPilotHandler(vision_client=vision, planner=planner)

    audio = await handler.get_narration_audio("Hello, I am clicking the button now.")
    assert isinstance(audio, bytes), f"Expected bytes, got {type(audio)}"
    assert len(audio) > 100, f"Audio too small ({len(audio)} bytes) — likely empty/error"


# ---------------------------------------------------------------------------
# 3. Full server round-trip with real Gemini (no stub)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_server():
    """Server using real Gemini API (no stub)."""
    with _real_server_ctx() as urls:
        yield urls


async def test_real_gemini_task_produces_action(real_server):
    """Full WS round-trip: submit task -> server calls real Gemini -> get action/done.

    This is the ultimate integration test — no stubs, no mocks. The server
    calls Gemini with the screenshot and returns a real action plan."""
    http_url, ws_url = real_server
    sid = _create_session(http_url)

    async with websockets.connect(f"{ws_url}/webpilot/ws/{sid}") as ws:
        await ws.send(json.dumps({
            "type": "task",
            "intent": "Describe what you see on the screen",
            "screenshot": BLANK,
        }))

        messages = []
        for _ in range(10):
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = json.loads(raw)
            messages.append(msg)

            if msg["type"] in ("done", "stopped", "error"):
                break
            if msg["type"] == "action":
                # Send another screenshot to keep the loop going
                await ws.send(json.dumps({
                    "type": "screenshot", "screenshot": BLANK,
                }))

        msg_types = [m["type"] for m in messages]
        assert "thinking" in msg_types, f"No thinking message: {msg_types}"
        # Must get at least one action or done (real Gemini responded)
        assert any(t in ("action", "done") for t in msg_types), (
            f"No action or done from real Gemini: {msg_types}"
        )

        # Stop the task
        await ws.send(json.dumps({"type": "stop"}))


async def test_real_tts_endpoint(real_server):
    """TTS endpoint with real server returns base64 audio."""
    http_url, _ = real_server
    r = httpx.post(
        f"{http_url}/webpilot/tts",
        json={"text": "Testing voice narration."},
        timeout=15.0,
    )
    assert r.status_code == 200, f"TTS failed: {r.status_code} {r.text}"
    data = r.json()
    assert "audio" in data
    audio_bytes = base64.b64decode(data["audio"])
    assert len(audio_bytes) > 100, f"Audio too small: {len(audio_bytes)} bytes"

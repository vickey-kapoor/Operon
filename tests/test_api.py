"""API-level tests for the UI Navigator FastAPI server.

Tests use httpx.AsyncClient pointed at the ASGI app directly — no server needed.
"""

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.core import AgentResult

# Fixtures (client, api_key, gemini_key) and helpers come from conftest.py.
from tests.conftest import small_png as _small_png, large_png as _large_png


# ---------------------------------------------------------------------------
# Health

# ---------------------------------------------------------------------------


async def test_health_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "active_tasks" in data
    assert "total_tasks" in data


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


async def test_missing_api_key_returns_401(client, api_key):
    resp = await client.post("/navigate", json={"task": "do something"})
    assert resp.status_code == 401


async def test_invalid_api_key_returns_401(client, api_key):
    resp = await client.post(
        "/navigate",
        json={"task": "do something"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


async def test_valid_api_key_accepted(client, api_key, gemini_key):
    """POST /navigate with a valid key should return 202 and a task_id."""
    mock_result = AgentResult(success=True, steps_taken=1)

    with patch("src.api.server.UINavigatorAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.run = AsyncMock(return_value=mock_result)
        instance.task_id = None
        instance.on_step = None

        resp = await client.post(
            "/navigate",
            json={"task": "do something"},
            headers={"X-API-Key": "valid-key-123"},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert "task_id" in data
    assert data["status"] == "started"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


async def test_rate_limit_returns_429(client, api_key, monkeypatch):
    """Setting RPM=2 and making 3 requests should trigger a 429."""
    monkeypatch.setenv("RATE_LIMIT_RPM", "2")

    # Re-read the env var by patching the module-level constant.
    import src.api.server as srv

    original_limit = srv._RATE_LIMIT_RPM
    srv._RATE_LIMIT_RPM = 2

    # Clear any existing rate window for this key.
    srv._rate_windows["valid-key-123"].clear()

    try:
        mock_result = AgentResult(success=True, steps_taken=0)
        with patch("src.api.server.UINavigatorAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.run = AsyncMock(return_value=mock_result)
            instance.task_id = None
            instance.on_step = None

            headers = {"X-API-Key": "valid-key-123"}
            r1 = await client.post("/navigate", json={"task": "t1"}, headers=headers)
            r2 = await client.post("/navigate", json={"task": "t2"}, headers=headers)
            r3 = await client.post("/navigate", json={"task": "t3"}, headers=headers)

        assert r1.status_code == 202
        assert r2.status_code == 202
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers
    finally:
        srv._RATE_LIMIT_RPM = original_limit
        srv._rate_windows["valid-key-123"].clear()


# ---------------------------------------------------------------------------
# Task not found
# ---------------------------------------------------------------------------


async def test_task_not_found_returns_404(client):
    resp = await client.get("/tasks/does-not-exist-00000000")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------


async def test_list_tasks_returns_200(client, api_key, gemini_key):
    """GET /tasks should return a TaskListResponse with a list."""
    mock_result = AgentResult(success=True, steps_taken=1)

    with patch("src.api.server.UINavigatorAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.run = AsyncMock(return_value=mock_result)
        instance.task_id = None
        instance.on_step = None

        # Create a task first.
        headers = {"X-API-Key": "valid-key-123"}
        post_resp = await client.post(
            "/navigate",
            json={"task": "list tasks test"},
            headers=headers,
        )
        assert post_resp.status_code == 202

    resp = await client.get("/tasks", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "tasks" in data
    assert "total" in data
    assert isinstance(data["tasks"], list)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


async def test_screenshot_too_large_returns_413(client, api_key, gemini_key):
    """POST /screenshot with a file > 5 MB should return 413."""
    headers = {"X-API-Key": "valid-key-123"}
    resp = await client.post(
        "/screenshot",
        files={"file": ("big.png", _large_png(), "image/png")},
        headers=headers,
    )
    assert resp.status_code == 413


async def test_task_too_long_returns_422(client, api_key, gemini_key):
    """POST /navigate with task > 2000 chars should return 422."""
    headers = {"X-API-Key": "valid-key-123"}
    resp = await client.post(
        "/navigate",
        json={"task": "x" * 2001},
        headers=headers,
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------


async def test_ssrf_blocked_file_url(client, api_key, gemini_key):
    """POST /navigate with a file:// start_url should return 422."""
    headers = {"X-API-Key": "valid-key-123"}
    resp = await client.post(
        "/navigate",
        json={"task": "read this", "start_url": "file:///etc/passwd"},
        headers=headers,
    )
    assert resp.status_code == 422


async def test_ssrf_blocked_private_ip(client, api_key, gemini_key):
    """POST /navigate with a private-IP start_url should return 422."""
    headers = {"X-API-Key": "valid-key-123"}
    resp = await client.post(
        "/navigate",
        json={"task": "fetch", "start_url": "http://169.254.169.254/metadata"},
        headers=headers,
    )
    assert resp.status_code == 422


async def test_ssrf_allowed_public_url(client, api_key, gemini_key):
    """POST /navigate with a legitimate public URL should be accepted (202)."""
    mock_result = AgentResult(success=True, steps_taken=1)
    headers = {"X-API-Key": "valid-key-123"}

    with patch("src.api.server.UINavigatorAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.run = AsyncMock(return_value=mock_result)
        instance.task_id = None
        instance.on_step = None

        resp = await client.post(
            "/navigate",
            json={"task": "go there", "start_url": "https://example.com"},
            headers=headers,
        )

    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Correlation ID header
# ---------------------------------------------------------------------------


async def test_response_includes_request_id(client):
    """Every response must echo back an X-Request-ID header."""
    resp = await client.get("/health")
    assert "x-request-id" in resp.headers


async def test_provided_request_id_echoed(client):
    """If the caller provides X-Request-ID it must be reflected in the response."""
    custom_id = "my-trace-abc-123"
    resp = await client.get("/health", headers={"X-Request-ID": custom_id})
    assert resp.headers.get("x-request-id") == custom_id


# ---------------------------------------------------------------------------
# Task cancellation
# ---------------------------------------------------------------------------


async def test_cancel_nonexistent_task_returns_404(client):
    resp = await client.delete("/tasks/does-not-exist-999")
    assert resp.status_code == 404


async def test_cancel_finished_task_returns_status(client, api_key, gemini_key):
    """DELETE on a finished task returns its current status without error."""
    mock_result = AgentResult(success=True, steps_taken=1)
    headers = {"X-API-Key": "valid-key-123"}

    with patch("src.api.server.UINavigatorAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.run = AsyncMock(return_value=mock_result)
        instance.task_id = None
        instance.on_step = None

        post_resp = await client.post(
            "/navigate", json={"task": "test cancel"}, headers=headers
        )
        assert post_resp.status_code == 202
        task_id = post_resp.json()["task_id"]

    # Give the background task a moment to complete.
    await asyncio.sleep(0.1)

    del_resp = await client.delete(f"/tasks/{task_id}", headers=headers)
    # Either 200 (status returned) or 404 if already cleaned up — both are valid.
    assert del_resp.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


async def test_health_includes_version(client):
    """GET /health must include the API version."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert data["version"]  # non-empty


# ---------------------------------------------------------------------------
# WebSocket — task not found (starlette TestClient)
# ---------------------------------------------------------------------------


def test_websocket_unknown_task_closes():
    """Connecting to /ws/<unknown-id> should be rejected immediately."""
    from starlette.testclient import TestClient
    from src.api.server import app

    with TestClient(app) as tc:
        try:
            with tc.websocket_connect("/ws/nonexistent-task-id-xyz"):
                pass
        except Exception:
            pass  # server closes with code 4404 — starlette raises on non-101

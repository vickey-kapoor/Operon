"""Tests for the ADK Chrome Extension session routes."""

import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.agent.planner import ActionPlan
from src.executor.actions import Action, ActionType


def _make_valid_image_b64() -> str:
    """Return a base64-encoded 1x1 PNG."""
    img = Image.new("RGB", (64, 64), color=(0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_plan(done: bool = False) -> ActionPlan:
    return ActionPlan(
        observation="A test page is visible.",
        reasoning="Test reasoning.",
        actions=[Action(type=ActionType.SCREENSHOT, description="observe")],
        done=done,
        result="done!" if done else None,
    )


# Fixtures (client, api_key, gemini_key) come from conftest.py.


# ---------------------------------------------------------------------------
# POST /sessions — create
# ---------------------------------------------------------------------------


async def test_create_session_returns_201(client):
    """POST /sessions must return 201 and a session_id."""
    with patch("src.api.session_routes.adk_agent.create_session", new=AsyncMock(return_value="sess-abc")):
        resp = await client.post("/sessions")
    assert resp.status_code == 201
    data = resp.json()
    assert "session_id" in data
    assert data["session_id"] == "sess-abc"


# ---------------------------------------------------------------------------
# POST /sessions/{id}/step
# ---------------------------------------------------------------------------


async def test_step_returns_action_plan(client):
    """POST /sessions/{id}/step with a valid screenshot returns an ActionPlan dict."""
    plan = _make_plan()
    with (
        patch("src.api.session_routes.adk_agent.session_exists", new=AsyncMock(return_value=True)),
        patch("src.api.session_routes.adk_agent.step", new=AsyncMock(return_value=plan)),
    ):
        resp = await client.post(
            "/sessions/sess-abc/step",
            json={"image_b64": _make_valid_image_b64(), "task": "do something"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "actions" in data
    assert isinstance(data["actions"], list)


async def test_step_unknown_session_returns_404(client):
    """POST /sessions/{id}/step with unknown session_id must return 404."""
    with patch("src.api.session_routes.adk_agent.session_exists", new=AsyncMock(return_value=False)):
        resp = await client.post(
            "/sessions/nonexistent/step",
            json={"image_b64": _make_valid_image_b64(), "task": "test"},
        )
    assert resp.status_code == 404


async def test_step_invalid_base64_returns_400(client):
    """POST /sessions/{id}/step with invalid base64 must return 400."""
    with patch("src.api.session_routes.adk_agent.session_exists", new=AsyncMock(return_value=True)):
        resp = await client.post(
            "/sessions/sess-abc/step",
            json={"image_b64": "!!!not-base64!!!", "task": "test"},
        )
    assert resp.status_code == 400


async def test_step_task_too_long_returns_422(client):
    """POST /sessions/{id}/step with task > 2000 chars must return 422."""
    with patch("src.api.session_routes.adk_agent.session_exists", new=AsyncMock(return_value=True)):
        resp = await client.post(
            "/sessions/sess-abc/step",
            json={"image_b64": _make_valid_image_b64(), "task": "x" * 2001},
        )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /sessions/{id}/events
# ---------------------------------------------------------------------------


async def test_events_returns_204(client):
    """POST /sessions/{id}/events with a valid body must return 204."""
    # Create a real session first (session_event now validates existence)
    create_resp = await client.post("/sessions")
    session_id = create_resp.json()["session_id"]
    resp = await client.post(
        f"/sessions/{session_id}/events",
        json={"event": "click", "data": {"x": 100, "y": 200}},
    )
    assert resp.status_code == 204


async def test_events_missing_event_field_returns_422(client):
    """POST /sessions/{id}/events without required 'event' field must return 422."""
    resp = await client.post(
        "/sessions/sess-abc/events",
        json={"data": {"x": 100}},
    )
    assert resp.status_code == 422


async def test_events_event_too_long_returns_422(client):
    """POST /sessions/{id}/events with event name > 100 chars must return 422."""
    resp = await client.post(
        "/sessions/sess-abc/events",
        json={"event": "x" * 101},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /sessions/{id}
# ---------------------------------------------------------------------------


async def test_delete_session_returns_204(client):
    """DELETE /sessions/{id} for an existing session must return 204."""
    with patch("src.api.session_routes.adk_agent.delete_session", new=AsyncMock(return_value=True)):
        resp = await client.delete("/sessions/sess-abc")
    assert resp.status_code == 204


async def test_delete_unknown_session_returns_404(client):
    """DELETE /sessions/{id} for unknown session_id must return 404."""
    with patch("src.api.session_routes.adk_agent.delete_session", new=AsyncMock(return_value=False)):
        resp = await client.delete("/sessions/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Additional session tests
# ---------------------------------------------------------------------------


async def test_step_done_plan(client):
    """POST /sessions/{id}/step with done=True returns result field."""
    plan = _make_plan(done=True)
    with (
        patch('src.api.session_routes.adk_agent.session_exists', new=AsyncMock(return_value=True)),
        patch('src.api.session_routes.adk_agent.step', new=AsyncMock(return_value=plan)),
    ):
        resp = await client.post(
            '/sessions/sess-abc/step',
            json={'image_b64': _make_valid_image_b64(), 'task': 'test'},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data['done'] is True
    assert data['result'] == 'done!'


async def test_step_image_too_large_returns_413(client):
    """POST /sessions/{id}/step with image_b64 > 10 MB returns 413."""
    huge_b64 = 'A' * (11 * 1024 * 1024)  # > 10 MB
    with patch('src.api.session_routes.adk_agent.session_exists', new=AsyncMock(return_value=True)):
        resp = await client.post(
            '/sessions/sess-abc/step',
            json={'image_b64': huge_b64, 'task': 'test'},
        )
    assert resp.status_code == 413


async def test_events_nonexistent_session_returns_404(client):
    """POST /sessions/{id}/events for non-existent session returns 404."""
    with patch('src.api.session_routes.adk_agent.session_exists', new=AsyncMock(return_value=False)):
        resp = await client.post(
            '/sessions/nonexistent/events',
            json={'event': 'click', 'data': {'x': 1}},
        )
    assert resp.status_code == 404

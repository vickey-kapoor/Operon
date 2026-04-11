"""Tests for the desktop API routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models.common import RunResponse, RunStatus


@pytest.fixture
def client():
    """Create a test client with mocked desktop agent loop."""
    from src.api.server import app

    return TestClient(app)


def _mock_run_response(status: RunStatus = RunStatus.RUNNING) -> RunResponse:
    return RunResponse(
        run_id="test-run-1",
        status=status,
        intent="Open Calculator",
        step_count=1,
    )


def test_root_serves_task_console(client: TestClient) -> None:
    """GET / should return the task console UI."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Task Console" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_console_route_serves_same_content(client: TestClient) -> None:
    """GET /console should serve the same HTML as GET /."""
    root = client.get("/")
    console = client.get("/console")
    assert root.status_code == 200
    assert console.status_code == 200
    assert root.text == console.text


def test_desktop_run_task(client: TestClient) -> None:
    """POST /desktop/run-task should create a desktop run."""
    mock_response = _mock_run_response(RunStatus.PENDING)

    with patch("src.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_loop.return_value.start_run = AsyncMock(return_value=mock_response)
        mock_loop.return_value.executor.reset_desktop = AsyncMock()
        resp = client.post(
            "/desktop/run-task",
            json={"intent": "Open Calculator"},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["run_id"] == "test-run-1"
    assert data["intent"] == "Open Calculator"


def test_desktop_step(client: TestClient) -> None:
    """POST /desktop/step should advance a desktop run."""
    mock_response = _mock_run_response(RunStatus.RUNNING)

    with patch("src.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_loop.return_value.step_run = AsyncMock(return_value=mock_response)
        resp = client.post(
            "/desktop/step",
            json={"run_id": "test-run-1"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"


def test_desktop_step_not_found(client: TestClient) -> None:
    """POST /desktop/step with unknown run_id should return 404."""
    with patch("src.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_loop.return_value.step_run = AsyncMock(side_effect=ValueError("Run not found"))
        resp = client.post(
            "/desktop/step",
            json={"run_id": "nonexistent"},
        )

    assert resp.status_code == 404


def test_desktop_get_run(client: TestClient) -> None:
    """GET /desktop/run/{id} should return run state."""
    from src.models.state import AgentState

    mock_state = AgentState(
        run_id="test-run-1",
        intent="Open Calculator",
        status=RunStatus.RUNNING,
    )

    with patch("src.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_loop.return_value.run_store.get_run = AsyncMock(return_value=mock_state)
        resp = client.get("/desktop/run/test-run-1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "test-run-1"


def test_desktop_get_run_not_found(client: TestClient) -> None:
    """GET /desktop/run/{id} with unknown id should return 404."""
    with patch("src.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_loop.return_value.run_store.get_run = AsyncMock(return_value=None)
        resp = client.get("/desktop/run/nonexistent")

    assert resp.status_code == 404


def test_desktop_resume(client: TestClient) -> None:
    """POST /desktop/resume should resume a paused desktop run."""
    mock_response = _mock_run_response(RunStatus.RUNNING)

    with patch("src.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_loop.return_value.resume_run = AsyncMock(return_value=mock_response)
        resp = client.post(
            "/desktop/resume",
            json={"run_id": "test-run-1"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"


def test_desktop_resume_not_found(client: TestClient) -> None:
    """POST /desktop/resume with bad run_id should return 404."""
    with patch("src.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_loop.return_value.resume_run = AsyncMock(side_effect=ValueError("invalid"))
        resp = client.post(
            "/desktop/resume",
            json={"run_id": "bad-id"},
        )

    assert resp.status_code == 404


def test_desktop_cleanup(client: TestClient) -> None:
    """POST /desktop/cleanup should call executor.cleanup_run."""
    with patch("src.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_executor = mock_loop.return_value.executor
        mock_executor.cleanup_run.return_value = 2
        resp = client.post(
            "/desktop/cleanup",
            json={"run_id": "test-run-1"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "test-run-1"
    assert data["closed_count"] == 2
    assert "2 application" in data["detail"]


def test_desktop_cleanup_no_apps(client: TestClient) -> None:
    """POST /desktop/cleanup with nothing to close returns 0."""
    with patch("src.api.routes.get_desktop_agent_loop") as mock_loop:
        mock_executor = mock_loop.return_value.executor
        mock_executor.cleanup_run.return_value = 0
        resp = client.post(
            "/desktop/cleanup",
            json={"run_id": "test-run-empty"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["closed_count"] == 0


def test_browser_cleanup(client: TestClient) -> None:
    """POST /cleanup should call executor.cleanup_run for browser runs."""
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        mock_executor = mock_loop.return_value.executor
        mock_executor.cleanup_run.return_value = 1
        resp = client.post(
            "/cleanup",
            json={"run_id": "browser-run-1"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "browser-run-1"
    assert data["closed_count"] == 1
    assert "1 browser session" in data["detail"]


def test_browser_cleanup_without_support(client: TestClient) -> None:
    """POST /cleanup should return a no-op response if cleanup is unsupported."""
    with patch("src.api.routes.get_agent_loop") as mock_loop:
        del mock_loop.return_value.executor.cleanup_run
        resp = client.post(
            "/cleanup",
            json={"run_id": "browser-run-2"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "browser-run-2"
    assert data["closed_count"] == 0
    assert "does not support cleanup" in data["detail"]


# ── desktop agent loop configuration tests ──────────────────────


def test_desktop_agent_loop_uses_120s_timeout_and_combined_service() -> None:
    """get_desktop_agent_loop should use CombinedPerceptionPolicyService with 120s timeout."""
    import src.api.routes as routes_module

    # Reset the singleton so it gets rebuilt
    original = routes_module._desktop_agent_loop
    routes_module._desktop_agent_loop = None

    try:
        with (
            patch("src.api.routes.DesktopExecutor"),
            patch("src.api.routes.FileBackedRunStore"),
            patch("src.api.routes.FileBackedMemoryStore"),
            patch("src.api.routes.GeminiHttpClient") as mock_gemini,
            patch("src.api.routes.ScreenCaptureService"),
            patch("src.api.routes.CombinedPerceptionPolicyService"),
            patch("src.api.routes.PolicyCoordinator"),
            patch("src.api.routes.DeterministicVerifierService"),
            patch("src.api.routes.RuleBasedRecoveryManager"),
            patch("src.api.routes.AgentLoop"),
        ):
            routes_module.get_desktop_agent_loop()

            # Two GeminiHttpClients: combined + policy verifier, both with 120s timeout
            assert mock_gemini.call_count == 2
            calls = mock_gemini.call_args_list
            assert all(c.kwargs.get("timeout_seconds") == 120.0 for c in calls)
    finally:
        routes_module._desktop_agent_loop = original


# ── UIElementType._missing_ tests ──────────────────────────────


def test_ui_element_type_missing_accepts_arbitrary_strings() -> None:
    """UIElementType._missing_ should accept arbitrary strings like 'menu_item', 'toolbar'."""
    from src.models.perception import UIElementType

    menu_item = UIElementType("menu_item")
    assert menu_item.value == "menu_item"
    assert isinstance(menu_item, UIElementType)

    toolbar = UIElementType("toolbar")
    assert toolbar.value == "toolbar"
    assert isinstance(toolbar, UIElementType)

    taskbar_icon = UIElementType("taskbar_icon")
    assert taskbar_icon.value == "taskbar_icon"
    assert isinstance(taskbar_icon, UIElementType)


def test_ui_element_type_known_values_still_work() -> None:
    """Known UIElementType values should still resolve normally."""
    from src.models.perception import UIElementType

    assert UIElementType("button") is UIElementType.BUTTON
    assert UIElementType("window") is UIElementType.WINDOW
    assert UIElementType("unknown") is UIElementType.UNKNOWN


def test_ui_element_type_window_exists() -> None:
    """UIElementType.WINDOW should exist and have value 'window'."""
    from src.models.perception import UIElementType

    assert hasattr(UIElementType, "WINDOW")
    assert UIElementType.WINDOW.value == "window"


# ── ActionType desktop additions ────────────────────────────────


def test_action_type_has_desktop_actions() -> None:
    """ActionType should include LAUNCH_APP, HOTKEY, and WAIT_FOR_USER."""
    from src.models.policy import ActionType

    assert hasattr(ActionType, "LAUNCH_APP")
    assert ActionType.LAUNCH_APP.value == "launch_app"
    assert hasattr(ActionType, "HOTKEY")
    assert ActionType.HOTKEY.value == "hotkey"
    assert hasattr(ActionType, "WAIT_FOR_USER")
    assert ActionType.WAIT_FOR_USER.value == "wait_for_user"


# ── new desktop action type and validation tests ─────────────────


def test_action_type_has_all_new_types() -> None:
    """ActionType should include all 8 new desktop action types."""
    from src.models.policy import ActionType

    new_types = {
        "DOUBLE_CLICK": "double_click",
        "RIGHT_CLICK": "right_click",
        "DRAG": "drag",
        "SCROLL": "scroll",
        "HOVER": "hover",
        "READ_CLIPBOARD": "read_clipboard",
        "WRITE_CLIPBOARD": "write_clipboard",
        "SCREENSHOT_REGION": "screenshot_region",
    }
    for attr, value in new_types.items():
        assert hasattr(ActionType, attr), f"ActionType missing {attr}"
        assert ActionType(value).value == value


def test_double_click_action_valid() -> None:
    """DOUBLE_CLICK with x,y should be valid."""
    from src.models.policy import ActionType, AgentAction

    action = AgentAction(action_type=ActionType.DOUBLE_CLICK, x=100, y=200)
    assert action.action_type is ActionType.DOUBLE_CLICK
    assert action.x == 100
    assert action.y == 200


def test_double_click_rejects_text() -> None:
    """DOUBLE_CLICK with text should raise ValueError."""
    from src.models.policy import ActionType, AgentAction

    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.DOUBLE_CLICK, x=100, y=200, text="bad")


def test_drag_action_valid() -> None:
    """DRAG with all 4 coordinates should be valid."""
    from src.models.policy import ActionType, AgentAction

    action = AgentAction(action_type=ActionType.DRAG, x=10, y=20, x_end=300, y_end=400)
    assert action.x_end == 300
    assert action.y_end == 400


def test_drag_missing_x_end_fails() -> None:
    """DRAG without x_end should raise ValueError."""
    from src.models.policy import ActionType, AgentAction

    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.DRAG, x=10, y=20, y_end=400)


def test_scroll_action_valid() -> None:
    """SCROLL with scroll_amount and coordinates should be valid."""
    from src.models.policy import ActionType, AgentAction

    action = AgentAction(action_type=ActionType.SCROLL, scroll_amount=3, x=100, y=200)
    assert action.scroll_amount == 3


def test_scroll_zero_amount_fails() -> None:
    """SCROLL with scroll_amount=0 should raise ValueError."""
    from src.models.policy import ActionType, AgentAction

    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.SCROLL, scroll_amount=0, x=100, y=200)


def test_scroll_rejects_x_end() -> None:
    """SCROLL with x_end should raise ValueError."""
    from src.models.policy import ActionType, AgentAction

    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.SCROLL, scroll_amount=3, x=100, y=200, x_end=500)


def test_hover_action_valid() -> None:
    """HOVER with x,y should be valid."""
    from src.models.policy import ActionType, AgentAction

    action = AgentAction(action_type=ActionType.HOVER, x=100, y=200)
    assert action.action_type is ActionType.HOVER


def test_read_clipboard_action_valid() -> None:
    """READ_CLIPBOARD with no fields should be valid."""
    from src.models.policy import ActionType, AgentAction

    action = AgentAction(action_type=ActionType.READ_CLIPBOARD)
    assert action.action_type is ActionType.READ_CLIPBOARD


def test_read_clipboard_rejects_text() -> None:
    """READ_CLIPBOARD with text should raise ValueError."""
    from src.models.policy import ActionType, AgentAction

    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.READ_CLIPBOARD, text="bad")


def test_write_clipboard_action_valid() -> None:
    """WRITE_CLIPBOARD with text should be valid."""
    from src.models.policy import ActionType, AgentAction

    action = AgentAction(action_type=ActionType.WRITE_CLIPBOARD, text="hello")
    assert action.text == "hello"


def test_write_clipboard_rejects_coords() -> None:
    """WRITE_CLIPBOARD with x,y should raise ValueError."""
    from src.models.policy import ActionType, AgentAction

    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.WRITE_CLIPBOARD, text="hello", x=100, y=200)


def test_screenshot_region_action_valid() -> None:
    """SCREENSHOT_REGION with all 4 coordinates should be valid."""
    from src.models.policy import ActionType, AgentAction

    action = AgentAction(action_type=ActionType.SCREENSHOT_REGION, x=10, y=20, x_end=200, y_end=300)
    assert action.x_end == 200
    assert action.y_end == 300


def test_existing_click_rejects_new_fields() -> None:
    """CLICK with x_end should raise ValueError."""
    from src.models.policy import ActionType, AgentAction

    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.CLICK, x=100, y=200, x_end=300)


def test_existing_stop_rejects_new_fields() -> None:
    """STOP with scroll_amount should raise ValueError."""
    from src.models.policy import ActionType, AgentAction

    with pytest.raises(Exception):
        AgentAction(action_type=ActionType.STOP, scroll_amount=5)

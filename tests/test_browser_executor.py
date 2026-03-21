"""Focused tests for Playwright-backed browser execution and target resolution."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from src.agent.capture import BrowserCaptureService
from src.executor.browser import (
    PlaywrightBrowserExecutor,
    _ensure_windows_process_env,
    _read_browser_debug_config,
)
from src.models.common import FailureCategory, RunStatus
from src.models.policy import ActionType, AgentAction
from src.models.state import AgentState


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def _skip_if_playwright_blocked(exc: Exception) -> None:
    if isinstance(exc, PermissionError):
        pytest.skip(f"Playwright subprocess launch is blocked in this environment: {exc}")
    raise exc


def test_browser_debug_config_uses_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BROWSER_HEADLESS", raising=False)
    monkeypatch.delenv("BROWSER_SLOW_MO_MS", raising=False)
    monkeypatch.delenv("BROWSER_DEVTOOLS", raising=False)

    config = _read_browser_debug_config()

    assert config.headless is True
    assert config.slow_mo_ms == 0
    assert config.devtools is False


def test_browser_debug_config_reads_env_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_HEADLESS", "false")
    monkeypatch.setenv("BROWSER_SLOW_MO_MS", "250")
    monkeypatch.setenv("BROWSER_DEVTOOLS", "true")

    config = _read_browser_debug_config()
    executor = PlaywrightBrowserExecutor(artifact_root=Path(".test-artifacts") / "config-only")

    assert config.headless is False
    assert config.slow_mo_ms == 250
    assert config.devtools is True
    assert executor.headless is False
    assert executor.slow_mo_ms == 250
    assert executor.devtools is True


def test_windows_process_env_restores_system_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SystemRoot", raising=False)
    monkeypatch.delenv("WINDIR", raising=False)
    monkeypatch.setenv("TEMP", "C:\\Temp")

    _ensure_windows_process_env()

    assert os.environ.get("SystemRoot") == "C:\\Windows"
    assert os.environ.get("WINDIR") == "C:\\Windows"


@pytest.mark.asyncio
async def test_browser_executor_dispatches_navigation_and_type_actions() -> None:
    root_dir = _local_test_dir("test-browser-executor-dispatch")
    executor = PlaywrightBrowserExecutor(artifact_root=root_dir / "scratch")

    try:
        navigate = AgentAction(
            action_type=ActionType.NAVIGATE,
            url="data:text/html,<html><body><input id='subject' value=''></body></html>",
        )
        navigate_result = await executor.execute(navigate)
        assert navigate_result.success is True
        assert navigate_result.artifact_path is not None
        assert Path(navigate_result.artifact_path).exists()

        type_action = AgentAction(
            action_type=ActionType.TYPE,
            target_element_id="subject",
            text="Draft subject",
        )
        type_result = await executor.execute(type_action)
        assert type_result.success is True

        key_action = AgentAction(
            action_type=ActionType.PRESS_KEY,
            key="Tab",
        )
        key_result = await executor.execute(key_action)
        assert key_result.success is True
    except Exception as exc:
        _skip_if_playwright_blocked(exc)
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_target_resolution_uses_explicit_selector_first() -> None:
    root_dir = _local_test_dir("test-target-resolution-selector")
    executor = PlaywrightBrowserExecutor(artifact_root=root_dir / "scratch")

    try:
        await executor.execute(
            AgentAction(
                action_type=ActionType.NAVIGATE,
                url="data:text/html,<html><body><input id='subject'><input id='fallback'></body></html>",
            )
        )
        result = await executor.execute(
            AgentAction(
                action_type=ActionType.TYPE,
                selector="#subject",
                target_element_id="missing-target",
                text="Hello Gmail",
            )
        )
        subject_value = await executor._page.locator("#subject").input_value()  # noqa: SLF001
        fallback_value = await executor._page.locator("#fallback").input_value()  # noqa: SLF001
        assert result.success is True
        assert subject_value == "Hello Gmail"
        assert fallback_value == ""
    except Exception as exc:
        _skip_if_playwright_blocked(exc)
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_target_resolution_falls_back_to_aria_lookup() -> None:
    root_dir = _local_test_dir("test-target-resolution-aria")
    executor = PlaywrightBrowserExecutor(artifact_root=root_dir / "scratch")

    try:
        await executor.execute(
            AgentAction(
                action_type=ActionType.NAVIGATE,
                url="data:text/html,<html><body><input aria-label='recipient'></body></html>",
            )
        )
        result = await executor.execute(
            AgentAction(
                action_type=ActionType.TYPE,
                target_element_id="recipient",
                text="alice@example.com",
            )
        )
        value = await executor._page.locator("input[aria-label='recipient']").input_value()  # noqa: SLF001
        assert result.success is True
        assert value == "alice@example.com"
    except Exception as exc:
        _skip_if_playwright_blocked(exc)
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_type_fails_cleanly_when_target_is_not_editable() -> None:
    root_dir = _local_test_dir("test-type-not-editable")
    executor = PlaywrightBrowserExecutor(artifact_root=root_dir / "scratch")

    try:
        await executor.execute(
            AgentAction(
                action_type=ActionType.NAVIGATE,
                url="data:text/html,<html><body><div id='subject' tabindex='0'>Subject</div></body></html>",
            )
        )
        result = await executor.execute(
            AgentAction(
                action_type=ActionType.TYPE,
                target_element_id="subject",
                text="Hello",
            )
        )
        assert result.success is False
        assert result.failure_category is FailureCategory.EXECUTION_TARGET_NOT_EDITABLE
        assert "not editable" in result.detail.lower()
    except Exception as exc:
        _skip_if_playwright_blocked(exc)
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_target_resolution_reports_failure_for_missing_target() -> None:
    root_dir = _local_test_dir("test-target-resolution-failure")
    executor = PlaywrightBrowserExecutor(artifact_root=root_dir / "scratch")

    try:
        await executor.execute(
            AgentAction(
                action_type=ActionType.NAVIGATE,
                url="data:text/html,<html><body><button id='compose'>Compose</button></body></html>",
            )
        )
        result = await executor.execute(
            AgentAction(
                action_type=ActionType.CLICK,
                target_element_id="missing-target",
            )
        )
        assert result.success is False
        assert "Unable to resolve click target" in result.detail
    except Exception as exc:
        _skip_if_playwright_blocked(exc)
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_capture_service_writes_planned_before_screenshot() -> None:
    root_dir = _local_test_dir("test-browser-capture-service")
    executor = PlaywrightBrowserExecutor(artifact_root=root_dir / "scratch")
    capture_service = BrowserCaptureService(browser_executor=executor, root_dir=root_dir / "runs")
    state = AgentState(run_id="run-capture", intent="Create draft", status=RunStatus.PENDING)

    try:
        await executor.execute(
            AgentAction(
                action_type=ActionType.NAVIGATE,
                url="data:text/html,<html><body><button id='compose'>Compose</button></body></html>",
            )
        )
        frame = await capture_service.capture(state)
        expected_path = root_dir / "runs" / "run-capture" / "step_1" / "before.png"
        assert frame.artifact_path == str(expected_path)
        assert expected_path.exists()
    except Exception as exc:
        _skip_if_playwright_blocked(exc)
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_browser_executor_capture_creates_real_screenshot_file() -> None:
    root_dir = _local_test_dir("test-browser-executor-capture")
    executor = PlaywrightBrowserExecutor(artifact_root=root_dir / "scratch")

    try:
        await executor.execute(
            AgentAction(
                action_type=ActionType.NAVIGATE,
                url="data:text/html,<html><body><h1>Hello</h1></body></html>",
            )
        )
        frame = await executor.capture()
        screenshot_path = Path(frame.artifact_path)
        assert screenshot_path.exists()
        assert screenshot_path.suffix == ".png"
        assert frame.width == 1280
        assert frame.height == 800
    except Exception as exc:
        _skip_if_playwright_blocked(exc)
    finally:
        await executor.close()

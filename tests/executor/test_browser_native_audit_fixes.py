"""KAN-151: browser executor fixes from the external audit (keys, load wait, upload)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from operon.browser.keys import normalize_key
from operon.executor.browser_native import NativeBrowserExecutor
from operon.models.common import FailureCategory
from operon.models.policy import ActionType, AgentAction


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ctrl+a", "Control+a"),
        ("control+shift+t", "Control+Shift+t"),
        ("Control+a", "Control+a"),
        ("cmd+c", "Meta+c"),
        ("alt+left", "Alt+ArrowLeft"),
        ("enter", "Enter"),
        ("f5", "F5"),
        ("a", "a"),
        ("+", "+"),
    ],
)
def test_normalize_key_spells_keys_the_way_playwright_accepts(raw: str, expected: str) -> None:
    assert normalize_key(raw) == expected


def _executor(tmp_path: Path, page) -> NativeBrowserExecutor:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    executor._current_page = AsyncMock(return_value=page)
    executor._capture_after = AsyncMock(return_value=str(tmp_path / "after.png"))
    return executor


async def test_hotkey_with_lowercase_modifiers_reaches_playwright_normalized(tmp_path: Path) -> None:
    keyboard = SimpleNamespace(press=AsyncMock())
    page = SimpleNamespace(keyboard=keyboard, wait_for_load_state=AsyncMock())

    result = await _executor(tmp_path, page).execute(AgentAction(action_type=ActionType.HOTKEY, key="ctrl+a"))

    assert result.success is True
    keyboard.press.assert_awaited_once_with("Control+a")


async def test_a_slow_page_load_after_a_click_is_not_a_failed_click(tmp_path: Path) -> None:
    mouse = SimpleNamespace(click=AsyncMock())
    page = SimpleNamespace(mouse=mouse, wait_for_load_state=AsyncMock(side_effect=TimeoutError("5000ms")))

    result = await _executor(tmp_path, page).execute(AgentAction(action_type=ActionType.CLICK, x=10, y=20))

    assert result.success is True
    mouse.click.assert_awaited_once()


async def test_upload_file_without_coordinates_fails_instead_of_using_the_dom(tmp_path: Path) -> None:
    page = SimpleNamespace(locator=AsyncMock(), expect_file_chooser=AsyncMock())

    result = await _executor(tmp_path, page).execute(
        AgentAction(action_type=ActionType.UPLOAD_FILE, text=str(tmp_path / "cv.pdf"))
    )

    assert result.success is False
    assert result.failure_category is FailureCategory.EXECUTION_TARGET_NOT_FOUND
    page.locator.assert_not_called()
    page.expect_file_chooser.assert_not_called()

"""Scrolling goes where and which way the model asked; JSON help requests parse (KAN-49)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from operon.agent.actions.translation import normalize_computer_use_function_call
from operon.agent.policy.service import parse_policy_output
from operon.models.policy import ActionType, AgentAction

# G12: the browser wheel scrolls what's under the model's point.


class _Mouse:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def move(self, x, y):
        self.calls.append(("move", x, y))

    async def wheel(self, dx, dy):
        self.calls.append(("wheel", dx, dy))


async def _scroll_in_browser(tmp_path: Path, action: AgentAction) -> list[tuple]:
    from operon.executor.browser_native import NativeBrowserExecutor

    executor = NativeBrowserExecutor(artifact_dir=tmp_path)
    mouse = _Mouse()
    page = SimpleNamespace(mouse=mouse, url="about:blank")
    executor._current_page = AsyncMock(return_value=page)
    executor._capture_after = AsyncMock(return_value=str(tmp_path / "after.png"))
    try:
        await executor.execute(action)
    except Exception:
        pass  # only the mouse calls matter here
    return mouse.calls


async def test_browser_scroll_moves_to_the_point_first(tmp_path: Path) -> None:
    calls = await _scroll_in_browser(tmp_path, AgentAction(action_type=ActionType.SCROLL, x=900, y=300, scroll_amount=-400))

    assert calls[:2] == [("move", 900, 300), ("wheel", 0, 400)]


async def test_browser_horizontal_scroll_uses_delta_x(tmp_path: Path) -> None:
    calls = await _scroll_in_browser(
        tmp_path, AgentAction(action_type=ActionType.SCROLL, x=10, y=20, scroll_amount=300, horizontal=True)
    )

    assert calls[:2] == [("move", 10, 20), ("wheel", 300, 0)]


# G13: JSON "direction" follows the executors' convention (positive = up).


def _parse(action: dict):
    return parse_policy_output(json.dumps({"action": action, "rationale": "r", "confidence": 0.9, "active_subgoal": "s"}))


@pytest.mark.parametrize(
    ("direction", "amount", "expected", "horizontal"),
    [("up", 500, 500, None), ("up", -500, 500, None), ("down", 500, -500, None), ("left", 200, -200, True), ("right", 200, 200, True)],
)
def test_json_scroll_direction_sets_the_sign(direction, amount, expected, horizontal) -> None:
    decision = _parse({"action_type": "scroll", "x": 5, "y": 5, "scroll_amount": amount, "direction": direction})

    assert decision.action.scroll_amount == expected
    assert decision.action.horizontal is horizontal


# G16: WAIT_FOR_USER keeps its reason and STOP keeps its answer.


def test_json_wait_for_user_keeps_its_reason() -> None:
    decision = _parse({"action_type": "wait_for_user", "text": "Please solve the CAPTCHA", "x": 1, "y": 2})

    assert decision.action.action_type is ActionType.WAIT_FOR_USER
    assert decision.action.text == "Please solve the CAPTCHA"
    assert decision.action.x is None


def test_json_stop_keeps_its_answer() -> None:
    decision = _parse({"action_type": "stop", "text": "1991", "url": "https://x.test"})

    assert decision.action.text == "1991"
    assert decision.action.url is None


# G22: sideways scrolls stay sideways.


@pytest.mark.parametrize(
    ("call", "sign"),
    [
        ({"name": "scroll_at", "args": {"x": 500, "y": 500, "direction": "right", "magnitude": 400}}, 1),
        ({"name": "scroll_at", "args": {"x": 500, "y": 500, "direction": "left", "magnitude": 400}}, -1),
        ({"name": "scroll_document", "args": {"direction": "left"}}, -1),
        ({"name": "scroll", "args": {"x": 500, "y": 500, "direction": "right"}}, 1),
    ],
)
def test_gemini_sideways_scroll_is_horizontal(call, sign) -> None:
    action = normalize_computer_use_function_call(call, screen_width=1000, screen_height=800)

    assert action["horizontal"] is True
    assert (action["scroll_amount"] > 0) == (sign > 0)


def test_gemini_vertical_scroll_is_unchanged() -> None:
    action = normalize_computer_use_function_call(
        {"name": "scroll_at", "args": {"x": 500, "y": 500, "direction": "down", "magnitude": 400}}, screen_width=1000, screen_height=800
    )

    assert "horizontal" not in action
    assert action["scroll_amount"] < 0


def test_horizontal_is_only_valid_on_scroll() -> None:
    with pytest.raises(ValueError):
        AgentAction(action_type=ActionType.CLICK, x=1, y=1, horizontal=True)


def test_desktop_horizontal_scroll_does_not_scroll_vertically(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from operon.executor import desktop

    fake = MagicMock()
    monkeypatch.setattr(desktop, "pyautogui", fake)
    user32 = MagicMock()
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    monkeypatch.setattr(desktop.ctypes, "windll", SimpleNamespace(user32=user32), raising=False)

    desktop._hscroll(240, 10, 20)

    fake.moveTo.assert_called_once_with(10, 20)
    user32.mouse_event.assert_called_once_with(desktop._MOUSEEVENTF_HWHEEL, 0, 0, 240, 0)
    fake.scroll.assert_not_called()
    fake.hscroll.assert_not_called()

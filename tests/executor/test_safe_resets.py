"""Recovery resets never click blind, never alt+tab, and never navigate off the allowed start URL.

Pixels only: screen capture and pyautogui/Playwright are faked; the blank-region test,
candidate search and the loop's trust-gate plumbing are the production code.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from operon.agent.perception.screen_diff import reset_click_candidates
from operon.executor.desktop import DesktopExecutor
from operon.executor.linux_desktop import LinuxDesktopExecutor
from operon.models.common import RunStatus
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.state import AgentState
from tests.agent.test_loop import _loop
from tests.executor.test_desktop_servo import (
    H,
    W,
    _executor,
    _FakeMss,
    _textlike,
    _uniform,
)

DEFAULT = (int(W * 0.5), int(H / 3))


def _frame_with_content_at(*points: tuple[int, int]) -> np.ndarray:
    """A blank screen with text-like stripes in a 120px square around each point."""
    frame = _uniform()
    stripes = _textlike()
    for x, y in points:
        frame[y - 60 : y + 60, x - 60 : x + 60] = stripes[y - 60 : y + 60, x - 60 : x + 60]
    return frame


def _run(frame: np.ndarray, coro_fn):
    pa = MagicMock()
    with (
        patch("operon.executor.desktop._mss", side_effect=lambda: _FakeMss(frame)),
        patch("operon.executor.desktop.pyautogui", pa),
    ):
        asyncio.run(coro_fn())
    return pa


@pytest.mark.parametrize("cls", [DesktopExecutor, LinuxDesktopExecutor])
def test_blank_default_point_is_clicked(cls) -> None:
    frame = _uniform()
    ex = _executor(frame, cls)
    pa = _run(frame, ex.context_reset)
    pa.click.assert_called_once_with(*DEFAULT)
    assert any("clicked blank region" in a for a in ex.last_reset_actions)


def test_content_at_default_clicks_a_blank_alternative() -> None:
    frame = _frame_with_content_at(DEFAULT)
    ex = _executor(frame)
    pa = _run(frame, ex.context_reset)
    pa.click.assert_called_once()
    point = pa.click.call_args.args
    assert point != DEFAULT
    assert point in reset_click_candidates(W, H)


def test_no_blank_region_skips_the_click_but_keeps_the_keys() -> None:
    frame = _textlike()
    ex = _executor(frame)
    pa = _run(frame, ex.context_reset)
    pa.click.assert_not_called()
    assert [c.args for c in pa.press.call_args_list] == [("escape",), ("escape",)]
    pa.hotkey.assert_called_once_with("ctrl", "home")
    assert any("skipped focus click" in a for a in ex.last_reset_actions)


def test_desktop_session_reset_does_not_alt_tab() -> None:
    frame = _uniform()
    ex = _executor(frame)
    ex._current_run_id = "run-1"
    ex._work_windows["run-1"] = 42
    with patch("operon.executor.desktop._bring_to_front", return_value=True) as front:
        pa = _run(frame, ex.session_reset)
    assert ("alt", "tab") not in [c.args for c in pa.hotkey.call_args_list]
    front.assert_called_once_with(42)
    assert "restored the work window" in ex.last_reset_actions


# ── browser ──────────────────────────────────────────────────────


def _png(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()


def _browser(frame: np.ndarray, url: str = "https://app.test/deep"):
    from operon.executor.browser_native import NativeBrowserExecutor

    ex = NativeBrowserExecutor(headless=True)
    h, w = frame.shape
    ex._viewport_width, ex._viewport_height = w, h
    page = MagicMock()
    page.url = url
    for name in ("goto", "reload"):
        setattr(page, name, AsyncMock())
    page.screenshot = AsyncMock(return_value=_png(frame))
    page.keyboard.press = AsyncMock()
    page.mouse.click = AsyncMock()
    ex._current_page = AsyncMock(return_value=page)  # type: ignore[method-assign]
    return ex, page


@pytest.mark.asyncio
async def test_browser_blank_page_is_clicked_at_the_default_point() -> None:
    ex, page = _browser(_uniform())
    await ex.context_reset()
    page.mouse.click.assert_awaited_once_with(*DEFAULT)


@pytest.mark.asyncio
async def test_browser_content_at_default_clicks_alternative() -> None:
    ex, page = _browser(_frame_with_content_at(DEFAULT))
    await ex.context_reset()
    page.mouse.click.assert_awaited_once()
    assert page.mouse.click.await_args.args != DEFAULT


@pytest.mark.asyncio
async def test_browser_no_blank_region_means_keys_only() -> None:
    ex, page = _browser(_textlike())
    await ex.context_reset()
    page.mouse.click.assert_not_awaited()
    assert [c.args for c in page.keyboard.press.await_args_list] == [("Escape",), ("Home",)]
    assert any("skipped focus click" in a for a in ex.last_reset_actions)


@pytest.mark.asyncio
async def test_browser_session_reset_goes_to_start_url_or_reloads() -> None:
    ex, page = _browser(_uniform())
    await ex.session_reset("https://app.test/start")
    page.goto.assert_awaited_once()
    assert page.goto.await_args.args == ("https://app.test/start",)
    page.reload.assert_not_awaited()

    ex, page = _browser(_uniform())
    await ex.session_reset(None)
    page.goto.assert_not_awaited()
    page.reload.assert_awaited_once()
    assert "reloaded the current page" in ex.last_reset_actions[0]


# ── loop plumbing: trust gate and trace ──────────────────────────


def _state(url: str | None) -> AgentState:
    return AgentState(run_id="run-1", intent="x", status=RunStatus.RUNNING, start_url=url)


def _session() -> RecoveryDecision:
    return RecoveryDecision(strategy=RecoveryStrategy.SESSION_RESET, message="restart")


@pytest.mark.asyncio
async def test_denied_start_url_is_never_navigated_to(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_TRUST_ALLOW_DOMAINS", "good.test")
    loop = _loop()
    loop.executor = SimpleNamespace(session_reset=AsyncMock(), last_reset_actions=["reloaded"])
    loop._artifacts = MagicMock()

    await loop._apply_recovery_actions(_state("https://evil.test/x"), _session(), 3)

    loop.executor.session_reset.assert_awaited_once_with(None)
    args = loop._artifacts.persist_recovery_reset.call_args.args
    assert args[:3] == ("run-1", 3, "session_reset")
    assert "not allowed" in args[3][0]


@pytest.mark.asyncio
async def test_allowed_start_url_is_passed_and_trace_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_TRUST_ALLOW_DOMAINS", "good.test")
    loop = _loop()
    loop.executor = SimpleNamespace(session_reset=AsyncMock(), last_reset_actions=["navigated"])
    loop._artifacts = MagicMock()

    await loop._apply_recovery_actions(_state("https://good.test/start"), _session(), 2)

    loop.executor.session_reset.assert_awaited_once_with("https://good.test/start")
    loop._artifacts.persist_recovery_reset.assert_called_once()

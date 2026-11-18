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


def _run(frame: np.ndarray, coro_fn, *, rect=(0, 0, W, H), foreground=7):
    """Run a reset with a fake screen; ``rect=None`` means the run has no recorded window."""
    pa = MagicMock()
    with (
        patch("operon.executor.desktop._mss", side_effect=lambda: _FakeMss(frame)),
        patch("operon.executor.desktop.pyautogui", pa),
        patch("operon.executor.desktop._foreground_window", return_value=foreground),
        patch("operon.executor.desktop._window_rect", return_value=rect),
        patch("operon.executor.desktop._bring_to_front", return_value=False),
    ):
        asyncio.run(coro_fn())
    return pa


def _desktop(frame: np.ndarray, cls=DesktopExecutor, *, window: int | None = 7):
    ex = _executor(frame, cls)
    ex._current_run_id = "run-1"
    if window is not None:
        ex._work_windows["run-1"] = window
    return ex


@pytest.mark.parametrize("cls", [DesktopExecutor, LinuxDesktopExecutor])
def test_blank_default_point_is_clicked(cls) -> None:
    frame = _uniform()
    ex = _desktop(frame, cls)
    pa = _run(frame, ex.context_reset)
    pa.click.assert_called_once_with(*DEFAULT)
    assert any("clicked blank region" in a for a in ex.last_reset_actions)


def test_content_at_default_clicks_a_blank_alternative() -> None:
    frame = _frame_with_content_at(DEFAULT)
    ex = _desktop(frame)
    pa = _run(frame, ex.context_reset)
    pa.click.assert_called_once()
    point = pa.click.call_args.args
    assert point != DEFAULT
    assert point in reset_click_candidates(W, H)


def test_no_blank_region_skips_the_click_but_keeps_the_keys() -> None:
    frame = _textlike()
    ex = _desktop(frame)
    pa = _run(frame, ex.context_reset)
    pa.click.assert_not_called()
    assert [c.args for c in pa.press.call_args_list] == [("escape",), ("escape",)]
    pa.hotkey.assert_called_once_with("ctrl", "home")
    assert any("skipped focus click" in a for a in ex.last_reset_actions)


def test_desktop_session_reset_does_not_alt_tab() -> None:
    frame = _uniform()
    ex = _desktop(frame, window=42)
    pa = _run(frame, ex.session_reset, foreground=42)
    assert ("alt", "tab") not in [c.args for c in pa.hotkey.call_args_list]
    assert any("work window" in a for a in ex.last_reset_actions)


def test_desktop_candidates_stay_inside_the_work_window() -> None:
    frame = _uniform()
    ex = _desktop(frame)
    pa = _run(frame, ex.context_reset, rect=(800, 100, 400, 500))
    x, y = pa.click.call_args.args
    assert 800 <= x <= 1200 and 100 <= y <= 600


def test_desktop_no_recorded_window_means_keys_only() -> None:
    frame = _uniform()
    ex = _desktop(frame, window=None)
    pa = _run(frame, ex.context_reset)
    pa.click.assert_not_called()
    assert any("no recorded work window" in a for a in ex.last_reset_actions)


def test_desktop_window_not_in_front_means_keys_only() -> None:
    frame = _uniform()
    ex = _desktop(frame)
    pa = _run(frame, ex.context_reset, foreground=99)
    pa.click.assert_not_called()
    assert any("not in front" in a for a in ex.last_reset_actions)


def test_large_neutral_panel_is_not_a_click_target() -> None:
    """Uniform 100px at the default point, but a different colour from the background."""
    frame = _uniform()
    cx, cy = DEFAULT
    frame[cy - 90 : cy + 90, cx - 90 : cx + 90] = 120  # a flat grey button/card, 180px wide
    ex = _desktop(frame)
    pa = _run(frame, ex.context_reset)
    pa.click.assert_called_once()
    assert pa.click.call_args.args != DEFAULT


def test_wide_ring_with_content_rejects_a_uniform_core() -> None:
    frame = _uniform()
    cx, cy = DEFAULT
    frame[cy - 90 : cy + 90, cx + 80 : cx + 100] = 0  # content just outside the 100px core
    ex = _desktop(frame)
    pa = _run(frame, ex.context_reset)
    assert pa.click.call_args.args != DEFAULT


def test_dominant_color_is_the_histogram_mode() -> None:
    from operon.agent.perception.screen_diff import dominant_color

    img = np.full((50, 50, 3), 240, dtype=np.uint8)
    img[:10] = (10, 20, 30)
    assert dominant_color(img.tobytes(), stride=1) == (240, 240, 240)


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


@pytest.mark.asyncio
async def test_browser_neutral_panel_is_not_clicked() -> None:
    frame = _uniform()
    cx, cy = DEFAULT
    frame[cy - 90 : cy + 90, cx - 90 : cx + 90] = 120
    ex, page = _browser(frame)
    await ex.context_reset()
    page.mouse.click.assert_awaited_once()
    assert page.mouse.click.await_args.args != DEFAULT


@pytest.mark.asyncio
async def test_browser_hidpi_screenshot_is_sampled_in_css_pixels() -> None:
    """A 2x screenshot of a 1280x720 viewport: the click stays in CSS px at a blank spot."""
    big = np.full((H * 2, W * 2), 200, dtype=np.uint8)
    cx, cy = DEFAULT
    big[(cy - 90) * 2 : (cy + 90) * 2, (cx - 90) * 2 : (cx + 90) * 2] = 120
    ex, page = _browser(_uniform())
    page.screenshot = AsyncMock(return_value=_png(big))
    await ex.context_reset()
    page.mouse.click.assert_awaited_once()
    x, y = page.mouse.click.await_args.args
    assert (x, y) != DEFAULT and x <= W and y <= H


@pytest.mark.asyncio
async def test_browser_non_uniform_scale_skips_the_click() -> None:
    ex, page = _browser(_uniform())
    page.screenshot = AsyncMock(return_value=_png(np.full((H, W * 2), 200, dtype=np.uint8)))
    await ex.context_reset()
    page.mouse.click.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "expected"), [("ask", None), ("risky", "https://good.test/start"), ("auto", "https://good.test/start")])
async def test_permission_mode_gates_the_start_url(monkeypatch: pytest.MonkeyPatch, mode: str, expected) -> None:
    monkeypatch.setenv("OPERON_TRUST_ALLOW_DOMAINS", "good.test")
    loop = _loop()
    loop.executor = SimpleNamespace(session_reset=AsyncMock(), last_reset_actions=["x"])
    loop._artifacts = MagicMock()
    state = _state("https://good.test/start")
    state.permission_mode = mode

    await loop._apply_recovery_actions(state, _session(), 1)

    loop.executor.session_reset.assert_awaited_once_with(expected)


@pytest.mark.asyncio
async def test_desktop_run_skips_the_start_url_note(monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.core.contracts.perception import Environment as UnifiedEnvironment

    monkeypatch.setenv("OPERON_TRUST_ALLOW_DOMAINS", "good.test")
    loop = _loop()
    loop.environment = UnifiedEnvironment.DESKTOP
    loop.executor = SimpleNamespace(session_reset=AsyncMock(), last_reset_actions=["did a thing"])
    loop._artifacts = MagicMock()

    await loop._apply_recovery_actions(_state("https://evil.test/x"), _session(), 1)

    notes = loop._artifacts.persist_recovery_reset.call_args.args[3]
    assert notes == ["did a thing"]

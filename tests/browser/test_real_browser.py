"""Real Chromium regression tests; no LLM credentials or external sites required.

Run after ``playwright install chromium`` with:
    OPERON_RUN_BROWSER_TESTS=true python -m pytest tests/browser/test_real_browser.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from operon.executor.browser_native import NativeBrowserExecutor
from operon.models.policy import AgentAction

pytestmark = pytest.mark.skipif(
    os.getenv("OPERON_RUN_BROWSER_TESTS", "").lower() != "true",
    reason="Opt in with OPERON_RUN_BROWSER_TESTS=true; requires installed Chromium",
)


@pytest.mark.asyncio
async def test_popup_becomes_active_page_for_next_action(tmp_path: Path) -> None:
    """A real target=_blank click must direct the next action into the new tab."""
    (tmp_path / "index.html").write_text(
        '<html><body style="margin:0"><a href="popup.html" target="_blank" '
        'style="display:block;width:200px;height:100px">Open tab</a></body></html>',
        encoding="utf-8",
    )
    popup_path = tmp_path / "popup.html"
    popup_path.write_text(
        '<html><body style="margin:0"><input id="entry" '
        'style="width:300px;height:100px"></body></html>',
        encoding="utf-8",
    )
    executor = NativeBrowserExecutor(
        artifact_dir=tmp_path / "artifacts",
        headless=True,
        record_video=False,
    )
    run_id = "popup-regression"
    executor.configure_run(run_id, headless=True, mode="batch")
    executor.set_current_run_id(run_id)
    try:
        result = await executor.execute(
            AgentAction(action_type="navigate", url=(tmp_path / "index.html").as_uri())
        )
        assert result.success, result.detail
        original = executor._sessions[run_id].page
        async with original.expect_popup(timeout=5000) as popup_event:
            result = await executor.execute(AgentAction(action_type="click", x=50, y=50))
        assert result.success, result.detail
        popup = await popup_event.value
        await popup.wait_for_load_state("domcontentloaded")
        assert await executor.current_url_for_run(run_id) == popup_path.as_uri()
        result = await executor.execute(
            AgentAction(action_type="type", x=50, y=50, text="new tab input")
        )
        assert result.success, result.detail
        # DOM access is only a test oracle; Operon's actions use coordinates.
        assert await popup.input_value("#entry") == "new tab input"
    finally:
        await executor.aclose_run(run_id)
        await executor.aclose()


@pytest.mark.asyncio
async def test_static_page_streams_initial_and_last_changed_frame(tmp_path: Path) -> None:
    """Throttling must not discard the only frame of a static page or final update."""
    import asyncio
    import io
    import socket

    from PIL import Image
    from playwright.async_api import async_playwright

    from operon.api import ws_stream
    from operon.browser.manager import BrowserManager

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True, args=[f"--remote-debugging-port={port}"]
        )
        page = await browser.new_page(viewport={"width": 800, "height": 600})
        await page.set_content('<html><body style="background:rgb(25,70,130)"></body></html>')
        await page.screenshot()  # Wait for the static scene to render before attaching.
        manager = BrowserManager()
        frames = asyncio.Queue()
        ws_stream._subscribers.append(frames)

        async def wait_for_color(expected: tuple[int, int, int]) -> None:
            async with asyncio.timeout(5):
                while True:
                    frame = await frames.get()
                    image = Image.open(io.BytesIO(frame)).convert("RGB")
                    actual = image.getpixel((400, 300))
                    if all(abs(a - b) < 15 for a, b in zip(actual, expected)):
                        return

        try:
            await manager.connect(port)
            await manager.start_screencast(fps=15)
            await wait_for_color((25, 70, 130))
            await page.evaluate("document.body.style.background = 'rgb(40,150,70)'")
            await wait_for_color((40, 150, 70))
            await manager.stop_screencast()
            await manager.disconnect()
            assert browser.is_connected(), "Disconnect must preserve the external browser"
        finally:
            await manager.disconnect()
            ws_stream._subscribers.remove(frames)
            await browser.close()

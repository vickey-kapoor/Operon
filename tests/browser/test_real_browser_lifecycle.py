"""Opt-in real Chromium lifecycle coverage using local pages and WebSockets.

Run with OPERON_RUN_BROWSER_TESTS=true after playwright install chromium.
"""

from __future__ import annotations

import asyncio
import io
import os
import socket
from pathlib import Path

import pytest
from PIL import Image
from playwright.async_api import async_playwright

from operon.api import ws_stream
from operon.browser.manager import BrowserManager
from operon.executor.browser_native import NativeBrowserExecutor
from operon.models.policy import AgentAction

pytestmark = pytest.mark.skipif(
    os.getenv("OPERON_RUN_BROWSER_TESTS", "").lower() != "true",
    reason="Opt in with OPERON_RUN_BROWSER_TESTS=true; requires installed Chromium",
)


def drain(queue):
    while not queue.empty():
        queue.get_nowait()


async def color_frame(queue, expected):
    async with asyncio.timeout(5):
        while True:
            data = await queue.get()
            if not isinstance(data, bytes):
                continue
            image = Image.open(io.BytesIO(data)).convert("RGB")
            actual = image.getpixel((image.width // 2, image.height // 2))
            if all(abs(a - b) < 15 for a, b in zip(actual, expected)):
                return image


@pytest.fixture
async def connected_browser():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True, args=[f"--remote-debugging-port={port}"]
        )
        page = await browser.new_page(viewport={"width": 800, "height": 600})
        await page.set_content(
            '<body style="margin:0;background:rgb(25,70,130)"></body>'
        )
        await page.screenshot()
        manager = BrowserManager()
        queue = asyncio.Queue()
        ws_stream._subscribers.append(queue)
        try:
            await manager.connect(port)
            yield browser, page, manager, queue, port
        finally:
            await manager.disconnect()
            ws_stream._subscribers.remove(queue)
            await browser.close()


async def test_popup_chain_dialog_and_next_run(tmp_path: Path):
    (tmp_path / "index.html").write_text(
        '<body style="margin:0"><a href="child.html" target="_blank" style="display:block;width:200px;height:100px">child</a>'
    )
    (tmp_path / "child.html").write_text(
        '<body style="margin:0"><a href="grandchild.html" target="_blank" style="display:block;width:200px;height:100px">grandchild</a>'
    )
    (tmp_path / "grandchild.html").write_text(
        '<body style="margin:0"><input id="entry" style="width:300px;height:100px"><button style="display:block;width:300px;height:100px" onclick="alert(\'grandchild dialog\')">dialog</button>'
    )
    executor = NativeBrowserExecutor(
        artifact_dir=tmp_path / "artifacts", headless=True, record_video=False
    )
    run_id = "chain"
    executor.configure_run(run_id, headless=True, mode="batch")
    executor.set_current_run_id(run_id)
    try:
        result = await executor.execute(
            AgentAction(action_type="navigate", url=(tmp_path / "index.html").as_uri())
        )
        assert result.success, result.detail
        for target in ["child.html", "grandchild.html"]:
            page = executor._sessions[run_id].page
            async with page.expect_popup(timeout=5000) as event:
                result = await executor.execute(
                    AgentAction(action_type="click", x=50, y=50)
                )
            assert result.success, result.detail
            popup = await event.value
            await popup.wait_for_load_state()
            assert (
                await executor.current_url_for_run(run_id)
                == (tmp_path / target).as_uri()
            )
        result = await executor.execute(
            AgentAction(action_type="type", x=50, y=50, text="grandchild input")
        )
        assert result.success, result.detail
        assert await popup.input_value("#entry") == "grandchild input"
        result = await executor.execute(AgentAction(action_type="click", x=50, y=150))
        assert result.success, result.detail
        assert executor.pop_last_dialog_message(run_id) == "grandchild dialog"
        old_browser = executor._sessions[run_id].browser
        old_pages = list(executor._sessions[run_id].context.pages)
        assert len(old_pages) == 3
        executor.configure_run("next", headless=True, mode="batch")
        executor.set_current_run_id("next")
        result = await executor.execute(
            AgentAction(action_type="navigate", url=(tmp_path / "index.html").as_uri())
        )
        assert result.success, result.detail
        assert not old_browser.is_connected()
        assert all(p.is_closed() for p in old_pages)
        assert run_id not in executor._sessions
        assert len(executor._sessions["next"].context.pages) == 1
    finally:
        for key in list(executor._sessions):
            await executor.aclose_run(key)
        await executor.aclose()


async def test_stream_rate_stop_restart_reconnect(connected_browser):
    browser, page, manager, queue, port = connected_browser
    await manager.start_screencast(fps=5)
    await color_frame(queue, (25, 70, 130))
    # At 5fps, frequent rendering must be coalesced and still publish the final color.
    drain(queue)
    await page.evaluate("""() => { let n=0; window.animation=setInterval(() => {
      document.body.style.background = `rgb(${n++ % 255},70,130)`;
    }, 10); }""")
    times = []
    start = asyncio.get_running_loop().time()
    while asyncio.get_running_loop().time() - start < 1.2:
        await asyncio.wait_for(queue.get(), 2)
        times.append(asyncio.get_running_loop().time())
    assert 3 <= len(times) <= 8, times
    assert all(b - a > 0.15 for a, b in zip(times, times[1:])), times
    await page.evaluate(
        "clearInterval(window.animation); document.body.style.background='rgb(40,150,70)'"
    )
    await color_frame(queue, (40, 150, 70))
    # Queue a delayed frame, stop before its throttle deadline, and check cancellation.
    await page.evaluate("document.body.style.background='rgb(180,50,60)'")
    async with asyncio.timeout(3):
        while manager._pending_frame is None:
            await asyncio.sleep(0.005)
    await manager.stop_screencast()
    drain(queue)
    await asyncio.sleep(0.3)
    assert queue.empty(), "A stopped stream published a pending frame"
    assert manager._cdp is None
    assert manager.latest_frame is None
    for _ in range(3):
        await manager.start_screencast(fps=5)
        await manager.start_screencast(fps=5)
        await color_frame(queue, (180, 50, 60))
        await manager.stop_screencast()
        await manager.stop_screencast()
        drain(queue)
    await manager.disconnect()
    assert browser.is_connected()
    await manager.connect(port)
    await manager.start_screencast(fps=5)
    await color_frame(queue, (180, 50, 60))


async def test_task_tab_switch_stream_and_input(connected_browser):
    browser, original, manager, queue, _port = connected_browser
    await manager.start_screencast(fps=5)
    await color_frame(queue, (25, 70, 130))
    _, task = await manager.new_task_context()
    await task.set_content(
        '<body style="margin:0;background:rgb(40,150,70)"><input id="entry" style="position:absolute;left:70%;top:70%;width:20%;height:20%"></body>'
    )
    await color_frame(queue, (40, 150, 70))
    await manager.inject_input(0.8, 0.8, "type", text="task input")
    assert await task.input_value("#entry") == "task input"
    _, next_task = await manager.new_task_context()
    assert task.is_closed()
    assert not original.is_closed(), "User tab must survive task replacement"
    await next_task.set_content('<body style="background:rgb(180,50,60)"></body>')
    await color_frame(queue, (180, 50, 60))
    await manager.disconnect()
    assert browser.is_connected()
    assert not original.is_closed()


async def test_browser_websocket_clients_receive_static_frame_and_reconnect(
    connected_browser, monkeypatch, caplog
):
    import uvicorn
    from fastapi import FastAPI

    browser, page, manager, queue, _port = connected_browser
    monkeypatch.setenv("API_KEYS", "browser-regression-key")
    app = FastAPI()
    app.include_router(ws_stream.router)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    ws_port = sock.getsockname()[1]
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    prior_manager = ws_stream._browser_manager
    ws_stream.set_browser_manager(manager)
    clients = await browser.new_context()
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.02)
        await manager.start_screencast(fps=15)
        await color_frame(queue, (25, 70, 130))
        await asyncio.sleep(
            0.2
        )  # Let the static scene settle before either client connects.
        for _ in range(2):
            client = await clients.new_page()
            await client.evaluate(
                """url => new Promise((resolve, reject) => {
                window.framesReceived = 0;
                window.ws = new WebSocket(url, ['x-api-key', 'browser-regression-key']);
                ws.binaryType = 'arraybuffer';
                ws.onmessage = async event => {
                    if (event.data instanceof ArrayBuffer) {
                        const img = await createImageBitmap(new Blob([event.data], {type:'image/jpeg'}));
                        const canvas = document.createElement('canvas');
                        canvas.width=img.width; canvas.height=img.height;
                        const ctx=canvas.getContext('2d'); ctx.drawImage(img,0,0);
                        window.pixel=Array.from(ctx.getImageData(img.width/2,img.height/2,1,1).data);
                        window.framesReceived++;
                    }
                };
                ws.onopen = resolve;
                ws.onerror = reject;
            })""",
                f"ws://127.0.0.1:{ws_port}/ws/stream",
            )
            await client.wait_for_function("window.framesReceived > 0", timeout=3000)
            pixel = await client.evaluate("window.pixel")
            assert all(abs(a - b) < 15 for a, b in zip(pixel[:3], (25, 70, 130)))
            await client.evaluate(
                """() => new Promise(resolve => { ws.onclose=resolve; ws.close(); })"""
            )
            await client.close()
        async with asyncio.timeout(3):
            while len(ws_stream._subscribers) != 1:
                await asyncio.sleep(0.02)
    finally:
        await clients.close()
        server.should_exit = True
        await asyncio.wait_for(server_task, 10)
        sock.close()
        ws_stream.set_browser_manager(prior_manager)
    assert not [
        record
        for record in caplog.records
        if record.name == "uvicorn.error" and record.levelno >= 40
    ]

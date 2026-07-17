"""CDP browser startup helpers."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

from operon.api import ws_stream as _ws_stream

logger = logging.getLogger(__name__)


async def ensure_cdp_ready(mode: str = "batch") -> None:
    """Ensure a CDP browser is connected before the first browser step runs."""
    if mode != "observable":
        # Batch tasks launch their own isolated Playwright Chromium and close it on
        # completion. Routing them through a shared CDP browser prevents cleanup.
        return

    from operon.browser.manager import (
        BrowserManager,
        get_active_manager,
        set_active_manager,
    )

    bm = get_active_manager()
    if bm is not None and bm.is_connected:
        return

    port_reachable = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get("http://localhost:9222/json/version")
        port_reachable = response.status_code == 200
    except Exception:
        pass

    if not port_reachable:
        chrome_candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        if sys.platform == "darwin":
            chrome_candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
        elif sys.platform.startswith("linux"):
            chrome_candidates = ["google-chrome", "chromium-browser", "chromium"]

        chrome_exe = next((path for path in chrome_candidates if chrome_exists(path)), None)
        if chrome_exe is None:
            logger.warning("_ensure_cdp_ready: no Chrome found; cannot auto-launch")
            return

        profile_dir = tempfile.mkdtemp(prefix="operon-cdp-")
        args = [
            chrome_exe,
            "--remote-debugging-port=9222",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ]
        try:
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("_ensure_cdp_ready: Chrome launched, waiting for port 9222")
        except Exception as exc:
            logger.warning("_ensure_cdp_ready: Chrome launch failed; %s", exc)
            return

        deadline = asyncio.get_event_loop().time() + 12
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    response = await client.get("http://localhost:9222/json/version")
                if response.status_code == 200:
                    break
            except Exception:
                pass
        else:
            logger.warning("_ensure_cdp_ready: port 9222 not reachable after 12 s")
            return

    existing = get_active_manager()
    if existing is not None:
        try:
            await existing.disconnect()
        except Exception:
            pass

    manager = BrowserManager()
    try:
        await manager.connect(9222)
        await manager.start_screencast(fps=15)
    except Exception as exc:
        logger.warning("_ensure_cdp_ready: connect failed; %s", exc)
        return

    _ws_stream.set_browser_manager(manager)
    set_active_manager(manager)
    logger.info("_ensure_cdp_ready: CDP browser connected and screencast live")


def chrome_exists(path: str) -> bool:
    import shutil

    return Path(path).exists() or shutil.which(path) is not None


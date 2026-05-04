"""BrowserManager: CDP-attached Chrome session with live screencast and input injection.

Connects to an existing Chrome instance (user's real browser) via Playwright's
connect_over_cdp. Streams JPEG frames using the CDP Page.startScreencast protocol
and publishes them to all connected WebSocket clients via ws_stream.publish_frame().

Input injection translates normalised frontend coordinates (0.0–1.0) into real
browser mouse/keyboard events on the actual page viewport.

The accessibility/bounds snapshot queries the live DOM every N frames and publishes
bounding boxes for all interactive elements so the React frontend can render overlays.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency at module load time.
# ws_stream is imported inside methods that use it.
_ws_stream_module: Any = None


def _ws_stream():
    global _ws_stream_module
    if _ws_stream_module is None:
        from src.api import ws_stream as _m
        _ws_stream_module = _m
    return _ws_stream_module


# CDP selector for all interactive DOM elements worth overlaying.
_INTERACTIVE_SELECTOR = (
    'a, button, input, select, textarea, '
    '[role="button"], [role="link"], [role="checkbox"], [role="radio"], [role="menuitem"], '
    '[tabindex]:not([tabindex="-1"])'
)

_BOUNDS_JS = f"""
() => {{
    const sel = {repr(_INTERACTIVE_SELECTOR)};
    return Array.from(document.querySelectorAll(sel))
        .map(el => {{
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) return null;
            return {{
                tag:  el.tagName.toLowerCase(),
                role: el.getAttribute('role') || el.tagName.toLowerCase(),
                text: (el.innerText || el.value || el.placeholder
                       || el.getAttribute('aria-label') || '').slice(0, 60).trim(),
                x: Math.round(r.left),
                y: Math.round(r.top),
                w: Math.round(r.width),
                h: Math.round(r.height),
            }};
        }})
        .filter(Boolean)
        .slice(0, 200);
}}
"""


class BrowserManager:
    """Manages a CDP-attached Chrome session with live screencast streaming."""

    def __init__(self) -> None:
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._cdp: Any = None                   # CDPSession for Page.startScreencast
        self._screencast_running: bool = False
        self._frame_count: int = 0
        self._viewport_w: int = 1920
        self._viewport_h: int = 1080

    # ── Connection ──────────────────────────────────────────────────────────

    async def connect(self, port: int) -> None:
        """Attach to an existing Chrome instance via CDP on the given port."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("playwright is not installed") from exc

        if self._playwright is not None:
            await self.disconnect()

        self._playwright = await async_playwright().start()
        endpoint = f"http://localhost:{port}"

        logger.info("BrowserManager: connecting to %s", endpoint)
        self._browser = await self._playwright.chromium.connect_over_cdp(endpoint)

        # Use the first page that already exists in the user's browser.
        contexts = self._browser.contexts
        if contexts:
            pages = contexts[0].pages
            self._page = pages[0] if pages else await contexts[0].new_page()
        else:
            ctx = await self._browser.new_context()
            self._page = await ctx.new_page()

        # Discover actual rendered viewport dimensions.
        size = self._page.viewport_size
        if size:
            self._viewport_w = size["width"]
            self._viewport_h = size["height"]
        else:
            # Fallback: query via JS (works when Playwright can't report the size).
            try:
                dims = await self._page.evaluate(
                    "() => ({ w: window.innerWidth, h: window.innerHeight })"
                )
                self._viewport_w = dims.get("w", 1920)
                self._viewport_h = dims.get("h", 1080)
            except Exception:
                pass

        logger.info(
            "BrowserManager: attached — %s  viewport=%dx%d",
            self._page.url,
            self._viewport_w,
            self._viewport_h,
        )

    async def disconnect(self) -> None:
        """Stop the screencast and release the Playwright handle.

        Does NOT close the user's Chrome — we only clean up our own resources.
        """
        await self.stop_screencast()
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        self._browser = None
        self._page = None
        self._cdp = None
        logger.info("BrowserManager: disconnected")

    # ── Screencast ──────────────────────────────────────────────────────────

    async def start_screencast(self, fps: int = 15) -> None:
        """Begin streaming JPEG frames via CDP Page.startScreencast.

        Chrome captures internally at ~60fps; everyNthFrame throttles delivery
        to the requested fps. Each frame is base64-decoded and published to all
        connected WebSocket clients as binary data.
        """
        if self._page is None:
            raise RuntimeError("Not connected — call connect() first")
        if self._screencast_running:
            return

        self._cdp = await self._page.context.new_cdp_session(self._page)
        self._screencast_running = True
        self._frame_count = 0

        # Chrome exposes ~60fps internally; skip every N frames to hit target fps.
        every_nth = max(1, round(60 / fps))

        async def _on_frame(params: dict) -> None:
            if not self._screencast_running:
                return

            # ACK immediately — Chrome stops sending if we don't ACK promptly.
            session_id = params.get("sessionId", 0)
            try:
                await self._cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
            except Exception:
                pass

            raw_b64 = params.get("data", "")
            if not raw_b64:
                return

            jpeg_bytes = base64.b64decode(raw_b64)
            _ws_stream().publish_frame(jpeg_bytes)

            self._frame_count += 1
            # Every 30 frames (~2s at 15fps): push element bounding boxes.
            if self._frame_count % 30 == 1:
                asyncio.create_task(self._publish_element_bounds())

        self._cdp.on("Page.screencastFrame", _on_frame)

        await self._cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": 75,
            "maxWidth": self._viewport_w,
            "maxHeight": self._viewport_h,
            "everyNthFrame": every_nth,
        })

        logger.info("BrowserManager: screencast started at ~%dfps (everyNth=%d)", fps, every_nth)

    async def stop_screencast(self) -> None:
        if not self._screencast_running:
            return
        self._screencast_running = False
        if self._cdp is not None:
            try:
                await self._cdp.send("Page.stopScreencast")
            except Exception:
                pass
        logger.info("BrowserManager: screencast stopped after %d frames", self._frame_count)

    # ── Input injection ─────────────────────────────────────────────────────

    async def inject_input(
        self,
        x_ratio: float,
        y_ratio: float,
        input_type: str,
        **kwargs: Any,
    ) -> None:
        """Translate normalised frontend coordinates into real browser input events.

        Args:
            x_ratio: Horizontal position as a fraction of viewport width (0.0–1.0).
            y_ratio: Vertical position as a fraction of viewport height (0.0–1.0).
            input_type: "click" | "dblclick" | "type" | "scroll" | "key" | "hover"
            **kwargs:
                text (str)     — for input_type=="type"
                key (str)      — for input_type=="key" (Playwright key name)
                button (str)   — for input_type=="click" ("left"|"right"|"middle")
                delta_y (int)  — for input_type=="scroll" (positive = down)
        """
        if self._page is None:
            logger.warning("inject_input called but no page is connected")
            return

        page_x = x_ratio * self._viewport_w
        page_y = y_ratio * self._viewport_h

        try:
            if input_type == "click":
                button = kwargs.get("button", "left")
                await self._page.mouse.click(page_x, page_y, button=button)

            elif input_type == "dblclick":
                await self._page.mouse.dblclick(page_x, page_y)

            elif input_type == "hover":
                await self._page.mouse.move(page_x, page_y)

            elif input_type == "type":
                text = kwargs.get("text", "")
                # Click to focus, then type. Mirrors the atomic behaviour in
                # NativeBrowserExecutor: click → 50ms → keyboard dispatch.
                await self._page.mouse.click(page_x, page_y)
                await asyncio.sleep(0.05)
                await self._page.keyboard.type(text)

            elif input_type == "scroll":
                delta_y = int(kwargs.get("delta_y", 300))
                await self._page.mouse.move(page_x, page_y)
                await self._page.mouse.wheel(0, delta_y)

            elif input_type == "key":
                key = kwargs.get("key", "Enter")
                await self._page.keyboard.press(key)

            else:
                logger.warning("inject_input: unknown type %r", input_type)
                return

            logger.debug(
                "inject_input: %s at (%.3f, %.3f) → page (%.1f, %.1f)",
                input_type, x_ratio, y_ratio, page_x, page_y,
            )
        except Exception as exc:
            logger.warning("inject_input failed: %s", exc)

    # ── Accessibility / element bounds ──────────────────────────────────────

    async def snapshot_accessibility_tree(self) -> dict:
        """Return Playwright's accessibility tree snapshot for the current page."""
        if self._page is None:
            return {}
        try:
            return await self._page.accessibility.snapshot() or {}
        except Exception as exc:
            logger.warning("accessibility snapshot failed: %s", exc)
            return {}

    async def _publish_element_bounds(self) -> None:
        """Query the live DOM for interactive element bounding boxes and broadcast."""
        if self._page is None:
            return
        try:
            elements = await self._page.evaluate(_BOUNDS_JS)
            _ws_stream().publish_event({
                "type": "element_bounds",
                "viewport_w": self._viewport_w,
                "viewport_h": self._viewport_h,
                "elements": elements,
            })
        except Exception as exc:
            logger.debug("element bounds query failed: %s", exc)

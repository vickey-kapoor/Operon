"""Playwright browser executor for UI navigation actions."""

import asyncio
import base64
import io
import logging
from typing import Optional

from PIL import Image
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from .actions import Action, ActionResult, ActionType

logger = logging.getLogger(__name__)


class PlaywrightBrowserExecutor:
    """Manages a Playwright Chromium browser and executes UI actions."""

    def __init__(
        self,
        headless: bool = True,
        width: int = 1280,
        height: int = 800,
    ) -> None:
        self.headless = headless
        self.width = width
        self.height = height

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._started = False

    async def start(self) -> None:
        """Launch the browser and create a new page."""
        if self._started:
            logger.warning("Browser already started, skipping re-initialisation")
            return

        logger.info("Starting Playwright Chromium browser (headless=%s)", self.headless)
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--no-first-run",
                "--no-zygote",
                # Anti-bot-detection: hide automation signals
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--flag-switches-begin",
                "--disable-site-isolation-trials",
                "--flag-switches-end",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": self.width, "height": self.height},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            permissions=["geolocation"],
            java_script_enabled=True,
            bypass_csp=False,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        # Mask navigator.webdriver and other automation fingerprints.
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            window.chrome = { runtime: {} };
        """)

        self._page = await self._context.new_page()

        # Navigate to a blank page to ensure the page is usable immediately.
        await self._page.goto("about:blank")
        self._started = True
        logger.info("Browser started successfully")

    async def stop(self) -> None:
        """Close the browser and clean up all resources."""
        if not self._started:
            return

        logger.info("Stopping browser")
        # Each cleanup step is independent -- one failure should not block the rest.
        try:
            if self._page and not self._page.is_closed():
                await self._page.close()
        except Exception as exc:
            logger.debug("Error closing page: %s", exc)

        try:
            if self._context:
                await self._context.close()
        except Exception as exc:
            logger.debug("Error closing context: %s", exc)

        try:
            if self._browser:
                await self._browser.close()
        except Exception as exc:
            logger.debug("Error closing browser: %s", exc)

        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception as exc:
            logger.debug("Error stopping playwright: %s", exc)

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._started = False
        logger.info("Browser stopped")

    def _ensure_started(self) -> Page:
        """Return the active page, raising if the browser has not been started."""
        if not self._started or self._page is None:
            raise RuntimeError(
                "Browser has not been started. Call await executor.start() first."
            )
        return self._page

    async def _screenshot_raw(self) -> bytes:
        """Capture a raw PNG screenshot of the current viewport."""
        page = self._ensure_started()
        try:
            return await page.screenshot(
                type="png", full_page=False, timeout=60000
            )
        except PlaywrightTimeoutError:
            # Wait for page to settle and retry once
            await page.wait_for_load_state("networkidle", timeout=10000)
            return await page.screenshot(
                type="png", full_page=False, timeout=60000
            )

    async def screenshot(self) -> Image.Image:
        """Capture a full-page screenshot and return it as a PIL Image."""
        raw = await self._screenshot_raw()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        logger.debug("Screenshot captured (%dx%d)", img.width, img.height)
        return img

    async def screenshot_base64(self) -> str:
        """Capture a screenshot and return it as a base64-encoded PNG string."""
        raw = await self._screenshot_raw()
        return base64.b64encode(raw).decode("utf-8")

    # ------------------------------------------------------------------
    # Individual action helpers
    # ------------------------------------------------------------------

    async def _click(self, x: int, y: int) -> None:
        page = self._ensure_started()
        logger.debug("Click at (%d, %d)", x, y)
        await page.mouse.click(x, y)
        # Brief wait for any triggered navigation or animation.
        await asyncio.sleep(0.3)

    async def _type(self, text: str) -> None:
        page = self._ensure_started()
        logger.debug("Typing %d characters", len(text))
        await page.keyboard.type(text, delay=30)

    async def _key(self, key: str) -> None:
        page = self._ensure_started()
        logger.debug("Key press: %s", key)
        await page.keyboard.press(key)
        await asyncio.sleep(0.2)

    async def _scroll(
        self,
        x: int,
        y: int,
        direction: str,
        amount: int,
    ) -> None:
        page = self._ensure_started()
        pixels = amount * 100  # 100px per scroll unit
        delta_x = 0
        delta_y = 0

        if direction == "down":
            delta_y = pixels
        elif direction == "up":
            delta_y = -pixels
        elif direction == "right":
            delta_x = pixels
        elif direction == "left":
            delta_x = -pixels
        else:
            raise ValueError(f"Unknown scroll direction: {direction!r}")

        logger.debug(
            "Scroll at (%d, %d) direction=%s amount=%d", x, y, direction, amount
        )
        # Move mouse to the scroll target first so the scroll hits the right element.
        await page.mouse.move(x, y)
        await page.mouse.wheel(delta_x, delta_y)
        await asyncio.sleep(0.2)

    async def navigate(self, url: str) -> None:
        """Navigate to a URL. Only http(s) and about: schemes are allowed."""
        page = self._ensure_started()
        if not url.startswith(("http://", "https://", "about:")):
            url = "https://" + url
        logger.info("Navigating to %s", url)
        try:
            await page.goto(url, wait_until="load", timeout=60_000)
        except Exception:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            except Exception as exc:
                logger.warning("Navigation fallback failed: %s", exc)
        # Wait briefly for JS-rendered content to paint.
        await asyncio.sleep(1.5)
        # Extra wait if the body appears empty (SPA still rendering).
        try:
            await page.wait_for_function(
                "document.body && document.body.innerText.trim().length > 50",
                timeout=5_000,
            )
        except Exception:
            pass  # best-effort — take the screenshot regardless

    async def _wait(self, duration_ms: int) -> None:
        logger.debug("Waiting %d ms", duration_ms)
        await asyncio.sleep(duration_ms / 1000.0)

    # ------------------------------------------------------------------
    # Main execute method
    # ------------------------------------------------------------------

    async def execute(self, action: Action) -> ActionResult:
        """Execute a single action and return the result."""
        action_type = ActionType(action.type)

        try:
            if action_type == ActionType.CLICK:
                if not action.coordinate or len(action.coordinate) < 2:
                    return ActionResult(
                        success=False,
                        error="CLICK action requires 'coordinate' [x, y]",
                        action_type=action.type,
                    )
                await self._click(action.coordinate[0], action.coordinate[1])
                screenshot_b64 = await self.screenshot_base64()
                return ActionResult(
                    success=True,
                    screenshot=screenshot_b64,
                    action_type=action.type,
                )

            elif action_type == ActionType.TYPE:
                if action.text is None:
                    return ActionResult(
                        success=False,
                        error="TYPE action requires 'text'",
                        action_type=action.type,
                    )
                await self._type(action.text)
                return ActionResult(success=True, action_type=action.type)

            elif action_type == ActionType.KEY:
                if not action.key:
                    return ActionResult(
                        success=False,
                        error="KEY action requires 'key'",
                        action_type=action.type,
                    )
                await self._key(action.key)
                screenshot_b64 = await self.screenshot_base64()
                return ActionResult(
                    success=True,
                    screenshot=screenshot_b64,
                    action_type=action.type,
                )

            elif action_type == ActionType.SCROLL:
                coord = action.coordinate or [self.width // 2, self.height // 2]
                direction = action.scroll_direction or "down"
                amount = action.scroll_amount or 3
                await self._scroll(coord[0], coord[1], direction, amount)
                return ActionResult(success=True, action_type=action.type)

            elif action_type == ActionType.NAVIGATE:
                if not action.url:
                    return ActionResult(
                        success=False,
                        error="NAVIGATE action requires 'url'",
                        action_type=action.type,
                    )
                await self.navigate(action.url)
                screenshot_b64 = await self.screenshot_base64()
                return ActionResult(
                    success=True,
                    screenshot=screenshot_b64,
                    action_type=action.type,
                )

            elif action_type == ActionType.WAIT:
                duration = action.duration_ms or 1000
                await self._wait(duration)
                return ActionResult(success=True, action_type=action.type)

            elif action_type == ActionType.SCREENSHOT:
                screenshot_b64 = await self.screenshot_base64()
                return ActionResult(
                    success=True,
                    screenshot=screenshot_b64,
                    action_type=action.type,
                )

            elif action_type == ActionType.DONE:
                logger.info("DONE action received — task signalled as complete")
                return ActionResult(success=True, action_type=action.type)

            else:
                return ActionResult(
                    success=False,
                    error=f"Unknown action type: {action.type!r}",
                    action_type=action.type,
                )

        except Exception as exc:
            logger.exception("Error executing action %s: %s", action.type, exc)
            return ActionResult(
                success=False,
                error=str(exc),
                action_type=action.type,
            )

    async def current_url(self) -> str:
        """Return the current page URL."""
        page = self._ensure_started()
        return page.url

    async def page_title(self) -> str:
        """Return the current page title."""
        page = self._ensure_started()
        return await page.title()

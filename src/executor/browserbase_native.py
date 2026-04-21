"""Browserbase-backed browser executor using Playwright CDP connection."""

from __future__ import annotations

import logging
import os

from src.executor.browser_native import NativeBrowserExecutor, _BrowserSession

logger = logging.getLogger(__name__)


class BrowserbaseNativeBrowserExecutor(NativeBrowserExecutor):
    """Browser executor that connects to a Browserbase remote session via CDP.

    Drop-in replacement for NativeBrowserExecutor. The only behavioural
    difference is in session creation: instead of launching a local Chromium
    process we create a Browserbase session and connect via CDP. All action
    execution, screenshot capture, and cleanup logic is inherited unchanged.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        **kwargs,
    ) -> None:
        # Force record_video=False — Browserbase handles session replay natively.
        kwargs.setdefault("record_video", False)
        super().__init__(**kwargs)
        self._bb_api_key = api_key or os.getenv("BROWSERBASE_API_KEY", "")
        self._bb_project_id = project_id or os.getenv("BROWSERBASE_PROJECT_ID", "")
        # Maps run_id → Browserbase session id for cleanup.
        self._bb_session_ids: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _ensure_session(self, run_id: str, *, foreground: bool = True) -> _BrowserSession:
        existing = self._sessions.get(run_id)
        if existing is not None:
            return existing

        await self._close_other_sessions(run_id)

        try:
            from browserbase import Browserbase
        except ImportError as exc:
            raise RuntimeError(
                "browserbase is not installed. Run: pip install browserbase"
            ) from exc

        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("playwright is not installed for browser execution.") from exc

        if not self._bb_api_key:
            raise RuntimeError("BROWSERBASE_API_KEY env var is required for browserbase backend.")
        if not self._bb_project_id:
            raise RuntimeError("BROWSERBASE_PROJECT_ID env var is required for browserbase backend.")

        bb = Browserbase(api_key=self._bb_api_key)
        bb_session = bb.sessions.create(project_id=self._bb_project_id)
        self._bb_session_ids[run_id] = bb_session.id
        logger.info("Created Browserbase session %s for run %s", bb_session.id, run_id)

        cdp_url = (
            f"wss://connect.browserbase.com"
            f"?apiKey={self._bb_api_key}&sessionId={bb_session.id}"
        )

        playwright = await async_playwright().start()
        try:
            browser = await playwright.chromium.connect_over_cdp(cdp_url)
        except Exception:
            await playwright.stop()
            raise

        # CDP connections reuse the pre-provisioned context/page.
        context = browser.contexts[0] if browser.contexts else await browser.new_context(
            viewport={"width": self._viewport_width, "height": self._viewport_height}
        )
        page = context.pages[0] if context.pages else await context.new_page()

        session = _BrowserSession(
            playwright=playwright,
            browser=browser,
            context=context,
            page=page,
            video_dir=None,
            browser_pid=None,
        )
        self._sessions[run_id] = session
        self._fresh_session_run_id = run_id
        return session

    async def _close_session(self, session: _BrowserSession) -> None:
        await super()._close_session(session)
        # Stop the Browserbase session by releasing whichever run_id owns it.
        for run_id, s in list(self._sessions.items()):
            if s is session:
                self._stop_bb_session(run_id)
                break
        # Also check already-popped sessions (run_id already removed from _sessions).
        for run_id, bb_sid in list(self._bb_session_ids.items()):
            if run_id not in self._sessions:
                self._stop_bb_session(run_id)

    def _stop_bb_session(self, run_id: str) -> None:
        bb_sid = self._bb_session_ids.pop(run_id, None)
        if bb_sid is None:
            return
        try:
            from browserbase import Browserbase

            bb = Browserbase(api_key=self._bb_api_key)
            bb.sessions.update(bb_sid, status="REQUEST_RELEASE")
            logger.info("Released Browserbase session %s (run %s)", bb_sid, run_id)
        except Exception as exc:
            logger.warning("Failed to release Browserbase session %s: %s", bb_sid, exc)

    async def aclose_run(self, run_id: str) -> int:
        result = await super().aclose_run(run_id)
        self._stop_bb_session(run_id)
        return result

    def cleanup_run(self, run_id: str) -> int:
        result = super().cleanup_run(run_id)
        self._stop_bb_session(run_id)
        return result

"""Browserbase-backed browser executor using Playwright CDP connection."""

from __future__ import annotations

import logging
import os

from src.executor.browser_native import NativeBrowserExecutor, _BrowserSession
from src.models.capture import CaptureFrame
from src.models.common import FailureCategory
from src.models.execution import ExecutedAction
from src.models.policy import AgentAction

logger = logging.getLogger(__name__)


def _is_session_closed(exc: Exception) -> bool:
    """Return True if the exception signals a closed/expired Browserbase session."""
    name = type(exc).__name__
    msg = str(exc)
    return "TargetClosedError" in name or "Target page" in msg or "Target closed" in msg


class BrowserbaseNativeBrowserExecutor(NativeBrowserExecutor):
    """Browser executor that connects to a Browserbase remote session via CDP.

    Drop-in replacement for NativeBrowserExecutor. Session creation connects
    to a Browserbase cloud browser via CDP instead of launching local Chromium.
    If the remote session expires mid-run, capture() and execute() evict the
    dead session and reconnect transparently on the next call.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        **kwargs,
    ) -> None:
        # Browserbase records sessions natively; skip local video recording.
        kwargs.setdefault("record_video", False)
        super().__init__(**kwargs)
        self._bb_api_key = api_key or os.getenv("BROWSERBASE_API_KEY", "")
        self._bb_project_id = project_id or os.getenv("BROWSERBASE_PROJECT_ID", "")
        # Maps run_id → Browserbase session id for cleanup.
        self._bb_session_ids: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Resilience wrapper
    # ------------------------------------------------------------------

    def _evict_dead_session(self) -> None:
        """Remove the current run's session so the next call reconnects."""
        run_id = self._current_run_id
        if run_id is None:
            return
        dead = self._sessions.pop(run_id, None)
        self._stop_bb_session(run_id)
        if dead is not None:
            logger.warning(
                "Browserbase session expired for run %s — evicted, will reconnect on next step",
                run_id,
            )
            # Best-effort teardown of the dead playwright handle.
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(dead.playwright.stop())
                else:
                    loop.run_until_complete(dead.playwright.stop())
            except Exception:
                pass

    async def capture(self) -> CaptureFrame:
        try:
            return await super().capture()
        except Exception as exc:
            if not _is_session_closed(exc):
                raise
            self._evict_dead_session()
            # Return a blank frame — the loop will retry on the next step with a fresh session.
            from uuid import uuid4

            blank_path = self._artifact_dir / f"browser_{uuid4().hex[:8]}.png"
            self._write_blank_png(blank_path)
            return CaptureFrame(
                artifact_path=str(blank_path),
                width=self._viewport_width,
                height=self._viewport_height,
                mime_type="image/png",
            )

    async def execute(self, action: AgentAction) -> ExecutedAction:
        try:
            return await super().execute(action)
        except Exception as exc:
            if not _is_session_closed(exc):
                raise
            self._evict_dead_session()
            return self._fail(
                action,
                "Browserbase session expired — reconnecting on next step",
                FailureCategory.EXECUTION_ERROR,
            )

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

        live_urls = bb.sessions.debug(bb_session.id)
        cdp_url = live_urls.ws_url

        playwright = await async_playwright().start()
        try:
            browser = await playwright.chromium.connect_over_cdp(cdp_url)
        except Exception:
            await playwright.stop()
            raise

        # CDP connections reuse the pre-provisioned context/page.
        context = (
            browser.contexts[0]
            if browser.contexts
            else await browser.new_context(
                viewport={"width": self._viewport_width, "height": self._viewport_height}
            )
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
        for run_id, s in list(self._sessions.items()):
            if s is session:
                self._stop_bb_session(run_id)
                break
        for run_id in list(self._bb_session_ids):
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

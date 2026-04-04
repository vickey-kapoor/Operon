"""Native browser executor using Playwright for browser-only automation."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from src.executor.browser import Executor
from src.models.capture import CaptureFrame
from src.models.common import FailureCategory, LoopStage
from src.models.execution import ExecutedAction, ExecutionAttemptTrace, ExecutionTrace
from src.models.policy import ActionType, AgentAction


@dataclass(slots=True)
class _BrowserSession:
    playwright: object
    browser: object
    context: object
    page: object
    video_dir: Path | None = None


class NativeBrowserExecutor(Executor):
    """Browser executor backed by a controlled Playwright Chromium session."""

    def __init__(
        self,
        *,
        artifact_dir: str | Path = ".browser-artifacts",
        viewport_width: int | None = None,
        viewport_height: int | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        post_action_delay: float = 0.5,
    ) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._viewport_width = viewport_width or int(os.getenv("BROWSER_WIDTH", "1440"))
        self._viewport_height = viewport_height or int(os.getenv("BROWSER_HEIGHT", "900"))
        if headless is None:
            headless = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
        self._headless = headless
        if record_video is None:
            record_video = True
        self._record_video = record_video
        self._post_action_delay = post_action_delay
        self._current_run_id: str | None = None
        self._sessions: dict[str, _BrowserSession] = {}

    def set_current_run_id(self, run_id: str) -> None:
        self._current_run_id = run_id

    async def reset_desktop(self) -> None:
        """No-op for native browser control."""
        return None

    async def aclose_run(self, run_id: str) -> int:
        session = self._sessions.pop(run_id, None)
        if session is None:
            return 0
        await self._close_session(session)
        return 1

    def cleanup_run(self, run_id: str) -> int:
        session = self._sessions.pop(run_id, None)
        if session is None:
            return 0
        try:
            asyncio.run(self._close_session(session))
        except RuntimeError:
            # If already inside an event loop, close best-effort in the background.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return 0
            loop.create_task(self._close_session(session))
        return 1

    async def capture(self) -> CaptureFrame:
        page = await self._current_page()
        filename = f"browser_{uuid4().hex[:8]}.png"
        filepath = self._artifact_dir / filename
        await page.screenshot(path=str(filepath))
        return CaptureFrame(
            artifact_path=str(filepath),
            width=self._viewport_width,
            height=self._viewport_height,
            mime_type="image/png",
        )

    async def execute(self, action: AgentAction) -> ExecutedAction:
        page = await self._current_page()
        if action.action_type is ActionType.BATCH:
            return await self._execute_batch(action)
        at = action.action_type
        try:
            if at is ActionType.NAVIGATE:
                if action.url is None:
                    return self._fail(action, "navigate requires url", FailureCategory.EXECUTION_ERROR)
                await page.goto(action.url)
            elif at is ActionType.LAUNCH_APP:
                if not action.text or action.text.lower() != "browser":
                    return self._fail(action, "native browser executor only supports launch_app for browser", FailureCategory.EXECUTION_ERROR)
                after_path = await self._capture_after()
                return self._ok(action, "Browser session already active", after_path)
            elif at is ActionType.CLICK:
                if action.x is None or action.y is None:
                    return self._fail(action, "click requires x,y coordinates", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
                await page.mouse.click(action.x, action.y)
            elif at is ActionType.HOVER:
                if action.x is None or action.y is None:
                    return self._fail(action, "hover requires x,y coordinates", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
                await page.mouse.move(action.x, action.y)
            elif at is ActionType.TYPE:
                if action.text is None:
                    return self._fail(action, "type requires text", FailureCategory.EXECUTION_ERROR)
                if action.x is not None and action.y is not None:
                    await page.mouse.click(action.x, action.y)
                if action.clear_before_typing:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Backspace")
                await page.keyboard.type(action.text)
                if action.press_enter:
                    await page.keyboard.press("Enter")
            elif at is ActionType.PRESS_KEY:
                if action.key is None:
                    return self._fail(action, "press_key requires key", FailureCategory.EXECUTION_ERROR)
                await page.keyboard.press(action.key)
            elif at is ActionType.HOTKEY:
                if action.key is None:
                    return self._fail(action, "hotkey requires key", FailureCategory.EXECUTION_ERROR)
                normalized = action.key.lower().replace(" ", "")
                if normalized == "alt+left":
                    await page.go_back()
                elif normalized == "alt+right":
                    await page.go_forward()
                else:
                    await page.keyboard.press("+".join(k.strip() for k in action.key.split("+")))
            elif at is ActionType.SCROLL:
                amount = action.scroll_amount or 800
                await page.mouse.wheel(0, -amount)
            elif at is ActionType.DRAG:
                if None in (action.x, action.y, action.x_end, action.y_end):
                    return self._fail(action, "drag requires x,y,x_end,y_end", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
                await page.mouse.move(action.x, action.y)
                await page.mouse.down()
                await page.mouse.move(action.x_end, action.y_end)
                await page.mouse.up()
            elif at is ActionType.WAIT:
                await asyncio.sleep((action.wait_ms or 1000) / 1000)
            elif at in {ActionType.STOP, ActionType.WAIT_FOR_USER}:
                return self._ok(action, "Acknowledged stop" if at is ActionType.STOP else f"Waiting for user: {action.text}", None)
            else:
                return self._fail(action, f"Action '{at}' is not supported on native browser executor", FailureCategory.EXECUTION_ERROR)
            await page.wait_for_load_state(timeout=5000)
            await asyncio.sleep(self._post_action_delay)
            after_path = await self._capture_after()
            return self._ok(action, f"Executed {at.value}", after_path)
        except Exception as exc:
            return self._fail(action, f"{at.value} failed: {exc}", FailureCategory.EXECUTION_ERROR)

    async def _execute_batch(self, action: AgentAction) -> ExecutedAction:
        if not action.actions:
            return self._fail(action, "batch requires actions", FailureCategory.EXECUTION_ERROR)
        completed: list[str] = []
        last_artifact_path: str | None = None
        for child_action in action.actions:
            result = await self.execute(child_action)
            completed.append(f"{child_action.action_type.value}:{'ok' if result.success else 'failed'}")
            if not result.success:
                return ExecutedAction(
                    action=action,
                    success=False,
                    detail=f"Batch failed on {child_action.action_type.value}: {result.detail}",
                    artifact_path=result.artifact_path,
                    execution_trace=result.execution_trace,
                    failure_category=result.failure_category,
                    failure_stage=result.failure_stage,
                )
            last_artifact_path = result.artifact_path
        return self._ok(action, f"Executed batch: {', '.join(completed)}", last_artifact_path)

    async def get_current_url(self) -> str | None:
        page = await self._current_page()
        return page.url

    async def execute_with_recording(
        self, action: AgentAction, step_dir: Path,
    ) -> tuple[ExecutedAction, Path | None]:
        result = await self.execute(action)
        return result, None

    async def _capture_after(self) -> str:
        frame = await self.capture()
        return frame.artifact_path

    async def _current_page(self):
        if self._current_run_id is None:
            raise RuntimeError("No active browser run id set.")
        session = await self._ensure_session(self._current_run_id)
        return session.page

    async def _ensure_session(self, run_id: str) -> _BrowserSession:
        existing = self._sessions.get(run_id)
        if existing is not None:
            return existing
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("playwright is not installed for native browser execution.") from exc
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=self._headless)
        video_dir = self._video_dir_for_run(run_id) if self._record_video else None
        context_kwargs = {
            "viewport": {"width": self._viewport_width, "height": self._viewport_height},
        }
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
            context_kwargs["record_video_dir"] = str(video_dir)
            context_kwargs["record_video_size"] = {
                "width": self._viewport_width,
                "height": self._viewport_height,
            }
        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        session = _BrowserSession(
            playwright=playwright,
            browser=browser,
            context=context,
            page=page,
            video_dir=video_dir,
        )
        self._sessions[run_id] = session
        return session

    async def _close_session(self, session: _BrowserSession) -> None:
        await session.context.close()
        await session.browser.close()
        await session.playwright.stop()
        self._finalize_recorded_video(session)

    def _video_dir_for_run(self, run_id: str) -> Path:
        return self._artifact_dir / run_id / "session_video"

    def recorded_video_path_for_run(self, run_id: str) -> Path | None:
        video_dir = self._video_dir_for_run(run_id)
        finalized_webm = video_dir / "session.webm"
        if finalized_webm.exists():
            return finalized_webm
        finalized_mp4 = video_dir / "session.mp4"
        if finalized_mp4.exists():
            return finalized_mp4
        candidates = sorted(video_dir.glob("*.webm")) + sorted(video_dir.glob("*.mp4"))
        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _finalize_recorded_video(session: _BrowserSession) -> None:
        if session.video_dir is None or not session.video_dir.exists():
            return
        video_files = sorted(session.video_dir.glob("*.webm")) + sorted(session.video_dir.glob("*.mp4"))
        if len(video_files) != 1:
            return
        source = video_files[0]
        final_path = session.video_dir / f"session{source.suffix}"
        if final_path.exists():
            return
        source.rename(final_path)

    @staticmethod
    def _ok(action: AgentAction, detail: str, after_path: str | None) -> ExecutedAction:
        trace = ExecutionTrace(
            attempts=[
                ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="browser_action_no_revalidation",
                    verification_result="success",
                )
            ],
            final_outcome="success",
        )
        return ExecutedAction(
            action=action,
            success=True,
            detail=detail,
            artifact_path=after_path,
            execution_trace=trace,
        )

    @staticmethod
    def _fail(action: AgentAction, detail: str, failure_category: FailureCategory) -> ExecutedAction:
        trace = ExecutionTrace(
            attempts=[
                ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="browser_action_no_revalidation",
                    verification_result="failure",
                    failure_category=failure_category,
                )
            ],
            final_outcome="failure",
            final_failure_category=failure_category,
        )
        return ExecutedAction(
            action=action,
            success=False,
            detail=detail,
            execution_trace=trace,
            failure_category=failure_category,
            failure_stage=LoopStage.EXECUTE,
        )

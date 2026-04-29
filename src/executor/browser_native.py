"""Native browser executor using Playwright for browser-only automation."""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from src.executor.browser import Executor
from src.executor.os_picker_macro import PickerOutcome, run_os_picker_macro
from src.models.capture import CaptureFrame
from src.models.common import FailureCategory, LoopStage
from src.models.execution import ExecutedAction, ExecutionAttemptTrace, ExecutionTrace
from src.models.policy import ActionType, AgentAction

logger = logging.getLogger(__name__)


# Playwright expects PascalCase key names. Normalise common variants the LLM may emit.
_PLAYWRIGHT_KEY_MAP: dict[str, str] = {
    "escape": "Escape",
    "esc": "Escape",
    "enter": "Enter",
    "return": "Enter",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "tab": "Tab",
    "space": "Space",
    "arrowup": "ArrowUp",
    "up": "ArrowUp",
    "arrowdown": "ArrowDown",
    "down": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "left": "ArrowLeft",
    "arrowright": "ArrowRight",
    "right": "ArrowRight",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
}


def _normalize_key_playwright(key: str) -> str:
    return _PLAYWRIGHT_KEY_MAP.get(key.lower(), key)


@dataclass(slots=True)
class _BrowserSession:
    playwright: object
    browser: object
    context: object
    page: object
    video_dir: Path | None = None
    browser_pid: int | None = None


class NativeBrowserExecutor(Executor):
    """Browser executor backed by a controlled Playwright Chromium session."""

    def __init__(
        self,
        *,
        artifact_dir: str | Path = ".browser-artifacts",
# ✅ Healed 2026-04-26T22:45:36Z | Added --disable-blink-features=AutomationControlled to prevent detection and add
        viewport_width: int | None = None,
        viewport_height: int | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        post_action_delay: float = 0.5,
    ) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._viewport_width = viewport_width or int(os.getenv("BROWSER_WIDTH", "1920"))
        self._viewport_height = viewport_height or int(os.getenv("BROWSER_HEIGHT", "1080"))
        if headless is None:
            headless = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
        self._headless = headless
        if record_video is None:
            record_video = True
        self._record_video = record_video
        self._post_action_delay = post_action_delay
        self._current_run_id: str | None = None
        self._sessions: dict[str, _BrowserSession] = {}
        self._run_headless: dict[str, bool | None] = {}
        self._fresh_session_run_id: str | None = None
        # Stores the last JavaScript dialog message per run_id so the loop can
        # surface form-submit success alerts that headless mode auto-dismisses.
        self._last_dialog_message: dict[str, str] = {}

    def set_current_run_id(self, run_id: str) -> None:
        self._current_run_id = run_id

    def configure_run(self, run_id: str, *, headless: bool | None = None) -> None:
        self._run_headless[run_id] = headless

    async def reset_desktop(self) -> None:
        """No-op for native browser control."""
        return None

    async def aclose_run(self, run_id: str) -> int:
        session = self._sessions.pop(run_id, None)
        self._run_headless.pop(run_id, None)
        if session is None:
            return 0
        await self._close_session(session)
        return 1

    def cleanup_run(self, run_id: str) -> int:
        session = self._sessions.pop(run_id, None)
        self._run_headless.pop(run_id, None)
        if session is None:
            return 0
        try:
            asyncio.run(self._close_session(session))
        except RuntimeError:
            # Already inside a running event loop — schedule and keep a reference
            # so the task is not garbage-collected before completion.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return 0
            task = loop.create_task(self._close_session(session))
            # Prevent silent discard: log any unexpected error from the background close.
            task.add_done_callback(
                lambda t: logger.warning("cleanup_run background close error: %s", t.exception())
                if not t.cancelled() and t.exception() is not None
                else None
            )
        return 1

    async def capture(self) -> CaptureFrame:
        filename = f"browser_{uuid4().hex[:8]}.png"
        filepath = self._artifact_dir / filename
        if self._current_run_id is None or self._current_run_id not in self._sessions:
            self._write_blank_png(filepath)
        else:
            page = await self._current_page(foreground=False)
            await page.screenshot(path=str(filepath))
        return CaptureFrame(
            artifact_path=str(filepath),
            width=self._viewport_width,
            height=self._viewport_height,
            mime_type="image/png",
        )

    async def live_frame_png(self, run_id: str) -> bytes | None:
        session = self._sessions.get(run_id)
        if session is None:
            return None
        return await session.page.screenshot(type="png")

    async def execute(self, action: AgentAction) -> ExecutedAction:
        if action.action_type is ActionType.BATCH:
            return await self._execute_batch(action)
        at = action.action_type
        try:
            if at is ActionType.NAVIGATE:
                if action.url is None:
                    return self._fail(action, "navigate requires url", FailureCategory.EXECUTION_ERROR)
                page = await self._current_page(foreground=False)
                await page.goto(action.url)
                await self._foreground_if_fresh_session()
            elif at is ActionType.LAUNCH_APP:
                if not action.text or action.text.lower() != "browser":
                    return self._fail(action, "native browser executor only supports launch_app for browser", FailureCategory.EXECUTION_ERROR)
                existing = self._current_run_id is not None and self._current_run_id in self._sessions
                if not existing:
                    await self._current_page(foreground=False)
                detail = "Opened browser session" if not existing else "Browser session already active"
                return self._ok(action, detail, None)
            elif at is ActionType.CLICK:
                page = await self._current_page()
                if action.selector:
                    await page.locator(action.selector).first.click(timeout=5000)
                else:
                    point = self._action_point(action)
                    if point is None:
                        return self._fail(action, "click requires selector or x,y coordinates", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
                    await page.mouse.click(*point)
                await self._adopt_new_tab_if_opened()
            elif at is ActionType.HOVER:
                page = await self._current_page()
                point = self._action_point(action)
                if point is None:
                    return self._fail(action, "hover requires x,y coordinates", FailureCategory.EXECUTION_TARGET_NOT_FOUND)
                await page.mouse.move(*point)
            elif at is ActionType.TYPE:
                page = await self._current_page()
                if action.text is None:
                    return self._fail(action, "type requires text", FailureCategory.EXECUTION_ERROR)
                point = self._action_point(action)
                _typed = False
                if point is not None:
                    # Use elementFromPoint to target the exact element at the given
                    # coordinates, then fill via ElementHandle.fill() which is more
                    # reliable than click→keyboard.type() (avoids focus-stealing issues
                    # where clicking coordinates A triggers focus on element B via JS).
                    try:
                        handle = await page.evaluate_handle(
                            f"document.elementFromPoint({point[0]}, {point[1]})"
                        )
                        el = handle.as_element()
                        if el is not None:
                            tag = await page.evaluate("el => el.tagName.toLowerCase()", handle)
                            # For email-address fills: always find the VISIBLE email input
                            # by name/id (not type=email which often finds hidden WP fields),
                            # scroll it into view, and type via keyboard (trusted events).
                            # Also handles the textarea-mismatch case.
                            if "@" in action.text:
                                # elementFromPoint returned a textarea but we're filling
                                # an email value — the form DOM layout differs from what
                                # Gemini perceives. Use Playwright's ElementHandle to find
                                # the actual email input, scroll it into view, click to
                                # focus it, then type via keyboard so all native events
                                # fire correctly (fixing form validation recognition).
                                # After typing, clear any textarea that got contaminated.
                                try:
                                    import json as _json
                                    # Target visible email input: prefer exact name/id match
                                    # (input[type=email] often matches hidden contact-form
                                    # inputs on WordPress pages; input[name=email] or #email
                                    # finds the visible form field).
                                    email_handle = await page.evaluate_handle(
                                        "Array.from(document.querySelectorAll('input[name*=\"email\" i], input[id*=\"email\" i]'))"
                                        ".find(el => {"
                                        "  const r = el.getBoundingClientRect();"
                                        "  return r.width > 0 && r.height > 0;"
                                        "}) || null"
                                    )
                                    email_el = email_handle.as_element()
                                    if email_el is not None:
                                        await email_el.scroll_into_view_if_needed()
                                        await email_el.click()
                                        await page.keyboard.press("Control+A")
                                        await page.keyboard.press("Backspace")
                                        await page.keyboard.type(action.text)
                                        # Clear any textarea that got contaminated by JS
                                        await page.evaluate(
                                            f"""document.querySelectorAll('textarea').forEach(t => {{
                                                if (t.value === {_json.dumps(action.text)}) t.value = '';
                                            }});"""
                                        )
                                        _typed = True
                                        logger.info(
                                            "Email redirect: found VISIBLE input[name*=email] "
                                            "at (%s), typed via keyboard",
                                            await page.evaluate("el => `${el.id||el.name}@(${Math.round(el.getBoundingClientRect().y)}px)`", email_el)
                                        )
                                except Exception as _email_err:
                                    logger.debug("Email redirect via ElementHandle failed (%s)", _email_err)
                            if not _typed and tag in ("input", "textarea"):
                                # ElementHandle.fill() selects-all, fills, dispatches events.
                                await el.fill(action.text)
                                _typed = True
                                logger.debug(
                                    "elementFromPoint fill: tag=%s coords=(%d,%d) text=%r",
                                    tag, point[0], point[1], action.text[:30],
                                )
                    except Exception as _efp_err:
                        logger.debug("elementFromPoint fill failed (%s) — falling back", _efp_err)
                    if not _typed:
                        await page.mouse.click(*point)
                        await asyncio.sleep(0.05)
                if not _typed:
                    if action.clear_before_typing:
                        await page.keyboard.press("Control+A")
                        await page.keyboard.press("Backspace")
                    await page.keyboard.type(action.text)
                if action.press_enter:
                    await page.keyboard.press("Enter")
            elif at is ActionType.PRESS_KEY:
                page = await self._current_page()
                if action.key is None:
                    return self._fail(action, "press_key requires key", FailureCategory.EXECUTION_ERROR)
                key = _normalize_key_playwright(action.key)
                point = self._action_point(action)
                if point is not None:
                    await page.mouse.click(*point)
                await page.keyboard.press(key)
            elif at is ActionType.HOTKEY:
                page = await self._current_page()
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
                page = await self._current_page()
                amount = action.scroll_amount or 800
                await page.mouse.wheel(0, -amount)
            elif at is ActionType.DRAG:
                page = await self._current_page()
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
            elif at is ActionType.UPLOAD_FILE:
                file_path = action.text
                if not file_path:
                    return self._fail(action, "upload_file requires text (file path)", FailureCategory.EXECUTION_ERROR)
                page = await self._current_page()
                point = self._action_point(action)
                async with page.expect_file_chooser(timeout=8000) as fc_info:
                    if point is not None:
                        await page.mouse.click(*point)
                    else:
                        await page.locator("input[type=file]").first.click(force=True)
                file_chooser = await fc_info.value
                await file_chooser.set_files(file_path)
            elif at is ActionType.READ_TEXT:
                page = await self._current_page()
                css = action.selector or "main, article, #content, body"
                extracted: str = await page.evaluate(
                    """(sel) => {
                        const el = document.querySelector(sel);
                        if (!el) return "";
                        const paras = Array.from(el.querySelectorAll("p"))
                            .map(p => p.innerText.trim())
                            .filter(t => t.length > 20);
                        return paras.length > 0 ? paras.join("\\n\\n") : el.innerText.trim();
                    }""",
                    css,
                )
                if not extracted:
                    return self._fail(action, f"No text found at selector {css!r}", FailureCategory.EXECUTION_ERROR)
                output_path = action.text
                if output_path:
                    import os as _os
                    parent = _os.path.dirname(_os.path.abspath(output_path))
                    if parent:
                        _os.makedirs(parent, exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as fh:
                        fh.write(extracted)
                after_path = await self._capture_after()
                return self._ok(action, extracted[:500], after_path)
            elif at is ActionType.UPLOAD_FILE_NATIVE:
                file_path = action.text
                if not file_path:
                    return self._fail(
                        action,
                        "upload_file_native requires text (absolute file path)",
                        FailureCategory.EXECUTION_ERROR,
                    )

                run_id = self._current_run_id or ""
                run_headless = self._run_headless.get(run_id, self._headless)
                if run_headless:
                    return self._fail(
                        action,
                        "upload_file_native requires headed browser mode (OS file picker unavailable in headless)",
                        FailureCategory.EXECUTION_ERROR,
                    )

                # Click the upload control to trigger the native OS file picker.
                page = await self._current_page()
                point = self._action_point(action)
                if point is not None:
                    await page.mouse.click(*point)
                elif action.selector is not None:
                    await page.locator(action.selector).first.click()
                elif action.target_element_id is not None:
                    await page.locator(self._target_element_locator(action.target_element_id)).first.click()
                else:
                    return self._fail(
                        action,
                        "upload_file_native requires coordinates, CSS selector, target context, or target_element_id",
                        FailureCategory.EXECUTION_TARGET_NOT_FOUND,
                    )

                # Wait for OS picker to appear, then run deterministic macro.
                await asyncio.sleep(1.0)
                macro_result = await asyncio.to_thread(run_os_picker_macro, file_path)

                after_path = await self._capture_after()
                if macro_result.outcome is PickerOutcome.SUCCESS:
                    return self._ok(action, macro_result.detail, after_path)

                category_map = {
                    PickerOutcome.PICKER_NOT_DETECTED: FailureCategory.PICKER_NOT_DETECTED,
                    PickerOutcome.FILE_NOT_REFLECTED: FailureCategory.FILE_NOT_REFLECTED,
                    PickerOutcome.UNAVAILABLE: FailureCategory.EXECUTION_ERROR,
                }
                failed = self._fail(
                    action, macro_result.detail, category_map[macro_result.outcome]
                )
                return ExecutedAction(
                    action=failed.action,
                    success=False,
                    detail=failed.detail,
                    artifact_path=after_path,
                    execution_trace=failed.execution_trace,
                    failure_category=failed.failure_category,
                    failure_stage=failed.failure_stage,
                )
            else:
                return self._fail(action, f"Action '{at}' is not supported on native browser executor", FailureCategory.EXECUTION_ERROR)
            # Only wait for page load after actions that can trigger navigation.
            if at in {ActionType.NAVIGATE, ActionType.CLICK, ActionType.PRESS_KEY, ActionType.HOTKEY}:
                await page.wait_for_load_state(timeout=5000)
            await asyncio.sleep(self._action_delay(at))
            after_path = await self._capture_after()
            return self._ok(action, f"Executed {at.value}", after_path)
        except Exception as exc:
            return self._fail(action, f"{at.value} failed: {exc}", FailureCategory.EXECUTION_ERROR)

    def _action_delay(self, action_type: ActionType) -> float:
        """Return the post-action settle delay appropriate for each action type."""
        delays = {
            ActionType.NAVIGATE: 1.0,
            ActionType.CLICK: 0.3,
            ActionType.DOUBLE_CLICK: 0.3,
            ActionType.TYPE: 0.2,
            ActionType.UPLOAD_FILE: 0.5,
            ActionType.UPLOAD_FILE_NATIVE: 0.5,
            ActionType.PRESS_KEY: 0.15,
            ActionType.HOTKEY: 0.15,
            ActionType.SCROLL: 0.1,
            ActionType.DRAG: 0.2,
            ActionType.HOVER: 0.05,
            ActionType.SELECT: 0.2,
        }
        return delays.get(action_type, self._post_action_delay)

    async def _adopt_new_tab_if_opened(self) -> None:
        """Switch session.page to a newly opened tab after a click, if any."""
        if self._current_run_id is None:
            return
        session = self._sessions.get(self._current_run_id)
        if session is None:
            return
        pages = session.context.pages
        if len(pages) <= 1:
            return
        new_page = pages[-1]
        try:
            await new_page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        session.page = new_page
        logger.info("New tab detected after click — switched session to %s", new_page.url)

    @staticmethod
    def _action_point(action: AgentAction) -> tuple[int, int] | None:
        if action.x is not None and action.y is not None:
            return action.x, action.y
        target_context = action.target_context
        if target_context is None:
            return None
        original_target = getattr(target_context, "original_target", None)
        if original_target is None:
            return None
        x = getattr(original_target, "x", None)
        y = getattr(original_target, "y", None)
        width = getattr(original_target, "width", None)
        height = getattr(original_target, "height", None)
        if not all(isinstance(value, int) for value in (x, y, width, height)):
            return None
        return x + max(1, width // 2), y + max(1, height // 2)

    @staticmethod
    def _target_element_locator(target_element_id: str) -> str:
        escaped = target_element_id.replace("\\", "\\\\").replace('"', '\\"')
        return (
            f'[id="{escaped}"], '
            f'[data-element-id="{escaped}"], '
            f'[data-testid="{escaped}"], '
            f'[name="{escaped}"]'
        )

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

    def pop_last_dialog_message(self, run_id: str) -> str | None:
        """Return and clear the last JS dialog message for a run (or None)."""
        return self._last_dialog_message.pop(run_id, None)

    async def get_current_url(self) -> str | None:
        page = await self._current_page()
        return page.url

    async def current_url_for_run(self, run_id: str) -> str | None:
        session = self._sessions.get(run_id)
        if session is None:
            return None
        return session.page.url

    async def execute_with_recording(
        self, action: AgentAction, step_dir: Path,
    ) -> tuple[ExecutedAction, Path | None]:
        result = await self.execute(action)
        return result, None

    async def context_reset(self) -> None:
        """Dismiss open modals/dropdowns, clear focus traps, scroll to top."""
        try:
            page = await self._current_page(foreground=False)
        except Exception:
            return
        try:
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.2)
            await page.mouse.click(self._viewport_width // 2, self._viewport_height // 3)
            await asyncio.sleep(0.1)
            await page.keyboard.press("Home")
        except Exception:
            logger.debug("context_reset: browser reset failed", exc_info=True)

    async def session_reset(self, start_url: str | None = None) -> None:
        """Navigate to start URL (or current URL) to reset page context."""
        try:
            page = await self._current_page(foreground=False)
        except Exception:
            return
        target_url = start_url or page.url
        if not target_url:
            await self.context_reset()
            return
        try:
            await page.goto(target_url, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(0.5)
        except Exception:
            logger.debug("session_reset: goto %r failed, falling back to context_reset", target_url, exc_info=True)
            await self.context_reset()

    async def _capture_after(self) -> str:
        frame = await self.capture()
        return frame.artifact_path

    async def _current_page(self, *, foreground: bool = True):
        if self._current_run_id is None:
            raise RuntimeError("No active browser run id set.")
        session = await self._ensure_session(self._current_run_id, foreground=foreground)
        return session.page

    async def _ensure_session(self, run_id: str, *, foreground: bool = True) -> _BrowserSession:
        existing = self._sessions.get(run_id)
        if existing is not None:
            return existing
        await self._close_other_sessions(run_id)
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("playwright is not installed for native browser execution.") from exc
        playwright = await async_playwright().start()
        try:
            chromium_executable = getattr(playwright.chromium, "executable_path", None)
            before_pids = self._chrome_process_ids(chromium_executable) if chromium_executable else set()
            launch_headless = self._run_headless.get(run_id)
            if launch_headless is None:
                launch_headless = self._headless
            if os.getenv("OPERON_TEST_SAFE_MODE", "false").lower() == "true":
                launch_headless = True
            # Homeostasis baseline: lock to 1920x1080 in all modes.
            # --start-maximized caused clipping inconsistency → perception_low_quality.
            launch_args = [
                f"--window-size={self._viewport_width},{self._viewport_height}",
                "--window-position=0,0",
            ]
            browser = await playwright.chromium.launch(
                headless=launch_headless,
                args=launch_args,
            )
        except Exception:
            await playwright.stop()
            raise
        video_dir = self._video_dir_for_run(run_id) if self._record_video else None
        context_kwargs = {
            "viewport": {"width": self._viewport_width, "height": self._viewport_height},
            "device_scale_factor": 1,
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

        # Diagnostic listeners — surface browser console output and network
        # failures to the Python log so blank-screen causes are visible.
        page.on("console", lambda msg: logger.debug("BROWSER_LOG [%s]: %s", msg.type, msg.text))
        page.on("requestfailed", lambda req: logger.warning(
            "NETWORK_FAIL: %s — %s", req.url, req.failure
        ))

        # Auto-accept JavaScript dialogs (alert/confirm/prompt) and record the
        # message so the loop can treat it as a success signal. Without this,
        # headless Playwright silently dismisses alerts and the form-submit
        # success message is never seen.
        async def _on_dialog(dialog) -> None:
            msg = dialog.message or ""
            logger.info("BROWSER_DIALOG [%s]: %s", dialog.type, msg)
            self._last_dialog_message[run_id] = msg
            await dialog.accept()

        page.on("dialog", _on_dialog)

        if foreground and hasattr(page, "bring_to_front"):
            await page.bring_to_front()
        if not launch_headless and foreground:
            await self._reset_browser_zoom(page)
        session = _BrowserSession(
            playwright=playwright,
            browser=browser,
            context=context,
            page=page,
            video_dir=video_dir,
            browser_pid=self._detect_browser_pid(chromium_executable, before_pids) if chromium_executable else None,
        )
        self._sessions[run_id] = session
        self._fresh_session_run_id = run_id
        if not launch_headless and foreground:
            await self._bring_browser_to_foreground(session.browser_pid)
            await asyncio.sleep(1.5)
        return session

    async def focus_window(self) -> None:
        """Bring the active browser window to the foreground. No-op in headless mode."""
        if self._current_run_id is None:
            return
        run_headless = self._run_headless.get(self._current_run_id, self._headless)
        if run_headless:
            return
        session = self._sessions.get(self._current_run_id)
        if session is None:
            return
        await self._bring_browser_to_foreground(session.browser_pid)

    async def _foreground_if_fresh_session(self) -> None:
        if self._current_run_id is None:
            return
        if self._fresh_session_run_id != self._current_run_id:
            return
        run_headless = self._run_headless.get(self._current_run_id, self._headless)
        session = self._sessions.get(self._current_run_id)
        if session is None or run_headless:
            return
        await self._bring_browser_to_foreground(session.browser_pid)
        await asyncio.sleep(0.5)

    def _write_blank_png(self, path: Path) -> None:
        frame = np.full((self._viewport_height, self._viewport_width, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(path), frame)

    async def _bring_browser_to_foreground(self, browser_pid: int | None = None) -> None:
        for _ in range(8):
            if self._focus_browser_window(browser_pid):
                return
            await asyncio.sleep(0.2)
        self._app_activate_browser_window(browser_pid)

    def _headed_launch_size(self) -> tuple[int, int]:
        if os.name != "nt":
            return self._viewport_width, self._viewport_height
        try:
            user32 = ctypes.windll.user32
            screen_width = int(user32.GetSystemMetrics(0))
            screen_height = int(user32.GetSystemMetrics(1))
        except Exception:
            return self._viewport_width, self._viewport_height
        if screen_width <= 0 or screen_height <= 0:
            return self._viewport_width, self._viewport_height
        return screen_width, screen_height

    @staticmethod
    async def _reset_browser_zoom(page: object) -> None:
        keyboard = getattr(page, "keyboard", None)
        if keyboard is None or not hasattr(keyboard, "press"):
            return
        try:
            await keyboard.press("Control+0")
        except Exception:
            return

    def _focus_browser_window(self, browser_pid: int | None = None) -> bool:
        if os.name != "nt":
            return False
        handles = self._find_browser_window_handles(browser_pid)
        for hwnd in reversed(handles):
            if self._activate_window_handle(hwnd):
                return True
        return False

    @staticmethod
    def _find_browser_window_handles(browser_pid: int | None = None) -> list[int]:
        user32 = ctypes.windll.user32
        handles: list[int] = []

        enum_windows_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def callback(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            process_id = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if browser_pid is not None and int(process_id.value) != browser_pid:
                return True
            class_buffer = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buffer, 256)
            window_class = class_buffer.value
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                if window_class.startswith("Chrome_WidgetWin"):
                    handles.append(int(hwnd))
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value
            if "Google Chrome for Testing" in title or window_class.startswith("Chrome_WidgetWin"):
                handles.append(int(hwnd))
            return True

        user32.EnumWindows(enum_windows_proc(callback), 0)
        return handles

    @staticmethod
    def _activate_window_handle(hwnd: int) -> bool:
        if os.name != "nt":
            return False
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        SW_RESTORE = 9
        SW_SHOW = 5
        ASFW_ANY = 0xFFFFFFFF
        VK_MENU = 0x12
        KEYEVENTF_KEYUP = 0x0002
        foreground = user32.GetForegroundWindow()
        foreground_thread = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
        current_thread = kernel32.GetCurrentThreadId()
        attached = False
        if hasattr(user32, "AllowSetForegroundWindow"):
            user32.AllowSetForegroundWindow(ASFW_ANY)
        if foreground_thread and foreground_thread != current_thread:
            attached = bool(user32.AttachThreadInput(foreground_thread, current_thread, True))
        if hasattr(user32, "IsIconic") and user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        else:
            user32.ShowWindow(hwnd, SW_SHOW)
        if hasattr(user32, "keybd_event"):
            user32.keybd_event(VK_MENU, 0, 0, 0)
            user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        user32.BringWindowToTop(hwnd)
        if hasattr(user32, "SetActiveWindow"):
            user32.SetActiveWindow(hwnd)
        if hasattr(user32, "SetFocus"):
            user32.SetFocus(hwnd)
        result = bool(user32.SetForegroundWindow(hwnd))
        if attached:
            user32.AttachThreadInput(foreground_thread, current_thread, False)
        return result

    @staticmethod
    def _app_activate_browser_window(browser_pid: int | None = None) -> bool:
        if os.name != "nt":
            return False
        activation_target = str(browser_pid) if browser_pid is not None else "Google Chrome for Testing"
        command = (
            "$wshell = New-Object -ComObject WScript.Shell; "
            f"if ($wshell.AppActivate('{activation_target}')) {{ exit 0 }} "
            "else { exit 1 }"
        )
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-WindowStyle",
                    "Hidden",
                    "-Command",
                    command,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except Exception:
            return False
        return result.returncode == 0

    @staticmethod
    def _chrome_process_ids(executable_path: str) -> set[int]:
        if os.name != "nt":
            return set()
        escaped_path = executable_path.replace("'", "''")
        command = (
            "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" "
            f"| Where-Object {{ $_.ExecutablePath -eq '{escaped_path}' }} "
            "| Select-Object -ExpandProperty ProcessId "
            "| ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except Exception:
            return set()
        if result.returncode != 0 or not result.stdout.strip():
            return set()
        try:
            parsed = json.loads(result.stdout)
        except Exception:
            return set()
        if isinstance(parsed, int):
            return {parsed}
        if isinstance(parsed, list):
            return {int(value) for value in parsed if isinstance(value, int)}
        return set()

    def _detect_browser_pid(self, executable_path: str, before_pids: set[int]) -> int | None:
        after_pids = self._chrome_process_ids(executable_path)
        new_pids = sorted(pid for pid in after_pids if pid not in before_pids)
        if new_pids:
            return new_pids[-1]
        existing_pids = sorted(after_pids)
        if existing_pids:
            return existing_pids[-1]
        return None

    async def _close_other_sessions(self, current_run_id: str) -> None:
        stale_run_ids = [run_id for run_id in self._sessions if run_id != current_run_id]
        for stale_run_id in stale_run_ids:
            session = self._sessions.pop(stale_run_id, None)
            self._run_headless.pop(stale_run_id, None)
            if session is None:
                continue
            try:
                await self._close_session(session)
            except Exception:
                continue

    def _consume_fresh_session_flag(self) -> bool:
        if self._current_run_id is None:
            return False
        was_fresh = self._fresh_session_run_id == self._current_run_id
        if was_fresh:
            self._fresh_session_run_id = None
        return was_fresh

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

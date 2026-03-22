"""Browser executor interface for browser-only execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from playwright.async_api import (
    Browser,
    BrowserContext,
    ElementHandle,
    Locator,
    Page,
    Playwright,
    async_playwright,
)

from src.models.capture import CaptureFrame
from src.models.common import FailureCategory, LoopStage
from src.models.execution import (
    ExecutedAction,
    ExecutionAttemptTrace,
    ExecutionTargetSnapshot,
    ExecutionTrace,
)
from src.models.policy import ActionType, AgentAction

_SHIFT_TOLERANCE_PX = 56.0


@dataclass(frozen=True)
class BrowserDebugConfig:
    """Environment-driven Playwright launch options for local debugging."""

    headless: bool
    slow_mo_ms: int
    devtools: bool


@dataclass
class _ResolvedTarget:
    locator: Locator | None = None
    handle: ElementHandle | None = None


class BrowserExecutor(ABC):
    @abstractmethod
    async def capture(self) -> CaptureFrame:
        """Capture a browser frame for the current run."""

    @abstractmethod
    async def execute(self, action: AgentAction) -> ExecutedAction:
        """Execute a typed browser action."""


def _read_browser_debug_config() -> BrowserDebugConfig:
    """Read local Playwright debug settings from the environment."""

    return BrowserDebugConfig(
        headless=_get_env_bool("BROWSER_HEADLESS", default=True),
        slow_mo_ms=_get_env_int("BROWSER_SLOW_MO_MS", default=0),
        devtools=_get_env_bool("BROWSER_DEVTOOLS", default=False),
    )


def _get_env_bool(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _get_env_int(name: str, *, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return max(0, int(raw_value.strip()))
    except ValueError:
        return default


def _ensure_windows_process_env() -> None:
    """Restore Windows process env vars Playwright's Node driver expects."""

    if os.name != "nt":
        return

    drive = (
        os.getenv("SystemDrive")
        or Path(os.getenv("TEMP") or "").drive
        or Path.cwd().drive
        or "C:"
    )
    windows_dir = f"{drive}\\Windows"
    if os.path.isdir(windows_dir):
        os.environ.setdefault("SystemRoot", windows_dir)
        os.environ.setdefault("WINDIR", windows_dir)


def _read_user_data_dir_from_env() -> Path | None:
    raw = os.getenv("BROWSER_USER_DATA_DIR")
    if raw and raw.strip():
        return Path(raw.strip())
    return None


class PlaywrightBrowserExecutor(BrowserExecutor):
    """Playwright-backed browser executor with deterministic hardening."""

    def __init__(
        self,
        *,
        headless: bool | None = None,
        slow_mo_ms: int | None = None,
        devtools: bool | None = None,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        artifact_root: str | Path = ".browser-artifacts",
        user_data_dir: str | Path | None = None,
    ) -> None:
        debug_config = _read_browser_debug_config()
        self.headless = debug_config.headless if headless is None else headless
        self.slow_mo_ms = debug_config.slow_mo_ms if slow_mo_ms is None else max(0, slow_mo_ms)
        self.devtools = debug_config.devtools if devtools is None else devtools
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.user_data_dir: Path | None = (
            Path(user_data_dir) if user_data_dir is not None
            else _read_user_data_dir_from_env()
        )
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def start(self) -> None:
        if self._page is not None:
            return

        _ensure_windows_process_env()
        self._playwright = await async_playwright().start()

        if self.user_data_dir is not None:
            # Persistent context: cookies, localStorage, and auth state survive across runs
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            launch_kwargs: dict = {
                "headless": self.headless,
                "slow_mo": self.slow_mo_ms,
                "viewport": {"width": self.viewport_width, "height": self.viewport_height},
            }
            if self.devtools:
                launch_kwargs["args"] = ["--auto-open-devtools-for-tabs"]
            self._context = await self._playwright.chromium.launch_persistent_context(
                str(self.user_data_dir), **launch_kwargs
            )
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            # Ephemeral context: clean browser with no saved state
            launch_kwargs = {
                "headless": self.headless,
                "slow_mo": self.slow_mo_ms,
            }
            if self.devtools:
                launch_kwargs["args"] = ["--auto-open-devtools-for-tabs"]
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
            self._context = await self._browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height}
            )
            self._page = await self._context.new_page()

    async def close(self) -> None:
        if self._page is not None:
            await self._page.close()
            self._page = None
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def capture(self) -> CaptureFrame:
        page = await self._require_page()
        artifact_path = self._next_artifact_path("capture")
        await page.screenshot(path=str(artifact_path), type="png")
        return CaptureFrame(
            artifact_path=str(artifact_path),
            width=self.viewport_width,
            height=self.viewport_height,
            mime_type="image/png",
        )

    async def _require_page(self) -> Page:
        await self.start()
        if self._page is None:
            raise RuntimeError("Playwright page did not initialize.")
        return self._page

    def _next_artifact_path(self, prefix: str) -> Path:
        return self.artifact_root / f"{prefix}_{uuid4().hex}.png"

    async def execute(self, action: AgentAction) -> ExecutedAction:
        page = await self._require_page()
        attempt_trace: ExecutionAttemptTrace
        success = True
        detail = f"Executed {action.action_type.value}"
        failure_category = None
        failure_stage = None

        try:
            if action.action_type is ActionType.CLICK:
                attempt_trace, detail, failure_category = await self._execute_click(page, action)
                success = failure_category is None
            elif action.action_type is ActionType.TYPE:
                attempt_trace, detail, failure_category = await self._execute_type(page, action)
                success = failure_category is None
            elif action.action_type is ActionType.SELECT:
                attempt_trace, detail, failure_category = await self._execute_select(page, action)
                success = failure_category is None
            elif action.action_type is ActionType.PRESS_KEY:
                before_signature = await self._page_signature(page)
                await page.keyboard.press(action.key or "")
                after_signature = await self._page_signature(page)
                attempt_trace = ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="not_applicable",
                    verification_result="pressed_key",
                    no_progress_detected=before_signature == after_signature,
                )
                detail = f"Pressed key {action.key}."
            elif action.action_type is ActionType.NAVIGATE:
                await page.goto(action.url or "about:blank", wait_until="domcontentloaded")
                attempt_trace = ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="not_applicable",
                    verification_result="navigated",
                )
                detail = f"Navigated to {action.url}."
            elif action.action_type is ActionType.WAIT:
                before_signature = await self._page_signature(page)
                await asyncio.sleep((action.wait_ms or 0) / 1000)
                after_signature = await self._page_signature(page)
                attempt_trace = ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="not_applicable",
                    verification_result="waited",
                    no_progress_detected=before_signature == after_signature,
                )
                detail = f"Waited {action.wait_ms}ms."
            else:
                attempt_trace = ExecutionAttemptTrace(
                    attempt_index=1,
                    revalidation_result="not_applicable",
                    verification_result="stop_acknowledged",
                )
                detail = "Stop action acknowledged."
        except Exception as exc:
            attempt_trace = ExecutionAttemptTrace(
                attempt_index=1,
                revalidation_result="exception",
                verification_result="execution_exception",
                failure_category=self._classify_execution_failure(exc),
            )
            success = False
            detail = f"Execution failed: {exc}"
            failure_category = attempt_trace.failure_category
            failure_stage = LoopStage.EXECUTE

        artifact_path = None
        try:
            artifact_path = str(self._next_artifact_path("after"))
            await page.screenshot(path=artifact_path, type="png")
        except Exception:
            artifact_path = None

        if failure_category is None and attempt_trace.failure_category is not None:
            failure_category = attempt_trace.failure_category
            success = False
            failure_stage = LoopStage.EXECUTE
        elif failure_category is not None:
            failure_stage = LoopStage.EXECUTE

        trace = ExecutionTrace(
            attempts=[attempt_trace],
            final_outcome="success" if success else "failure",
            final_failure_category=failure_category,
        )
        return ExecutedAction(
            action=action,
            success=success,
            detail=detail,
            artifact_path=artifact_path,
            execution_trace=trace,
            failure_category=failure_category,
            failure_stage=failure_stage,
        )

    async def _execute_click(
        self,
        page: Page,
        action: AgentAction,
    ) -> tuple[ExecutionAttemptTrace, str, FailureCategory | None]:
        resolved = await self._resolve_target(page, action)
        before_snapshot, revalidation_failure = await self._revalidate_target(page, resolved, action)
        if revalidation_failure is not None:
            return (
                ExecutionAttemptTrace(
                    attempt_index=1,
                    selected_target_before_action=before_snapshot,
                    revalidation_result=revalidation_failure.value,
                    verification_result="click_not_attempted",
                    failure_category=revalidation_failure,
                ),
                f"Execution failed: {revalidation_failure.value.replace('_', ' ')}.",
                revalidation_failure,
            )

        before_signature = await self._page_signature(page)
        before_focus = await self._active_element_signature(page)
        await self._click_resolved_target(page, resolved)
        after_snapshot = await self._inspect_target(page, action)
        after_signature = await self._page_signature(page)
        after_focus = await self._active_element_signature(page)
        verification_result = "click_verified"
        failure_category = None
        no_progress = before_signature == after_signature and before_focus == after_focus

        if self._is_toggle_input(before_snapshot):
            if after_snapshot is None or after_snapshot.checked == before_snapshot.checked:
                verification_result = "checkbox_verification_failed"
                failure_category = FailureCategory.CHECKBOX_VERIFICATION_FAILED
            else:
                verification_result = "checkbox_state_changed"
        elif no_progress:
            verification_result = "click_no_effect"
            failure_category = FailureCategory.CLICK_NO_EFFECT

        return (
            ExecutionAttemptTrace(
                attempt_index=1,
                selected_target_before_action=before_snapshot,
                selected_target_after_action=after_snapshot,
                revalidation_result="ok",
                verification_result=verification_result,
                no_progress_detected=no_progress,
                failure_category=failure_category,
            ),
            "Clicked target." if failure_category is None else f"Execution failed: {verification_result.replace('_', ' ')}.",
            failure_category,
        )

    @staticmethod
    def _is_toggle_input(snapshot: ExecutionTargetSnapshot | None) -> bool:
        if snapshot is None:
            return False
        return snapshot.tag_name == "input" and snapshot.input_type in {"checkbox", "radio"}

    async def _execute_type(
        self,
        page: Page,
        action: AgentAction,
    ) -> tuple[ExecutionAttemptTrace, str, FailureCategory | None]:
        resolved = await self._resolve_target(page, action)
        before_snapshot, revalidation_failure = await self._revalidate_target(page, resolved, action)
        if revalidation_failure is not None:
            return (
                ExecutionAttemptTrace(
                    attempt_index=1,
                    selected_target_before_action=before_snapshot,
                    revalidation_result=revalidation_failure.value,
                    verification_result="type_not_attempted",
                    failure_category=revalidation_failure,
                ),
                f"Execution failed: {revalidation_failure.value.replace('_', ' ')}.",
                revalidation_failure,
            )

        focus_result, focus_failure = await self._ensure_focus(page, resolved)
        if focus_failure is not None:
            return (
                ExecutionAttemptTrace(
                    attempt_index=1,
                    selected_target_before_action=before_snapshot,
                    revalidation_result="ok",
                    focus_verification_result=focus_result,
                    verification_result="type_not_attempted",
                    failure_category=focus_failure,
                ),
                f"Execution failed: {focus_failure.value.replace('_', ' ')}.",
                focus_failure,
            )

        if not await self._resolved_target_is_editable(page, resolved):
            return (
                ExecutionAttemptTrace(
                    attempt_index=1,
                    selected_target_before_action=before_snapshot,
                    revalidation_result="ok",
                    focus_verification_result=focus_result,
                    verification_result="target_not_editable",
                    failure_category=FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
                ),
                "Execution failed: resolved type target is not editable.",
                FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
            )

        await self._fill_or_type(page, resolved, action.text or "")
        after_snapshot = await self._inspect_target(page, action)
        if self._typed_value_verified(after_snapshot, action.text or ""):
            return (
                ExecutionAttemptTrace(
                    attempt_index=1,
                    selected_target_before_action=before_snapshot,
                    selected_target_after_action=after_snapshot,
                    revalidation_result="ok",
                    focus_verification_result=focus_result,
                    verification_result="typed_value_verified",
                    no_progress_detected=before_snapshot == after_snapshot,
                ),
                "Typed text.",
                None,
            )

        retry_focus_result, retry_focus_failure = await self._ensure_focus(page, resolved)
        if retry_focus_failure is None:
            await self._fill_or_type(page, resolved, action.text or "")
            retry_snapshot = await self._inspect_target(page, action)
            if self._typed_value_verified(retry_snapshot, action.text or ""):
                return (
                    ExecutionAttemptTrace(
                        attempt_index=1,
                        selected_target_before_action=before_snapshot,
                        selected_target_after_action=retry_snapshot,
                        revalidation_result="ok",
                        focus_verification_result=f"{focus_result};retry={retry_focus_result}",
                        verification_result="typed_value_verified_after_refocus",
                    ),
                    "Typed text after re-focusing target.",
                    None,
                )

        return (
            ExecutionAttemptTrace(
                attempt_index=1,
                selected_target_before_action=before_snapshot,
                selected_target_after_action=await self._inspect_target(page, action),
                revalidation_result="ok",
                focus_verification_result=f"{focus_result};retry={retry_focus_result}",
                verification_result="type_verification_failed",
                failure_category=FailureCategory.TYPE_VERIFICATION_FAILED,
            ),
            "Execution failed: typed value was not reflected in the target.",
            FailureCategory.TYPE_VERIFICATION_FAILED,
        )

    async def _execute_select(
        self,
        page: Page,
        action: AgentAction,
    ) -> tuple[ExecutionAttemptTrace, str, FailureCategory | None]:
        resolved = await self._resolve_target(page, action)
        before_snapshot, revalidation_failure = await self._revalidate_target(page, resolved, action)
        if revalidation_failure is not None:
            return (
                ExecutionAttemptTrace(
                    attempt_index=1,
                    selected_target_before_action=before_snapshot,
                    revalidation_result=revalidation_failure.value,
                    verification_result="select_not_attempted",
                    failure_category=revalidation_failure,
                ),
                f"Execution failed: {revalidation_failure.value.replace('_', ' ')}.",
                revalidation_failure,
            )

        if resolved.locator is None:
            return (
                ExecutionAttemptTrace(
                    attempt_index=1,
                    selected_target_before_action=before_snapshot,
                    revalidation_result="ok",
                    verification_result="select_target_unresolved",
                    failure_category=FailureCategory.SELECT_VERIFICATION_FAILED,
                ),
                "Execution failed: select target could not be resolved as a locator.",
                FailureCategory.SELECT_VERIFICATION_FAILED,
            )

        await resolved.locator.select_option(label=action.text or "")
        after_snapshot = await self._inspect_target(page, action)
        if after_snapshot is not None and (after_snapshot.selected_value == action.text or after_snapshot.value == action.text):
            return (
                ExecutionAttemptTrace(
                    attempt_index=1,
                    selected_target_before_action=before_snapshot,
                    selected_target_after_action=after_snapshot,
                    revalidation_result="ok",
                    verification_result="selected_value_verified",
                ),
                "Selected option.",
                None,
            )

        return (
            ExecutionAttemptTrace(
                attempt_index=1,
                selected_target_before_action=before_snapshot,
                selected_target_after_action=after_snapshot,
                revalidation_result="ok",
                verification_result="select_verification_failed",
                failure_category=FailureCategory.SELECT_VERIFICATION_FAILED,
            ),
            "Execution failed: selected option was not reflected in the element.",
            FailureCategory.SELECT_VERIFICATION_FAILED,
        )

    async def _resolve_target(self, page: Page, action: AgentAction) -> _ResolvedTarget:
        locator = await self._resolve_locator(page, action)
        if locator is not None:
            return _ResolvedTarget(locator=locator)

        handle = await self._resolve_element_handle(page, action)
        if handle is not None:
            return _ResolvedTarget(handle=handle)
        return _ResolvedTarget()

    async def _revalidate_target(
        self,
        page: Page,
        resolved: _ResolvedTarget,
        action: AgentAction,
    ) -> tuple[ExecutionTargetSnapshot | None, FailureCategory | None]:
        snapshot = await self._inspect_resolved_target(page, resolved, action)
        if snapshot is None:
            return None, FailureCategory.TARGET_LOST_BEFORE_ACTION
        if not snapshot.is_visible or not snapshot.is_interactable:
            return snapshot, FailureCategory.STALE_TARGET_BEFORE_ACTION
        # Approximate model coordinates are useful when we fall back to elementFromPoint,
        # but they are too noisy to hard-fail a locator that has already been resolved.
        if (
            resolved.locator is None
            and action.x is not None
            and action.y is not None
            and snapshot.x is not None
            and snapshot.y is not None
        ):
            center_x = snapshot.x + ((snapshot.width or 0) / 2.0)
            center_y = snapshot.y + ((snapshot.height or 0) / 2.0)
            if ((center_x - action.x) ** 2 + (center_y - action.y) ** 2) ** 0.5 > _SHIFT_TOLERANCE_PX:
                return snapshot, FailureCategory.TARGET_SHIFTED_BEFORE_ACTION
        return snapshot, None

    async def _resolve_locator(self, page: Page, action: AgentAction) -> Locator | None:
        for selector in self._candidate_selectors(action):
            locator = page.locator(selector).first
            if await locator.count() > 0:
                return locator
        return None

    async def _resolve_element_handle(self, page: Page, action: AgentAction) -> ElementHandle | None:
        if action.x is None or action.y is None:
            return None
        handle = await page.evaluate_handle(
            "({ x, y }) => document.elementFromPoint(x, y)",
            {"x": action.x, "y": action.y},
        )
        return handle.as_element()

    async def _inspect_target(self, page: Page, action: AgentAction) -> ExecutionTargetSnapshot | None:
        resolved = await self._resolve_target(page, action)
        return await self._inspect_resolved_target(page, resolved, action)

    async def _inspect_resolved_target(
        self,
        page: Page,
        resolved: _ResolvedTarget,
        action: AgentAction,
    ) -> ExecutionTargetSnapshot | None:
        if resolved.locator is not None:
            try:
                return await self._snapshot_locator(page, resolved.locator, action.target_element_id)
            except Exception:
                return None
        if resolved.handle is not None:
            try:
                return await self._snapshot_handle(page, resolved.handle, action.target_element_id)
            except Exception:
                return None
        return None

    async def _snapshot_locator(self, page: Page, locator: Locator, target_element_id: str | None) -> ExecutionTargetSnapshot:
        box = await locator.bounding_box()
        payload = await locator.evaluate(
            """
            element => ({
              dom_id: element.id || null,
              name: element.getAttribute('name'),
              tag_name: element.tagName.toLowerCase(),
              input_type: element.getAttribute('type'),
              value: 'value' in element ? String(element.value ?? '') : null,
              checked: 'checked' in element ? Boolean(element.checked) : null,
              selected_value: element.tagName.toLowerCase() === 'select' ? String(element.value ?? '') : null,
              is_visible: !!(element.offsetWidth || element.offsetHeight || element.getClientRects().length),
              is_interactable: !element.disabled && !(element.getAttribute('aria-disabled') === 'true'),
              is_focused: document.activeElement === element,
            })
            """
        )
        return ExecutionTargetSnapshot(
            target_element_id=target_element_id,
            dom_id=payload["dom_id"],
            name=payload["name"],
            tag_name=payload["tag_name"],
            input_type=payload["input_type"],
            x=box["x"] if box is not None else None,
            y=box["y"] if box is not None else None,
            width=box["width"] if box is not None else None,
            height=box["height"] if box is not None else None,
            is_visible=bool(payload["is_visible"]) and box is not None,
            is_interactable=bool(payload["is_interactable"]),
            is_focused=bool(payload["is_focused"]),
            value=payload["value"],
            checked=payload["checked"],
            selected_value=payload["selected_value"],
            page_signature=await self._page_signature(page),
            page_url=page.url,
        )

    async def _snapshot_handle(self, page: Page, handle: ElementHandle, target_element_id: str | None) -> ExecutionTargetSnapshot:
        box = await handle.bounding_box()
        payload = await handle.evaluate(
            """
            element => ({
              dom_id: element.id || null,
              name: element.getAttribute('name'),
              tag_name: element.tagName.toLowerCase(),
              input_type: element.getAttribute('type'),
              value: 'value' in element ? String(element.value ?? '') : null,
              checked: 'checked' in element ? Boolean(element.checked) : null,
              selected_value: element.tagName.toLowerCase() === 'select' ? String(element.value ?? '') : null,
              is_visible: !!(element.offsetWidth || element.offsetHeight || element.getClientRects().length),
              is_interactable: !element.disabled && !(element.getAttribute('aria-disabled') === 'true'),
              is_focused: document.activeElement === element,
            })
            """
        )
        return ExecutionTargetSnapshot(
            target_element_id=target_element_id,
            dom_id=payload["dom_id"],
            name=payload["name"],
            tag_name=payload["tag_name"],
            input_type=payload["input_type"],
            x=box["x"] if box is not None else None,
            y=box["y"] if box is not None else None,
            width=box["width"] if box is not None else None,
            height=box["height"] if box is not None else None,
            is_visible=bool(payload["is_visible"]) and box is not None,
            is_interactable=bool(payload["is_interactable"]),
            is_focused=bool(payload["is_focused"]),
            value=payload["value"],
            checked=payload["checked"],
            selected_value=payload["selected_value"],
            page_signature=await self._page_signature(page),
            page_url=page.url,
        )

    async def _ensure_focus(self, page: Page, resolved: _ResolvedTarget) -> tuple[str, FailureCategory | None]:
        if await self._resolved_target_is_focused(page, resolved):
            return "already_focused", None
        try:
            await self._click_resolved_target(page, resolved)
        except Exception:
            return "click_before_type_failed", FailureCategory.CLICK_BEFORE_TYPE_FAILED
        if await self._resolved_target_is_focused(page, resolved):
            return "focused_after_click", None
        try:
            await self._click_resolved_target(page, resolved)
        except Exception:
            return "click_before_type_failed", FailureCategory.CLICK_BEFORE_TYPE_FAILED
        if await self._resolved_target_is_focused(page, resolved):
            return "focused_after_retry_click", None
        return "focus_verification_failed", FailureCategory.FOCUS_VERIFICATION_FAILED

    async def _click_resolved_target(self, page: Page, resolved: _ResolvedTarget) -> None:
        if resolved.locator is not None:
            await resolved.locator.click()
            return
        if resolved.handle is not None:
            await resolved.handle.click()
            return
        raise RuntimeError("Unable to resolve click target.")

    async def _resolved_target_is_focused(self, page: Page, resolved: _ResolvedTarget) -> bool:
        if resolved.locator is not None:
            return await resolved.locator.evaluate("element => document.activeElement === element")
        if resolved.handle is not None:
            return await resolved.handle.evaluate("element => document.activeElement === element")
        return False

    async def _resolved_target_is_editable(self, page: Page, resolved: _ResolvedTarget) -> bool:
        if resolved.locator is not None:
            return await resolved.locator.evaluate(
                "element => element.isContentEditable || ['input', 'textarea'].includes(element.tagName.toLowerCase())"
            )
        if resolved.handle is not None:
            return await resolved.handle.evaluate(
                "element => element.isContentEditable || ['input', 'textarea'].includes(element.tagName.toLowerCase())"
            )
        return False

    async def _fill_or_type(self, page: Page, resolved: _ResolvedTarget, text: str) -> None:
        if resolved.locator is not None:
            can_fill = await resolved.locator.evaluate(
                "element => ['input', 'textarea'].includes(element.tagName.toLowerCase())"
            )
            if can_fill:
                await resolved.locator.fill(text)
                return
            await page.keyboard.press("Control+A")
            await page.keyboard.type(text)
            return
        if resolved.handle is not None:
            can_fill = await resolved.handle.evaluate(
                "element => ['input', 'textarea'].includes(element.tagName.toLowerCase())"
            )
            if can_fill:
                await resolved.handle.fill(text)
                return
            await page.keyboard.press("Control+A")
            await page.keyboard.type(text)
            return
        raise RuntimeError("Unable to resolve type target.")

    @staticmethod
    def _typed_value_verified(snapshot: ExecutionTargetSnapshot | None, text: str) -> bool:
        if snapshot is None:
            return False
        value = snapshot.value or ""
        selected_value = snapshot.selected_value or ""
        return text in value or value == text or selected_value == text

    async def _page_signature(self, page: Page) -> str:
        payload = await page.evaluate(
            """
            () => {
              const active = document.activeElement;
              const fields = Array.from(document.querySelectorAll('input, textarea, select')).slice(0, 20).map(element => ({
                id: element.id || null,
                name: element.getAttribute('name'),
                tag: element.tagName.toLowerCase(),
                value: 'value' in element ? String(element.value ?? '') : '',
                checked: 'checked' in element ? Boolean(element.checked) : null,
              }));
              return {
                url: location.href,
                title: document.title,
                active: active ? {
                  id: active.id || null,
                  name: active.getAttribute('name'),
                  tag: active.tagName.toLowerCase(),
                } : null,
                fields,
                text: (document.body?.innerText || '').slice(0, 1000),
              };
            }
            """
        )
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    async def _active_element_signature(self, page: Page) -> str:
        payload = await page.evaluate(
            """
            () => {
              const active = document.activeElement;
              if (!active) return '';
              return [
                active.tagName.toLowerCase(),
                active.id || '',
                active.getAttribute('name') || '',
                active.getAttribute('aria-label') || '',
              ].join('|');
            }
            """
        )
        return str(payload)

    def _candidate_selectors(self, action: AgentAction) -> list[str]:
        selectors: list[str] = []
        if action.selector is not None:
            selectors.append(action.selector)

        target = action.target_element_id
        if target is None:
            target = None
        else:
            selectors.extend(self._derived_selector_candidates(target))
            selectors.extend(
                [
                    f"[aria-label='{self._escape_attr_value(target)}']",
                    f"[name='{self._escape_attr_value(target)}']",
                    f"[data-testid='{self._escape_attr_value(target)}']",
                ]
            )

        for semantic_name in self._semantic_target_names(action):
            escaped = self._escape_attr_value(semantic_name)
            selectors.extend(
                [
                    f"[aria-label='{escaped}']",
                    f"[placeholder='{escaped}']",
                    f"[title='{escaped}']",
                    f"[name='{escaped}']",
                    f"[data-testid='{escaped}']",
                ]
            )

        return list(dict.fromkeys(selectors))

    @staticmethod
    def _semantic_target_names(action: AgentAction) -> list[str]:
        names: list[str] = []
        context = action.target_context
        if context is not None:
            if context.intent.target_text:
                names.append(context.intent.target_text)
            if context.original_target.primary_name:
                names.append(context.original_target.primary_name)
        return [name for name in dict.fromkeys(name.strip() for name in names if name and name.strip())]

    def _derived_selector_candidates(self, target: str) -> list[str]:
        escaped = self._escape_attr_value(target)
        candidates = [f"[id='{escaped}']"]
        if self._is_css_identifier(target):
            candidates.insert(0, f"#{target}")
        return candidates

    @staticmethod
    def _escape_attr_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _is_css_identifier(value: str) -> bool:
        if not value:
            return False
        allowed = set("-_:")
        return all(character.isalnum() or character in allowed for character in value)

    @staticmethod
    def _classify_execution_failure(exc: Exception) -> FailureCategory:
        message = str(exc)
        if "not editable" in message:
            return FailureCategory.EXECUTION_TARGET_NOT_EDITABLE
        if "Unable to resolve" in message:
            return FailureCategory.EXECUTION_TARGET_NOT_FOUND
        return FailureCategory.EXECUTION_ERROR

"""Browser executor interface for browser-only execution."""

from __future__ import annotations

import asyncio
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
from src.models.execution import ExecutedAction
from src.models.policy import ActionType, AgentAction


@dataclass(frozen=True)
class BrowserDebugConfig:
    """Environment-driven Playwright launch options for local debugging."""

    headless: bool
    slow_mo_ms: int
    devtools: bool


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


class PlaywrightBrowserExecutor(BrowserExecutor):
    """Minimal Playwright-backed browser executor for the Gmail draft MVP."""

    def __init__(
        self,
        *,
        headless: bool | None = None,
        slow_mo_ms: int | None = None,
        devtools: bool | None = None,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        artifact_root: str | Path = ".browser-artifacts",
    ) -> None:
        debug_config = _read_browser_debug_config()
        self.headless = debug_config.headless if headless is None else headless
        self.slow_mo_ms = debug_config.slow_mo_ms if slow_mo_ms is None else max(0, slow_mo_ms)
        self.devtools = debug_config.devtools if devtools is None else devtools
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def start(self) -> None:
        if self._page is not None:
            return

        self._playwright = await async_playwright().start()
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
        await self._page.goto("about:blank")

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

    async def execute(self, action: AgentAction) -> ExecutedAction:
        page = await self._require_page()
        detail = f"Executed {action.action_type.value}"
        success = True
        failure_category = None
        failure_stage = None

        try:
            if action.action_type is ActionType.CLICK:
                await self._execute_click(page, action)
                detail = "Clicked target."
            elif action.action_type is ActionType.TYPE:
                await self._execute_type(page, action)
                detail = "Typed text."
            elif action.action_type is ActionType.PRESS_KEY:
                await page.keyboard.press(action.key or "")
                detail = f"Pressed key {action.key}."
            elif action.action_type is ActionType.NAVIGATE:
                await page.goto(action.url or "about:blank", wait_until="domcontentloaded")
                detail = f"Navigated to {action.url}."
            elif action.action_type is ActionType.WAIT:
                await asyncio.sleep((action.wait_ms or 0) / 1000)
                detail = f"Waited {action.wait_ms}ms."
            elif action.action_type is ActionType.STOP:
                detail = "Stop action acknowledged."
        except Exception as exc:
            success = False
            detail = f"Execution failed: {exc}"
            failure_category = self._classify_execution_failure(exc)
            failure_stage = LoopStage.EXECUTE

        artifact_path = None
        try:
            artifact_path = str(self._next_artifact_path("after"))
            await page.screenshot(path=artifact_path, type="png")
        except Exception:
            artifact_path = None

        return ExecutedAction(
            action=action,
            success=success,
            detail=detail,
            artifact_path=artifact_path,
            failure_category=failure_category,
            failure_stage=failure_stage,
        )

    async def _require_page(self) -> Page:
        await self.start()
        if self._page is None:
            raise RuntimeError("Playwright page did not initialize.")
        return self._page

    def _next_artifact_path(self, prefix: str) -> Path:
        return self.artifact_root / f"{prefix}_{uuid4().hex}.png"

    async def _execute_click(self, page: Page, action: AgentAction) -> None:
        locator = await self._resolve_locator(page, action)
        if locator is not None:
            await locator.click()
            return

        handle = await self._resolve_element_handle(page, action)
        if handle is not None:
            await handle.click()
            return

        if action.x is not None and action.y is not None:
            await page.mouse.click(action.x, action.y)
            return

        raise RuntimeError("Unable to resolve click target.")

    async def _execute_type(self, page: Page, action: AgentAction) -> None:
        locator = await self._resolve_locator(page, action)
        if locator is not None:
            if await self._locator_is_fillable(locator):
                await locator.click()
                await locator.fill(action.text or "")
                return
            if await self._locator_can_focus_editable(page, locator):
                await page.keyboard.press("Control+A")
                await page.keyboard.type(action.text or "")
                return
            raise RuntimeError("Resolved type target is not editable.")

        handle = await self._resolve_element_handle(page, action)
        if handle is not None:
            if await self._handle_is_fillable(handle):
                await handle.click()
                await handle.fill(action.text or "")
                return
            if await self._handle_can_focus_editable(page, handle):
                await page.keyboard.press("Control+A")
                await page.keyboard.type(action.text or "")
                return
            raise RuntimeError("Resolved type target is not editable.")

        raise RuntimeError("Unable to resolve type target.")

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
            """
            ({ x, y }) => document.elementFromPoint(x, y)
            """,
            {"x": action.x, "y": action.y},
        )
        return handle.as_element()

    async def _locator_is_fillable(self, locator: Locator) -> bool:
        return await locator.evaluate(
            "element => ['input', 'textarea'].includes(element.tagName.toLowerCase())"
        )

    async def _locator_can_focus_editable(self, page: Page, locator: Locator) -> bool:
        await locator.click()
        return await self._active_element_is_editable(page)

    async def _handle_is_fillable(self, handle: ElementHandle) -> bool:
        return await handle.evaluate(
            "element => ['input', 'textarea'].includes(element.tagName.toLowerCase())"
        )

    async def _handle_can_focus_editable(self, page: Page, handle: ElementHandle) -> bool:
        await handle.click()
        return await self._active_element_is_editable(page)

    async def _active_element_is_editable(self, page: Page) -> bool:
        return await page.evaluate(
            """
            () => {
              const active = document.activeElement;
              if (!active) return false;
              const tag = active.tagName.toLowerCase();
              return active.isContentEditable || tag === 'input' || tag === 'textarea';
            }
            """
        )

    def _candidate_selectors(self, action: AgentAction) -> list[str]:
        selectors: list[str] = []
        if action.selector is not None:
            selectors.append(action.selector)

        target = action.target_element_id
        if target is None:
            return selectors

        selectors.extend(self._derived_selector_candidates(target))
        selectors.extend(
            [
                f"[aria-label='{self._escape_attr_value(target)}']",
                f"[name='{self._escape_attr_value(target)}']",
                f"[data-testid='{self._escape_attr_value(target)}']",
            ]
        )
        return selectors

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

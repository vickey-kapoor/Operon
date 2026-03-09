"""Core UINavigatorAgent — the main execution loop."""

import asyncio
import base64
import io
import logging
import os
import time
from typing import Callable, List, Optional

from pydantic import BaseModel

from src.executor.actions import ActionType
from src.executor.browser import PlaywrightBrowserExecutor
from src import metrics, tracing
from .vision import GeminiVisionClient, VisionUnavailableError
from .planner import ActionPlan, ActionPlanner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class AgentResult(BaseModel):
    """Outcome returned by UINavigatorAgent.run()."""

    success: bool
    result: Optional[str] = None
    steps_taken: int = 0
    screenshots: List[str] = []  # base64-encoded PNG strings
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Progress event helper
# ---------------------------------------------------------------------------


class StepEvent(BaseModel):
    """Emitted after each agent step to report progress."""

    step: int
    observation: str
    reasoning: str
    actions_taken: List[str]
    screenshot: Optional[str] = None  # base64


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class UINavigatorAgent:
    """
    AI agent that autonomously navigates browser UIs using Gemini multimodal.

    The agent loop:
      1. Take a screenshot of the current browser viewport.
      2. Send the screenshot + task to Gemini to get an ActionPlan.
      3. Execute each action in the plan.
      4. Repeat until ``done`` is True or ``max_steps`` is reached.
    """

    def __init__(
        self,
        mode: str = "browser",
        api_key: Optional[str] = None,
        headless: Optional[bool] = None,
        browser_width: int = 1280,
        browser_height: int = 800,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        mode:
            Execution mode.  Currently only "browser" is supported.
        api_key:
            Gemini API key.  Falls back to the ``GOOGLE_API_KEY`` env var.
        headless:
            Whether to run Chromium in headless mode.  Defaults to the value
            of the ``BROWSER_HEADLESS`` env var, or True if unset.
        browser_width / browser_height:
            Viewport dimensions in pixels.
        model:
            Gemini model name to use.  Falls back to ``GeminiVisionClient.MODEL_NAME``.
        system_prompt:
            Optional override for the Gemini system prompt.
        """
        if mode != "browser":
            raise ValueError(f"Unsupported mode {mode!r}. Only 'browser' is supported.")

        self.mode = mode
        self.task_id: Optional[str] = None  # set by server for structured logging

        self._headless = (
            headless
            if headless is not None
            else os.environ.get("BROWSER_HEADLESS", "true").lower() != "false"
        )
        self._browser_width = int(os.environ.get("BROWSER_WIDTH", browser_width))
        self._browser_height = int(os.environ.get("BROWSER_HEIGHT", browser_height))

        self._vision = GeminiVisionClient(api_key=api_key, model=model)
        if system_prompt:
            from .vision import SYSTEM_PROMPT
            from google.genai import types
            self._vision._generation_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self._vision._generation_config.temperature,
                top_p=self._vision._generation_config.top_p,
                max_output_tokens=self._vision._generation_config.max_output_tokens,
                response_mime_type=self._vision._generation_config.response_mime_type,
            )
        self._planner = ActionPlanner(vision_client=self._vision)
        self._executor: Optional[PlaywrightBrowserExecutor] = None

        # Optional callback called after each step with a StepEvent.
        self.on_step: Optional[Callable[[StepEvent], None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        start_url: Optional[str] = None,
        max_steps: int = 20,
    ) -> AgentResult:
        """
        Execute a browser navigation task.

        Parameters
        ----------
        task:
            High-level user intent.
        start_url:
            Optional URL to navigate to before starting the loop.
        max_steps:
            Maximum number of plan→execute cycles before stopping.

        Returns
        -------
        AgentResult
        """
        executor = PlaywrightBrowserExecutor(
            headless=self._headless,
            width=self._browser_width,
            height=self._browser_height,
        )
        self._executor = executor

        screenshots: List[str] = []
        history: List[dict] = []
        steps_taken = 0
        log_ctx = {"task_id": self.task_id}

        try:
            metrics.emit("tasks_started")
            await executor.start()

            if start_url:
                logger.info("Navigating to start URL: %s", start_url, extra=log_ctx)
                await executor._navigate(start_url)

            for step in range(1, max_steps + 1):
                steps_taken = step
                step_ctx = {**log_ctx, "step": step}
                logger.info("--- Step %d/%d ---", step, max_steps, extra=step_ctx)

                # 1. Capture screenshot.
                img = await executor.screenshot()
                screenshot_b64 = self._image_to_base64(img)
                screenshots.append(screenshot_b64)

                # 2. Plan next actions via Gemini + execute, wrapped in a trace span.
                t_step = time.time()
                with tracing.span("agent_step", {"step": str(step), "task_id": self.task_id or ""}):
                    t_plan = time.time()
                    try:
                        plan: ActionPlan = await self._planner.plan(
                            image=img,
                            task=task,
                            history=history if history else None,
                        )
                    except VisionUnavailableError:
                        logger.error(
                            "Gemini vision unavailable — aborting task",
                            extra=step_ctx,
                        )
                        metrics.emit("tasks_failed", labels={"reason": "vision_unavailable"})
                        return AgentResult(
                            success=False,
                            steps_taken=steps_taken,
                            screenshots=screenshots,
                            error="vision_unavailable",
                        )

                    plan_ms = int((time.time() - t_plan) * 1000)
                    logger.info(
                        "Plan ready",
                        extra={
                            **step_ctx,
                            "observation_preview": plan.observation[:120],
                            "actions": [a.type for a in plan.actions],
                            "plan_latency_ms": plan_ms,
                        },
                    )

                # 3. Execute the plan's actions.
                action_descriptions: List[str] = []
                last_screenshot_b64: Optional[str] = screenshot_b64

                for action in plan.actions:
                    t_action = time.time()
                    logger.info(
                        "Executing action",
                        extra={
                            **step_ctx,
                            "action_type": action.type,
                            "description": action.description,
                        },
                    )
                    result = await executor.execute(action)
                    action_ms = int((time.time() - t_action) * 1000)

                    if result.screenshot:
                        last_screenshot_b64 = result.screenshot
                        screenshots.append(result.screenshot)
                    action_descriptions.append(
                        f"{action.type}: {action.description}"
                    )
                    if not result.success:
                        logger.warning(
                            "Action failed",
                            extra={
                                **step_ctx,
                                "action_type": action.type,
                                "error": result.error,
                                "duration_ms": action_ms,
                            },
                        )
                    else:
                        logger.debug(
                            "Action succeeded",
                            extra={
                                **step_ctx,
                                "action_type": action.type,
                                "duration_ms": action_ms,
                            },
                        )

                # 4. Emit progress event.
                event = StepEvent(
                    step=step,
                    observation=plan.observation,
                    reasoning=plan.reasoning,
                    actions_taken=action_descriptions,
                    screenshot=last_screenshot_b64,
                )
                if self.on_step:
                    try:
                        self.on_step(event)
                    except Exception as cb_exc:
                        logger.warning(
                            "on_step callback raised: %s", cb_exc, extra=step_ctx
                        )

                # 5. Update conversation history for Gemini context.
                history = self._update_history(history, plan, screenshot_b64)

                # 6. Check termination.
                if plan.done:
                    logger.info(
                        "Task complete",
                        extra={**step_ctx, "result_preview": (plan.result or "")[:120]},
                    )
                    metrics.emit("tasks_completed", labels={"success": "true"})
                    return AgentResult(
                        success=True,
                        result=plan.result or "Task completed successfully.",
                        steps_taken=steps_taken,
                        screenshots=screenshots,
                    )

                step_ms = int((time.time() - t_step) * 1000)
                metrics.emit("step_latency_ms", step_ms)

                # Brief yield so other coroutines can run.
                await asyncio.sleep(0.1)

            # Reached max_steps without done=True.
            logger.warning(
                "Reached max_steps without completing task",
                extra={**log_ctx, "max_steps": max_steps},
            )
            metrics.emit("tasks_failed", labels={"reason": "max_steps"})
            return AgentResult(
                success=False,
                result=None,
                steps_taken=steps_taken,
                screenshots=screenshots,
                error=f"Task not completed within {max_steps} steps.",
            )

        except Exception as exc:
            logger.exception("Agent run failed: %s", exc, extra=log_ctx)
            return AgentResult(
                success=False,
                steps_taken=steps_taken,
                screenshots=screenshots,
                error=str(exc),
            )
        finally:
            await executor.stop()
            self._executor = None

    async def take_and_analyze_screenshot(
        self,
        task: str,
        start_url: Optional[str] = None,
    ) -> dict:
        """
        One-shot: navigate to a URL (optional), take a screenshot, and analyse it.

        Returns a dict with keys: screenshot (base64), analysis (ActionPlan dict).
        """
        executor = PlaywrightBrowserExecutor(
            headless=self._headless,
            width=self._browser_width,
            height=self._browser_height,
        )
        try:
            await executor.start()
            if start_url:
                await executor._navigate(start_url)

            img = await executor.screenshot()
            plan: ActionPlan = await self._planner.plan(image=img, task=task)
            screenshot_b64 = self._image_to_base64(img)

            return {
                "screenshot": screenshot_b64,
                "analysis": plan.model_dump(),
            }
        finally:
            await executor.stop()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_base64(img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _update_history(
        history: list,
        plan: ActionPlan,
        screenshot_b64: str,
    ) -> list:
        """
        Append the latest model response to the conversation history.

        We keep history bounded to the last 10 turns to avoid exceeding the
        Gemini context window for long tasks.
        """
        from google.genai import types as genai_types

        MAX_HISTORY_TURNS = 10

        history = list(history)
        history.append(
            genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text=plan.model_dump_json())],
            )
        )
        # Trim to last N turns (each turn = user + model = 2 entries).
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]
        return history

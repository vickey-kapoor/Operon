"""Tests for UI Navigator agent components."""

import base64
import json
import os
from io import BytesIO
from typing import List
from unittest.mock import MagicMock, patch

# Set BROWSER_HEADLESS=false to watch tests run in a visible browser window.
_HEADLESS = os.environ.get("BROWSER_HEADLESS", "true").lower() != "false"

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 1280, height: int = 800) -> Image.Image:
    """Create a solid-colour PIL Image for testing."""
    return Image.new("RGB", (width, height), color=(30, 30, 30))


def _pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


VALID_PLAN_JSON = json.dumps(
    {
        "observation": "A blank browser page is visible.",
        "reasoning": "I need to navigate to example.com to complete the task.",
        "actions": [
            {
                "type": "navigate",
                "url": "https://example.com",
                "description": "Navigate to example.com",
            }
        ],
        "done": False,
        "result": None,
    }
)

DONE_PLAN_JSON = json.dumps(
    {
        "observation": "Example Domain page is fully loaded.",
        "reasoning": "The page title 'Example Domain' is visible. Task complete.",
        "actions": [{"type": "done", "description": "Task complete"}],
        "done": True,
        "result": "Page title is 'Example Domain'.",
    }
)


# ---------------------------------------------------------------------------
# ActionPlanner tests
# ---------------------------------------------------------------------------


class TestActionPlannerParsing:
    """Tests for ActionPlanner JSON parsing and error recovery."""

    def _make_planner(self, responses: List[str]):
        """Create an ActionPlanner with a mocked vision client."""
        from src.agent.planner import ActionPlanner

        vision_mock = MagicMock()

        async def analyze_screen(image, task, history=None):
            return responses.pop(0)

        vision_mock.analyze_screen = analyze_screen
        return ActionPlanner(vision_client=vision_mock)

    @pytest.mark.asyncio
    async def test_parse_valid_json(self):
        """ActionPlanner successfully parses a valid JSON response."""
        from src.agent.planner import ActionPlan

        planner = self._make_planner([VALID_PLAN_JSON])
        img = _make_pil_image()
        plan = await planner.plan(image=img, task="Navigate to example.com")

        assert isinstance(plan, ActionPlan)
        assert "blank" in plan.observation.lower()
        assert len(plan.actions) == 1
        assert plan.actions[0].type == "navigate"
        assert plan.actions[0].url == "https://example.com"
        assert plan.done is False

    @pytest.mark.asyncio
    async def test_parse_done_plan(self):
        """ActionPlanner correctly recognises a done=True plan."""
        planner = self._make_planner([DONE_PLAN_JSON])
        img = _make_pil_image()
        plan = await planner.plan(image=img, task="Report page title")

        assert plan.done is True
        assert plan.result == "Page title is 'Example Domain'."

    @pytest.mark.asyncio
    async def test_parse_json_wrapped_in_markdown(self):
        """ActionPlanner extracts JSON from markdown code fences."""
        wrapped = f"Sure, here you go:\n```json\n{VALID_PLAN_JSON}\n```"
        planner = self._make_planner([wrapped])
        img = _make_pil_image()
        plan = await planner.plan(image=img, task="Navigate to example.com")

        assert plan.actions[0].url == "https://example.com"

    @pytest.mark.asyncio
    async def test_malformed_json_returns_fallback_after_retries(self):
        """ActionPlanner returns a fallback screenshot action on repeated bad JSON."""
        bad_responses = [
            "This is not JSON at all.",
            "Still not JSON {{bad}}",
            "```json\n{ invalid: true\n```",
        ]
        planner = self._make_planner(bad_responses)
        img = _make_pil_image()
        plan = await planner.plan(image=img, task="Navigate to example.com")

        # Should get a fallback plan with a screenshot action.
        assert plan.done is False
        assert any(a.type == "screenshot" for a in plan.actions)

    @pytest.mark.asyncio
    async def test_parse_with_extra_action_fields_ignored(self):
        """ActionPlanner ignores unexpected fields in action dicts."""
        plan_with_extras = json.dumps(
            {
                "observation": "A page.",
                "reasoning": "Click something.",
                "actions": [
                    {
                        "type": "click",
                        "coordinate": [100, 200],
                        "description": "Click button",
                        "unknown_field": "should_be_ignored",
                    }
                ],
                "done": False,
                "result": None,
            }
        )
        planner = self._make_planner([plan_with_extras])
        img = _make_pil_image()
        plan = await planner.plan(image=img, task="Click something")

        assert plan.actions[0].coordinate == [100, 200]


# ---------------------------------------------------------------------------
# Action model tests
# ---------------------------------------------------------------------------


class TestActionModels:
    def test_action_type_values(self):
        from src.executor.actions import ActionType

        assert ActionType.CLICK == "click"
        assert ActionType.TYPE == "type"
        assert ActionType.NAVIGATE == "navigate"
        assert ActionType.DONE == "done"

    def test_action_model_click(self):
        from src.executor.actions import Action, ActionType

        action = Action(
            type=ActionType.CLICK,
            coordinate=[640, 400],
            description="Click centre of screen",
        )
        assert action.coordinate == [640, 400]

    def test_action_result_success(self):
        from src.executor.actions import ActionResult

        r = ActionResult(success=True, action_type="click")
        assert r.success is True
        assert r.error is None

    def test_action_result_failure(self):
        from src.executor.actions import ActionResult

        r = ActionResult(success=False, error="Element not found", action_type="click")
        assert r.success is False
        assert "not found" in r.error


# ---------------------------------------------------------------------------
# PlaywrightBrowserExecutor tests (integration — requires Chromium)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPlaywrightBrowserExecutor:
    """Integration tests that spin up a real headless Chromium browser."""

    async def test_start_and_stop(self):
        from src.executor.browser import PlaywrightBrowserExecutor

        executor = PlaywrightBrowserExecutor(headless=_HEADLESS)
        await executor.start()
        assert executor._started is True
        await executor.stop()
        assert executor._started is False

    async def test_screenshot_returns_pil_image(self):
        from src.executor.browser import PlaywrightBrowserExecutor

        executor = PlaywrightBrowserExecutor(headless=_HEADLESS)
        await executor.start()
        try:
            img = await executor.screenshot()
            assert isinstance(img, Image.Image)
            assert img.width == 1280
            assert img.height == 800
        finally:
            await executor.stop()

    async def test_navigate_action(self):
        from src.executor.actions import Action, ActionType
        from src.executor.browser import PlaywrightBrowserExecutor

        executor = PlaywrightBrowserExecutor(headless=_HEADLESS)
        await executor.start()
        try:
            action = Action(
                type=ActionType.NAVIGATE,
                url="https://example.com",
                description="Navigate to example.com",
            )
            result = await executor.execute(action)
            assert result.success is True
            assert result.screenshot is not None

            url = await executor.current_url()
            assert "example.com" in url
        finally:
            await executor.stop()

    async def test_scroll_action(self):
        from src.executor.actions import Action, ActionType
        from src.executor.browser import PlaywrightBrowserExecutor

        executor = PlaywrightBrowserExecutor(headless=_HEADLESS)
        await executor.start()
        try:
            await executor.navigate("https://example.com")
            action = Action(
                type=ActionType.SCROLL,
                coordinate=[640, 400],
                scroll_direction="down",
                scroll_amount=3,
                description="Scroll down",
            )
            result = await executor.execute(action)
            assert result.success is True
        finally:
            await executor.stop()

    async def test_screenshot_action(self):
        from src.executor.actions import Action, ActionType
        from src.executor.browser import PlaywrightBrowserExecutor

        executor = PlaywrightBrowserExecutor(headless=_HEADLESS)
        await executor.start()
        try:
            action = Action(
                type=ActionType.SCREENSHOT,
                description="Capture screen",
            )
            result = await executor.execute(action)
            assert result.success is True
            assert result.screenshot is not None
            # Verify it is valid base64.
            raw = base64.b64decode(result.screenshot)
            assert raw[:4] == b"\x89PNG"
        finally:
            await executor.stop()


# ---------------------------------------------------------------------------
# Full agent loop test (mocked Gemini)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUINavigatorAgent:
    """Full agent loop tests with Gemini API mocked."""

    async def test_navigate_and_report_page_title(self):
        """
        The agent should navigate to example.com and return the page title.
        Gemini responses are mocked so no real API key is needed.
        """
        from src.agent.core import UINavigatorAgent

        # Two Gemini turns:
        # 1. First screenshot → navigate to example.com
        # 2. Second screenshot → report title and set done=True
        gemini_responses = iter(
            [
                json.dumps(
                    {
                        "observation": "A blank browser page.",
                        "reasoning": "I need to open example.com.",
                        "actions": [
                            {
                                "type": "navigate",
                                "url": "https://example.com",
                                "description": "Open example.com",
                            }
                        ],
                        "done": False,
                        "result": None,
                    }
                ),
                json.dumps(
                    {
                        "observation": "The Example Domain page is loaded.",
                        "reasoning": "The page title is visible. Task complete.",
                        "actions": [{"type": "done", "description": "Done"}],
                        "done": True,
                        "result": "Page title is 'Example Domain'.",
                    }
                ),
            ]
        )

        with patch(
            "src.agent.vision.GeminiVisionClient._call_with_retry",
            side_effect=lambda *args, **kwargs: next(gemini_responses),
        ):
            agent = UINavigatorAgent(
                mode="browser",
                api_key="fake-api-key",
                headless=_HEADLESS,
            )
            result = await agent.run(
                task="Navigate to https://example.com and tell me the page title.",
                max_steps=10,
            )

        assert result.success is True
        assert "Example Domain" in (result.result or "")
        assert result.steps_taken <= 10
        assert len(result.screenshots) >= 1

    async def test_agent_handles_max_steps_exceeded(self):
        """Agent returns success=False when done is never set within max_steps."""
        from src.agent.core import UINavigatorAgent

        # Always return a non-done plan.
        never_done = json.dumps(
            {
                "observation": "Still loading…",
                "reasoning": "Waiting.",
                "actions": [{"type": "wait", "duration_ms": 100, "description": "Wait"}],
                "done": False,
                "result": None,
            }
        )

        with patch(
            "src.agent.vision.GeminiVisionClient._call_with_retry",
            return_value=never_done,
        ):
            agent = UINavigatorAgent(
                mode="browser",
                api_key="fake-api-key",
                headless=_HEADLESS,
            )
            result = await agent.run(task="This task never ends.", max_steps=2)

        assert result.success is False
        assert result.error is not None
        assert result.steps_taken == 2

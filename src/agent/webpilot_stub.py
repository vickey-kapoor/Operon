"""Deterministic stub implementation of WebPilotHandler for testing.

Use by setting WEBPILOT_STUB=<scenario_name> when starting the server.
No Gemini API calls are made; scripted WebPilotAction responses are returned
in order, looping back to the start when the script is exhausted.

Supported named scenarios
--------------------------
search                — navigate to bing, type query, click search, done
navigate_and_done     — navigate to a URL, immediately done
confirm_flow          — navigate, then confirm_required, then done
interrupt_redirect    — navigate, wait (simulates mid-task pause), done
navigate_with_redirect — navigate → login_required (simulates redirect to login)
slow_navigate         — navigate (delay_ms=12000) → done (simulates heavy page)
modal_then_proceed    — click × 2 (close modals) → click (target) → done
stuck_loop            — three identical wait actions (simulates stuck state)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from src.api.webpilot_models import InterruptionType, WebPilotAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-defined scenario scripts
# Each entry is a plain dict that matches the WebPilotAction fields.
# ---------------------------------------------------------------------------

_SCENARIOS: Dict[str, List[dict]] = {
    "search": [
        {
            "action": "navigate",
            "url": "https://www.bing.com",
            "narration": "Navigating to Bing to perform the search.",
            "action_label": "Open Bing",
            "is_irreversible": False,
            "observation": "Blank starting page.",
        },
        {
            "action": "click",
            "x": 640,
            "y": 300,
            "narration": "Clicking the search box.",
            "action_label": "Click search box",
            "is_irreversible": False,
            "observation": "Bing homepage is loaded.",
        },
        {
            "action": "type",
            "text": "test query",
            "narration": "Typing the search query.",
            "action_label": "Type query",
            "is_irreversible": False,
            "observation": "Search box is focused.",
        },
        {
            "action": "done",
            "narration": "Search submitted successfully.",
            "action_label": "Done",
            "is_irreversible": False,
            "observation": "Search results are visible.",
        },
    ],
    "navigate_and_done": [
        {
            "action": "navigate",
            "url": "https://example.com",
            "narration": "Navigating to the requested URL.",
            "action_label": "Navigate",
            "is_irreversible": False,
            "observation": "Starting navigation.",
        },
        {
            "action": "done",
            "narration": "Navigation complete. Task finished.",
            "action_label": "Done",
            "is_irreversible": False,
            "observation": "Target page is loaded.",
        },
    ],
    "confirm_flow": [
        {
            "action": "navigate",
            "url": "https://example.com/checkout",
            "narration": "Navigating to checkout.",
            "action_label": "Navigate to checkout",
            "is_irreversible": False,
            "observation": "Blank starting page.",
        },
        {
            "action": "confirm_required",
            "narration": "About to submit the order — this is irreversible. Please confirm.",
            "action_label": "Confirm order",
            "is_irreversible": True,
            "observation": "Checkout page is loaded.",
        },
        {
            "action": "done",
            "narration": "Order placed successfully.",
            "action_label": "Done",
            "is_irreversible": False,
            "observation": "Order confirmation page is shown.",
        },
    ],
    "interrupt_redirect": [
        {
            "action": "navigate",
            "url": "https://example.com",
            "narration": "Starting navigation as requested.",
            "action_label": "Navigate",
            "is_irreversible": False,
            "observation": "Blank starting page.",
        },
        {
            "action": "wait",
            "duration": 500,
            "narration": "Pausing briefly before next step.",
            "action_label": "Wait",
            "is_irreversible": False,
            "observation": "Page loading.",
        },
        {
            "action": "done",
            "narration": "Task completed after redirect.",
            "action_label": "Done",
            "is_irreversible": False,
            "observation": "Final page loaded.",
        },
    ],
    "navigate_with_redirect": [
        {
            "action": "navigate",
            "url": "https://mail.google.com",
            "narration": "Navigating to Gmail.",
            "action_label": "Navigate to Gmail",
            "is_irreversible": False,
            "observation": "Starting navigation.",
        },
        {
            "action": "login_required",
            "narration": "Login screen detected — please sign in.",
            "action_label": "Login required",
            "is_irreversible": False,
            "observation": "Redirected to login page.",
        },
    ],
    "slow_navigate": [
        {
            "action": "navigate",
            "url": "https://heavy-page.example.com",
            "narration": "Navigating to a heavy page.",
            "action_label": "Navigate",
            "is_irreversible": False,
            "observation": "Starting navigation.",
            "delay_ms": 12000,
        },
        {
            "action": "done",
            "narration": "Page loaded successfully.",
            "action_label": "Done",
            "is_irreversible": False,
            "observation": "Heavy page loaded.",
        },
    ],
    "modal_then_proceed": [
        {
            "action": "click",
            "x": 900,
            "y": 50,
            "narration": "Closing the cookie consent modal.",
            "action_label": "Close modal",
            "is_irreversible": False,
            "observation": "Modal overlay is visible.",
        },
        {
            "action": "click",
            "x": 900,
            "y": 50,
            "narration": "Dismissing the newsletter popup.",
            "action_label": "Dismiss popup",
            "is_irreversible": False,
            "observation": "Newsletter popup appeared.",
        },
        {
            "action": "click",
            "x": 640,
            "y": 400,
            "narration": "Clicking the target button.",
            "action_label": "Click target",
            "is_irreversible": False,
            "observation": "Overlays dismissed, target visible.",
        },
        {
            "action": "done",
            "narration": "Task completed after dismissing modals.",
            "action_label": "Done",
            "is_irreversible": False,
            "observation": "Target action executed.",
        },
    ],
    "stuck_loop": [
        {
            "action": "wait",
            "duration": 1000,
            "narration": "Waiting — page appears unchanged.",
            "action_label": "Wait",
            "is_irreversible": False,
            "observation": "Page has not changed.",
        },
        {
            "action": "wait",
            "duration": 1000,
            "narration": "Still waiting — retrying.",
            "action_label": "Wait",
            "is_irreversible": False,
            "observation": "Page still unchanged.",
        },
        {
            "action": "wait",
            "duration": 1000,
            "narration": "Stuck — same screenshot for three consecutive steps.",
            "action_label": "Wait",
            "is_irreversible": False,
            "observation": "Page identical to previous frames.",
        },
    ],
}


class WebPilotStubHandler:
    """
    Deterministic drop-in replacement for WebPilotHandler.

    Accepts either a named scenario string or an explicit list of
    WebPilotAction dicts.  Responses are returned in order; when the
    script is exhausted it loops back to index 0.
    """

    def __init__(self, scenario: str | List[dict] = "navigate_and_done") -> None:
        if isinstance(scenario, str):
            if scenario not in _SCENARIOS:
                raise ValueError(
                    f"Unknown stub scenario {scenario!r}. "
                    f"Available: {sorted(_SCENARIOS)}"
                )
            raw = _SCENARIOS[scenario]
            self._script: List[WebPilotAction] = [
                WebPilotAction(**{k: v for k, v in d.items() if k != "delay_ms"}) for d in raw
            ]
            self._raw_script: List[dict] = list(raw)
            logger.info("WebPilotStubHandler initialised with scenario=%r", scenario)
        else:
            self._script = [WebPilotAction(**{k: v for k, v in d.items() if k != "delay_ms"}) for d in scenario]
            self._raw_script = list(scenario)
            logger.info(
                "WebPilotStubHandler initialised with custom script (%d steps)",
                len(self._script),
            )
        self._index = 0
        self.call_log: list[dict] = []  # records args per get_next_action call

    # ------------------------------------------------------------------
    # Public API — mirrors WebPilotHandler
    # ------------------------------------------------------------------

    async def get_next_action(
        self,
        image_b64: str,  # noqa: ARG002 — accepted but ignored
        intent: str,     # noqa: ARG002
        history: list,   # noqa: ARG002
        stuck: bool = False,
        viewport_width: int = 1280,   # noqa: ARG002
        viewport_height: int = 800,   # noqa: ARG002
        current_url: str = "",        # noqa: ARG002
    ) -> WebPilotAction:
        action = self._script[self._index]
        delay_ms = self._raw_script[self._index].get("delay_ms", 0)
        self.call_log.append(
            {"call_number": len(self.call_log), "stuck": stuck, "returned_action": action.action, "current_url": current_url}
        )
        logger.debug(
            "WebPilotStubHandler step %d/%d stuck=%s → action=%r",
            self._index,
            len(self._script),
            stuck,
            action.action,
        )
        self._index = (self._index + 1) % len(self._script)
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000)
        return action

    async def get_interruption_replan(
        self,
        image_b64: str,           # noqa: ARG002
        original_intent: str,     # noqa: ARG002
        new_instruction: str,     # noqa: ARG002
        history: list,            # noqa: ARG002
        interrupt_type: Optional[InterruptionType] = None,  # noqa: ARG002
        viewport_width: int = 1280,   # noqa: ARG002
        viewport_height: int = 800,   # noqa: ARG002
    ) -> WebPilotAction:
        """Return the next scripted action regardless of interruption type."""
        return await self.get_next_action(
            image_b64="", intent=new_instruction, history=[]
        )

    async def get_narration_audio(self, text: str) -> bytes:  # noqa: ARG002
        """Return silent bytes — no TTS call."""
        return b""

    @staticmethod
    def classify_interruption_type(instruction: str) -> InterruptionType:
        """Delegate to the real static method — no Gemini needed."""
        # Lazy import to avoid circular import: webpilot_stub → webpilot_handler
        # → src.api.webpilot_models → src.api.__init__ → server → webpilot_routes
        # → webpilot_handler (circular). By the time this method is called, all
        # modules are fully initialised so the import is safe.
        from src.agent.webpilot_handler import WebPilotHandler
        return WebPilotHandler.classify_interruption_type(instruction)

"""WebPilot-specific Gemini handler for single-action-at-a-time browser control."""
from __future__ import annotations

import base64
import json
import logging
from typing import List, Optional

from google.genai import types

from src.agent.vision import GeminiVisionClient
from src.agent.planner import _extract_json_from_text
from src.api.webpilot_models import InterruptionType, WebPilotAction

logger = logging.getLogger(__name__)

# FIX 3: Removed hardcoded 1280x800 — dimensions are now injected per-call dynamically.
WEBPILOT_SYSTEM_PROMPT = """\
You are WebPilot, an AI agent that controls web browsers by analyzing screenshots.

Your task: Given a screenshot of the current UI state and a user's intent, determine
the SINGLE NEXT ACTION to take. Return ONE action at a time as a JSON object.

RESPONSE FORMAT (always return exactly this JSON structure, no markdown fences, no prose):
{
  "observation": "<one sentence: describe exactly what you currently see on screen>",
  "action": "<action_type>",
  "x": <integer pixel x coordinate, or null if not applicable>,
  "y": <integer pixel y coordinate, or null if not applicable>,
  "text": "<text to type, or null if not applicable>",
  "url": "<URL to navigate to, or null if not applicable>",
  "direction": "<up or down, or null if not applicable>",
  "duration": <milliseconds to wait, or null if not applicable>,
  "narration": "<short human-readable description of what you are doing and why>",
  "action_label": "<very short action label, e.g. 'Click Search', 'Type email'>",
  "is_irreversible": <true if action cannot be undone, false otherwise>
}

LOOK BEFORE YOU ACT:
- You MUST fill in "observation" first, describing exactly what is on screen RIGHT NOW.
- Only after observing should you decide x/y coordinates.
- Never guess coordinates — only click elements you can clearly see in the screenshot.

VALID ACTION TYPES:
- "click": Click at coordinates (x, y). Required: x, y.
- "type": Type text into the currently focused or a specified element. Required: text.
- "scroll": Scroll the page. Required: direction ("up" or "down").
- "wait": Wait for a specified duration. Required: duration (milliseconds).
- "navigate": Navigate the browser to a URL. Required: url.
- "done": The task is fully complete. No further actions needed.
- "confirm_required": The next logical action is irreversible (e.g., purchase, deletion,
  sending an email, submitting a form with real-world consequences). Pause and ask the user
  to confirm before proceeding.

COORDINATE RULES:
- Coordinates are pixel positions from the top-left of the viewport.
- Exact viewport dimensions are provided in each user message — stay within those bounds.
- Be precise — click on the center of the target element.
- Only use coordinates of elements you can clearly see in the screenshot.

IRREVERSIBILITY RULES:
- Set is_irreversible=true for actions that cannot be undone: purchases, payments,
  sending emails/messages, deleting data, submitting orders, confirming bookings.
- When is_irreversible=true, also set action="confirm_required" so the user can approve first.
- Form fills, navigation, clicks on non-destructive UI elements are reversible (is_irreversible=false).

COMPLETION:
- When the user's task is fully accomplished, return action="done".
- Include a clear narration explaining what was achieved.
- After a "navigate" action lands on the correct page, return action="done" immediately
  if there is nothing else the user explicitly asked to do on that page.
- Do NOT take extra actions (scroll, click, etc.) unless the user's intent requires them.

IMPORTANT:
- Respond ONLY with the JSON object — no markdown fences, no extra prose.
- If the page shows a CAPTCHA or bot detection, navigate to a different site.
- Avoid google.com for searches — use bing.com, duckduckgo.com, or navigate directly.
- If a page is blank or white, immediately navigate to the most appropriate website.
"""


class WebPilotHandler:
    """
    Handles WebPilot-specific Gemini interactions for single-action-at-a-time control.

    Uses the underlying genai client from GeminiVisionClient to call Gemini with
    the WebPilot system prompt, returning one WebPilotAction per call.
    """

    MODEL_NAME = "gemini-2.5-flash"

    def __init__(self, vision_client: GeminiVisionClient, planner) -> None:
        """
        Parameters
        ----------
        vision_client:
            An initialised GeminiVisionClient whose ._client will be reused.
        planner:
            An ActionPlanner instance (kept for potential future use; not used directly).
        """
        self._client = vision_client._client
        self._planner = planner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_next_action(
        self,
        image_b64: str,
        intent: str,
        history: list,
        stuck: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 800,
    ) -> WebPilotAction:
        """
        Call Gemini with the current screenshot and intent to get the next action.

        Parameters
        ----------
        image_b64:
            Base64-encoded PNG screenshot of the current browser state.
        intent:
            The user's high-level task or goal.
        history:
            List of prior types.Content objects (user + model turns).
        stuck:
            If True, inject a hint that the page hasn't changed and a new approach is needed.
        viewport_width:
            Actual width of the browser viewport in pixels (FIX 3: dynamic, not hardcoded).
        viewport_height:
            Actual height of the browser viewport in pixels (FIX 3: dynamic, not hardcoded).

        Returns
        -------
        WebPilotAction
            A validated single action for the extension to execute.
        """
        user_content = self._build_user_content(
            image_b64, intent, history, stuck=stuck,
            viewport_width=viewport_width, viewport_height=viewport_height,
        )
        contents = list(history) + [user_content]

        response = await self._client.aio.models.generate_content(
            model=self.MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=WEBPILOT_SYSTEM_PROMPT,
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=1024,
                # FIX 1: Allow thinking — Gemini 2.5 Flash needs reasoning budget for
                # spatial vision tasks. thinking_budget=0 was disabling this entirely.
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
            contents=contents,
        )

        raw_text = response.text
        if not raw_text:
            raise ValueError("Gemini returned an empty response")

        action = self._parse_action(raw_text)

        # Append this turn to history (mutates the caller's list via append).
        history.append(user_content)
        history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=raw_text)],
            )
        )

        return action

    async def get_interruption_replan(
        self,
        image_b64: str,
        original_intent: str,
        new_instruction: str,
        history: list,
        interrupt_type: Optional[InterruptionType] = None,
        viewport_width: int = 1280,
        viewport_height: int = 800,
    ) -> WebPilotAction:
        """
        Replan after a user interruption, injecting the new instruction into context.

        Parameters
        ----------
        image_b64:
            Base64-encoded PNG screenshot of the current browser state.
        original_intent:
            The original task the agent was working on.
        new_instruction:
            The user's new or updated instruction.
        history:
            List of prior types.Content objects (user + model turns).
        interrupt_type:
            Classification of the interruption (REFINEMENT, REDIRECT, or ABORT).
            If None, defaults to REFINEMENT behaviour.
        viewport_width:
            Actual width of the browser viewport in pixels.
        viewport_height:
            Actual height of the browser viewport in pixels.

        Returns
        -------
        WebPilotAction
            A validated single action based on the new instruction.
        """
        img_bytes = base64.b64decode(image_b64)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")

        if interrupt_type == InterruptionType.REDIRECT:
            instruction_text = (
                f"New goal (replaces previous): {new_instruction}. "
                "Replan from current screen."
            )
        elif interrupt_type == InterruptionType.REFINEMENT:
            instruction_text = (
                f"Add this constraint to the original goal: {new_instruction}. "
                f"Previous goal still applies: {original_intent}."
            )
        else:
            instruction_text = (
                f"Original intent: {original_intent}\n"
                f"New instruction: {new_instruction}"
            )

        text_part = types.Part.from_text(
            text=(
                f"INTERRUPTION — the user has changed or clarified their intent.\n\n"
                f"{instruction_text}\n\n"
                f"Viewport dimensions: {viewport_width}x{viewport_height}px\n"
                "Observe the current screenshot, then determine the single next action. "
                "Respond with a JSON action object including an 'observation' field."
            )
        )
        user_content = types.Content(role="user", parts=[image_part, text_part])
        contents = list(history) + [user_content]

        response = await self._client.aio.models.generate_content(
            model=self.MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=WEBPILOT_SYSTEM_PROMPT,
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=1024,
                # FIX 1: Allow thinking here too.
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
            contents=contents,
        )

        raw_text = response.text
        if not raw_text:
            raise ValueError("Gemini returned an empty response for interruption replan")

        action = self._parse_action(raw_text)

        history.append(user_content)
        history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=raw_text)],
            )
        )

        return action

    async def get_narration_audio(self, text: str) -> bytes:
        """Generate speech audio for narration using Gemini TTS."""
        response = await self._client.aio.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                    )
                ),
            ),
        )
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        return audio_data

    @staticmethod
    def classify_interruption_type(instruction: str) -> InterruptionType:
        """
        Classify a user interruption instruction into ABORT, REDIRECT, or REFINEMENT.

        Redirect is checked before abort so that phrases like "Actually cancel" (which
        contains both a redirect keyword and an abort keyword) resolve to REDIRECT.
        """
        lower = instruction.strip().lower()
        redirect_keywords = {"instead", "forget", "new goal", "start over", "different", "actually"}
        if any(kw in lower for kw in redirect_keywords):
            return InterruptionType.REDIRECT
        abort_keywords = {"stop", "abort", "quit", "never mind", "nevermind", "forget it"}
        if any(kw in lower for kw in abort_keywords):
            return InterruptionType.ABORT
        return InterruptionType.REFINEMENT

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_content(
        image_b64: str,
        intent: str,
        history: list = None,
        stuck: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 800,
    ) -> types.Content:
        """Construct a user Content turn with screenshot, intent, action history summary,
        and viewport dimensions."""
        img_bytes = base64.b64decode(image_b64)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")

        # FIX 2: Summarize prior actions from history so Gemini knows what it already
        # tried — previously this was just a useless step count number.
        prior_actions: list[str] = []
        if history:
            for turn in history:
                if turn.role == "model":
                    try:
                        data = json.loads(_extract_json_from_text(turn.parts[0].text))
                        label = data.get("action_label") or data.get("action", "")
                        narration = data.get("narration", "")
                        if label or narration:
                            prior_actions.append(f"- {label}: {narration}")
                    except Exception:
                        pass  # skip unparseable turns silently

        prior_summary = "\n".join(prior_actions) if prior_actions else "None yet."

        stuck_note = (
            "\n⚠️ STUCK: The page has not changed after multiple attempts. "
            "Try a completely different element or approach."
            if stuck else ""
        )

        # FIX 3: Inject actual viewport dimensions so coordinates are accurate.
        text_part = types.Part.from_text(
            text=(
                f"Viewport dimensions: {viewport_width}x{viewport_height}px\n"
                f"User intent: {intent}\n"
                f"Actions already taken:\n{prior_summary}"
                f"{stuck_note}\n\n"
                "Observe the screenshot carefully, then return the SINGLE NEXT action as JSON."
            )
        )
        return types.Content(role="user", parts=[image_part, text_part])

    @staticmethod
    def _parse_action(raw_text: str) -> WebPilotAction:
        """Parse raw Gemini text into a WebPilotAction, raising on failure."""
        json_str = _extract_json_from_text(raw_text)
        data = json.loads(json_str)
        try:
            return WebPilotAction(**data)
        except Exception as exc:
            logger.warning(
                "WebPilotAction validation failed: %s — raw data: %s", exc, data
            )
            raise ValueError(f"Could not parse WebPilotAction from Gemini response: {exc}") from exc
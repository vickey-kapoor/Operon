"""WebPilot-specific Gemini handler for single-action-at-a-time browser control."""
from __future__ import annotations

import base64
import json
import logging
import os
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
  "plan": ["<step 1>", "<step 2>", ...],
  "steps_completed": <integer: how many plan steps are done so far>,
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
- "key": Press a keyboard key or shortcut. Required: text (key name, e.g. "Enter", "Tab", "Escape", "ArrowDown").
- "done": The task is fully complete. No further actions needed.
- "confirm_required": The next logical action is irreversible (e.g., purchase, deletion,
  sending an email, submitting a form with real-world consequences). Pause and ask the user
  to confirm before proceeding.
- "captcha_detected": The page shows a CAPTCHA or bot detection challenge. Pause and let the user solve it.
- "login_required": The page requires login/authentication. Pause and let the user sign in.

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

PLANNING (Phase 1 — first call only, when no actions have been taken yet):
- Before taking any action, generate a "plan" field: a list of concrete steps needed
  to fully complete the user's goal. Be specific and actionable.
  Example for "find a flight from Austin to Tokyo":
    ["navigate to google flights", "enter Austin as origin", "enter Tokyo as destination",
     "set travel date", "click search", "read flight results"]
  Example for "go to Gmail":
    ["navigate to gmail.com"]
- The plan defines your completion criteria. You are done ONLY when all steps are complete.

PROGRESS CHECK (Phase 2 — every call including the first):
- On every response, set "steps_completed" to the number of plan steps fully finished so far.
- Carry forward the same "plan" array from your first response on every subsequent response.
- After a navigate action, trust that navigation succeeded — do NOT take an extra
  screenshot to verify the page loaded. But if the plan has more steps after navigation,
  continue with the next step.
- Do NOT take extra actions (scroll, click, etc.) unless the plan requires them.

CRITICAL: You may ONLY return action="done" when steps_completed equals the TOTAL number
of steps in your plan. If steps_completed < total plan steps, you MUST continue taking
actions. Never return done mid-plan.

Example:
- plan has 7 steps
- steps_completed = 1  → MUST continue, do NOT return done
- steps_completed = 7  → MAY return done

Current progress check before every response:
  remaining = len(plan) - steps_completed
  if remaining > 0: continue taking actions
  if remaining == 0: return done

IMPORTANT:
- Respond ONLY with the JSON object — no markdown fences, no extra prose.
- If you see a CAPTCHA, use action="captcha_detected". If you see a login wall, use action="login_required".
- Avoid google.com for searches — use bing.com, duckduckgo.com, or navigate directly.
- If a page is blank or white, immediately navigate to the most appropriate website.
"""


_BROWSER_ACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "observation": {"type": "STRING", "description": "One sentence describing what you currently see on screen."},
        "plan": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Ordered list of concrete steps to complete the goal. Set on first call, carry forward on subsequent calls."},
        "steps_completed": {"type": "INTEGER", "description": "Number of plan steps fully completed so far."},
        "action": {
            "type": "STRING",
            "enum": ["click", "type", "scroll", "wait", "navigate", "key", "done",
                     "confirm_required", "captcha_detected", "login_required"],
            "description": "The single next action to take.",
        },
        "x": {"type": "INTEGER", "description": "Pixel x coordinate (for click/type). Null if N/A.", "nullable": True},
        "y": {"type": "INTEGER", "description": "Pixel y coordinate (for click/type). Null if N/A.", "nullable": True},
        "text": {"type": "STRING", "description": "Text to type, or key name for 'key' action. Null if N/A.", "nullable": True},
        "url": {"type": "STRING", "description": "URL for navigate action. Null if N/A.", "nullable": True},
        "direction": {"type": "STRING", "enum": ["up", "down"], "description": "Scroll direction. Null if N/A.", "nullable": True},
        "duration": {"type": "INTEGER", "description": "Milliseconds to wait. Null if N/A.", "nullable": True},
        "narration": {"type": "STRING", "description": "Short human-readable description of what you are doing and why."},
        "action_label": {"type": "STRING", "description": "Very short action label, e.g. 'Click Search'."},
        "is_irreversible": {"type": "BOOLEAN", "description": "True if action cannot be undone."},
    },
    "required": ["observation", "plan", "steps_completed", "action", "narration", "action_label", "is_irreversible"],
}

BROWSER_ACTION_TOOL = types.FunctionDeclaration(
    name="browser_action",
    description="Execute a single browser action. Call this function with the next action to perform.",
    parameters=_BROWSER_ACTION_SCHEMA,
)


class WebPilotHandler:
    """
    Live API handler for WebPilot — maintains a persistent Gemini Live session
    per WebPilot session. Context accumulates automatically (no manual history).

    Falls back to LegacyWebPilotHandler if Live API connect fails.
    """

    MODEL_NAME = os.environ.get("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-preview")

    def __init__(self, client) -> None:
        self._client = client
        self._session = None
        self._resumption_handle = None

    async def connect(self, intent: str) -> None:
        """Open a Live session for the given user intent."""
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction=WEBPILOT_SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=[BROWSER_ACTION_TOOL])],
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            session_resumption=types.SessionResumptionConfig(transparent=True),
            temperature=0.2,
        )
        try:
            self._session = await self._client.aio.live.connect(
                model=self.MODEL_NAME,
                config=config,
            )
            # Send the initial intent as context
            await self._session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"User intent: {intent}")],
                ),
                turn_complete=True,
            )
            logger.info("Live session connected for intent: %s", intent[:80])
        except Exception as exc:
            logger.warning("Live API connect failed: %s", exc)
            self._session = None
            raise

    async def send_screenshot_and_get_action(
        self,
        image_b64: str,
        intent: str,
        stuck: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        current_url: str = "",
    ) -> WebPilotAction:
        """Send screenshot to Live session, get next action via function call."""
        if not self._session:
            raise RuntimeError("Live session not connected")

        img_bytes = base64.b64decode(image_b64)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")

        stuck_note = (
            "\n⚠️ STUCK: The page has not changed after multiple attempts. "
            "Try a completely different element or approach.\n"
            "Previous clicks may have failed. Consider keyboard alternatives:\n"
            "- Tab to move focus, Enter to activate, Escape to dismiss\n"
            "- Arrow keys for lists/menus, Space for checkboxes/buttons\n"
            'Use action="key" with the appropriate key value.'
            if stuck else ""
        )

        url_line = f"Current URL: {current_url}\n" if current_url else ""
        text_part = types.Part.from_text(
            text=(
                f"Viewport dimensions: {viewport_width}x{viewport_height}px\n"
                f"{url_line}"
                f"User intent: {intent}"
                f"{stuck_note}\n\n"
                "Observe the screenshot carefully, then call browser_action with the SINGLE NEXT action."
            )
        )

        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[image_part, text_part]),
            turn_complete=True,
        )

        # Iterate responses looking for a tool_call
        async for msg in self._session.receive():
            # Handle session resumption updates
            if hasattr(msg, 'session_resumption_update') and msg.session_resumption_update:
                if hasattr(msg.session_resumption_update, 'handle'):
                    self._resumption_handle = msg.session_resumption_update.handle

            # Handle go_away (session expiring)
            if hasattr(msg, 'go_away') and msg.go_away:
                logger.warning("Live session go_away received — reconnecting")
                await self._reconnect(intent)
                return await self.send_screenshot_and_get_action(
                    image_b64, intent, stuck, viewport_width, viewport_height
                )

            # Check for tool calls
            if hasattr(msg, 'tool_call') and msg.tool_call:
                for fc in msg.tool_call.function_calls:
                    if fc.name == "browser_action":
                        try:
                            action = WebPilotAction(**fc.args)
                            logger.info(
                                "Live action=%s plan_len=%d steps_completed=%d",
                                action.action,
                                len(action.plan) if action.plan else 0,
                                action.steps_completed or 0,
                            )
                            # Send tool response to acknowledge
                            await self._session.send_tool_response(
                                function_responses=[types.FunctionResponse(
                                    name="browser_action",
                                    response={"status": "executed"},
                                )]
                            )
                            return action
                        except Exception as exc:
                            logger.warning("Function call args invalid: %s — args: %s", exc, fc.args)
                            raise ValueError(f"Invalid browser_action args: {exc}") from exc

            # Check for text response (fallback — model didn't use function calling)
            if hasattr(msg, 'server_content') and msg.server_content:
                if hasattr(msg.server_content, 'model_turn') and msg.server_content.model_turn:
                    for part in msg.server_content.model_turn.parts:
                        if hasattr(part, 'text') and part.text:
                            # Try to parse as JSON action (fallback)
                            try:
                                return LegacyWebPilotHandler._parse_action(part.text)
                            except Exception:
                                pass
                # Check turn_complete
                if hasattr(msg.server_content, 'turn_complete') and msg.server_content.turn_complete:
                    break

        raise ValueError("Live session did not return a browser_action tool call")

    async def send_interruption(self, instruction: str) -> None:
        """Send an interruption instruction into the active Live session."""
        if not self._session:
            raise RuntimeError("Live session not connected")
        await self._session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=f"INTERRUPTION — the user has changed or clarified their intent.\n\n{instruction}"
                )],
            ),
            turn_complete=True,
        )

    async def verify_completion(
        self,
        image_b64: str,
        intent: str,
        viewport_width: int = 1280,
        viewport_height: int = 800,
    ) -> bool:
        """Verify task completion via the Live session."""
        if not self._session:
            return True

        img_bytes = base64.b64decode(image_b64)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        text_part = types.Part.from_text(
            text=(
                f"Viewport dimensions: {viewport_width}x{viewport_height}px\n"
                f"The user's original goal was: {intent}\n\n"
                "You just reported the task as 'done'. Look at this screenshot and determine "
                "whether the goal has ACTUALLY been achieved.\n"
                "Reply with ONLY: VERIFIED or NOT_VERIFIED followed by a brief reason."
            )
        )
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[image_part, text_part]),
            turn_complete=True,
        )

        try:
            async for msg in self._session.receive():
                if hasattr(msg, 'server_content') and msg.server_content:
                    if hasattr(msg.server_content, 'model_turn') and msg.server_content.model_turn:
                        for part in msg.server_content.model_turn.parts:
                            if hasattr(part, 'text') and part.text:
                                text = part.text.strip().upper()
                                if "NOT_VERIFIED" in text:
                                    logger.info("Live completion verification: NOT_VERIFIED")
                                    return False
                                if "VERIFIED" in text:
                                    logger.info("Live completion verification: VERIFIED")
                                    return True
                    if hasattr(msg.server_content, 'turn_complete') and msg.server_content.turn_complete:
                        break
        except Exception as exc:
            logger.warning("Live verify_completion error, accepting done: %s", exc)
        return True

    async def get_narration_audio(self, text: str) -> bytes:
        """Generate speech audio for narration using Gemini TTS (separate from Live)."""
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
        try:
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            return audio_data
        except (IndexError, AttributeError) as exc:
            raise ValueError(f"TTS response missing audio data: {exc}") from exc

    async def close(self) -> None:
        """Close the Live session."""
        if self._session:
            try:
                await self._session.close()
            except Exception as exc:
                logger.debug("Live session close error: %s", exc)
            self._session = None
            self._resumption_handle = None

    async def _reconnect(self, intent: str) -> None:
        """Reconnect after go_away, using resumption handle if available."""
        old_session = self._session
        self._session = None
        try:
            if old_session:
                await old_session.close()
        except Exception:
            pass

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction=WEBPILOT_SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=[BROWSER_ACTION_TOOL])],
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            session_resumption=types.SessionResumptionConfig(
                transparent=True,
                handle=self._resumption_handle,
            ),
            temperature=0.2,
        )
        self._session = await self._client.aio.live.connect(
            model=self.MODEL_NAME,
            config=config,
        )
        logger.info("Live session reconnected after go_away")

    @staticmethod
    def classify_interruption_type(instruction: str) -> InterruptionType:
        """Classify a user interruption instruction into ABORT, REDIRECT, or REFINEMENT."""
        lower = instruction.strip().lower()
        abort_keywords = {"stop", "abort", "quit", "never mind", "nevermind", "forget it", "forget about it"}
        if any(kw in lower for kw in abort_keywords):
            return InterruptionType.ABORT
        redirect_keywords = {"instead", "new goal", "start over", "different", "actually"}
        if any(kw in lower for kw in redirect_keywords):
            return InterruptionType.REDIRECT
        return InterruptionType.REFINEMENT


class LegacyWebPilotHandler:
    """
    Legacy handler using per-call generate_content(). Kept as fallback when
    the Live API is unavailable.
    """

    MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def __init__(self, vision_client: GeminiVisionClient, planner) -> None:
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
        current_url: str = "",
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
            current_url=current_url,
        )
        contents = list(history) + [user_content]

        response = await self._client.aio.models.generate_content(
            model=self.MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=WEBPILOT_SYSTEM_PROMPT,
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=1024,
                # Reasoning budget — 1024 gives Gemini enough room to generate a plan
                # and reason about progress against it on each step.
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
            contents=contents,
        )

        raw_text = response.text
        if not raw_text:
            raise ValueError("Gemini returned an empty response")

        action = self._parse_action(raw_text)
        logger.info(
            "Legacy action=%s plan_len=%d steps_completed=%d",
            action.action,
            len(action.plan) if action.plan else 0,
            action.steps_completed or 0,
        )

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
        logger.info(
            "Replan action=%s plan_len=%d steps_completed=%d",
            action.action,
            len(action.plan) if action.plan else 0,
            action.steps_completed or 0,
        )

        history.append(user_content)
        history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=raw_text)],
            )
        )

        return action

    async def verify_completion(
        self,
        image_b64: str,
        intent: str,
        viewport_width: int = 1280,
        viewport_height: int = 800,
    ) -> bool:
        """
        Verify via screenshot that the user's goal was actually achieved.

        Returns True if goal appears met, False if not. On parse failure,
        returns True to avoid blocking on verification errors.
        """
        img_bytes = base64.b64decode(image_b64)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        text_part = types.Part.from_text(
            text=(
                f"Viewport dimensions: {viewport_width}x{viewport_height}px\n"
                f"The user's original goal was: {intent}\n\n"
                "The AI agent just reported the task as 'done'. "
                "Look at this screenshot carefully and determine whether the goal "
                "has ACTUALLY been achieved.\n\n"
                "Respond with ONLY a JSON object:\n"
                '{"verified": true, "reason": "..."} if the goal is met\n'
                '{"verified": false, "reason": "..."} if the goal is NOT met'
            )
        )
        user_content = types.Content(role="user", parts=[image_part, text_part])

        try:
            response = await self._client.aio.models.generate_content(
                model=self.MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction="You are a verification assistant. Determine if a browser task was completed successfully by examining screenshots.",
                    temperature=0.1,
                    max_output_tokens=256,
                    thinking_config=types.ThinkingConfig(thinking_budget=512),
                ),
                contents=[user_content],
            )
            raw_text = response.text
            if not raw_text:
                return True
            json_str = _extract_json_from_text(raw_text)
            data = json.loads(json_str)
            verified = data.get("verified", True)
            logger.info(
                "Completion verification: verified=%s reason=%s",
                verified, data.get("reason", ""),
            )
            return bool(verified)
        except Exception as exc:
            logger.warning("Completion verification failed, accepting done: %s", exc)
            return True

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
        try:
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            return audio_data
        except (IndexError, AttributeError) as exc:
            raise ValueError(f"TTS response missing audio data: {exc}") from exc

    @staticmethod
    def classify_interruption_type(instruction: str) -> InterruptionType:
        """
        Classify a user interruption instruction into ABORT, REDIRECT, or REFINEMENT.

        Redirect is checked before abort so that phrases like "Actually cancel" (which
        contains both a redirect keyword and an abort keyword) resolve to REDIRECT.
        """
        lower = instruction.strip().lower()
        # Check abort first for explicit "forget it" / "forget about it" phrases,
        # then check redirect (which includes "instead", "start over", etc.).
        abort_keywords = {"stop", "abort", "quit", "never mind", "nevermind", "forget it", "forget about it"}
        if any(kw in lower for kw in abort_keywords):
            return InterruptionType.ABORT
        redirect_keywords = {"instead", "new goal", "start over", "different", "actually"}
        if any(kw in lower for kw in redirect_keywords):
            return InterruptionType.REDIRECT
        return InterruptionType.REFINEMENT

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_content(
        image_b64: str,
        intent: str,
        history: Optional[list] = None,
        stuck: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        current_url: str = "",
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
                    except Exception as exc:
                        logger.debug("Skipping unparseable history turn: %s", exc)

        prior_summary = "\n".join(prior_actions) if prior_actions else "None yet."

        stuck_note = (
            "\n⚠️ STUCK: The page has not changed after multiple attempts. "
            "Try a completely different element or approach.\n"
            "Previous clicks may have failed. Consider keyboard alternatives:\n"
            "- Tab to move focus, Enter to activate, Escape to dismiss\n"
            "- Arrow keys for lists/menus, Space for checkboxes/buttons\n"
            'Use action="key" with the appropriate key value.'
            if stuck else ""
        )

        # FIX 3: Inject actual viewport dimensions so coordinates are accurate.
        url_line = f"Current URL: {current_url}\n" if current_url else ""
        text_part = types.Part.from_text(
            text=(
                f"Viewport dimensions: {viewport_width}x{viewport_height}px\n"
                f"{url_line}"
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
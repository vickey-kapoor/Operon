"""Policy service interface for next-action selection."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from pydantic import ValidationError

from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, PolicyDecision
from operon.models.state import AgentState

logger = logging.getLogger(__name__)


class PolicyError(RuntimeError):
    """Raised when Gemini policy output cannot be parsed into strict schemas."""


class PolicyService(ABC):
    @abstractmethod
    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        """Choose the next action from current state and perception."""

    def supports_resampling(self, run_id: str = "") -> bool:
        """Whether a second ``choose_action`` for the same step yields a fresh
        decision (Best-of-N sampling needs it)."""
        return True


def parse_policy_output(raw_output: str) -> PolicyDecision:
    cleaned = _strip_json_fence(raw_output)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise PolicyError("Gemini policy output was not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise PolicyError("Gemini policy output must be a JSON object.")

    parsed = _normalize_policy_payload(parsed)

    if "expected_change" not in parsed:
        logger.warning("[PLANNER] Response missing expected_change; defaulting to empty string")
        parsed["expected_change"] = "none"

    try:
        return PolicyDecision.model_validate(parsed)
    except ValidationError as exc:
        raise PolicyError("Gemini policy output did not match the strict schema.") from exc



def _strip_json_fence(raw_output: str) -> str:
    cleaned = raw_output.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_policy_payload(parsed: dict[str, object]) -> dict[str, object]:
    action = parsed.get("action")
    if not isinstance(action, dict):
        return parsed

    normalized_action = dict(action)
    action_type = normalized_action.get("action_type")
    normalized_payload = dict(parsed)

    nested_rationale = normalized_action.get("rationale")
    if isinstance(nested_rationale, str):
        if not isinstance(normalized_payload.get("rationale"), str):
            normalized_payload["rationale"] = nested_rationale
        normalized_action.pop("rationale", None)

    # STOP/WAIT_FOR_USER must not carry payload fields (LLMs sometimes put answer text in text=)
    _PAYLOAD_FIELDS = {"text", "url", "key", "selector", "x", "y", "wait_ms", "scroll_amount"}
    if action_type in (ActionType.STOP.value, ActionType.WAIT_FOR_USER.value):
        for _f in _PAYLOAD_FIELDS:
            normalized_action.pop(_f, None)

    # SCROLL: LLMs often put the scroll distance in y= instead of scroll_amount=,
    # emit scroll_direction/direction="up"/"down" instead of a signed scroll_amount,
    # use "pixels" instead of "scroll_amount", or omit x,y coordinates. Normalise all.
    if action_type == ActionType.SCROLL.value:
        # Accept both "scroll_direction" and bare "direction" as direction hints
        direction = normalized_action.pop("scroll_direction", None) or normalized_action.pop("direction", None)
        # Accept "pixels" as an alias for scroll_amount
        if normalized_action.get("scroll_amount") is None:
            pixels = normalized_action.pop("pixels", None)
            if isinstance(pixels, int) and pixels != 0:
                normalized_action["scroll_amount"] = pixels
            else:
                normalized_action.pop("pixels", None)
                candidate = normalized_action.get("y")
                if isinstance(candidate, int) and candidate != 0:
                    normalized_action["scroll_amount"] = candidate
                    normalized_action.pop("y", None)
        else:
            normalized_action.pop("pixels", None)
        # Apply direction sign: "up" → negative, "down" → positive
        if isinstance(direction, str) and normalized_action.get("scroll_amount") is not None:
            amt = normalized_action["scroll_amount"]
            if direction.lower() == "up" and isinstance(amt, int) and amt > 0:
                normalized_action["scroll_amount"] = -amt
        if normalized_action.get("x") is None:
            normalized_action["x"] = 960
        if normalized_action.get("y") is None:
            normalized_action["y"] = 540

    # CLICK/DOUBLE_CLICK/RIGHT_CLICK cannot carry text, key, url, or wait_ms.
    # LLMs sometimes annotate a click with the label of the element being clicked
    # (e.g. text="Water") — strip those fields before Pydantic validation rejects them.
    if action_type in (ActionType.CLICK.value, ActionType.DOUBLE_CLICK.value, ActionType.TRIPLE_CLICK.value, ActionType.RIGHT_CLICK.value):
        for _f in ("text", "key", "url", "wait_ms"):
            normalized_action.pop(_f, None)

    wait_ms = normalized_action.get("wait_ms")
    if action_type == ActionType.WAIT.value:
        if isinstance(wait_ms, int) and wait_ms <= 0:
            normalized_action["wait_ms"] = 1
    else:
        # Gemini sometimes emits wait_ms on non-wait actions — strip it
        if wait_ms is not None:
            normalized_action.pop("wait_ms", None)

    key_value = normalized_action.get("key")
    if action_type == ActionType.TYPE.value and isinstance(key_value, str):
        if key_value.strip().lower() == "enter":
            normalized_action["press_enter"] = True
            normalized_action.pop("key", None)

    text_value = normalized_action.get("text")
    if action_type == ActionType.TYPE.value and isinstance(text_value, str):
        stripped_text = text_value.rstrip("\r\n")
        if stripped_text != text_value:
            normalized_action["text"] = stripped_text
            if stripped_text:
                normalized_action["press_enter"] = True

    if normalized_action != action:
        normalized_payload["action"] = normalized_action
        return normalized_payload
    return normalized_payload



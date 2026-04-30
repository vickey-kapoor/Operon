"""Policy service interface for next-action selection."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import ValidationError

from src.clients.gemini import GeminiClient, GeminiClientError
from src.models.logs import ModelDebugArtifacts
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, PolicyDecision
from src.models.state import AgentState
from src.store.background_writer import bg_writer

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


class GeminiPolicyService(PolicyService):
    """Gemini-backed policy implementation for one-step next-action selection."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        prompt_path: Path | None = None,
    ) -> None:
        self.gemini_client = gemini_client
        self.prompt_path = prompt_path or Path(__file__).resolve().parents[2] / "prompts" / "policy_prompt.txt"
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self._last_debug_artifacts: ModelDebugArtifacts | None = None
        self._advisory_hints: dict[str, list[tuple[str, str]]] = {}

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        prompt = self._render_prompt(state, perception)
        step_dir = Path(perception.capture_artifact_path).resolve().parent
        debug_artifacts = self._artifact_paths(step_dir)
        bg_writer.enqueue(debug_artifacts.prompt_artifact_path, prompt)
        try:
            raw_output = await self.gemini_client.generate_policy(prompt)
        except GeminiClientError:
            raise
        bg_writer.enqueue(debug_artifacts.raw_response_artifact_path, raw_output)
        decision = parse_policy_output(raw_output)
        bg_writer.enqueue(debug_artifacts.parsed_artifact_path, decision.model_dump_json())
        self._last_debug_artifacts = ModelDebugArtifacts(
            prompt_artifact_path=str(debug_artifacts.prompt_artifact_path),
            raw_response_artifact_path=str(debug_artifacts.raw_response_artifact_path),
            parsed_artifact_path=str(debug_artifacts.parsed_artifact_path),
            usage_artifact_path=str(debug_artifacts.usage_artifact_path),
            usage=_latest_usage(self.gemini_client, debug_artifacts.usage_artifact_path),
        )
        return decision

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    def _reset_advisory_hints_for_test(self, hints: list[str]) -> None:
        """Reset hints to a known state. Test use only."""
        self._advisory_hints = {"": [(h, "") for h in hints if h]}

    def add_advisory_hints(self, hints: list[str], source: str = "", run_id: str = "") -> None:
        """Append hints scoped to run_id so concurrent runs don't cross-contaminate."""
        incoming = [(h, source) for h in hints if h]
        bucket = self._advisory_hints.setdefault(run_id, [])
        bucket.extend(incoming)
        logger.debug(
            "add_advisory_hints(%s): run=%r source=%r incoming=%d total=%d",
            self.__class__.__name__, run_id, source, len(incoming), len(bucket),
        )

    def clear_advisory_hints(self, run_id: str = "") -> None:
        self._advisory_hints.pop(run_id, None)

    def _render_prompt(self, state: AgentState, perception: ScreenPerception) -> str:
        last_verification = "none"
        if state.verification_history:
            last_vr = state.verification_history[-1]
            parts = [f"status={last_vr.status}", f"reason={last_vr.reason!r}"]
            if last_vr.recovery_hint:
                parts.append(f"recovery_hint={last_vr.recovery_hint!r}")
            last_verification = ", ".join(parts)
        prompt = self._prompt_template.format(
            intent=state.intent,
            current_subgoal=state.current_subgoal or "not set",
            step_count=state.step_count,
            retry_counts=json.dumps(state.retry_counts, sort_keys=True),
            perception_json=perception.model_dump_json(),
            last_verification=last_verification,
        )
        hints = self._advisory_hints.pop(state.run_id, None) or self._advisory_hints.pop("", None)
        if hints:
            _counts: dict[str, int] = {}
            for _, _src in hints:
                _label = _src or "unknown"
                _counts[_label] = _counts.get(_label, 0) + 1
            logger.debug("hints consumed (%s): [%s]", self.__class__.__name__, ", ".join(f"{k}:{v}" for k, v in _counts.items()))
            prompt = f"{prompt}\n\nAdvisory memory hints:\n" + "\n".join(f"- {h}" for h, _ in hints)
        return prompt

    @staticmethod
    def _artifact_paths(step_dir: Path) -> "_StageArtifactPaths":
        step_dir.mkdir(parents=True, exist_ok=True)
        return _StageArtifactPaths(
            prompt_artifact_path=step_dir / "policy_prompt.txt",
            raw_response_artifact_path=step_dir / "policy_raw.txt",
            parsed_artifact_path=step_dir / "policy_decision.json",
            usage_artifact_path=step_dir / "policy_usage.json",
        )


class _StageArtifactPaths:
    def __init__(self, prompt_artifact_path: Path, raw_response_artifact_path: Path, parsed_artifact_path: Path, usage_artifact_path: Path) -> None:
        self.prompt_artifact_path = prompt_artifact_path
        self.raw_response_artifact_path = raw_response_artifact_path
        self.parsed_artifact_path = parsed_artifact_path
        self.usage_artifact_path = usage_artifact_path


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
    if action_type in (ActionType.CLICK.value, ActionType.DOUBLE_CLICK.value, ActionType.RIGHT_CLICK.value):
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


def _latest_usage(client: GeminiClient, usage_artifact_path: Path):
    if not hasattr(client, "latest_usage"):
        return None
    usage = client.latest_usage()
    if usage is not None:
        bg_writer.enqueue(usage_artifact_path, usage.model_dump_json(indent=2))
    return usage

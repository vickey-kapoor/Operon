"""Policy service interface for next-action selection."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import ValidationError

from src.clients.gemini import GeminiClient, GeminiClientError
from src.models.logs import ModelDebugArtifacts
from src.models.perception import ScreenPerception, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.state import AgentState
from src.store.background_writer import bg_writer


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
        self._advisory_hints: list[str] = []

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
        decision = self._apply_focus_first_guardrail(state, perception, decision)
        bg_writer.enqueue(debug_artifacts.parsed_artifact_path, decision.model_dump_json())
        self._last_debug_artifacts = ModelDebugArtifacts(
            prompt_artifact_path=str(debug_artifacts.prompt_artifact_path),
            raw_response_artifact_path=str(debug_artifacts.raw_response_artifact_path),
            parsed_artifact_path=str(debug_artifacts.parsed_artifact_path),
        )
        return decision

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    def set_advisory_hints(self, hints: list[str]) -> None:
        self._advisory_hints = [hint for hint in hints if hint]

    def _render_prompt(self, state: AgentState, perception: ScreenPerception) -> str:
        prompt = self._prompt_template.format(
            intent=state.intent,
            current_subgoal=state.current_subgoal or "not set",
            step_count=state.step_count,
            retry_counts=json.dumps(state.retry_counts, sort_keys=True),
            perception_json=perception.model_dump_json(),
        )
        if self._advisory_hints:
            prompt = f"{prompt}\n\nAdvisory memory hints:\n" + "\n".join(f"- {hint}" for hint in self._advisory_hints)
        self._advisory_hints = []
        return prompt

    def _apply_focus_first_guardrail(
        self,
        state: AgentState,
        perception: ScreenPerception,
        decision: PolicyDecision,
    ) -> PolicyDecision:
        action = decision.action
        if action.action_type is not ActionType.TYPE:
            return decision
        if action.selector is not None:
            return decision

        target_id = action.target_element_id
        if target_id is None:
            return decision

        target = next((element for element in perception.visible_elements if element.element_id == target_id), None)
        if target is None or target.element_type is not UIElementType.INPUT or not target.usable_for_targeting:
            return decision

        if perception.focused_element_id == target_id:
            return decision

        click_action = AgentAction(
            action_type=ActionType.CLICK,
            target_element_id=target_id,
            x=target.x + max(1, target.width // 2),
            y=target.y + max(1, target.height // 2),
        )
        return PolicyDecision(
            action=click_action,
            rationale=f"Focus {target.primary_name} before typing.",
            confidence=min(decision.confidence, 0.8),
            active_subgoal=f"focus {target_id}",
        )

    @staticmethod
    def _artifact_paths(step_dir: Path) -> "_StageArtifactPaths":
        step_dir.mkdir(parents=True, exist_ok=True)
        return _StageArtifactPaths(
            prompt_artifact_path=step_dir / "policy_prompt.txt",
            raw_response_artifact_path=step_dir / "policy_raw.txt",
            parsed_artifact_path=step_dir / "policy_decision.json",
        )


class _StageArtifactPaths:
    def __init__(self, prompt_artifact_path: Path, raw_response_artifact_path: Path, parsed_artifact_path: Path) -> None:
        self.prompt_artifact_path = prompt_artifact_path
        self.raw_response_artifact_path = raw_response_artifact_path
        self.parsed_artifact_path = parsed_artifact_path


def parse_policy_output(raw_output: str) -> PolicyDecision:
    cleaned = _strip_json_fence(raw_output)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise PolicyError("Gemini policy output was not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise PolicyError("Gemini policy output must be a JSON object.")

    parsed = _normalize_policy_payload(parsed)

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

    wait_ms = normalized_action.get("wait_ms")
    if action_type == ActionType.WAIT.value:
        if isinstance(wait_ms, int) and wait_ms <= 0:
            normalized_action["wait_ms"] = 1
    else:
        # Gemini sometimes emits wait_ms on non-wait actions — strip it
        if wait_ms is not None:
            normalized_action.pop("wait_ms", None)

    if normalized_action != action:
        normalized_payload = dict(parsed)
        normalized_payload["action"] = normalized_action
        return normalized_payload
    return parsed

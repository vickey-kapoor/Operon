"""Combined perception+policy service that uses a single Gemini call per step."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.perception import (
    PerceptionError,
    PerceptionLowQualityError,
    PerceptionService,
    _apply_weak_canonicalization,
    _low_quality_reason,
    _normalize_visible_elements,
    _quality_metrics,
    _strip_json_fence,
    parse_perception_output,
)
from src.agent.policy import PolicyError, PolicyService, parse_policy_output
from src.clients.gemini import GeminiClient, GeminiClientError
from src.models.capture import CaptureFrame
from src.models.logs import ModelDebugArtifacts
from src.models.perception import ScreenPerception
from src.models.policy import PolicyDecision
from src.models.state import AgentState
from src.store.background_writer import bg_writer

logger = logging.getLogger(__name__)

_MAX_COMBINED_RETRIES = 1


class CombinedPerceptionPolicyService(PerceptionService, PolicyService):
    """Single Gemini call that returns both perception and policy.

    During ``perceive()``, sends the screenshot with a combined prompt and
    caches the policy decision.  The subsequent ``choose_action()`` returns
    the cached decision without a second network round-trip.

    Supports a ``perception_only`` flag for retry paths where the loop
    needs fresh perception without overwriting the cached policy decision.
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        prompt_path: Path | None = None,
    ) -> None:
        self.gemini_client = gemini_client
        self.prompt_path = prompt_path or Path(__file__).resolve().parents[2] / "prompts" / "desktop_combined_prompt.txt"
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self._cached_decision: PolicyDecision | None = None
        self._last_debug_artifacts: ModelDebugArtifacts | None = None
        self._advisory_hints: list[str] = []
        self._perception_only: bool = False

    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> ScreenPerception:
        """Run combined perception+policy and cache both results.

        If ``_perception_only`` is set, still calls Gemini (combined prompt)
        but discards the policy decision to avoid corrupting cached state.
        """
        step_dir = Path(screenshot.artifact_path).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = step_dir / "combined_prompt.txt"
        raw_path = step_dir / "combined_raw.txt"
        parsed_path = step_dir / "combined_parsed.json"

        for attempt in range(_MAX_COMBINED_RETRIES + 1):
            prompt = self._render_prompt(state, semantic_retry=attempt > 0)
            bg_writer.enqueue(prompt_path, prompt)

            try:
                raw_output = await self.gemini_client.generate_perception(prompt, screenshot.artifact_path)
            except GeminiClientError:
                raise

            bg_writer.enqueue(raw_path, raw_output)

            perception, decision = self._parse_combined_output(raw_output, screenshot.artifact_path)

            # Quality gate — same checks as GeminiPerceptionService
            quality_metrics = _quality_metrics(perception)
            low_quality_reason = _low_quality_reason(perception, quality_metrics=quality_metrics)
            if low_quality_reason is not None:
                logger.warning(
                    "Combined perception low quality (%s), attempt %d/%d.",
                    low_quality_reason, attempt + 1, _MAX_COMBINED_RETRIES + 1,
                )
                if attempt < _MAX_COMBINED_RETRIES:
                    continue
                # Last attempt — try label inference salvage
                perception = _apply_weak_canonicalization(perception)

            # Cache decision only on primary calls, not retry-path perception
            if not self._perception_only:
                self._cached_decision = decision

            bg_writer.enqueue(parsed_path, json.dumps({
                "perception": perception.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }))

            self._last_debug_artifacts = ModelDebugArtifacts(
                prompt_artifact_path=str(prompt_path),
                raw_response_artifact_path=str(raw_path),
                parsed_artifact_path=str(parsed_path),
            )
            return perception

        # Should not reach here, but satisfy type checker
        raise PerceptionError("Combined perception failed after all retries.")

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        """Return the cached policy decision from the combined call."""
        if self._cached_decision is not None:
            decision = self._cached_decision
            self._cached_decision = None
            return decision
        raise PolicyError("No cached decision available from combined call")

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    def set_advisory_hints(self, hints: list[str]) -> None:
        self._advisory_hints = [hint for hint in hints if hint]

    def set_perception_only(self, value: bool) -> None:
        """When True, perceive() won't overwrite the cached policy decision."""
        self._perception_only = value

    def _render_prompt(self, state: AgentState, *, semantic_retry: bool = False) -> str:
        previous_summary = state.observation_history[-1].summary if state.observation_history else "none"
        # Build action history so the model knows what already happened
        action_log = self._format_action_history(state)
        hints_text = ""
        if self._advisory_hints:
            hints_text = "Advisory memory hints:\n" + "\n".join(f"- {hint}" for hint in self._advisory_hints)
            self._advisory_hints = []
        if action_log:
            hints_text = f"Actions already completed this run:\n{action_log}\n\n{hints_text}"
        prompt = self._prompt_template.format(
            intent=state.intent,
            current_subgoal=state.current_subgoal or "not set",
            step_count=state.step_count,
            previous_summary=previous_summary,
            retry_counts=json.dumps(state.retry_counts, sort_keys=True),
            advisory_hints=hints_text,
        )
        if semantic_retry:
            prompt += (
                "\n\nRetry: Previous response had low-quality perception. "
                "Ensure all interactive elements have labels, use null for missing fields, "
                "and include precise bounding box coordinates."
            )
        return prompt

    @staticmethod
    def _format_action_history(state: AgentState) -> str:
        """Format recent action history so the model knows what already succeeded."""
        if not state.action_history:
            return ""
        lines = []
        for i, action in enumerate(state.action_history[-6:], start=max(1, len(state.action_history) - 5)):
            a = action.action
            detail = action.detail or ""
            text = a.text or a.key or a.url or ""
            ok = "OK" if action.success else "FAILED"
            lines.append(f"  Step {i}: {a.action_type.value}({text}) -> {ok}: {detail[:80]}")
        return "\n".join(lines)

    @staticmethod
    def _parse_combined_output(raw_output: str, screenshot_path: str) -> tuple[ScreenPerception, PolicyDecision]:
        """Parse the combined JSON response into separate perception and policy objects."""
        cleaned = _strip_json_fence(raw_output)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PerceptionError("Combined output was not valid JSON.") from exc

        if not isinstance(parsed, dict):
            raise PerceptionError("Combined output must be a JSON object.")

        # Extract perception
        perception_data = parsed.get("perception")
        if not isinstance(perception_data, dict):
            raise PerceptionError("Combined output missing 'perception' object.")
        perception_data["capture_artifact_path"] = screenshot_path
        _normalize_visible_elements(perception_data)
        perception = parse_perception_output(
            json.dumps(perception_data), screenshot_path
        )

        # Extract policy decision
        action_data = parsed.get("action")
        if not isinstance(action_data, dict):
            raise PolicyError("Combined output missing 'action' object.")
        decision_data = {
            "action": action_data,
            "rationale": parsed.get("rationale", "combined call"),
            "confidence": parsed.get("confidence", 0.5),
            "active_subgoal": parsed.get("active_subgoal", "combined step"),
        }
        decision = parse_policy_output(json.dumps(decision_data))

        return perception, decision

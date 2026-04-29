"""Verifier service interface for post-execution checks."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import ValidationError

from src.clients.anthropic import AnthropicClientError
from src.clients.gemini import GeminiClient, GeminiClientError
from src.models.common import FailureCategory, LoopStage, StopReason
from src.models.execution import ExecutedAction
from src.models.logs import ModelDebugArtifacts
from src.models.perception import PageHint
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.state import AgentState
from src.models.verification import (
    VerificationFailureType,
    VerificationResult,
    VerificationStatus,
)
from src.store.background_writer import bg_writer

logger = logging.getLogger(__name__)


class VerifierService(ABC):
    """Typed interface for verification and stop-before-send checks."""

    @abstractmethod
    async def verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
    ) -> VerificationResult:
        """Verify whether the executed action achieved the expected outcome."""


class DeterministicVerifierService(VerifierService):
    """Deterministic verifier for general-purpose agent tasks."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        prompt_path: Path | None = None,
    ) -> None:
        self.gemini_client = gemini_client
        self.prompt_path = prompt_path or Path(__file__).resolve().parents[2] / "prompts" / "critic_prompt.txt"
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self._last_debug_artifacts: ModelDebugArtifacts | None = None

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    async def verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
    ) -> VerificationResult:
        """Evaluate typed inputs using deterministic rules."""
        action = decision.action
        latest_perception = state.observation_history[-1] if state.observation_history else None

        # Terminal-state check: look for visual evidence of the goal before checking
        # action success. This means the loop ends as soon as the goal is visible —
        # even if the triggering action was a WAIT or a mid-task CLICK.
        if latest_perception is not None:
            terminal = self.check_terminal_state(state, latest_perception)
            if terminal is not None:
                return terminal

        if action.action_type is ActionType.STOP:
            if self._suspicious_early_stop(state, decision, latest_perception):
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    expected_outcome_met=False,
                    stop_condition_met=False,
                    reason="Stop was proposed before the browser task showed enough evidence of completion.",
                    failure_type=VerificationFailureType.EXPECTED_OUTCOME_NOT_MET,
                    failure_category=FailureCategory.EXPECTED_OUTCOME_NOT_MET,
                    failure_stage=LoopStage.VERIFY,
                )
            if decision.active_subgoal == "stop for benchmark setup":
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    expected_outcome_met=False,
                    stop_condition_met=True,
                    reason="Benchmark precondition failed: authenticated start state was required.",
                    failure_type=VerificationFailureType.BENCHMARK_PRECONDITION_FAILED,
                    failure_category=FailureCategory.BENCHMARK_PRECONDITION_FAILED,
                    failure_stage=LoopStage.CHOOSE_ACTION,
                    stop_reason=StopReason.BENCHMARK_PRECONDITION_FAILED,
                )
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason="Intentional task stop boundary met.",
                failure_type=VerificationFailureType.STOP_BOUNDARY_REACHED,
                stop_reason=self._stop_reason_for_intentional_stop(state, decision),
            )

        # Self-terminating actions: when the chosen action IS the goal
        if executed_action.success and self._is_goal_completing_action(state, action):
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason=f"Goal-completing action {action.action_type.value} succeeded.",
                failure_type=VerificationFailureType.STOP_BOUNDARY_REACHED,
                stop_reason=StopReason.TASK_COMPLETED,
            )

        if not executed_action.success:
            return VerificationResult(
                status=VerificationStatus.FAILURE,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="Executed action reported failure.",
                failure_type=VerificationFailureType.ACTION_FAILED,
                failure_category=executed_action.failure_category or FailureCategory.EXECUTION_ERROR,
                failure_stage=executed_action.failure_stage or LoopStage.EXECUTE,
            )

        if self._is_page_loading(action, latest_perception):
            return VerificationResult(
                status=VerificationStatus.PENDING,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="Page appears to be loading or mid-transition; waiting before re-verifying.",
                failure_type=VerificationFailureType.PAGE_LOADING,
                failure_category=FailureCategory.UNCERTAIN_SCREEN_STATE,
                failure_stage=LoopStage.VERIFY,
            )

        if self._passive_wait_needs_more_signal(state, action, latest_perception):
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="Wait completed, but the browser task still lacks enough evidence of useful progress.",
                failure_type=VerificationFailureType.UNCERTAIN_SCREEN_STATE,
                failure_category=FailureCategory.UNCERTAIN_SCREEN_STATE,
                failure_stage=LoopStage.VERIFY,
            )

        model_result = await self._model_verify(state, decision, executed_action)
        if model_result is not None:
            return model_result

        if decision.confidence < 0.5:
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                expected_outcome_met=False,
                stop_condition_met=False,
                reason="Policy confidence is too low to confirm the expected outcome.",
                failure_type=VerificationFailureType.UNCERTAIN_SCREEN_STATE,
                failure_category=FailureCategory.UNCERTAIN_SCREEN_STATE,
                failure_stage=LoopStage.VERIFY,
                critic_fallback_reason="critic_unavailable_or_unusable",
            )

        return VerificationResult(
            status=VerificationStatus.SUCCESS,
            expected_outcome_met=True,
            stop_condition_met=False,
            reason="Executed action succeeded and the expected outcome is treated as met.",
            critic_fallback_reason="critic_unavailable_or_unusable",
        )

    async def _model_verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
    ) -> VerificationResult | None:
        screenshot_path = executed_action.artifact_path
        if not screenshot_path:
            self._last_debug_artifacts = None
            return None

        prompt = self._render_prompt(state, decision, executed_action)
        step_dir = Path(screenshot_path).resolve().parent
        debug_artifacts = self._artifact_paths(step_dir)
        bg_writer.enqueue(debug_artifacts.prompt_artifact_path, prompt)
        raw_output: str
        try:
            if hasattr(self.gemini_client, "generate_verification"):
                raw_output = await self.gemini_client.generate_verification(prompt, screenshot_path)
            elif hasattr(self.gemini_client, "generate_policy"):
                raw_output = await self.gemini_client.generate_policy(prompt)
            else:
                self._write_fallback_debug(debug_artifacts, "client_missing_verification_method")
                return None
        except (AnthropicClientError, GeminiClientError, NotImplementedError, RuntimeError) as exc:
            self._write_fallback_debug(debug_artifacts, f"critic_error: {exc}")
            return None
        except Exception as exc:
            logger.error("Unexpected error in critic call (possible code bug): %s: %s", type(exc).__name__, exc, exc_info=True)
            self._write_fallback_debug(debug_artifacts, f"critic_error_unexpected: {type(exc).__name__}: {exc}")
            return None

        bg_writer.enqueue(debug_artifacts.raw_response_artifact_path, raw_output)
        parsed = _parse_verification_output(raw_output)
        if parsed is None:
            self._write_fallback_debug(debug_artifacts, "critic_parse_failed", raw_output=raw_output)
            return None
        normalized = _normalize_verification_result(parsed).model_copy(
            update={"critic_model_used": True, "critic_fallback_reason": None}
        )
        bg_writer.enqueue(debug_artifacts.parsed_artifact_path, normalized.model_dump_json())
        self._last_debug_artifacts = debug_artifacts.model_copy(
            update={
                "usage_artifact_path": str(Path(debug_artifacts.parsed_artifact_path).resolve().parent / "verification_usage.json"),
                "usage": _latest_usage(self.gemini_client, Path(debug_artifacts.parsed_artifact_path).resolve().parent / "verification_usage.json"),
            }
        )
        return normalized

    def _render_prompt(
        self,
        state: AgentState,
        decision: PolicyDecision,
        executed_action: ExecutedAction,
    ) -> str:
        latest_perception = state.observation_history[-1] if state.observation_history else None
        return self._prompt_template.format(
            intent=state.intent,
            current_subgoal=decision.active_subgoal,
            action_json=decision.action.model_dump_json(),
            rationale=decision.rationale,
            confidence=decision.confidence,
            execution_detail=executed_action.detail,
            previous_summary=latest_perception.summary if latest_perception is not None else "none",
        )

    @staticmethod
    def _artifact_paths(step_dir: Path) -> ModelDebugArtifacts:
        step_dir.mkdir(parents=True, exist_ok=True)
        return ModelDebugArtifacts(
            prompt_artifact_path=str(step_dir / "verification_prompt.txt"),
            raw_response_artifact_path=str(step_dir / "verification_raw.txt"),
            parsed_artifact_path=str(step_dir / "verification_result.json"),
            retry_log_artifact_path=str(step_dir / "verification_retry_log.txt"),
            diagnostics_artifact_path=str(step_dir / "verification_diagnostics.json"),
            usage_artifact_path=str(step_dir / "verification_usage.json"),
        )

    def _write_fallback_debug(
        self,
        debug_artifacts: ModelDebugArtifacts,
        fallback_reason: str,
        *,
        raw_output: str | None = None,
    ) -> None:
        if raw_output is not None:
            bg_writer.enqueue(debug_artifacts.raw_response_artifact_path, raw_output)
        bg_writer.enqueue(
            debug_artifacts.diagnostics_artifact_path,
            json.dumps({"critic_model_used": False, "critic_fallback_reason": fallback_reason}),
        )
        self._last_debug_artifacts = debug_artifacts

    @staticmethod
    def _stop_reason_for_intentional_stop(state: AgentState, decision: PolicyDecision) -> StopReason:
        latest_perception = state.observation_history[-1] if state.observation_history else None
        if decision.active_subgoal == "verify_success" or (
            latest_perception is not None and latest_perception.page_hint is PageHint.FORM_SUCCESS
        ):
            return StopReason.FORM_SUBMITTED_SUCCESS
        if decision.active_subgoal == "stop for benchmark setup":
            return StopReason.BENCHMARK_PRECONDITION_FAILED
        return StopReason.TASK_COMPLETED

    # Navigation-triggering action types that can leave the page briefly blank.
    _NAV_ACTION_TYPES = frozenset([
        ActionType.CLICK,
        ActionType.PRESS_KEY,
        ActionType.HOTKEY,
        ActionType.NAVIGATE,
        ActionType.DOUBLE_CLICK,
    ])

    @staticmethod
    def _is_page_loading(action, latest_perception) -> bool:
        """Return True when the page looks mid-transition: sparse elements + unknown hint."""
        if latest_perception is None:
            return False
        if action.action_type not in DeterministicVerifierService._NAV_ACTION_TYPES:
            return False
        if latest_perception.page_hint is not PageHint.UNKNOWN:
            return False
        return len(latest_perception.visible_elements) <= 2

    @staticmethod
    def _passive_wait_needs_more_signal(
        state: AgentState,
        action: AgentAction,
        latest_perception,
    ) -> bool:
        if action.action_type is not ActionType.WAIT:
            return False
        if latest_perception is None:
            return False
        intent = state.intent.lower()
        if not any(token in intent for token in ("inspect", "look", "find", "open", "check", "view", "read")):
            return False
        if latest_perception.page_hint is not PageHint.UNKNOWN:
            return False
        return not latest_perception.visible_elements

    @staticmethod
    def _suspicious_early_stop(
        state: AgentState,
        decision: PolicyDecision,
        latest_perception,
    ) -> bool:
        if state.step_count > 1:
            return False
        if latest_perception is None:
            return False
        if latest_perception.page_hint is PageHint.FORM_SUCCESS:
            return False
        if latest_perception.visible_elements:
            return False
        if decision.active_subgoal in {"verify_success", "complete task"}:
            return True
        return latest_perception.page_hint is PageHint.UNKNOWN

    @staticmethod
    def _is_goal_completing_action(state: AgentState, action: AgentAction) -> bool:
        """Check if this action type directly fulfills the task intent."""
        intent = state.intent.lower()
        if action.action_type is ActionType.READ_TEXT and any(
            kw in intent for kw in ("read", "extract", "save", "get", "find", "fetch")
        ):
            return True
        if action.action_type is ActionType.READ_CLIPBOARD and ("clipboard" in intent or "read" in intent):
            return True
        if action.action_type is ActionType.HOVER and "hover" in intent:
            return True
        if action.action_type is ActionType.RIGHT_CLICK and ("right-click" in intent or "right click" in intent):
            return True
        if action.action_type is ActionType.DOUBLE_CLICK and "double-click" in intent:
            return True
        if action.action_type is ActionType.SCREENSHOT_REGION and ("screenshot" in intent or "capture" in intent):
            return True
        if action.action_type is ActionType.DRAG and "drag" in intent:
            return True
        if action.action_type is ActionType.WRITE_CLIPBOARD and ("clipboard" in intent or "copy" in intent):
            return True
        # Hotkey that matches the last phrase of the intent (e.g. "copy it with Ctrl+C")
        if action.action_type is ActionType.HOTKEY and action.key:
            key_lower = action.key.lower().replace("+", "")
            if key_lower in intent.replace("+", "").replace("-", ""):
                # Only if this is the LAST action in the intent
                last_phrase = intent.split(",")[-1].strip().lower() if "," in intent else intent
                if key_lower in last_phrase.replace("+", "").replace("-", ""):
                    return True
        return False

    def check_terminal_state(
        self,
        state: AgentState,
        perception,
    ) -> VerificationResult | None:
        """Check whether the current screen shows visual evidence of goal completion.

        Returns a terminal VerificationResult when the goal is confirmed, or None
        when there is no conclusive evidence yet. This is a pure visual predicate —
        it does not inspect whether the last action succeeded, only whether the
        terminal state is now visually present.

        Checks (in priority order):
        1. Generic success page hint or benchmark success tokens
        2. Goal-specific terminal signals extracted from the intent
        """
        if self._task_success_visible(perception, state.benchmark):
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason="Terminal state confirmed: task success is visually present on screen.",
                failure_type=VerificationFailureType.STOP_BOUNDARY_REACHED,
                stop_reason=StopReason.FORM_SUBMITTED_SUCCESS,
            )

        # Intent-derived terminal signals: extract goal keywords and check visibility
        terminal_signal = self._intent_terminal_signal(state.intent, perception)
        if terminal_signal is not None:
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=True,
                reason=f"Terminal state confirmed: '{terminal_signal}' is visible — goal achieved.",
                failure_type=VerificationFailureType.STOP_BOUNDARY_REACHED,
                stop_reason=StopReason.TASK_COMPLETED,
            )

        return None

    @staticmethod
    def _intent_terminal_signal(intent: str, perception) -> str | None:
        """Return the matched terminal signal token if the intent's goal is visually confirmed."""
        intent_lower = intent.lower()
        summary_lower = perception.summary.lower()
        element_names = " ".join(
            e.primary_name.lower() for e in perception.visible_elements
        )

        # Build goal-specific terminal indicators from the intent
        _TERMINAL_PATTERNS: list[tuple[str, list[str]]] = [
            ("submitted", ["submitted", "submission received", "thank you", "success"]),
            ("saved", ["saved", "changes saved", "save successful"]),
            ("created", ["created", "issue created", "record created", "added"]),
            ("deleted", ["deleted", "removed", "record deleted"]),
            ("uploaded", ["uploaded", "upload complete", "file uploaded"]),
            ("sent", ["sent", "message sent", "email sent"]),
            ("logged in", ["dashboard", "welcome", "logged in", "sign out", "log out"]),
            ("logged out", ["signed out", "logged out", "login", "sign in"]),
            ("searched", ["search results", "results for", "no results"]),
            ("found", ["found", "result", "match"]),
        ]
        for intent_keyword, visual_signals in _TERMINAL_PATTERNS:
            if intent_keyword not in intent_lower:
                continue
            for signal in visual_signals:
                if signal in summary_lower or signal in element_names:
                    return signal

        return None

    @staticmethod
    def _task_success_visible(perception, benchmark: str | None) -> bool:
        from src.benchmarks.registry import BENCHMARK_REGISTRY
        if perception.page_hint is PageHint.FORM_SUCCESS:
            return True
        success_tokens = BENCHMARK_REGISTRY.get_success_tokens(benchmark)
        if not success_tokens:
            return False
        if any(token in perception.summary.lower() for token in success_tokens):
            return True
        return any(any(token in element.primary_name.lower() for token in success_tokens) for element in perception.visible_elements)

def _parse_verification_output(raw_output: str) -> VerificationResult | None:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    try:
        return VerificationResult.model_validate(payload)
    except ValidationError:
        return None


def _normalize_verification_result(result: VerificationResult) -> VerificationResult:
    updates: dict[str, object] = {}
    if result.status is VerificationStatus.FAILURE:
        if result.failure_type is None:
            updates["failure_type"] = VerificationFailureType.EXPECTED_OUTCOME_NOT_MET
        if result.failure_category is None:
            updates["failure_category"] = FailureCategory.EXPECTED_OUTCOME_NOT_MET
        if result.failure_stage is None:
            updates["failure_stage"] = LoopStage.VERIFY
    elif result.status in {VerificationStatus.UNCERTAIN, VerificationStatus.PENDING}:
        if result.failure_type is None:
            updates["failure_type"] = (
                VerificationFailureType.PAGE_LOADING
                if result.status is VerificationStatus.PENDING
                else VerificationFailureType.UNCERTAIN_SCREEN_STATE
            )
        if result.failure_category is None:
            updates["failure_category"] = FailureCategory.UNCERTAIN_SCREEN_STATE
        if result.failure_stage is None:
            updates["failure_stage"] = LoopStage.VERIFY
    elif result.status is VerificationStatus.SUCCESS and result.stop_condition_met and result.stop_reason is None:
        updates["stop_reason"] = StopReason.TASK_COMPLETED
    return result.model_copy(update=updates) if updates else result


def _latest_usage(client: GeminiClient, usage_artifact_path: Path):
    if not hasattr(client, "latest_usage"):
        return None
    usage = client.latest_usage()
    if usage is not None:
        bg_writer.enqueue(usage_artifact_path, usage.model_dump_json(indent=2))
    return usage

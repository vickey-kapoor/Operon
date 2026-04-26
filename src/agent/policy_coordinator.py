"""Thin coordinator that runs explicit rules before LLM-backed policy."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.agent.policy import PolicyService
from src.agent.policy_rules import PolicyRuleEngine
from src.models.common import FailureCategory
from src.models.episode import Episode, EpisodeReplayState
from src.models.logs import ModelDebugArtifacts
from src.models.memory import MemoryHint
from src.models.perception import ScreenPerception
from src.models.policy import ActionType, PolicyDecision
from src.models.selector import SelectorTrace
from src.models.state import AgentState
from src.store.background_writer import bg_writer
from src.store.memory import MemoryStore, normalize_intent

logger = logging.getLogger(__name__)


class PolicyCoordinator(PolicyService):
    """Run explicit rules first, then fall back to the existing policy service."""

    def __init__(
        self,
        *,
        delegate: PolicyService,
        memory_store: MemoryStore,
        rule_engine: PolicyRuleEngine | None = None,
    ) -> None:
        self.delegate = delegate
        self.memory_store = memory_store
        self.rule_engine = rule_engine or PolicyRuleEngine()
        self._last_debug_artifacts: ModelDebugArtifacts | None = None
        self._active_episode: Episode | None = None
        self._replay_state: EpisodeReplayState | None = None
        # Single-step hint cache: avoids re-loading memory JSONL twice per step.
        self._cached_hints: list[MemoryHint] | None = None
        self._cached_hints_key: tuple | None = None

    def _get_hints(self, state: AgentState, perception: ScreenPerception | None) -> list[MemoryHint]:
        """Fetch hints, using a single-step in-memory cache to avoid double JSONL reads."""
        benchmark = state.benchmark or "generic_task"
        page_hint = perception.page_hint if perception else None
        cache_key = (benchmark, page_hint, state.current_subgoal, self._recent_failure_category(state))
        if self._cached_hints is not None and self._cached_hints_key == cache_key:
            return self._cached_hints
        hints = self.memory_store.get_hints(
            benchmark=benchmark,
            page_hint=page_hint,
            subgoal=state.current_subgoal,
            recent_failure_category=self._recent_failure_category(state),
        )
        self._cached_hints = hints
        self._cached_hints_key = cache_key
        return hints

    def prepare_hints(self, state: AgentState, perception: ScreenPerception) -> None:
        """Fetch memory hints and inject them into the delegate BEFORE perceive().

        In combined mode the Gemini call happens during perceive(), so hints
        must be set before that call.  In separate mode this is a harmless no-op
        because choose_action() will re-fetch and set hints anyway.
        """
        # Clear the step cache so this step's hints are freshly loaded.
        self._cached_hints = None
        self._cached_hints_key = None
        memory_hints = self._get_hints(state, perception)
        if hasattr(self.delegate, "add_advisory_hints"):
            self.delegate.add_advisory_hints([hint.hint for hint in memory_hints], source="memory", run_id=state.run_id)

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        # Inject episode advisory hint if a matching trajectory exists
        self._try_episode_hint(state, perception)

        memory_hints = self._get_hints(state, perception)
        benchmark = state.benchmark

        decision = self.rule_engine.choose_action(state, perception, memory_hints, benchmark_name=benchmark)
        selector_traces = self.rule_engine.latest_selector_traces()
        if decision is not None:
            if _is_rule_success_stop(decision):
                confirmed = await self._confirm_success(state, perception, decision)
                if confirmed is not decision:
                    # LLM overrode — propagate its decision and its debug artifacts
                    if hasattr(self.delegate, "latest_debug_artifacts"):
                        self._last_debug_artifacts = self.delegate.latest_debug_artifacts()
                    return confirmed
            self._last_debug_artifacts = self._write_rule_debug_artifacts(state, perception, memory_hints, decision, selector_traces)
            return decision

        if hasattr(self.delegate, "add_advisory_hints"):
            self.delegate.add_advisory_hints([hint.hint for hint in memory_hints], source="memory", run_id=state.run_id)

        decision = await self.delegate.choose_action(state, perception)
        decision = await self._reject_hallucinated_target(state, perception, decision)
        decision = await self._reject_premature_stop(state, perception, decision)
        if hasattr(self.delegate, "latest_debug_artifacts"):
            self._last_debug_artifacts = self.delegate.latest_debug_artifacts()
        self._last_debug_artifacts = self._attach_selector_trace(perception, self._last_debug_artifacts, selector_traces)
        return decision

    async def _confirm_success(
        self,
        state: AgentState,
        perception: ScreenPerception,
        rule_stop: PolicyDecision,
    ) -> PolicyDecision:
        """Ask the delegate LLM to confirm or override a rule-sourced success stop.

        Returns the original rule_stop when the LLM agrees (preserves rationale and
        subgoal for the verifier). Returns the LLM's recovery decision when it
        disagrees. Falls back to rule_stop on any LLM error so a transient API
        failure never blocks task completion.
        """
        hint = (
            "CONFIRMATION REQUIRED: The success-detection rule believes the task is "
            f"complete (page_hint={perception.page_hint.value!r}, "
            f"summary={perception.summary!r}). "
            "Examine the current screen carefully. "
            "If the task is genuinely complete, respond with STOP. "
            "If this is an error message, a partial state, or the task is not done, "
            "respond with the appropriate recovery action instead."
        )
        if hasattr(self.delegate, "add_advisory_hints"):
            self.delegate.add_advisory_hints([hint], source="success_confirm", run_id=state.run_id)

        try:
            llm_decision = await self.delegate.choose_action(state, perception)
        except Exception as exc:
            logger.warning(
                "Success confirmation LLM call failed (%s: %s) — accepting rule stop.",
                type(exc).__name__, exc,
            )
            return rule_stop

        if llm_decision.action.action_type is ActionType.STOP:
            logger.info(
                "Success confirmed by LLM policy (page_hint=%r).",
                perception.page_hint.value,
            )
            return rule_stop

        logger.warning(
            "LLM rejected rule-based success stop (page_hint=%r). Recovery: %s.",
            perception.page_hint.value,
            llm_decision.action.action_type.value,
        )
        return llm_decision

    async def _reject_premature_stop(
        self,
        state: AgentState,
        perception: ScreenPerception,
        decision: PolicyDecision,
    ) -> PolicyDecision:
        """Re-plan once if the LLM issues STOP before taking any meaningful action.

        A STOP is premature when the agent has performed zero substantive actions
        (TYPE, CLICK, NAVIGATE, HOTKEY, PRESS_KEY, LAUNCH_APP) in the current run.
        This catches the pattern where the LLM reads information on an already-open
        page and declares the task done without having done any work itself.

        The re-plan hint is derived from the action history gap — no domain
        knowledge or static keyword lists are used.
        """
        if decision.action.action_type is not ActionType.STOP:
            return decision

        substantive = {
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.TYPE,
            ActionType.PRESS_KEY,
            ActionType.HOTKEY,
            ActionType.NAVIGATE,
            ActionType.LAUNCH_APP,
            ActionType.DRAG,
            ActionType.SELECT,
        }
        actions_taken = [h.action.action_type for h in state.action_history]
        if any(a in substantive for a in actions_taken):
            return decision

        logger.warning(
            "Policy issued STOP with no prior substantive actions (history=%s). Re-planning.",
            [a.value for a in actions_taken] or "[]",
        )
        correction = (
            "CORRECTION: you issued STOP before performing any actions. "
            "The task has not been started yet — you have not navigated, typed, clicked, "
            "or taken any action to complete it. "
            f"Original task: {state.intent!r}. "
            "Plan and execute the next concrete step toward completing this task."
        )
        if hasattr(self.delegate, "add_advisory_hints"):
            self.delegate.add_advisory_hints([correction], source="validation", run_id=state.run_id)

        replanned = await self.delegate.choose_action(state, perception)
        return replanned

    async def _reject_hallucinated_target(
        self,
        state: AgentState,
        perception: ScreenPerception,
        decision: PolicyDecision,
    ) -> PolicyDecision:
        """Re-plan once if the LLM produced a target_element_id absent from perception."""
        target_id = decision.action.target_element_id
        if target_id is None:
            return decision
        known_ids = {e.element_id for e in perception.visible_elements}
        if target_id in known_ids:
            return decision

        logger.warning(
            "Policy hallucinated element_id %r (not in current perception). Re-planning.",
            target_id,
        )
        known_list = ", ".join(sorted(known_ids)) if known_ids else "none"
        correction = (
            f"CORRECTION: element_id '{target_id}' does not exist on the current screen. "
            f"You MUST use one of these actual element IDs: [{known_list}]. "
            "Do not invent element IDs."
        )
        if hasattr(self.delegate, "add_advisory_hints"):
            self.delegate.add_advisory_hints([correction], source="validation", run_id=state.run_id)

        replanned = await self.delegate.choose_action(state, perception)
        if (
            replanned.action.target_element_id is not None
            and replanned.action.target_element_id not in known_ids
        ):
            logger.warning(
                "Re-planned policy still references unknown element_id %r. Proceeding anyway.",
                replanned.action.target_element_id,
            )
        return replanned

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    @staticmethod
    def _recent_failure_category(state: AgentState) -> FailureCategory | None:
        if state.verification_history:
            failure = state.verification_history[-1].failure_category
            if failure is not None:
                return failure
        if state.action_history:
            return state.action_history[-1].failure_category
        return None

    def _write_rule_debug_artifacts(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
        decision: PolicyDecision,
        selector_traces: list[SelectorTrace],
    ) -> ModelDebugArtifacts:
        step_dir = Path(perception.capture_artifact_path).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = step_dir / "policy_prompt.txt"
        raw_path = step_dir / "policy_raw.txt"
        parsed_path = step_dir / "policy_decision.json"
        selector_trace_path = self._write_selector_trace(step_dir, selector_traces)
        bg_writer.enqueue(prompt_path, self._render_rule_context(state, perception, memory_hints))
        bg_writer.enqueue(raw_path, json.dumps({
            "source": "rule",
            "decision_rationale": decision.rationale,
            "subgoal": decision.active_subgoal,
        }))
        bg_writer.enqueue(parsed_path, decision.model_dump_json())
        return ModelDebugArtifacts(
            prompt_artifact_path=str(prompt_path),
            raw_response_artifact_path=str(raw_path),
            parsed_artifact_path=str(parsed_path),
            selector_trace_artifact_path=str(selector_trace_path) if selector_trace_path is not None else None,
        )

    @staticmethod
    def _render_rule_context(
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> str:
        lines = [
            "Policy coordinator selected a rule-based action.",
            f"intent: {state.intent}",
            f"subgoal: {state.current_subgoal or 'not set'}",
            f"page_hint: {perception.page_hint.value}",
            "memory_hints:",
        ]
        if not memory_hints:
            lines.append("- none")
        else:
            lines.extend(f"- {hint.key}: {hint.hint}" for hint in memory_hints)
        return "\n".join(lines)

    @staticmethod
    def _write_selector_trace(step_dir: Path, selector_traces: list[SelectorTrace]) -> Path | None:
        if not selector_traces:
            return None
        path = step_dir / "selector_trace.json"
        payload = [trace.model_dump(mode="json") for trace in selector_traces]
        bg_writer.enqueue(path, json.dumps(payload))
        return path

    def _attach_selector_trace(
        self,
        perception: ScreenPerception,
        debug_artifacts: ModelDebugArtifacts | None,
        selector_traces: list[SelectorTrace],
    ) -> ModelDebugArtifacts | None:
        if debug_artifacts is None:
            return None
        selector_trace_path = self._write_selector_trace(Path(perception.capture_artifact_path).resolve().parent, selector_traces)
        if selector_trace_path is None:
            return debug_artifacts
        return debug_artifacts.model_copy(update={"selector_trace_artifact_path": str(selector_trace_path)})

    # ------------------------------------------------------------------
    # Episode replay (advisory hints)
    # ------------------------------------------------------------------

    def _try_episode_hint(self, state: AgentState, perception: ScreenPerception) -> None:
        """Inject an advisory hint from a matching episode trajectory."""
        # Initialize replay on first step
        if self._replay_state is None and state.step_count <= 1:
            benchmark = state.benchmark or "generic_task"
            norm = normalize_intent(state.intent)
            episode = self.memory_store.get_episode(norm, benchmark)
            if episode is not None:
                self._active_episode = episode
                self._replay_state = EpisodeReplayState(episode_id=episode.episode_id)
                logger.info(
                    "Episode replay activated: %s (%d steps, %dx success)",
                    episode.episode_id[:8],
                    len(episode.steps),
                    episode.success_count,
                )

        if self._replay_state is None or not self._replay_state.active:
            return

        episode = self._active_episode
        if episode is None:
            return

        idx = self._replay_state.current_step_index
        if idx >= len(episode.steps):
            self._replay_state.active = False
            return

        expected = episode.steps[idx]

        # Divergence check: page_hint mismatch
        if perception.page_hint != expected.page_hint:
            self._replay_state.deviations += 1
            if self._replay_state.deviations >= self._replay_state.max_deviations:
                self._replay_state.active = False
                logger.info("Episode replay deactivated: too many divergences")
                return

        # Build advisory hint
        total = len(episode.steps)
        parts = [f"Episode replay (step {idx + 1}/{total}):"]
        parts.append(f"perform {expected.action_type.value}")
        if expected.target_description:
            parts.append(f"on '{expected.target_description}'")
        if expected.text:
            parts.append(f"with text '{expected.text}'")
        if expected.key:
            parts.append(f"key '{expected.key}'")
        parts.append(f"[subgoal: {expected.subgoal}]")
        parts.append("Follow this if the screen matches.")
        hint = " ".join(parts)

        if hasattr(self.delegate, "add_advisory_hints"):
            self.delegate.add_advisory_hints([hint], source="episode", run_id=state.run_id)

        self._replay_state.current_step_index += 1


def _is_rule_success_stop(decision: PolicyDecision) -> bool:
    """True when a rule engine decision is a success-flagged STOP needing LLM confirmation."""
    return (
        decision.action.action_type is ActionType.STOP
        and decision.active_subgoal == "verify_success"
    )

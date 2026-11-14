"""Thin coordinator that runs explicit rules before LLM-backed policy."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from operon.agent.policy import PolicyService
from operon.agent.policy.rules import PolicyRuleEngine
from operon.models.common import FailureCategory
from operon.models.logs import ModelDebugArtifacts
from operon.models.memory import MemoryHint
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState
from operon.store.background_writer import bg_writer
from operon.store.memory import MemoryStore

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
        # Single-step hint cache: avoids re-loading memory JSONL twice per step.
        self._cached_hints: list[MemoryHint] | None = None
        self._cached_hints_key: tuple | None = None
        self._hints_prepared_for: str | None = None

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

    def reset_run_context(self, run_id: str) -> None:
        """Reset per-run coordinator state for a new task.

        Called at the start of each new run so hint caches and form-fill state
        from a previous task do not bleed into the new one.

        Auth-related browser state (cookies, session tokens) is NOT cleared here —
        that is managed at the browser-executor level and intentionally preserved
        across runs of authenticated benchmarks (e.g. WebArena/GitLab).

        Clears:
        - Hint cache (benchmark/page_hint context changes per run)
        - Last rule trace (stale rule context from a prior run)
        """
        self._cached_hints = None
        self._cached_hints_key = None
        logger.debug("PolicyCoordinator: run context reset for run_id=%r", run_id)

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
            # The rule trace goes in now too: perceive() is this step's model call,
            # so a trace added after it would reach the model a step late.
            if state.last_rule_trace:
                self.delegate.add_advisory_hints([state.last_rule_trace], source="rule_trace", run_id=state.run_id)
        self._hints_prepared_for = state.run_id

    async def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        self._delegate_decision: PolicyDecision | None = None
        decision = await self._choose_action(state, perception)
        own = self._delegate_decision
        if own is not None and own.action.action_type is ActionType.WAIT_FOR_USER and decision is not own:
            # The model asked for a human (e.g. a provider safety check): no rule or
            # guard may run something else in its place, or the pause is skipped.
            logger.info("Keeping the model's request for a human over %s.", _describe(decision))
            decision = own
        same_stop = own is not None and own.action.action_type is ActionType.STOP \
            and decision.action.action_type is ActionType.STOP
        if decision is not own and not same_stop:
            # A rule or guard ran something else: a backend holding the model's tool
            # call tells the model, so it does not hear its own action reported as run.
            self.note_replaced(state.run_id, _describe(decision))
        return decision

    async def _ask_delegate(self, state: AgentState, perception: ScreenPerception) -> PolicyDecision:
        if getattr(self, "_delegate_decision", None) is not None:
            # Asked again in the same step (a guard rejected the last proposal): that
            # proposal never ran, so the model must not hear it reported as done.
            self.note_rejected(state.run_id, "Operon's harness rejected it and asked you again")
        decision = await self.delegate.choose_action(state, perception)
        self._delegate_decision = decision
        return decision

    async def _choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision:
        memory_hints = self._get_hints(state, perception)

        decision = self.rule_engine.choose_action(state, perception, memory_hints)
        if decision is not None:
            # Stamp which rule fired so the loop can build a rule trace for the next LLM prompt
            fired_rule = self.rule_engine.last_fired_rule_name()
            if fired_rule and decision.rule_name is None:
                decision = decision.model_copy(update={"rule_name": fired_rule})
            if _is_rule_success_stop(decision):
                confirmed = await self._confirm_success(state, perception, decision)
                if confirmed is not decision:
                    # LLM overrode — propagate its decision and its debug artifacts
                    if hasattr(self.delegate, "latest_debug_artifacts"):
                        self._last_debug_artifacts = self.delegate.latest_debug_artifacts()
                    return confirmed
            self._last_debug_artifacts = self._write_rule_debug_artifacts(state, perception, memory_hints, decision)
            return decision

        # prepare_hints() already queued this step's memory hints and rule trace
        # before perceive(); queueing them again would repeat them next step.
        prepared = getattr(self, "_hints_prepared_for", None) == state.run_id
        self._hints_prepared_for = None
        if not prepared and hasattr(self.delegate, "add_advisory_hints"):
            self.delegate.add_advisory_hints([hint.hint for hint in memory_hints], source="memory", run_id=state.run_id)

            # Rule-Augmented Generation: if a rule fired last step, inject its outcome
            # so the LLM planner knows what was tried and whether it succeeded.
            if state.last_rule_trace:
                self.delegate.add_advisory_hints([state.last_rule_trace], source="rule_trace", run_id=state.run_id)

        decision = await self._ask_delegate(state, perception)
        decision = await self._reject_premature_stop(state, perception, decision)
        if hasattr(self.delegate, "latest_debug_artifacts"):
            self._last_debug_artifacts = self.delegate.latest_debug_artifacts()
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
        resample = getattr(self.delegate, "supports_resampling", None)
        if resample is not None and not resample(state.run_id):
            # The delegate's decision for this screen was made in perceive() and it
            # cannot be asked again (Computer Use): that independent decision is
            # the confirmation. A queued question would only leak into next step.
            try:
                llm_decision = await self._ask_delegate(state, perception)
            except Exception as exc:  # noqa: BLE001 - never block completion on it
                logger.warning("Success confirmation failed (%s: %s) — accepting rule stop.", type(exc).__name__, exc)
                return rule_stop
            return rule_stop if llm_decision.action.action_type is ActionType.STOP else llm_decision
        # Drop the decision perceive() cached so the question gets a fresh answer.
        discard = getattr(self.delegate, "discard_cached_decision", None)
        if discard is not None:
            discard(state.run_id)
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
            llm_decision = await self._ask_delegate(state, perception)
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
            ActionType.TRIPLE_CLICK,
            ActionType.RIGHT_CLICK,
            ActionType.UPLOAD_FILE,
            ActionType.UPLOAD_FILE_NATIVE,
            ActionType.WRITE_CLIPBOARD,
            ActionType.TYPE,
            ActionType.PRESS_KEY,
            ActionType.HOTKEY,
            ActionType.NAVIGATE,
            ActionType.LAUNCH_APP,
            ActionType.DRAG,
            ActionType.SELECT,
        }
        # A batch's children count one by one (a run done in batches did work).
        actions_taken = [leaf.action_type for h in state.action_history for leaf in _leaves(h.action)]
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

        replanned = await self._ask_delegate(state, perception)
        return replanned

    def latest_debug_artifacts(self) -> ModelDebugArtifacts | None:
        return self._last_debug_artifacts

    def supports_resampling(self, run_id: str = "") -> bool:
        return self.delegate.supports_resampling(run_id)

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
    ) -> ModelDebugArtifacts:
        step_dir = Path(perception.capture_artifact_path).resolve().parent
        step_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = step_dir / "policy_prompt.txt"
        raw_path = step_dir / "policy_raw.txt"
        parsed_path = step_dir / "policy_decision.json"
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


def _is_rule_success_stop(decision: PolicyDecision) -> bool:
    """True when a rule engine decision is a success-flagged STOP needing LLM confirmation."""
    return (
        decision.action.action_type is ActionType.STOP
        and decision.active_subgoal == "verify_success"
    )


def _describe(decision: PolicyDecision) -> str:
    """A short name for an action the harness ran in the model's place."""
    action = decision.action
    what = action.action_type.value
    if action.x is not None and action.y is not None:
        what += f" at ({action.x}, {action.y})"
    by = f"rule {decision.rule_name}" if decision.rule_name else "a harness guard"
    return f"{what}, chosen by {by}"


def _leaves(action: AgentAction) -> list[AgentAction]:
    """A batch's leaf actions (nested batches flattened); any other action itself."""
    if action.action_type is not ActionType.BATCH:
        return [action]
    return [leaf for child in action.actions or [] for leaf in _leaves(child)]

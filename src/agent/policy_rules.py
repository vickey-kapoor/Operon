"""Explicit rule-first policy checks for deterministic agent cases."""

from __future__ import annotations

import re
from collections.abc import Callable

from src.agent.selector import DeterministicTargetSelector
from src.models.common import FailureCategory
from src.models.memory import MemoryHint
from src.models.perception import PageHint, ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.selector import (
    SelectorConfidenceBand,
    SelectorTrace,
    TargetEvidence,
    TargetIntent,
    TargetIntentAction,
)
from src.models.state import AgentState
from src.models.verification import VerificationStatus

# A benchmark rule plugin is any callable that matches this signature.
# Return a PolicyDecision to short-circuit the engine, or None to pass through.
BenchmarkRulePlugin = Callable[
    [AgentState, ScreenPerception, list[MemoryHint]],
    PolicyDecision | None,
]


# ---------------------------------------------------------------------------
# Gmail-specific rules
# ---------------------------------------------------------------------------

def gmail_login_page_guardrail(
    state: AgentState,
    perception: ScreenPerception,
    memory_hints: list[MemoryHint],
) -> PolicyDecision | None:
    if perception.page_hint != "google_sign_in":
        return None
    if _has_hint(memory_hints, "authenticated_start_required"):
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.STOP),
            rationale="Benchmark requires an authenticated Gmail start state; login/auth screens are out of scope.",
            confidence=1.0,
            active_subgoal="stop for benchmark setup",
        )
    return PolicyDecision(
        action=AgentAction(
            action_type=ActionType.WAIT_FOR_USER,
            text="Login page detected. Please sign in using the browser window, then click Resume.",
        ),
        rationale="Login/auth page detected — requires user credentials that Operon cannot provide autonomously.",
        confidence=1.0,
        active_subgoal="wait for user authentication",
    )


def gmail_compose_already_visible_rule(
    state: AgentState,
    perception: ScreenPerception,
    memory_hints: list[MemoryHint],
) -> PolicyDecision | None:
    current_subgoal = (state.current_subgoal or "").lower()
    if "open compose" not in current_subgoal and "compose" not in current_subgoal:
        return None
    if perception.page_hint != "gmail_compose" and not _compose_form_visible(perception):
        return None
    target = _preferred_compose_input(perception)
    if target is None:
        return None
    return _click_decision(target, "Compose form is already visible; move to the form instead of reopening compose.")


# ---------------------------------------------------------------------------
# Form benchmark-specific rules
# ---------------------------------------------------------------------------

def form_submit_when_ready_rule(
    state: AgentState,
    perception: ScreenPerception,
    memory_hints: list[MemoryHint],
) -> PolicyDecision | None:
    if perception.page_hint is not PageHint.FORM_PAGE:
        return None
    if not _form_fields_completed(state, perception):
        return None
    submit_button = _submit_button(perception)
    if submit_button is None:
        return None
    return PolicyDecision(
        action=AgentAction(
            action_type=ActionType.CLICK,
            target_element_id=submit_button.element_id,
            x=submit_button.x + max(1, submit_button.width // 2),
            y=submit_button.y + max(1, submit_button.height // 2),
        ),
        rationale="Required form fields are already filled; submit the form.",
        confidence=0.97,
        active_subgoal="submit_form",
    )


# ---------------------------------------------------------------------------
# Default plugin registry — benchmark name → plugins
# ---------------------------------------------------------------------------

BENCHMARK_PLUGINS: dict[str, list[BenchmarkRulePlugin]] = {
    "gmail_draft_authenticated": [
        gmail_login_page_guardrail,
        gmail_compose_already_visible_rule,
    ],
    "auth_free_form": [
        form_submit_when_ready_rule,
    ],
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PolicyRuleEngine:
    """Small explicit rule engine that runs before the LLM-backed policy.

    Engine primitives run unconditionally for every benchmark.
    Benchmark-specific plugins are registered per benchmark name and only
    run when that benchmark is active.
    """

    def __init__(
        self,
        selector: DeterministicTargetSelector | None = None,
        plugins: dict[str, list[BenchmarkRulePlugin]] | None = None,
    ) -> None:
        self.selector = selector or DeterministicTargetSelector()
        self._plugins = plugins if plugins is not None else dict(BENCHMARK_PLUGINS)
        self._latest_selector_traces: list[SelectorTrace] = []
        self._cached_intermediates: tuple[list[UIElement], list[list[UIElement]]] | None = None

    def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
        benchmark_name: str | None = None,
    ) -> PolicyDecision | None:
        self._latest_selector_traces = []
        self._cached_intermediates = (
            self.selector._label_like_text_candidates(perception),
            self.selector._visual_groups(perception),
        )

        # Benchmark-specific plugins run first (higher priority / more specific)
        if benchmark_name is not None:
            for plugin in self._plugins.get(benchmark_name, []):
                decision = plugin(state, perception, memory_hints)
                if decision is not None:
                    return decision

        # Engine primitives — always active
        return (
            self._task_success_stop_rule(perception)
            or self._avoid_identical_type_retry(state, perception, memory_hints)
            or self._search_query_rule(state, perception)
            or self._focus_before_type_rule(state, perception, memory_hints)
        )

    def register_plugins(self, benchmark_name: str, plugins: list[BenchmarkRulePlugin]) -> None:
        """Register additional plugins for a benchmark at runtime."""
        self._plugins.setdefault(benchmark_name, []).extend(plugins)

    def latest_selector_traces(self) -> list[SelectorTrace]:
        return list(self._latest_selector_traces)

    # ------------------------------------------------------------------
    # Engine primitives
    # ------------------------------------------------------------------

    def _avoid_identical_type_retry(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> PolicyDecision | None:
        if not _has_hint(memory_hints, "avoid_identical_type_retry"):
            return None

        last_action = state.action_history[-1].action if state.action_history else None
        if last_action is None or last_action.action_type is not ActionType.TYPE:
            return None

        last_failure = None
        if state.verification_history:
            last_failure = state.verification_history[-1].failure_category
        if last_failure is None and state.action_history:
            last_failure = state.action_history[-1].failure_category
        if last_failure not in {
            FailureCategory.EXECUTION_TARGET_NOT_FOUND,
            FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        }:
            return None

        if last_action.target_element_id is None:
            return None
        if not _has_recent_target_failure_signal(state, last_action.target_element_id, last_failure):
            return None

        target = self._resolve_target_element(
            perception,
            target_element_id=last_action.target_element_id,
            subgoal=state.current_subgoal,
        )
        if target is None:
            target = self._fallback_input_target(perception, state.current_subgoal, last_action.target_element_id)

        if target is not None and perception.focused_element_id != target.element_id:
            return _click_decision(target, f"Re-establish focus on {target.primary_name} before retrying type.")

        return PolicyDecision(
            action=AgentAction(action_type=ActionType.WAIT, wait_ms=500),
            rationale="Previous type action failed on the same target; avoid repeating the identical type action blindly.",
            confidence=0.93,
            active_subgoal=f"re-assess {last_action.target_element_id or 'input'}",
        )

    def _task_success_stop_rule(self, perception: ScreenPerception) -> PolicyDecision | None:
        if perception.page_hint is not PageHint.FORM_SUCCESS and not _has_success_signal(perception):
            return None
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.STOP),
            rationale="Task success state is already visible.",
            confidence=1.0,
            active_subgoal="verify_success",
        )

    def _focus_before_type_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> PolicyDecision | None:
        if not _has_hint(memory_hints, "click_before_type"):
            return None
        target = self._resolve_target_element(
            perception,
            target_element_id=None,
            subgoal=state.current_subgoal,
        )
        if target is None or perception.focused_element_id == target.element_id:
            return None
        return _click_decision(target, f"Focus {target.primary_name} before typing because input focus is not established.")

    def _search_query_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision | None:
        query = _extract_search_query(state)
        if query is None:
            return None

        search_input = self._search_input_target(perception)
        if search_input is not None:
            if perception.focused_element_id != search_input.element_id:
                return _click_decision(
                    search_input,
                    f"Focus {search_input.primary_name} so the search query can be entered.",
                )
            return PolicyDecision(
                action=AgentAction(
                    action_type=ActionType.TYPE,
                    target_element_id=search_input.element_id,
                    text=query,
                    press_enter=True,
                ),
                rationale=f"Type the search query '{query}' into {search_input.primary_name} and submit it.",
                confidence=0.97,
                active_subgoal=f"search for {query}",
            )

        search_trigger = _search_trigger_target(perception)
        if search_trigger is None:
            return None
        return PolicyDecision(
            action=AgentAction(
                action_type=ActionType.CLICK,
                target_element_id=search_trigger.element_id,
                x=search_trigger.x + max(1, search_trigger.width // 2),
                y=search_trigger.y + max(1, search_trigger.height // 2),
            ),
            rationale=f"Activate {search_trigger.primary_name} so the search input becomes available.",
            confidence=0.94,
            active_subgoal=f"focus {search_trigger.element_id}",
        )

    # ------------------------------------------------------------------
    # Selector helpers
    # ------------------------------------------------------------------

    def _resolve_target_element(
        self,
        perception: ScreenPerception,
        *,
        target_element_id: str | None,
        subgoal: str | None,
    ) -> UIElement | None:
        if target_element_id is not None:
            target = next((e for e in perception.visible_elements if e.element_id == target_element_id), None)
            if target is not None and target.element_type is UIElementType.INPUT and target.usable_for_targeting:
                self._latest_selector_traces.append(
                    SelectorTrace(
                        intent=_input_target_intent(subgoal, target_element_id),
                        candidate_count=1,
                        top_candidates=[
                            TargetEvidence(
                                element_id=target.element_id,
                                element_type=target.element_type,
                                primary_name=target.primary_name,
                                total_score=999.0,
                                matched_signals=["exact_target_element_id"],
                                rejected_by=[],
                                action_compatible=True,
                                exact_semantic_match=True,
                                uses_unlabeled_fallback=target.is_unlabeled,
                                nearest_matched_text_candidate_id=None,
                                spatial_grounding_contributed=False,
                                confidence_band=SelectorConfidenceBand.HIGH,
                            )
                        ],
                        selected_element_id=target.element_id,
                        decision_reason="Accepted exact target_element_id match before semantic scoring.",
                        rejection_reason=None,
                        score_margin=None,
                    )
                )
                return target

        intent = _input_target_intent(subgoal, target_element_id)
        return self._select_target(perception, intent)

    def _fallback_input_target(
        self,
        perception: ScreenPerception,
        subgoal: str | None,
        target_element_id: str | None,
    ) -> UIElement | None:
        intent = _input_target_intent(subgoal, target_element_id)
        return self._select_target(perception, intent)

    def _search_input_target(self, perception: ScreenPerception) -> UIElement | None:
        candidates = [e for e in _input_candidates(perception) if "search" in e.primary_name.lower()]
        return candidates[0] if candidates else None

    def _select_target(self, perception: ScreenPerception, intent: TargetIntent) -> UIElement | None:
        result = self.selector.select(perception, intent, _cached_intermediates=self._cached_intermediates)
        self._latest_selector_traces.append(result.trace)
        return result.selected


# ---------------------------------------------------------------------------
# Module-level helpers shared by engine primitives and benchmark plugins
# ---------------------------------------------------------------------------

def _has_hint(memory_hints: list[MemoryHint], key: str) -> bool:
    return any(hint.key == key for hint in memory_hints)


def _has_recent_target_failure_signal(
    state: AgentState,
    target_element_id: str,
    failure_category: FailureCategory,
) -> bool:
    key = f"type:{target_element_id}:{failure_category.value}"
    return state.target_failure_counts.get(key, 0) > 0


def _has_success_signal(perception: ScreenPerception) -> bool:
    success_tokens = (
        "thank you",
        "submitted successfully",
        "submission successful",
        "submission complete",
        "task completed",
    )
    for element in perception.visible_elements:
        if any(token in element.primary_name.lower() for token in success_tokens):
            return True
    return any(token in perception.summary.lower() for token in success_tokens)


def _input_candidates(perception: ScreenPerception) -> list[UIElement]:
    return [e for e in perception.visible_elements if e.element_type is UIElementType.INPUT and e.usable_for_targeting]


def _click_decision(target: UIElement, rationale: str) -> PolicyDecision:
    return PolicyDecision(
        action=AgentAction(
            action_type=ActionType.CLICK,
            target_element_id=target.element_id,
            x=target.x + max(1, target.width // 2),
            y=target.y + max(1, target.height // 2),
        ),
        rationale=rationale,
        confidence=0.95,
        active_subgoal=f"focus {target.element_id}",
    )


def _match_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = "".join(c.lower() if c.isalnum() else " " for c in value)
    return {t for t in normalized.split() if len(t) >= 2}


def _input_target_intent(subgoal: str | None, target_element_id: str | None) -> TargetIntent:
    lowered_subgoal = (subgoal or "").lower()
    expected_section: str | None = None
    target_text = None

    subgoal_map = (
        ("name", {"name"}),
        ("email", {"email", "recipient", "to"}),
        ("message", {"message", "body", "draft"}),
        ("subject", {"subject"}),
    )
    for canonical, tokens in subgoal_map:
        if any(token in lowered_subgoal for token in tokens):
            target_text = canonical
            break

    if target_text is None:
        id_tokens = _match_tokens(target_element_id)
        for canonical, tokens in subgoal_map:
            if id_tokens & tokens:
                target_text = canonical
                break

    if "compose" in lowered_subgoal or "gmail" in lowered_subgoal:
        expected_section = "compose"
    elif "form" in lowered_subgoal or "submit" in lowered_subgoal:
        expected_section = "form"

    return TargetIntent(
        action=TargetIntentAction.CLICK,
        target_text=target_text,
        expected_element_types=[UIElementType.INPUT],
        expected_section=expected_section,
    )


def _submit_button(perception: ScreenPerception) -> UIElement | None:
    from src.agent.selector import DeterministicTargetSelector
    result = DeterministicTargetSelector().select(
        perception,
        TargetIntent(
            action=TargetIntentAction.CLICK,
            target_text="submit",
            expected_element_types=[UIElementType.BUTTON, UIElementType.LINK],
            expected_section="form",
        ),
    )
    return result.selected


def _extract_search_query(state: AgentState) -> str | None:
    for source in (state.current_subgoal, state.intent):
        if not source:
            continue
        match = re.search(r"search for\s+(.+?)(?:,|\sand\s|\sthen\s|$)", source, flags=re.IGNORECASE)
        if match is None:
            continue
        query = match.group(1).strip(" .")
        if query:
            return query
    return None


def _search_trigger_target(perception: ScreenPerception) -> UIElement | None:
    for element in perception.visible_elements:
        if not element.usable_for_targeting:
            continue
        if element.element_type not in {UIElementType.BUTTON, UIElementType.LINK, UIElementType.ICON}:
            continue
        if "search" in element.primary_name.lower():
            return element
    return None


def _preferred_compose_input(perception: ScreenPerception) -> UIElement | None:
    from src.agent.selector import DeterministicTargetSelector
    selector = DeterministicTargetSelector()
    intents = (
        TargetIntent(action=TargetIntentAction.CLICK, target_text="recipient", expected_element_types=[UIElementType.INPUT], expected_section="compose"),
        TargetIntent(action=TargetIntentAction.CLICK, target_text="to", expected_element_types=[UIElementType.INPUT], expected_section="compose"),
        TargetIntent(action=TargetIntentAction.CLICK, target_text="subject", expected_element_types=[UIElementType.INPUT], expected_section="compose"),
        TargetIntent(action=TargetIntentAction.CLICK, target_text="message", expected_element_types=[UIElementType.INPUT], expected_section="compose"),
    )
    for intent in intents:
        result = selector.select(perception, intent)
        if result.selected is not None:
            return result.selected
    return None


def _compose_form_visible(perception: ScreenPerception) -> bool:
    return _preferred_compose_input(perception) is not None


def _form_fields_completed(state: AgentState, perception: ScreenPerception) -> bool:
    required_targets: dict[str, str] = {}
    for element in _input_candidates(perception):
        label = element.primary_name.lower()
        if "name" in label:
            required_targets["name"] = element.element_id
        elif "email" in label:
            required_targets["email"] = element.element_id
        elif "message" in label:
            required_targets["message"] = element.element_id

    if len(required_targets) < 3:
        return False

    completed_targets = {
        executed.action.target_element_id
        for executed, verification in zip(state.action_history, state.verification_history)
        if executed.action.action_type is ActionType.TYPE
        and verification.status is VerificationStatus.SUCCESS
        and executed.action.target_element_id is not None
    }
    return all(target_id in completed_targets for target_id in required_targets.values())

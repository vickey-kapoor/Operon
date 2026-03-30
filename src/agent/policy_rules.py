"""Explicit rule-first policy checks for deterministic agent cases."""

from __future__ import annotations

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


class PolicyRuleEngine:
    """Small explicit rule engine that runs before the LLM-backed policy."""

    def __init__(self, selector: DeterministicTargetSelector | None = None) -> None:
        self.selector = selector or DeterministicTargetSelector()
        self._latest_selector_traces: list[SelectorTrace] = []
        self._cached_intermediates: tuple[list[UIElement], list[list[UIElement]]] | None = None

    def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> PolicyDecision | None:
        self._latest_selector_traces = []
        # Precompute selector intermediates once for all rules in this step
        self._cached_intermediates = (
            self.selector._label_like_text_candidates(perception),
            self.selector._visual_groups(perception),
        )
        return (
            self._login_page_guardrail(state, perception, memory_hints)
            or self._task_success_stop_rule(perception)
            or self._avoid_identical_type_retry(state, perception, memory_hints)
            or self._compose_already_visible_rule(state, perception)
            or self._submit_form_when_ready_rule(state, perception)
            or self._focus_before_type_rule(state, perception, memory_hints)
        )

    def latest_selector_traces(self) -> list[SelectorTrace]:
        return list(self._latest_selector_traces)

    def _login_page_guardrail(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> PolicyDecision | None:
        if perception.page_hint is not PageHint.GOOGLE_SIGN_IN:
            return None
        if self._has_hint(memory_hints, "authenticated_start_required"):
            return PolicyDecision(
                action=AgentAction(action_type=ActionType.STOP),
                rationale="Benchmark requires an authenticated Gmail start state; login/auth screens are out of scope.",
                confidence=1.0,
                active_subgoal="stop for benchmark setup",
            )
        # General-purpose run: pause and ask the user to handle authentication
        return PolicyDecision(
            action=AgentAction(
                action_type=ActionType.WAIT_FOR_USER,
                text="Login page detected. Please sign in using the browser window, then click Resume.",
            ),
            rationale="Login/auth page detected — requires user credentials that Operon cannot provide autonomously.",
            confidence=1.0,
            active_subgoal="wait for user authentication",
        )

    def _avoid_identical_type_retry(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> PolicyDecision | None:
        if not self._has_hint(memory_hints, "avoid_identical_type_retry"):
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
        if not self._has_recent_target_failure_signal(state, last_action.target_element_id, last_failure):
            return None

        target = self._resolve_target_element(
            perception,
            target_element_id=last_action.target_element_id,
            subgoal=state.current_subgoal,
        )
        if target is None:
            target = self._fallback_input_target(perception, state.current_subgoal, last_action.target_element_id)

        if target is not None and perception.focused_element_id != target.element_id:
            return self._click_decision(target, f"Re-establish focus on {target.primary_name} before retrying type.")

        return PolicyDecision(
            action=AgentAction(action_type=ActionType.WAIT, wait_ms=500),
            rationale="Previous type action failed on the same target; avoid repeating the identical type action blindly.",
            confidence=0.93,
            active_subgoal=f"re-assess {last_action.target_element_id or 'input'}",
        )

    def _task_success_stop_rule(self, perception: ScreenPerception) -> PolicyDecision | None:
        if perception.page_hint is not PageHint.FORM_SUCCESS and not self._has_success_signal(perception):
            return None
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.STOP),
            rationale="Task success state is already visible.",
            confidence=1.0,
            active_subgoal="verify_success",
        )

    def _compose_already_visible_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision | None:
        current_subgoal = (state.current_subgoal or "").lower()
        if "open compose" not in current_subgoal and "compose" not in current_subgoal:
            return None
        if perception.page_hint is not PageHint.GMAIL_COMPOSE and not self._compose_form_visible(perception):
            return None

        target = self._preferred_compose_input(perception)
        if target is None:
            return None
        return self._click_decision(target, "Compose form is already visible; move to the form instead of reopening compose.")

    def _focus_before_type_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> PolicyDecision | None:
        if not self._has_hint(memory_hints, "click_before_type"):
            return None

        target = self._resolve_target_element(
            perception,
            target_element_id=None,
            subgoal=state.current_subgoal,
        )
        if target is None or perception.focused_element_id == target.element_id:
            return None
        return self._click_decision(target, f"Focus {target.primary_name} before typing because input focus is not established.")

    def _submit_form_when_ready_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision | None:
        if perception.page_hint is not PageHint.FORM_PAGE:
            return None
        if not self._required_form_fields_completed(state, perception):
            return None
        submit_button = self._submit_button(perception)
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

    def _resolve_target_element(
        self,
        perception: ScreenPerception,
        *,
        target_element_id: str | None,
        subgoal: str | None,
    ) -> UIElement | None:
        if target_element_id is not None:
            target = next((element for element in perception.visible_elements if element.element_id == target_element_id), None)
            if target is not None and target.element_type is UIElementType.INPUT and target.usable_for_targeting:
                self._latest_selector_traces.append(
                    SelectorTrace(
                        intent=self._input_target_intent(subgoal, target_element_id),
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

        intent = self._input_target_intent(subgoal, target_element_id)
        return self._select_target(perception, intent)

    def _fallback_input_target(
        self,
        perception: ScreenPerception,
        subgoal: str | None,
        target_element_id: str | None,
    ) -> UIElement | None:
        intent = self._input_target_intent(subgoal, target_element_id)
        return self._select_target(perception, intent)

    def _required_form_fields_completed(self, state: AgentState, perception: ScreenPerception) -> bool:
        required_targets: dict[str, str] = {}
        for element in self._input_candidates(perception):
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

    @staticmethod
    def _has_success_signal(perception: ScreenPerception) -> bool:
        success_tokens = ("success", "thank you", "submitted")
        for element in perception.visible_elements:
            label = element.primary_name.lower()
            if any(token in label for token in success_tokens):
                return True
        return any(token in perception.summary.lower() for token in success_tokens)

    def _preferred_compose_input(self, perception: ScreenPerception) -> UIElement | None:
        intents = (
            TargetIntent(
                action=TargetIntentAction.CLICK,
                target_text="recipient",
                expected_element_types=[UIElementType.INPUT],
                expected_section="compose",
            ),
            TargetIntent(
                action=TargetIntentAction.CLICK,
                target_text="to",
                expected_element_types=[UIElementType.INPUT],
                expected_section="compose",
            ),
            TargetIntent(
                action=TargetIntentAction.CLICK,
                target_text="subject",
                expected_element_types=[UIElementType.INPUT],
                expected_section="compose",
            ),
            TargetIntent(
                action=TargetIntentAction.CLICK,
                target_text="message",
                expected_element_types=[UIElementType.INPUT],
                expected_section="compose",
            ),
        )
        for intent in intents:
            target = self._select_target(perception, intent)
            if target is not None:
                return target
        return None

    def _compose_form_visible(self, perception: ScreenPerception) -> bool:
        return self._preferred_compose_input(perception) is not None

    @staticmethod
    def _input_candidates(perception: ScreenPerception) -> list[UIElement]:
        return [
            element
            for element in perception.visible_elements
            if element.element_type is UIElementType.INPUT and element.usable_for_targeting
        ]

    @staticmethod
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

    @staticmethod
    def _has_hint(memory_hints: list[MemoryHint], key: str) -> bool:
        return any(hint.key == key for hint in memory_hints)

    @staticmethod
    def _has_recent_target_failure_signal(
        state: AgentState,
        target_element_id: str,
        failure_category: FailureCategory,
    ) -> bool:
        key = f"type:{target_element_id}:{failure_category.value}"
        return state.target_failure_counts.get(key, 0) > 0

    @staticmethod
    def _match_tokens(value: str | None) -> set[str]:
        if not value:
            return set()
        normalized = "".join(character.lower() if character.isalnum() else " " for character in value)
        return {token for token in normalized.split() if len(token) >= 2}

    def _submit_button(self, perception: ScreenPerception) -> UIElement | None:
        return self._select_target(
            perception,
            TargetIntent(
                action=TargetIntentAction.CLICK,
                target_text="submit",
                expected_element_types=[UIElementType.BUTTON, UIElementType.LINK],
                expected_section="form",
            ),
        )

    def _input_target_intent(self, subgoal: str | None, target_element_id: str | None) -> TargetIntent:
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
            id_tokens = self._match_tokens(target_element_id)
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

    def _select_target(self, perception: ScreenPerception, intent: TargetIntent) -> UIElement | None:
        result = self.selector.select(perception, intent, _cached_intermediates=self._cached_intermediates)
        self._latest_selector_traces.append(result.trace)
        return result.selected

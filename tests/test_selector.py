"""Focused tests for deterministic target selection and spatial grounding."""

from __future__ import annotations

from src.agent.selector import DeterministicTargetSelector
from src.models.common import FailureCategory, StopReason
from src.models.perception import ScreenPerception, UIElement, UIElementType
from src.models.selector import SelectorFinalDecision, SelectorRecoveryStrategy, TargetIntent, TargetIntentAction


def _perception(*elements: UIElement) -> ScreenPerception:
    return ScreenPerception(
        summary="Form page visible.",
        page_hint="form_page",
        visible_elements=list(elements),
        capture_artifact_path="runs/test/step_1/before.png",
        confidence=0.95,
    )


def _element(
    element_id: str,
    *,
    element_type: UIElementType = UIElementType.INPUT,
    label: str | None = None,
    text: str | None = None,
    placeholder: str | None = None,
    name: str | None = None,
    x: int = 100,
    y: int = 100,
    width: int = 180,
    height: int = 28,
    is_interactable: bool = True,
    confidence: float = 0.95,
) -> UIElement:
    return UIElement(
        element_id=element_id,
        element_type=element_type,
        label=label,
        text=text,
        placeholder=placeholder,
        name=name,
        x=x,
        y=y,
        width=width,
        height=height,
        is_interactable=is_interactable,
        confidence=confidence,
    )


def _text(element_id: str, text: str, *, x: int, y: int, width: int = 80, height: int = 20) -> UIElement:
    return _element(
        element_id,
        element_type=UIElementType.TEXT,
        text=text,
        x=x,
        y=y,
        width=width,
        height=height,
        is_interactable=False,
    )


def _intent(target_text: str, action: TargetIntentAction = TargetIntentAction.TYPE) -> TargetIntent:
    return TargetIntent(
        action=action,
        target_text=target_text,
        expected_element_types=[UIElementType.INPUT] if action is TargetIntentAction.TYPE else [UIElementType.BUTTON],
        expected_section="form",
    )


def test_exact_semantic_match_beats_fuzzy_match() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("email-input", label="Email"),
        _element("email-address-input", label="Email address", y=140),
    )

    result = selector.select(perception, _intent("Email"))

    assert result.selected is not None
    assert result.selected.element_id == "email-input"
    assert result.trace.rejection_reason is None
    assert result.trace.recovery_attempted is False
    assert result.trace.top_candidates[0].exact_semantic_match is True


def test_placeholder_exact_match_beats_unlabeled_fallback() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("field-1", label=None, placeholder=None, y=100),
        _element("name-input", label=None, placeholder="Name", y=140),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is not None
    assert result.selected.element_id == "name-input"
    assert result.trace.top_candidates[0].uses_unlabeled_fallback is False
    assert "exact_primary_name" in result.trace.top_candidates[0].matched_signals


def test_action_incompatible_candidate_is_filtered() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("submit-button", element_type=UIElementType.BUTTON, label="Submit"),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is None
    assert result.trace.rejection_reason is FailureCategory.TARGET_ACTION_INCOMPATIBLE


def test_unlabeled_candidate_is_rejected_without_grounding() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("field-1", label=None, placeholder=None),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is None
    assert result.trace.rejection_reason is FailureCategory.TARGET_UNLABELED_INSUFFICIENT_GROUNDING


def test_ambiguous_close_score_candidates_fail() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _text("label-1", "Name", x=100, y=80),
        _element("field-1", label=None, placeholder=None, x=100, y=108),
        _text("label-2", "Name", x=360, y=80),
        _element("field-2", label=None, placeholder=None, x=360, y=108),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is None
    assert result.trace.rejection_reason is FailureCategory.AMBIGUOUS_TARGET_CANDIDATES
    assert result.trace.score_margin is not None


def test_no_candidates_after_filtering_returns_explicit_failure() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("status-text", element_type=UIElementType.TEXT, text="Ready", is_interactable=False),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is None
    assert result.trace.rejection_reason is FailureCategory.SELECTOR_NO_CANDIDATES_AFTER_FILTERING


def test_label_directly_above_input_selects_input() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _text("name-label", "Name", x=120, y=80),
        _element("name-input", label=None, placeholder=None, x=100, y=108),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is not None
    assert result.selected.element_id == "name-input"
    assert "nearby_label_above_exact_match" in result.trace.top_candidates[0].matched_signals


def test_label_directly_left_of_input_selects_input() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _text("email-label", "Email", x=40, y=104, width=50),
        _element("email-input", label=None, placeholder=None, x=110, y=100),
    )

    result = selector.select(perception, _intent("Email"))

    assert result.selected is not None
    assert result.selected.element_id == "email-input"
    assert "nearby_label_left_exact_match" in result.trace.top_candidates[0].matched_signals


def test_unlabeled_input_resolved_via_nearby_label_above() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _text("message-label", "Message", x=120, y=150),
        _element("message-field", label=None, placeholder=None, x=100, y=178),
    )

    result = selector.select(perception, _intent("Message"))

    assert result.selected is not None
    assert result.selected.element_id == "message-field"
    assert result.trace.top_candidates[0].nearest_matched_text_candidate_id == "message-label"
    assert result.trace.top_candidates[0].spatial_grounding_contributed is True


def test_unlabeled_input_resolved_via_nearby_label_left() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _text("email-label", "Email", x=10, y=204, width=60),
        _element("email-field", label=None, placeholder=None, x=90, y=200),
    )

    result = selector.select(perception, _intent("Email"))

    assert result.selected is not None
    assert result.selected.element_id == "email-field"
    assert result.trace.top_candidates[0].nearest_matched_text_candidate_id == "email-label"


def test_exact_element_semantics_beat_contextual_only_match() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _text("name-label", "Name", x=100, y=80),
        _element("field-1", label=None, placeholder=None, x=100, y=108),
        _element("name-input", label="Name", x=360, y=108),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is not None
    assert result.selected.element_id == "name-input"
    assert result.trace.top_candidates[0].exact_semantic_match is True


def test_spatial_signals_appear_in_selector_trace() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _text("name-label", "Name", x=120, y=80),
        _element("name-input", label=None, placeholder=None, x=100, y=108),
    )

    result = selector.select(perception, _intent("Name"))

    evidence = result.trace.top_candidates[0]
    assert "nearest_label_match" in evidence.matched_signals
    assert "same_group_context_match" in evidence.matched_signals
    assert evidence.nearest_matched_text_candidate_id == "name-label"
    assert evidence.spatial_grounding_contributed is True


def test_threshold_based_recovery_succeeds() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("email-input", label="Email", x=100, y=108, confidence=0.5),
    )

    result = selector.select(perception, _intent("Primary Email"))

    assert result.selected is not None
    assert result.selected.element_id == "email-input"
    assert result.trace.recovery_attempted is True
    assert result.trace.recovery_strategy_used is SelectorRecoveryStrategy.THRESHOLD_RELAXATION
    assert result.trace.adjusted_acceptance_threshold == 70.0
    assert result.trace.final_decision is SelectorFinalDecision.SUCCESS
    assert result.trace.final_stop_reason is StopReason.SELECTOR_RECOVERY_USED


def test_margin_based_recovery_succeeds() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("email-input", label="Email", x=100, y=108),
        _element("email-secondary", text="Email", x=360, y=108),
    )

    result = selector.select(perception, _intent("Email"))

    assert result.selected is not None
    assert result.selected.element_id == "email-input"
    assert result.trace.recovery_attempted is True
    assert result.trace.initial_failure_reason is FailureCategory.AMBIGUOUS_TARGET_CANDIDATES
    assert result.trace.recovery_strategy_used is SelectorRecoveryStrategy.MARGIN_RELAXATION
    assert result.trace.adjusted_ambiguity_margin == 6.0


def test_contextual_recovery_succeeds_for_unlabeled_element() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _text("name-label", "Name Profile", x=120, y=80),
        _element("name-field", label=None, placeholder=None, x=100, y=108),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is not None
    assert result.selected.element_id == "name-field"
    assert result.trace.recovery_attempted is True
    assert result.trace.recovery_strategy_used is SelectorRecoveryStrategy.CONTEXTUAL_BOOST
    assert result.trace.recovery_changed_selected_candidate is True


def test_recovery_does_not_trigger_for_hard_failures() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("submit-button", element_type=UIElementType.BUTTON, label="Submit"),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is None
    assert result.trace.recovery_attempted is False
    assert result.trace.recovery_strategy_used is None
    assert result.trace.rejection_reason is FailureCategory.TARGET_ACTION_INCOMPATIBLE


def test_recovery_does_not_violate_safety_constraints() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _text("name-label", "Name", x=120, y=80),
        _element("field-1", label=None, placeholder=None, x=100, y=108),
        _text("email-label", "Name", x=380, y=80),
        _element("field-2", label=None, placeholder=None, x=360, y=108),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is None
    assert result.trace.recovery_attempted is True
    assert result.trace.final_decision is SelectorFinalDecision.FAILURE
    assert result.trace.final_stop_reason is StopReason.SELECTOR_RECOVERY_FAILED


def test_near_threshold_case_succeeds_via_recovery() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("email-input", label="Email", x=100, y=108, confidence=0.5),
    )

    result = selector.select(perception, _intent("Primary Email"))

    assert result.selected is not None
    assert result.trace.initial_failure_reason is FailureCategory.SELECTOR_SCORE_BELOW_THRESHOLD


def test_ambiguous_case_resolved_via_recovery() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("name-input", label="Name", x=100, y=108),
        _element("name-info", text="Name", x=360, y=108),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is not None
    assert result.selected.element_id == "name-input"
    assert result.trace.recovery_strategy_used is SelectorRecoveryStrategy.MARGIN_RELAXATION


def test_true_ambiguous_case_still_fails_after_recovery() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("name-input-a", label="Name", x=100, y=108),
        _element("name-input-b", label="Name", x=360, y=108),
    )

    result = selector.select(perception, _intent("Name"))

    assert result.selected is None
    assert result.trace.recovery_attempted is True
    assert result.trace.final_stop_reason is StopReason.SELECTOR_RECOVERY_FAILED


def test_selector_trace_includes_recovery_metadata() -> None:
    selector = DeterministicTargetSelector()
    perception = _perception(
        _element("email-input", label="Email", x=100, y=108, confidence=0.5),
    )

    result = selector.select(perception, _intent("Primary Email"))

    assert result.trace.initial_failure_reason is FailureCategory.SELECTOR_SCORE_BELOW_THRESHOLD
    assert result.trace.recovery_attempted is True
    assert result.trace.recovery_strategy_used is SelectorRecoveryStrategy.THRESHOLD_RELAXATION
    assert result.trace.adjusted_acceptance_threshold == 70.0
    assert result.trace.final_decision is SelectorFinalDecision.SUCCESS

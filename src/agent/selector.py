"""Deterministic multi-signal target selector used by rule-first policy logic."""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.geometry import (
    GROUP_VERTICAL_GAP_PX,
    NEARBY_LABEL_MAX_DISTANCE_PX,
    bbox_distance,
    horizontal_overlap,
    is_above,
    is_left_of,
    same_column,
    same_row,
)
from src.models.common import FailureCategory, StopReason
from src.models.perception import (
    ScreenPerception,
    UIElement,
    UIElementNameSource,
    UIElementType,
)
from src.models.selector import (
    OriginalTargetSignature,
    SelectorConfidenceBand,
    SelectorFinalDecision,
    SelectorMode,
    SelectorRecoveryStrategy,
    SelectorTrace,
    TargetEvidence,
    TargetIntent,
    TargetIntentAction,
    TargetSelectionContext,
)

_ACCEPTANCE_THRESHOLD = 80.0
_RECOVERY_ACCEPTANCE_THRESHOLD = 70.0
_AMBIGUITY_MARGIN = 12.0
_RECOVERY_AMBIGUITY_MARGIN = 6.0
_RECOVERY_RATIO_THRESHOLD = 1.5
_TOP_CANDIDATE_COUNT = 3
_DEFAULT_SPATIAL_CAP = 22.0
_RECOVERY_SPATIAL_CAP = 36.0
_RERESOLUTION_PRIOR_ID_EXACT_WEIGHT = 6.0
_RERESOLUTION_PRIOR_ID_TOKEN_WEIGHT = 3.0
_RERESOLUTION_PRIOR_NAME_EXACT_WEIGHT = 7.0
_RERESOLUTION_PRIOR_NAME_TOKEN_WEIGHT = 4.0
_RERESOLUTION_REGION_NEAR_WEIGHT = 6.0
_RERESOLUTION_REGION_MID_WEIGHT = 3.0
_RERESOLUTION_SIGNAL_CONTINUITY_CAP = 5.0
_ACTION_COMPATIBILITY: dict[TargetIntentAction, set[UIElementType]] = {
    TargetIntentAction.CLICK: {UIElementType.BUTTON, UIElementType.LINK, UIElementType.INPUT},
    TargetIntentAction.TYPE: {UIElementType.INPUT},
    TargetIntentAction.SELECT: {UIElementType.INPUT},
    TargetIntentAction.CHECK: {UIElementType.INPUT, UIElementType.BUTTON},
    TargetIntentAction.UNCHECK: {UIElementType.INPUT, UIElementType.BUTTON},
}
_EXACT_FIELD_WEIGHTS = {
    "primary_name": 120.0,
    "label": 114.0,
    "text": 108.0,
    "placeholder": 112.0,
    "name": 104.0,
}
_FUZZY_FIELD_WEIGHTS = {
    "primary_name": 56.0,
    "label": 52.0,
    "text": 46.0,
    "placeholder": 50.0,
    "name": 42.0,
}
_SPATIAL_WEIGHTS = {
    "nearby_label_above_exact_match": 76.0,
    "nearby_label_left_exact_match": 58.0,
    "nearby_label_above_fuzzy_match": 34.0,
    "nearby_label_left_fuzzy_match": 28.0,
    "nearest_label_match": 12.0,
    "same_group_context_match": 8.0,
}


@dataclass(slots=True)
class SelectorResult:
    selected: UIElement | None
    trace: SelectorTrace


@dataclass(slots=True)
class _SpatialMatch:
    label: UIElement
    score: float
    signals: list[str]


@dataclass(slots=True)
class _SelectorConfig:
    acceptance_threshold: float
    ambiguity_margin: float
    spatial_cap: float
    recovery_mode: bool = False
    strategy: SelectorRecoveryStrategy | None = None
    allow_ratio_override: bool = False
    selector_mode: SelectorMode = SelectorMode.INITIAL
    selection_context: TargetSelectionContext | None = None


@dataclass(slots=True)
class _AttemptResult:
    selected: UIElement | None
    decision_reason: str
    failure_reason: FailureCategory | None
    score_margin: float | None
    candidate_count: int
    candidates_before_filtering: int
    selected_element_id: str | None
    top_candidates: list[TargetEvidence]
    ranked: list[tuple[UIElement, TargetEvidence]]


class DeterministicTargetSelector:
    """Score candidates with transparent signals and deterministic acceptance rules."""

    def build_selection_context(
        self,
        perception: ScreenPerception,
        intent: TargetIntent,
        target: UIElement,
        *,
        page_signature: str | None = None,
    ) -> TargetSelectionContext:
        text_candidates = self._label_like_text_candidates(perception)
        visual_groups = self._visual_groups(perception)
        evidence = self._score_candidate(
            perception,
            target,
            intent,
            text_candidates,
            visual_groups,
            config=_SelectorConfig(
                acceptance_threshold=_ACCEPTANCE_THRESHOLD,
                ambiguity_margin=_AMBIGUITY_MARGIN,
                spatial_cap=_DEFAULT_SPATIAL_CAP,
            ),
        )
        trace = self.select(perception, intent).trace
        return TargetSelectionContext(
            intent=intent,
            original_target=OriginalTargetSignature(
                element_id=target.element_id,
                element_type=target.element_type,
                primary_name=target.primary_name,
                role=target.role,
                x=target.x,
                y=target.y,
                width=target.width,
                height=target.height,
            ),
            selected_candidate_evidence=evidence,
            top_candidates=trace.top_candidates,
            original_matched_signals=list(evidence.matched_signals),
            original_page_signature=page_signature,
        )

    def select(self, perception: ScreenPerception, intent: TargetIntent) -> SelectorResult:
        initial = self._run_attempt(
            perception,
            intent,
            _SelectorConfig(
                acceptance_threshold=_ACCEPTANCE_THRESHOLD,
                ambiguity_margin=_AMBIGUITY_MARGIN,
                spatial_cap=_DEFAULT_SPATIAL_CAP,
            ),
        )
        if initial.selected is not None:
            return SelectorResult(selected=initial.selected, trace=self._build_trace(intent, initial))

        recovery_config = self._recovery_config(
            initial,
            config=_SelectorConfig(
                acceptance_threshold=_ACCEPTANCE_THRESHOLD,
                ambiguity_margin=_AMBIGUITY_MARGIN,
                spatial_cap=_DEFAULT_SPATIAL_CAP,
            ),
        )
        if recovery_config is None:
            return SelectorResult(selected=None, trace=self._build_trace(intent, initial))

        recovery = self._run_attempt(perception, intent, recovery_config)
        return SelectorResult(
            selected=recovery.selected,
            trace=self._build_trace(intent, recovery, initial=initial, recovery_config=recovery_config),
        )

    def reresolve(self, perception: ScreenPerception, selection_context: TargetSelectionContext) -> SelectorResult:
        initial_config = _SelectorConfig(
            acceptance_threshold=_ACCEPTANCE_THRESHOLD,
            ambiguity_margin=_AMBIGUITY_MARGIN,
            spatial_cap=_DEFAULT_SPATIAL_CAP,
            selector_mode=SelectorMode.RERESOLUTION,
            selection_context=selection_context,
        )
        initial = self._run_attempt(perception, selection_context.intent, initial_config)
        if initial.selected is not None:
            return SelectorResult(selected=initial.selected, trace=self._build_trace(selection_context.intent, initial, config=initial_config))

        recovery_config = self._recovery_config(initial, config=initial_config)
        if recovery_config is None:
            return SelectorResult(selected=None, trace=self._build_trace(selection_context.intent, initial, config=initial_config))

        recovery = self._run_attempt(perception, selection_context.intent, recovery_config)
        return SelectorResult(
            selected=recovery.selected,
            trace=self._build_trace(
                selection_context.intent,
                recovery,
                initial=initial,
                recovery_config=recovery_config,
                config=initial_config,
            ),
        )

    def _run_attempt(self, perception: ScreenPerception, intent: TargetIntent, config: _SelectorConfig) -> _AttemptResult:
        filtered_candidates: list[UIElement] = []
        evidences: list[TargetEvidence] = []
        incompatible_found = False
        unlabeled_rejected = False
        text_candidates = self._label_like_text_candidates(perception)
        visual_groups = self._visual_groups(perception)
        candidates_before_filtering = 0

        for element in perception.visible_elements:
            prefilter_eligible = self._passes_prefilter(element, intent)
            if prefilter_eligible:
                candidates_before_filtering += 1
            passes_filters, rejected_by, action_compatible = self._passes_hard_filters(perception, element, intent, config=config)
            if not action_compatible and element.is_interactable:
                incompatible_found = True
            if "unlabeled_without_grounding" in rejected_by:
                unlabeled_rejected = True
            if not passes_filters:
                continue
            filtered_candidates.append(element)
            evidences.append(self._score_candidate(perception, element, intent, text_candidates, visual_groups, config=config))

        if not evidences:
            rejection_reason = FailureCategory.SELECTOR_NO_CANDIDATES_AFTER_FILTERING
            if incompatible_found:
                rejection_reason = FailureCategory.TARGET_ACTION_INCOMPATIBLE
            elif unlabeled_rejected:
                rejection_reason = FailureCategory.TARGET_UNLABELED_INSUFFICIENT_GROUNDING
            return _AttemptResult(
                selected=None,
                decision_reason="No candidates survived deterministic selector hard filtering.",
                failure_reason=rejection_reason,
                score_margin=None,
                candidate_count=0,
                candidates_before_filtering=candidates_before_filtering,
                selected_element_id=None,
                top_candidates=[],
                ranked=[],
            )

        ranked = sorted(
            zip(filtered_candidates, evidences, strict=False),
            key=lambda item: (
                item[1].total_score,
                1 if item[1].exact_semantic_match else 0,
                1 if item[1].spatial_grounding_contributed else 0,
                -item[0].y,
                -item[0].x,
                item[0].element_id,
            ),
            reverse=True,
        )
        top_candidates = [evidence for _, evidence in ranked[:_TOP_CANDIDATE_COUNT]]
        winner, winner_evidence = ranked[0]
        margin = None if len(ranked) == 1 else round(winner_evidence.total_score - ranked[1][1].total_score, 2)

        if winner_evidence.total_score < config.acceptance_threshold:
            return _AttemptResult(
                selected=None,
                decision_reason="Top candidate score was below the deterministic acceptance threshold.",
                failure_reason=FailureCategory.SELECTOR_SCORE_BELOW_THRESHOLD,
                score_margin=margin,
                candidate_count=len(ranked),
                candidates_before_filtering=candidates_before_filtering,
                selected_element_id=None,
                top_candidates=top_candidates,
                ranked=ranked,
            )

        if margin is not None and margin < config.ambiguity_margin:
            if config.allow_ratio_override and ranked[1][1].total_score > 0:
                ratio = winner_evidence.total_score / ranked[1][1].total_score
                if ratio > _RECOVERY_RATIO_THRESHOLD:
                    return _AttemptResult(
                        selected=winner,
                        decision_reason="Recovery accepted the top candidate because it was materially stronger than the runner-up.",
                        failure_reason=None,
                        score_margin=margin,
                        candidate_count=len(ranked),
                        candidates_before_filtering=candidates_before_filtering,
                        selected_element_id=winner.element_id,
                        top_candidates=top_candidates,
                        ranked=ranked,
                    )
            return _AttemptResult(
                selected=None,
                decision_reason="Top candidates were too close to accept deterministically.",
                failure_reason=FailureCategory.AMBIGUOUS_TARGET_CANDIDATES,
                score_margin=margin,
                candidate_count=len(ranked),
                candidates_before_filtering=candidates_before_filtering,
                selected_element_id=None,
                top_candidates=top_candidates,
                ranked=ranked,
            )

        return _AttemptResult(
            selected=winner,
            decision_reason="Selected the highest-scoring candidate that cleared deterministic acceptance rules.",
            failure_reason=None,
            score_margin=margin,
            candidate_count=len(ranked),
            candidates_before_filtering=candidates_before_filtering,
            selected_element_id=winner.element_id,
            top_candidates=top_candidates,
            ranked=ranked,
        )

    def _recovery_config(self, initial: _AttemptResult, *, config: _SelectorConfig) -> _SelectorConfig | None:
        if initial.failure_reason is FailureCategory.SELECTOR_SCORE_BELOW_THRESHOLD:
            if initial.top_candidates and initial.top_candidates[0].spatial_grounding_contributed:
                return _SelectorConfig(
                    acceptance_threshold=_RECOVERY_ACCEPTANCE_THRESHOLD,
                    ambiguity_margin=_AMBIGUITY_MARGIN,
                    spatial_cap=_RECOVERY_SPATIAL_CAP,
                    recovery_mode=True,
                    strategy=SelectorRecoveryStrategy.CONTEXTUAL_BOOST,
                    selector_mode=config.selector_mode,
                    selection_context=config.selection_context,
                )
            if initial.top_candidates and initial.top_candidates[0].total_score >= _RECOVERY_ACCEPTANCE_THRESHOLD:
                return _SelectorConfig(
                    acceptance_threshold=_RECOVERY_ACCEPTANCE_THRESHOLD,
                    ambiguity_margin=_AMBIGUITY_MARGIN,
                    spatial_cap=_DEFAULT_SPATIAL_CAP,
                    recovery_mode=True,
                    strategy=SelectorRecoveryStrategy.THRESHOLD_RELAXATION,
                    selector_mode=config.selector_mode,
                    selection_context=config.selection_context,
                )
            return None

        if initial.failure_reason is FailureCategory.AMBIGUOUS_TARGET_CANDIDATES:
            if len(initial.top_candidates) >= 2 and initial.top_candidates[1].total_score > 0:
                ratio = initial.top_candidates[0].total_score / initial.top_candidates[1].total_score
                if ratio > _RECOVERY_RATIO_THRESHOLD:
                    return _SelectorConfig(
                        acceptance_threshold=_ACCEPTANCE_THRESHOLD,
                        ambiguity_margin=_RECOVERY_AMBIGUITY_MARGIN,
                        spatial_cap=_DEFAULT_SPATIAL_CAP,
                        recovery_mode=True,
                        strategy=SelectorRecoveryStrategy.FALLBACK_BEST_CANDIDATE,
                        allow_ratio_override=True,
                        selector_mode=config.selector_mode,
                        selection_context=config.selection_context,
                    )
            if initial.top_candidates and initial.top_candidates[0].total_score >= _ACCEPTANCE_THRESHOLD:
                return _SelectorConfig(
                    acceptance_threshold=_ACCEPTANCE_THRESHOLD,
                    ambiguity_margin=_RECOVERY_AMBIGUITY_MARGIN,
                    spatial_cap=_DEFAULT_SPATIAL_CAP,
                    recovery_mode=True,
                    strategy=SelectorRecoveryStrategy.MARGIN_RELAXATION,
                    selector_mode=config.selector_mode,
                    selection_context=config.selection_context,
                )
            return None

        if (
            initial.failure_reason is FailureCategory.SELECTOR_NO_CANDIDATES_AFTER_FILTERING
            and initial.candidates_before_filtering > 0
        ):
            return _SelectorConfig(
                acceptance_threshold=_RECOVERY_ACCEPTANCE_THRESHOLD,
                ambiguity_margin=_AMBIGUITY_MARGIN,
                spatial_cap=_RECOVERY_SPATIAL_CAP,
                recovery_mode=True,
                strategy=SelectorRecoveryStrategy.CONTEXTUAL_BOOST,
                selector_mode=config.selector_mode,
                selection_context=config.selection_context,
            )

        return None

    def _build_trace(
        self,
        intent: TargetIntent,
        attempt: _AttemptResult,
        *,
        initial: _AttemptResult | None = None,
        recovery_config: _SelectorConfig | None = None,
        config: _SelectorConfig | None = None,
    ) -> SelectorTrace:
        recovery_attempted = initial is not None
        final_success = attempt.selected is not None
        return SelectorTrace(
            selector_mode=(
                recovery_config.selector_mode
                if recovery_config is not None
                else config.selector_mode
                if config is not None
                else SelectorMode.INITIAL
            ),
            intent=intent,
            candidate_count=attempt.candidate_count,
            top_candidates=attempt.top_candidates,
            selected_element_id=attempt.selected_element_id,
            decision_reason=attempt.decision_reason,
            rejection_reason=attempt.failure_reason,
            score_margin=attempt.score_margin,
            initial_failure_reason=initial.failure_reason if initial is not None else None,
            recovery_attempted=recovery_attempted,
            recovery_strategy_used=recovery_config.strategy if recovery_config is not None else None,
            adjusted_acceptance_threshold=recovery_config.acceptance_threshold if recovery_config is not None else None,
            adjusted_ambiguity_margin=recovery_config.ambiguity_margin if recovery_config is not None else None,
            final_decision=SelectorFinalDecision.SUCCESS if final_success else SelectorFinalDecision.FAILURE,
            final_stop_reason=(
                StopReason.SELECTOR_RECOVERY_USED
                if recovery_attempted and final_success
                else StopReason.SELECTOR_RECOVERY_FAILED
                if recovery_attempted
                else None
            ),
            recovery_changed_selected_candidate=(initial is not None and initial.selected_element_id != attempt.selected_element_id),
        )

    def _passes_prefilter(self, element: UIElement, intent: TargetIntent) -> bool:
        if not element.is_interactable:
            return False
        if element.element_type not in _ACTION_COMPATIBILITY[intent.action]:
            return False
        if intent.expected_element_types and element.element_type not in intent.expected_element_types:
            return False
        return True

    def _passes_hard_filters(
        self,
        perception: ScreenPerception,
        element: UIElement,
        intent: TargetIntent,
        *,
        config: _SelectorConfig,
    ) -> tuple[bool, list[str], bool]:
        rejected_by: list[str] = []
        if not element.is_interactable:
            rejected_by.append("not_interactable")
            return False, rejected_by, False

        action_compatible = element.element_type in _ACTION_COMPATIBILITY[intent.action]
        if not action_compatible:
            rejected_by.append("action_incompatible")
            return False, rejected_by, False

        if intent.expected_element_types and element.element_type not in intent.expected_element_types:
            rejected_by.append("unexpected_element_type")
            return False, rejected_by, True

        if intent.target_role is not None and element.role is not None and self._normalize_text(element.role) != self._normalize_text(intent.target_role):
            rejected_by.append("unexpected_role")
            return False, rejected_by, True

        if element.is_unlabeled and not self._unlabeled_fallback_allowed(perception, element, intent, recovery_mode=config.recovery_mode):
            rejected_by.append("unlabeled_without_grounding")
            return False, rejected_by, True

        return True, rejected_by, True

    def _score_candidate(
        self,
        perception: ScreenPerception,
        element: UIElement,
        intent: TargetIntent,
        text_candidates: list[UIElement],
        visual_groups: dict[str, int],
        *,
        config: _SelectorConfig,
    ) -> TargetEvidence:
        score = 18.0
        matched_signals: list[str] = ["action_compatible"]
        exact_semantic_match = False
        target_tokens = self._tokenize(intent.target_text)
        values = {
            "primary_name": element.primary_name,
            "label": element.label,
            "text": element.text,
            "placeholder": element.placeholder,
            "name": element.name,
        }

        for field_name, value in values.items():
            if intent.target_text is None or not value:
                continue
            normalized_value = self._normalize_text(value)
            normalized_target = self._normalize_text(intent.target_text)
            if normalized_value == normalized_target:
                score += _EXACT_FIELD_WEIGHTS[field_name]
                matched_signals.append(f"exact_{field_name}")
                exact_semantic_match = True
                continue

            field_tokens = self._tokenize(value)
            overlap = self._token_overlap(target_tokens, field_tokens)
            if overlap >= 1.0:
                score += _FUZZY_FIELD_WEIGHTS[field_name]
                matched_signals.append(f"token_full_{field_name}")
            elif overlap >= 0.67:
                score += round(_FUZZY_FIELD_WEIGHTS[field_name] * 0.7, 2)
                matched_signals.append(f"token_strong_{field_name}")
            elif overlap >= 0.5:
                score += round(_FUZZY_FIELD_WEIGHTS[field_name] * 0.45, 2)
                matched_signals.append(f"token_partial_{field_name}")

        direct_semantics_present = exact_semantic_match or any(signal.startswith("token_") for signal in matched_signals)
        spatial_match = self._best_spatial_match(
            element=element,
            intent=intent,
            target_tokens=target_tokens,
            text_candidates=text_candidates,
            visual_groups=visual_groups,
        )
        if spatial_match is not None:
            spatial_score = spatial_match.score
            if direct_semantics_present:
                spatial_score = min(spatial_score, config.spatial_cap)
            score += spatial_score
            matched_signals.extend(spatial_match.signals)

        if intent.target_role is not None and element.role is not None:
            if self._normalize_text(intent.target_role) == self._normalize_text(element.role):
                score += 20.0
                matched_signals.append("exact_role")

        if intent.expected_section is not None:
            section_tokens = self._tokenize(intent.expected_section)
            haystack = self._tokenize(" ".join(value for value in (element.element_id, element.primary_name, element.label, element.name) if value))
            overlap = self._token_overlap(section_tokens, haystack)
            if overlap >= 0.5:
                score += 10.0
                matched_signals.append("expected_section")

        if target_tokens:
            id_overlap = self._token_overlap(target_tokens, self._tokenize(element.element_id))
            if id_overlap >= 1.0:
                score += 24.0
                matched_signals.append("exact_element_id_tokens")
            elif id_overlap >= 0.5:
                score += 12.0
                matched_signals.append("partial_element_id_tokens")

        if element.element_type in intent.expected_element_types:
            score += 10.0
            matched_signals.append("expected_element_type")

        if element.confidence < 0.65:
            score -= 12.0
            matched_signals.append("low_model_confidence_penalty")

        duplicate_penalty = self._duplicate_name_penalty(perception, element)
        if duplicate_penalty:
            score -= duplicate_penalty
            matched_signals.append("duplicate_name_penalty")

        uses_unlabeled_fallback = element.name_source is UIElementNameSource.SYNTHETIC
        if uses_unlabeled_fallback:
            score -= 28.0
            matched_signals.append("synthetic_name_penalty")

        if config.selection_context is not None:
            score += self._reresolution_support_score(element, matched_signals, config.selection_context)

        confidence_band = self._confidence_band(score)
        return TargetEvidence(
            element_id=element.element_id,
            element_type=element.element_type,
            primary_name=element.primary_name,
            total_score=round(score, 2),
            matched_signals=matched_signals,
            rejected_by=[],
            action_compatible=True,
            exact_semantic_match=exact_semantic_match,
            uses_unlabeled_fallback=uses_unlabeled_fallback,
            nearest_matched_text_candidate_id=spatial_match.label.element_id if spatial_match is not None else None,
            spatial_grounding_contributed=spatial_match is not None,
            confidence_band=confidence_band,
        )

    def _reresolution_support_score(
        self,
        element: UIElement,
        matched_signals: list[str],
        selection_context: TargetSelectionContext,
    ) -> float:
        score = 0.0
        original = selection_context.original_target

        if element.element_id == original.element_id:
            score += _RERESOLUTION_PRIOR_ID_EXACT_WEIGHT
            matched_signals.append("reresolution_exact_prior_element_id")
        else:
            id_overlap = self._token_overlap(self._tokenize(original.element_id), self._tokenize(element.element_id))
            if id_overlap >= 0.5:
                score += _RERESOLUTION_PRIOR_ID_TOKEN_WEIGHT
                matched_signals.append("reresolution_prior_element_id_tokens")

        if self._normalize_text(element.primary_name) == self._normalize_text(original.primary_name):
            score += _RERESOLUTION_PRIOR_NAME_EXACT_WEIGHT
            matched_signals.append("reresolution_exact_prior_primary_name")
        else:
            name_overlap = self._token_overlap(self._tokenize(original.primary_name), self._tokenize(element.primary_name))
            if name_overlap >= 0.67:
                score += _RERESOLUTION_PRIOR_NAME_TOKEN_WEIGHT
                matched_signals.append("reresolution_prior_primary_name_tokens")

        original_center_x = original.x + (original.width / 2.0)
        original_center_y = original.y + (original.height / 2.0)
        candidate_center_x = element.x + (element.width / 2.0)
        candidate_center_y = element.y + (element.height / 2.0)
        distance = ((candidate_center_x - original_center_x) ** 2 + (candidate_center_y - original_center_y) ** 2) ** 0.5
        if distance <= 120.0:
            score += _RERESOLUTION_REGION_NEAR_WEIGHT
            matched_signals.append("reresolution_prior_region_near")
        elif distance <= 240.0:
            score += _RERESOLUTION_REGION_MID_WEIGHT
            matched_signals.append("reresolution_prior_region_mid")

        continuity_signals = set(selection_context.original_matched_signals) & set(matched_signals)
        continuity_signals.discard("action_compatible")
        continuity_score = min(len(continuity_signals) * 1.5, _RERESOLUTION_SIGNAL_CONTINUITY_CAP)
        if continuity_score > 0:
            score += continuity_score
            matched_signals.append("reresolution_signal_continuity")

        return score

    def _best_spatial_match(
        self,
        *,
        element: UIElement,
        intent: TargetIntent,
        target_tokens: set[str],
        text_candidates: list[UIElement],
        visual_groups: dict[str, int],
    ) -> _SpatialMatch | None:
        if intent.target_text is None:
            return None

        matches: list[_SpatialMatch] = []
        normalized_target = self._normalize_text(intent.target_text)
        element_group = visual_groups.get(element.element_id)
        for label in text_candidates:
            relation_signal, relation_weight = self._spatial_relation(label, element, normalized_target, target_tokens)
            if relation_signal is None:
                continue

            signals = [relation_signal]
            score = relation_weight
            if bbox_distance(label, element) <= NEARBY_LABEL_MAX_DISTANCE_PX:
                score += _SPATIAL_WEIGHTS["nearest_label_match"]
                signals.append("nearest_label_match")
            if element_group is not None and visual_groups.get(label.element_id) == element_group:
                score += _SPATIAL_WEIGHTS["same_group_context_match"]
                signals.append("same_group_context_match")
            matches.append(_SpatialMatch(label=label, score=score, signals=signals))

        if not matches:
            return None

        matches.sort(key=lambda match: (match.score, -bbox_distance(match.label, element), match.label.element_id), reverse=True)
        return matches[0]

    def _spatial_relation(
        self,
        label: UIElement,
        element: UIElement,
        normalized_target: str,
        target_tokens: set[str],
    ) -> tuple[str | None, float]:
        normalized_label = self._normalize_text(label.primary_name)
        label_tokens = self._tokenize(label.primary_name)
        exact = normalized_label == normalized_target
        overlap = self._token_overlap(target_tokens, label_tokens)

        if is_above(label, element) and same_column(label, element):
            if exact:
                return "nearby_label_above_exact_match", _SPATIAL_WEIGHTS["nearby_label_above_exact_match"]
            if overlap >= 0.67:
                return "nearby_label_above_fuzzy_match", _SPATIAL_WEIGHTS["nearby_label_above_fuzzy_match"]

        if is_left_of(label, element) and same_row(label, element):
            if exact:
                return "nearby_label_left_exact_match", _SPATIAL_WEIGHTS["nearby_label_left_exact_match"]
            if overlap >= 0.67:
                return "nearby_label_left_fuzzy_match", _SPATIAL_WEIGHTS["nearby_label_left_fuzzy_match"]

        return None, 0.0

    def _label_like_text_candidates(self, perception: ScreenPerception) -> list[UIElement]:
        candidates: list[UIElement] = []
        for element in perception.visible_elements:
            if element.element_type in {UIElementType.BUTTON, UIElementType.LINK, UIElementType.INPUT, UIElementType.DIALOG, UIElementType.ICON}:
                continue
            if element.is_interactable:
                continue
            if not any((element.label, element.text, element.name, element.primary_name)):
                continue
            tokens = self._tokenize(element.primary_name)
            if tokens & {"submit", "send", "cancel", "close", "next", "back"}:
                continue
            candidates.append(element)
        return candidates

    def _visual_groups(self, perception: ScreenPerception) -> dict[str, int]:
        ordered = sorted(perception.visible_elements, key=lambda element: (element.y, element.x, element.element_id))
        groups: dict[str, int] = {}
        current_group = -1
        previous_y: int | None = None
        for element in ordered:
            if previous_y is None or abs(element.y - previous_y) > GROUP_VERTICAL_GAP_PX:
                current_group += 1
            groups[element.element_id] = current_group
            previous_y = element.y
        return groups

    @staticmethod
    def _unlabeled_fallback_allowed(
        perception: ScreenPerception,
        element: UIElement,
        intent: TargetIntent,
        *,
        recovery_mode: bool,
    ) -> bool:
        if intent.target_text is None:
            compatible_elements = [
                candidate
                for candidate in perception.visible_elements
                if candidate.is_interactable and candidate.element_type in _ACTION_COMPATIBILITY[intent.action]
            ]
            return len(compatible_elements) == 1

        target_tokens = DeterministicTargetSelector._tokenize(intent.target_text)
        grounding_tokens = (
            DeterministicTargetSelector._tokenize(element.element_id)
            | DeterministicTargetSelector._tokenize(element.role)
            | DeterministicTargetSelector._tokenize(element.placeholder)
        )
        minimum_overlap = 0.5 if recovery_mode else 0.67
        if bool(target_tokens) and DeterministicTargetSelector._token_overlap(target_tokens, grounding_tokens) >= minimum_overlap:
            return True

        for candidate in perception.visible_elements:
            if candidate.is_interactable:
                continue
            if candidate.element_type is not UIElementType.TEXT:
                continue
            relation = (is_above(candidate, element) and same_column(candidate, element)) or (
                is_left_of(candidate, element) and same_row(candidate, element)
            )
            if not relation:
                continue
            if bbox_distance(candidate, element) > NEARBY_LABEL_MAX_DISTANCE_PX:
                continue
            label_tokens = DeterministicTargetSelector._tokenize(candidate.primary_name)
            if target_tokens and DeterministicTargetSelector._token_overlap(target_tokens, label_tokens) >= minimum_overlap:
                return True
        return False

    @staticmethod
    def _duplicate_name_penalty(perception: ScreenPerception, element: UIElement) -> float:
        same_name_count = sum(
            1
            for candidate in perception.visible_elements
            if candidate.element_id != element.element_id
            and candidate.is_interactable
            and candidate.primary_name == element.primary_name
        )
        if same_name_count:
            return 10.0

        related_labels = [
            candidate
            for candidate in perception.visible_elements
            if candidate.element_id != element.element_id
            and candidate.element_type is UIElementType.TEXT
            and bbox_distance(candidate, element) <= NEARBY_LABEL_MAX_DISTANCE_PX
        ]
        close_label_count = sum(
            1
            for candidate in related_labels
            if same_row(candidate, element) or same_column(candidate, element) or horizontal_overlap(candidate, element) >= 0.4
        )
        return 8.0 if close_label_count > 1 else 0.0

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        if not value:
            return ""
        normalized = "".join(character.lower() if character.isalnum() else " " for character in value)
        return " ".join(normalized.split())

    @classmethod
    def _tokenize(cls, value: str | None) -> set[str]:
        return {token for token in cls._normalize_text(value).split() if len(token) >= 2}

    @staticmethod
    def _token_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left)

    @staticmethod
    def _confidence_band(score: float) -> SelectorConfidenceBand:
        if score >= 120.0:
            return SelectorConfidenceBand.HIGH
        if score >= 85.0:
            return SelectorConfidenceBand.MEDIUM
        return SelectorConfidenceBand.LOW

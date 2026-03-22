"""Typed schemas for deterministic target selection."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from src.models.common import FailureCategory, StopReason, StrictModel
from src.models.perception import UIElementType


class TargetIntentAction(StrEnum):
    """Normalized interaction kinds used by the deterministic selector."""

    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"


class SelectorConfidenceBand(StrEnum):
    """Human-readable confidence buckets for candidate evidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SelectorRecoveryStrategy(StrEnum):
    """Deterministic fallback strategies for one bounded recovery attempt."""

    THRESHOLD_RELAXATION = "threshold_relaxation"
    MARGIN_RELAXATION = "margin_relaxation"
    CONTEXTUAL_BOOST = "contextual_boost"
    FALLBACK_BEST_CANDIDATE = "fallback_best_candidate"


class SelectorFinalDecision(StrEnum):
    """Final selector outcome after the initial pass and optional recovery."""

    SUCCESS = "success"
    FAILURE = "failure"


class SelectorMode(StrEnum):
    """Whether the selector is resolving an initial target or re-resolving one."""

    INITIAL = "initial"
    RERESOLUTION = "reresolution"


class TargetIntent(StrictModel):
    """Deterministic selector intent derived from a subgoal and action context."""

    action: TargetIntentAction
    target_text: str | None = Field(default=None, min_length=1)
    target_role: str | None = Field(default=None, min_length=1)
    expected_element_types: list[UIElementType] = Field(default_factory=list)
    value_to_type: str | None = Field(default=None, min_length=1)
    expected_section: str | None = Field(default=None, min_length=1)

    @field_validator("target_text", "target_role", "value_to_type", "expected_section", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None


class OriginalTargetSignature(StrictModel):
    """Lightweight signature of the original resolved target."""

    element_id: str = Field(min_length=1)
    element_type: UIElementType
    primary_name: str = Field(min_length=1)
    role: str | None = Field(default=None, min_length=1)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class TargetEvidence(StrictModel):
    """Per-candidate selector reasoning for replay and debug artifacts."""

    element_id: str = Field(min_length=1)
    element_type: UIElementType
    primary_name: str = Field(min_length=1)
    total_score: float
    matched_signals: list[str] = Field(default_factory=list)
    rejected_by: list[str] = Field(default_factory=list)
    action_compatible: bool
    exact_semantic_match: bool
    uses_unlabeled_fallback: bool
    nearest_matched_text_candidate_id: str | None = Field(default=None, min_length=1)
    spatial_grounding_contributed: bool = False
    confidence_band: SelectorConfidenceBand


class TargetSelectionContext(StrictModel):
    """Serializable context carried forward for deterministic target re-resolution."""

    intent: TargetIntent
    original_target: OriginalTargetSignature
    selected_candidate_evidence: TargetEvidence
    top_candidates: list[TargetEvidence] = Field(default_factory=list)
    original_matched_signals: list[str] = Field(default_factory=list)
    original_page_signature: str | None = Field(default=None, min_length=1)


class SelectorTrace(StrictModel):
    """Compact decision trace for one selector attempt."""

    selector_mode: SelectorMode = SelectorMode.INITIAL
    intent: TargetIntent
    candidate_count: int = Field(ge=0)
    top_candidates: list[TargetEvidence] = Field(default_factory=list)
    selected_element_id: str | None = Field(default=None, min_length=1)
    decision_reason: str = Field(min_length=1)
    rejection_reason: FailureCategory | None = None
    score_margin: float | None = None
    initial_failure_reason: FailureCategory | None = None
    recovery_attempted: bool = False
    recovery_strategy_used: SelectorRecoveryStrategy | None = None
    adjusted_acceptance_threshold: float | None = None
    adjusted_ambiguity_margin: float | None = None
    final_decision: SelectorFinalDecision = SelectorFinalDecision.FAILURE
    final_stop_reason: StopReason | None = None
    recovery_changed_selected_candidate: bool = False

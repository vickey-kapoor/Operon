"""Tests for native failure-category → strategy lookup.

This module replaces the previous indirection through the unified-contract
layer. These tests pin the lookup behavior so future changes can't silently
reintroduce contract-driven retry decisions.
"""

from __future__ import annotations

from src.agent.adaptation import (
    FOCUS_CORRECTION_THEN_RETRY,
    REFRESH_STATE_AND_REPLAN,
    REPERCEIVE_AND_REPLAN,
    WAIT_THEN_RETRY,
    strategy_for_failure,
)
from src.models.common import FailureCategory


def test_specific_category_returns_specific_strategy() -> None:
    assert strategy_for_failure(FailureCategory.EXECUTION_TARGET_NOT_FOUND) == REPERCEIVE_AND_REPLAN
    assert strategy_for_failure(FailureCategory.FOCUS_VERIFICATION_FAILED) == FOCUS_CORRECTION_THEN_RETRY
    assert strategy_for_failure(FailureCategory.PICKER_NOT_DETECTED) == WAIT_THEN_RETRY
    assert strategy_for_failure(FailureCategory.PERCEPTION_LOW_QUALITY) == REFRESH_STATE_AND_REPLAN


def test_no_category_no_status_returns_none() -> None:
    assert strategy_for_failure(None) is None


def test_no_category_match_with_failure_falls_through_to_wait_then_retry() -> None:
    """Catch-all behavior preserved from the legacy adapter: a verification
    failure with no specific category still gets one wait_then_retry shot."""
    assert (
        strategy_for_failure(
            FailureCategory.EXPECTED_OUTCOME_NOT_MET, verification_failure=True
        )
        == WAIT_THEN_RETRY
    )
    assert strategy_for_failure(None, verification_failure=True) == WAIT_THEN_RETRY


def test_no_category_match_with_uncertain_falls_through_to_refresh() -> None:
    assert (
        strategy_for_failure(None, verification_uncertain=True) == REFRESH_STATE_AND_REPLAN
    )


def test_specific_match_wins_over_status_fallback() -> None:
    """If a specific category matches, the table entry wins over the catch-all."""
    assert (
        strategy_for_failure(
            FailureCategory.EXECUTION_TARGET_NOT_FOUND,
            verification_failure=True,
            verification_uncertain=True,
        )
        == REPERCEIVE_AND_REPLAN
    )

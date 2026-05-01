"""Native failure→strategy mapping for in-step adaptation retries.

Replaces the previous indirection through the unified contract layer
(LegacyOperonContractAdapter → UnifiedOrchestrator.adaptation_strategy_for)
with a direct dict lookup keyed by the legacy FailureCategory enum.

Behavior is preserved: the same FailureCategory inputs produce the same
strategy outputs as before, but the call no longer requires a contract
translation.
"""

from __future__ import annotations

from src.models.common import FailureCategory

# Strategies a downstream consumer might read; kept opaque (just strings)
# because the loop and recovery logic only needs to compare equality.
WAIT_THEN_RETRY = "wait_then_retry"
FOCUS_CORRECTION_THEN_RETRY = "focus_correction_then_retry"
REPERCEIVE_AND_REPLAN = "reperceive_and_replan"
REFRESH_STATE_AND_REPLAN = "refresh_state_and_replan"

# Combines the FailureCategory→FailureType mapping (was in
# src/runtime/legacy_adapter._map_failure_type) with the FailureType→strategy
# mapping (was in src/runtime/orchestrator.UnifiedOrchestrator.adaptation_strategy_for).
_STRATEGY_TABLE: dict[FailureCategory, str] = {
    FailureCategory.EXECUTION_TARGET_NOT_FOUND: REPERCEIVE_AND_REPLAN,
    FailureCategory.SELECTOR_NO_CANDIDATES_AFTER_FILTERING: REPERCEIVE_AND_REPLAN,
    FailureCategory.TARGET_RERESOLUTION_FAILED: REPERCEIVE_AND_REPLAN,
    FailureCategory.TARGET_LOST_BEFORE_ACTION: REPERCEIVE_AND_REPLAN,
    FailureCategory.FOCUS_VERIFICATION_FAILED: FOCUS_CORRECTION_THEN_RETRY,
    FailureCategory.CLICK_BEFORE_TYPE_FAILED: FOCUS_CORRECTION_THEN_RETRY,
    FailureCategory.TYPE_VERIFICATION_FAILED: WAIT_THEN_RETRY,
    FailureCategory.EXECUTION_TARGET_NOT_EDITABLE: WAIT_THEN_RETRY,
    FailureCategory.STALE_TARGET_BEFORE_ACTION: REPERCEIVE_AND_REPLAN,
    FailureCategory.TARGET_SHIFTED_BEFORE_ACTION: REPERCEIVE_AND_REPLAN,
    FailureCategory.UNCERTAIN_SCREEN_STATE: REFRESH_STATE_AND_REPLAN,
    FailureCategory.PERCEPTION_LOW_QUALITY: REFRESH_STATE_AND_REPLAN,
    FailureCategory.AMBIGUOUS_TARGET_CANDIDATES: REFRESH_STATE_AND_REPLAN,
    FailureCategory.PICKER_NOT_DETECTED: WAIT_THEN_RETRY,
    FailureCategory.FILE_NOT_REFLECTED: REPERCEIVE_AND_REPLAN,
}


def strategy_for_failure(
    category: FailureCategory | None,
    *,
    verification_failure: bool = False,
    verification_uncertain: bool = False,
) -> str | None:
    """Return the deterministic adaptation strategy for a failure category, or None.

    Preserves the previous catch-all behavior:
    - Specific category match → strategy from the table.
    - No category match but verification.status is FAILURE → wait_then_retry
      (treated as a transient timing issue worth one more shot).
    - No category match but verification.status is UNCERTAIN → refresh_state_and_replan
      (re-perceive on the assumption the screen state is ambiguous).
    - Otherwise → None (no auto-retry; outer recovery layer decides).
    """
    if category is not None:
        mapped = _STRATEGY_TABLE.get(category)
        if mapped is not None:
            return mapped
    if verification_failure:
        return WAIT_THEN_RETRY
    if verification_uncertain:
        return REFRESH_STATE_AND_REPLAN
    return None

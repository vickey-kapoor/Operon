"""Per-step retry resolution for executor failures.

Encapsulates the decision tree that the loop runs after a single execute
attempt fails with a recoverable category:

1. Should this failure trigger an in-step retry at all? (`should_retry`)
2. If yes, do we need a target re-resolution (stale/shifted) or just refreshed
   coordinates?  (`resolve_retry_action` -> `intent_reresolve_action` /
   `refresh_action_coordinates`)
3. Merge or apply-failure on the second attempt's trace so downstream logging
   reflects both attempts as one ExecutedAction.

Stateless aside from the injected `DeterministicTargetSelector`. The decision
table lives here so the AgentLoop only needs to call `harden(...)`-style
methods through delegators.
"""

from __future__ import annotations

from dataclasses import dataclass

from operon.agent.actions.grounding import DeterministicGrounder, element_center
from operon.models.common import FailureCategory, LoopStage
from operon.models.execution import ExecutionReresolutionTrace
from operon.models.policy import AgentAction


@dataclass(slots=True)
class RetryResolution:
    action: AgentAction | None
    trace: ExecutionReresolutionTrace | None


_RETRYABLE_CATEGORIES: frozenset[FailureCategory] = frozenset({
    FailureCategory.STALE_TARGET_BEFORE_ACTION,
    FailureCategory.TARGET_SHIFTED_BEFORE_ACTION,
    FailureCategory.TARGET_LOST_BEFORE_ACTION,
    FailureCategory.FOCUS_VERIFICATION_FAILED,
    FailureCategory.CLICK_BEFORE_TYPE_FAILED,
    FailureCategory.CLICK_NO_EFFECT,
    FailureCategory.CHECKBOX_VERIFICATION_FAILED,
    FailureCategory.SELECT_VERIFICATION_FAILED,
})

_RERESOLVE_CATEGORIES: frozenset[FailureCategory] = frozenset({
    FailureCategory.STALE_TARGET_BEFORE_ACTION,
    FailureCategory.TARGET_SHIFTED_BEFORE_ACTION,
    FailureCategory.TARGET_LOST_BEFORE_ACTION,
})


def should_retry(executed_action) -> bool:
    return executed_action.failure_category in _RETRYABLE_CATEGORIES


def refresh_action_coordinates(action, perception):
    target_id = action.target_element_id
    if target_id is None:
        return action
    target = next((element for element in perception.visible_elements if element.element_id == target_id), None)
    if target is None:
        return action
    cx, cy = element_center(target)
    return action.model_copy(update={"x": cx, "y": cy})


def merge_execution_retry(*, original, retried, retry_reason, target_reresolved, reresolution_trace):
    original_trace = original.execution_trace
    retried_trace = retried.execution_trace
    if original_trace is None or retried_trace is None:
        return retried
    merged_trace = original_trace.model_copy(
        update={
            "attempts": [*original_trace.attempts, *retried_trace.attempts],
            "target_reresolved": target_reresolved,
            "retry_attempted": True,
            "retry_reason": retry_reason,
            "reresolution_trace": reresolution_trace,
            "final_outcome": retried_trace.final_outcome,
            "final_failure_category": retried_trace.final_failure_category,
        }
    )
    return retried.model_copy(update={"execution_trace": merged_trace})


def apply_reresolution_failure(original, reresolution_trace: ExecutionReresolutionTrace | None):
    trace = original.execution_trace
    if trace is None or reresolution_trace is None:
        return original
    failure_category = (
        FailureCategory.TARGET_RERESOLUTION_AMBIGUOUS
        if reresolution_trace.selector_trace.rejection_reason is FailureCategory.AMBIGUOUS_TARGET_CANDIDATES
        else FailureCategory.TARGET_RERESOLUTION_FAILED
    )
    merged_trace = trace.model_copy(
        update={
            "retry_attempted": True,
            "retry_reason": reresolution_trace.trigger_reason,
            "reresolution_trace": reresolution_trace,
            "final_outcome": "failure",
            "final_failure_category": failure_category,
        }
    )
    return original.model_copy(
        update={
            "success": False,
            "detail": f"Execution failed: {failure_category.value.replace('_', ' ')}.",
            "failure_category": failure_category,
            "failure_stage": LoopStage.EXECUTE,
            "execution_trace": merged_trace,
        }
    )


class RetryHardening:
    """Owns retry-action resolution that requires the DeterministicTargetSelector."""

    def __init__(self, target_selector) -> None:
        self.target_selector = target_selector
        self._grounder = DeterministicGrounder(target_selector)

    def resolve_retry_action(self, *, action, perception, retry_reason: FailureCategory | None) -> RetryResolution:
        if retry_reason in _RERESOLVE_CATEGORIES:
            return self.intent_reresolve_action(action, perception, retry_reason)
        return RetryResolution(action=refresh_action_coordinates(action, perception), trace=None)

    def intent_reresolve_action(
        self,
        action: AgentAction,
        perception,
        retry_reason: FailureCategory,
    ) -> RetryResolution:
        if action.target_context is None:
            return RetryResolution(action=refresh_action_coordinates(action, perception), trace=None)

        result = self._grounder.ground(
            action.target_context.intent, perception, prior_context=action.target_context
        )
        trace = ExecutionReresolutionTrace(
            trigger_reason=retry_reason,
            original_target_element_id=action.target_context.original_target.element_id,
            original_intent=action.target_context.intent,
            original_target_signature=action.target_context.original_target,
            original_page_signature=action.target_context.original_page_signature,
            selector_trace=result.trace,
            reused_original_element_id=(
                result.grounded and result.element_id == action.target_context.original_target.element_id
            ),
            final_target_element_id=result.element_id,
            succeeded=result.grounded,
            detail=(
                f"Re-resolved target to {result.element_id} from original intent."
                if result.grounded
                else "Intent-based target re-resolution did not find a safe deterministic match."
            ),
        )
        if not result.grounded:
            return RetryResolution(action=None, trace=trace)

        return RetryResolution(
            action=action.model_copy(
                update={
                    "target_element_id": result.element_id,
                    "x": result.x,
                    "y": result.y,
                }
            ),
            trace=trace,
        )

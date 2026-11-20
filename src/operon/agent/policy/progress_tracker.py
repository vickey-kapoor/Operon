"""Loop-progress tracking: signatures, redundancy detection, and stop guards.

Decoupled from AgentLoop so the pure-function layer (signatures, novelty,
alternating-pattern detection) is independently testable. Stateful operations
that mutate `state.progress_state` live on `ProgressTracker` and take state as
an explicit argument — no hidden coupling to a god-object self.
"""

from __future__ import annotations

from pathlib import Path

from operon.core.settings import get_settings
from operon.models.common import FailureCategory, LoopStage, StopReason
from operon.models.policy import ActionType, AgentAction
from operon.models.progress import ProgressTrace

# Window size for recent_actions / recent_failures — small enough to detect
# alternating period-2/3 loops but not so large that one bad page early in a
# run permanently colors loop detection.
RECENT_WINDOW_SIZE = 6

# RFC 0003: after-screenshots kept per action signature, and how many signatures.
REVISIT_OUTCOMES_PER_ACTION = 3
REVISIT_MAX_ACTIONS = 20


def _screen_changed(screen_change_ratio: float | None) -> bool:
    from operon.agent.perception.screen_diff import SCREEN_CHANGE_THRESHOLD

    return screen_change_ratio is not None and screen_change_ratio >= SCREEN_CHANGE_THRESHOLD


def progress_revisit_enabled() -> bool:
    """``OPERON_PROGRESS_REVISIT`` (RFC 0003): off unless set to on/true/1/yes."""
    return get_settings().progress_revisit


# --- pure signature helpers ---------------------------------------------------


def page_signature(perception) -> str:
    top_elements = perception.visible_elements[:15]
    element_part = "|".join(f"{element.element_id}:{element.primary_name}" for element in top_elements) or "none"
    focused = perception.focused_element_id or "none"
    return f"{perception.page_hint.value}|{focused}|{element_part}"


def target_signature(action: AgentAction) -> str | None:
    if action.target_element_id is not None:
        return f"id:{action.target_element_id}"
    if action.selector is not None:
        return f"selector:{action.selector}"
    if action.x is not None and action.y is not None:
        # Bucket coordinates to 50px grid so nearby clicks are detected as repeats
        bx, by = action.x // 50 * 50, action.y // 50 * 50
        return f"xy:{bx}:{by}"
    return None


def action_signature(action: AgentAction) -> str:
    if action.action_type is ActionType.BATCH:
        # Two different batches are different actions: sign by their children.
        children = ";".join(action_signature(child) for child in action.actions or [])
        return f"{action.action_type.value}|[{children}]"
    target = target_signature(action) or "no_target"
    payload = action.text or action.key or action.url or (str(action.wait_ms) if action.wait_ms is not None else "")
    return f"{action.action_type.value}|{target}|{payload.strip().lower()}"


def subgoal_signature(subgoal: str | None) -> str:
    if not subgoal:
        return "unknown_subgoal"
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in subgoal)
    return "_".join(part for part in normalized.split("_") if part) or "unknown_subgoal"


def failure_signature(executed_action, verification, recovery, target_sig: str | None) -> str | None:
    category = recovery.failure_category or verification.failure_category or executed_action.failure_category
    if category is None:
        return None
    target_part = target_sig or "no_target"
    return f"{category.value}|{target_part}"


def stop_reason_for_failure(category: FailureCategory | None) -> StopReason | None:
    if category is None:
        return None
    return StopReason(category.value)


def append_window(entries: list[str], value: str) -> list[str]:
    return [*entries, value][-RECENT_WINDOW_SIZE:]


def alternating_action_loop(recent_actions: list[str]) -> bool:
    # Period-2: abab
    if len(recent_actions) >= 4:
        a, b, c, d = recent_actions[-4:]
        if a == c and b == d and a != b:
            return True
    # Period-3: abcabc
    if len(recent_actions) >= 6:
        tail = recent_actions[-6:]
        if tail[:3] == tail[3:] and len(set(tail[:3])) > 1:
            return True
    return False


def meaningful_progress(
    executed_action,
    verification,
    *,
    subgoal_changed: bool = True,
    screen_change_ratio: float | None = None,
    is_novel_action: bool = True,
) -> bool:
    """Progress requires visual change AND a novel action.

    Signals:
    - screen_change_ratio: did the screen actually change?
    - is_novel_action: is this a new action signature (not seen before)?
    - subgoal_changed: did the model advance its subgoal?

    Screen change + novel action = progress (typing text in Notepad).
    Screen change + repeated action = NOT progress (launch_app Notepad 5x).
    No screen change = NOT progress regardless.
    Subgoal change always counts as progress (model advancing intentionally).
    """
    if not executed_action.success:
        return False
    if verification.stop_condition_met:
        return True
    if subgoal_changed:
        return True

    if screen_change_ratio is not None:
        from operon.agent.perception.screen_diff import SCREEN_CHANGE_THRESHOLD
        screen_changed = screen_change_ratio >= SCREEN_CHANGE_THRESHOLD
        return screen_changed and is_novel_action

    # Screenshots unavailable — require novelty but don't treat as confirmed progress;
    # return False so the no-progress streak is not reset without visual evidence.
    return False


def should_mark_subgoal_complete(progress_state, executed_action, verification) -> bool:
    if not executed_action.success:
        return False
    if verification.stop_condition_met:
        return True

    action = executed_action.action
    previous_page = progress_state.previous_page_signature
    current_page = progress_state.latest_page_signature
    page_changed = previous_page is not None and current_page is not None and previous_page != current_page

    if action.action_type is ActionType.NAVIGATE:
        return True
    if action.action_type in {ActionType.TYPE, ActionType.SELECT}:
        return page_changed or not bool(action.press_enter)
    if action.action_type in {
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.RIGHT_CLICK,
        ActionType.PRESS_KEY,
        ActionType.HOTKEY,
        ActionType.LAUNCH_APP,
    }:
        return page_changed
    return page_changed


def apply_no_progress_detection(state, executed_action):
    trace = executed_action.execution_trace
    if trace is None or not trace.attempts:
        return executed_action
    latest_attempt = trace.attempts[-1]
    if not latest_attempt.no_progress_detected:
        return executed_action
    if not state.action_history:
        return executed_action
    previous = state.action_history[-1]
    previous_trace = previous.execution_trace
    if previous.action.target_element_id != executed_action.action.target_element_id:
        return executed_action
    if previous_trace is None or not previous_trace.attempts or not previous_trace.attempts[-1].no_progress_detected:
        return executed_action
    merged_trace = trace.model_copy(update={"final_failure_category": FailureCategory.EXECUTION_NO_PROGRESS})
    return executed_action.model_copy(
        update={
            "success": False,
            "detail": "Execution failed: repeated no-progress detected on the same target.",
            "failure_category": FailureCategory.EXECUTION_NO_PROGRESS,
            "failure_stage": LoopStage.EXECUTE,
            "execution_trace": merged_trace,
        }
    )


# --- ProgressTracker ---------------------------------------------------------


class ProgressTracker:
    """Stateless tracker — operates on `state.progress_state` passed in by the caller.

    Constants are class attributes so tests / loop-side code can introspect or
    parametrize them without instantiating the tracker.
    """

    MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS = 2
    MAX_REPEAT_SAME_TARGET_WITHOUT_PROGRESS = 2
    MAX_NO_PROGRESS_STEPS = 3
    MAX_REPEAT_SAME_FAILURE = 2

    # RFC 0003: a repeated action whose after-screen is within this ratio of an
    # after-screen it produced before reached a screen it already reached. Calibrated
    # on saved runs: legitimate repeats (scroll, switching back to an app) differ by 3%+.
    REVISIT_SAME_SCREEN_THRESHOLD = 0.01

    def sync_with_perception(self, state, perception) -> None:
        page_sig = page_signature(perception)
        progress_state = state.progress_state
        previous_signature = progress_state.latest_page_signature
        progress_state.previous_page_signature = previous_signature
        progress_state.latest_page_signature = page_sig
        if previous_signature is None or previous_signature == page_sig:
            return
        # Page changed — reset action/target repeat counters (but NOT no_progress_streak,
        # which is managed by meaningful_progress based on actual screen change detection)
        progress_state.repeated_action_count.clear()
        progress_state.repeated_target_count.clear()
        progress_state.recent_failures = []
        progress_state.loop_detected = False

    def update_progress_state(
        self,
        *,
        state,
        decision,
        executed_action,
        verification,
        recovery,
        step_index: int,
        before_artifact_path: str | None = None,
        screen_change_ratio: float | None = None,
    ) -> ProgressTrace:
        progress_state = state.progress_state
        action_sig = action_signature(executed_action.action)
        target_sig = target_signature(executed_action.action)
        subgoal_sig = subgoal_signature(decision.active_subgoal)
        failure_sig = failure_signature(executed_action, verification, recovery, target_sig)

        # Subgoal-based progress signal
        previous_subgoal = progress_state.latest_subgoal_signature
        subgoal_changed = previous_subgoal is None or subgoal_sig != previous_subgoal
        progress_state.latest_subgoal_signature = subgoal_sig

        # Use pre-computed ratio if available; compute only as fallback.
        if screen_change_ratio is None:
            after_path = executed_action.artifact_path if hasattr(executed_action, "artifact_path") else None
            if before_artifact_path and after_path:
                from operon.agent.perception.screen_diff import (
                    compute_screen_change_ratio,
                )
                screen_change_ratio = compute_screen_change_ratio(before_artifact_path, after_path)

        # An action is "novel" if its signature hasn't been seen before in this run
        is_novel_action = action_sig not in progress_state.repeated_action_count

        progress_made = meaningful_progress(
            executed_action, verification,
            subgoal_changed=subgoal_changed,
            screen_change_ratio=screen_change_ratio,
            is_novel_action=is_novel_action,
        )

        # RFC 0003: a repeat that lands on a screen it already produced is not progress.
        # The stop condition and a subgoal change still count first, as above.
        # A repeat that is already counted (not novel) but reaches a screen it never
        # produced is progress, so one revisit does not block a legitimate repeat
        # (scroll: A, B, A, C) for the rest of the run.
        revisited_screen = False
        if progress_revisit_enabled():
            after_path = getattr(executed_action, "artifact_path", None)
            screen_only = (
                executed_action.success
                and not verification.stop_condition_met
                and not subgoal_changed
            )
            if screen_only and progress_made:
                revisited_screen = self._known_screen_match(progress_state, action_sig, after_path) is True
                progress_made = not revisited_screen
            elif screen_only and not is_novel_action and _screen_changed(screen_change_ratio):
                progress_made = self._known_screen_match(progress_state, action_sig, after_path) is False
            if executed_action.success:
                self._remember_outcome(progress_state, action_sig, after_path)

        progress_state.recent_actions = append_window(progress_state.recent_actions, action_sig)
        loop_failure_category = None
        loop_pattern = None

        if progress_made:
            progress_state.no_progress_streak = 0
            progress_state.loop_detected = False
            progress_state.repeated_action_count.clear()
            progress_state.repeated_target_count.clear()
            progress_state.recent_failures = []
            progress_state.last_meaningful_progress_step = step_index
            self._mark_completed_progress(state, decision, executed_action, verification)
        else:
            progress_state.no_progress_streak += 1
            progress_state.repeated_action_count[action_sig] = progress_state.repeated_action_count.get(action_sig, 0) + 1
            if target_sig is not None:
                progress_state.repeated_target_count[target_sig] = progress_state.repeated_target_count.get(target_sig, 0) + 1
            if failure_sig is not None:
                progress_state.recent_failures = append_window(progress_state.recent_failures, failure_sig)

            loop_failure_category, loop_pattern = self.detect_loop_failure(
                progress_state=progress_state,
                action_signature=action_sig,
                target_signature=target_sig,
                failure_signature=failure_sig,
            )
            progress_state.loop_detected = loop_failure_category is not None

        # Persist the raw screen change ratio so block_redundant_action can
        # detect a Stalled State on the *next* step without recomputing it.
        if screen_change_ratio is not None:
            progress_state.last_screen_change_ratio = screen_change_ratio

        return ProgressTrace(
            step_index=step_index,
            page_signature=progress_state.latest_page_signature or "unknown_page",
            action_signature=action_sig,
            target_signature=target_sig,
            subgoal_signature=subgoal_sig,
            failure_signature=failure_sig,
            blocked_as_redundant=executed_action.failure_category
            in {
                FailureCategory.SUBGOAL_ALREADY_COMPLETED,
                FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS,
                FailureCategory.REPEATED_TARGET_WITHOUT_PROGRESS,
            },
            redundancy_reason=executed_action.failure_category
            if executed_action.failure_category
            in {
                FailureCategory.SUBGOAL_ALREADY_COMPLETED,
                FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS,
                FailureCategory.REPEATED_TARGET_WITHOUT_PROGRESS,
            }
            else None,
            loop_pattern_detected=loop_pattern,
            progress_made=progress_made,
            revisited_screen=revisited_screen,
            screen_change_ratio=screen_change_ratio,
            no_progress_streak=progress_state.no_progress_streak,
            final_failure_category=loop_failure_category,
            final_stop_reason=stop_reason_for_failure(loop_failure_category),
        )

    def _known_screen_match(self, progress_state, action_sig: str, after_path: str | None) -> bool | None:
        """Compare this after-screen with the ones the same action produced earlier in the run.

        True: near-identical to one of them (a revisit). False: different from all of them
        (a new screen). None: nothing to compare (no earlier outcome, or a screenshot is
        missing), so callers keep today's verdict. Missing files are checked first, since
        ``compute_screen_change_ratio`` reports 0.0 (identical) for a missing file."""
        earlier = [path for path in progress_state.action_outcomes.get(action_sig) or [] if Path(path).exists()]
        if not earlier or not after_path or not Path(after_path).exists():
            return None
        from operon.agent.perception.screen_diff import compute_screen_change_ratio

        return any(
            compute_screen_change_ratio(path, after_path) < self.REVISIT_SAME_SCREEN_THRESHOLD for path in earlier
        )

    @staticmethod
    def _remember_outcome(progress_state, action_sig: str, after_path: str | None) -> None:
        if not after_path:
            return
        outcomes = progress_state.action_outcomes
        paths = [*outcomes.pop(action_sig, []), after_path][-REVISIT_OUTCOMES_PER_ACTION:]
        outcomes[action_sig] = paths  # re-inserted last: most recently used
        while len(outcomes) > REVISIT_MAX_ACTIONS:
            del outcomes[next(iter(outcomes))]

    def detect_loop_failure(
        self,
        *,
        progress_state,
        action_signature: str,
        target_signature: str | None,
        failure_signature: str | None,
    ) -> tuple[FailureCategory | None, str | None]:
        if progress_state.repeated_action_count.get(action_signature, 0) > self.MAX_REPEAT_SAME_ACTION_WITHOUT_PROGRESS:
            return FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS, "same_action_repeated_without_progress"

        if (
            target_signature is not None
            and progress_state.repeated_target_count.get(target_signature, 0) > self.MAX_REPEAT_SAME_TARGET_WITHOUT_PROGRESS
        ):
            return FailureCategory.REPEATED_TARGET_WITHOUT_PROGRESS, "same_target_repeated_without_progress"

        if failure_signature is not None and progress_state.recent_failures.count(failure_signature) > self.MAX_REPEAT_SAME_FAILURE:
            return FailureCategory.REPEATED_FAILURE_LOOP, "repeated_identical_failure_signature"

        if alternating_action_loop(progress_state.recent_actions):
            return FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS, "alternating_action_pattern_without_progress"

        if progress_state.no_progress_streak >= self.MAX_NO_PROGRESS_STEPS:
            return FailureCategory.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS, "no_meaningful_progress_threshold_reached"

        return None, None

    def _mark_completed_progress(self, state, decision, executed_action, verification) -> None:
        progress_state = state.progress_state
        target_sig = target_signature(executed_action.action)
        subgoal_sig = subgoal_signature(decision.active_subgoal)
        page_sig = progress_state.latest_page_signature or "unknown_page"

        if target_sig is not None:
            if target_sig not in progress_state.completed_targets:
                progress_state.completed_targets.append(target_sig)
            progress_state.target_completion_page_signatures[target_sig] = page_sig
            if executed_action.action.action_type in {ActionType.TYPE, ActionType.SELECT} and executed_action.action.text is not None:
                progress_state.target_value_history[target_sig] = executed_action.action.text

        if should_mark_subgoal_complete(progress_state, executed_action, verification):
            if subgoal_sig not in progress_state.completed_subgoals:
                progress_state.completed_subgoals.append(subgoal_sig)
            progress_state.subgoal_completion_page_signatures[subgoal_sig] = page_sig

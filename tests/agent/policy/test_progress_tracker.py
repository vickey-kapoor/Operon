"""Loop-progress tracking, tested directly (operon.agent.policy.progress_tracker).

test_loop.py covers the common paths through AgentLoop's wrappers; these pin the
stall reset, the screen-change progress rule, no-progress merging and the
signature helpers that those paths rest on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from operon.agent.perception.screen_diff import SCREEN_CHANGE_THRESHOLD
from operon.agent.policy import progress_tracker as pt
from operon.models.common import FailureCategory, RunStatus, StopReason
from operon.models.execution import (
    ExecutedAction,
    ExecutionAttemptTrace,
    ExecutionTrace,
)
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.progress import ProgressState
from operon.models.recovery import RecoveryDecision, RecoveryStrategy
from operon.models.state import AgentState
from operon.models.verification import VerificationResult, VerificationStatus

LOG = logging.getLogger("test_progress_tracker")
PAGE = "form_page|none|name-input:Name"


def _state(**progress) -> AgentState:
    progress.setdefault("latest_page_signature", PAGE)
    return AgentState(
        run_id="run-1",
        intent="Fill the form",
        status=RunStatus.RUNNING,
        current_subgoal="fill name",
        progress_state=ProgressState(**progress),
    )


def _click(element: str = "submit") -> AgentAction:
    return AgentAction(action_type=ActionType.CLICK, target_element_id=element)


def _executed(action: AgentAction, success: bool = True, *, no_progress: bool | None = None) -> ExecutedAction:
    trace = None
    if no_progress is not None:
        trace = ExecutionTrace(
            attempts=[ExecutionAttemptTrace(attempt_index=1, revalidation_result="ok", verification_result="ok", no_progress_detected=no_progress)],
            final_outcome="done",
        )
    return ExecutedAction(action=action, success=success, detail="ok" if success else "failed", execution_trace=trace)


def _screen(tmp_path, name: str, color: tuple[int, int, int]) -> str:
    """A real after-screenshot on disk: a solid colour, so two colours differ completely."""
    from PIL import Image

    path = tmp_path / f"{name}.png"
    Image.new("RGB", (480, 270), color).save(path)
    return str(path)


def _verification(stop: bool = False) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.SUCCESS if stop else VerificationStatus.UNCERTAIN,
        expected_outcome_met=stop,
        stop_condition_met=stop,
        reason="checked",
    )


def _recovery() -> RecoveryDecision:
    return RecoveryDecision(strategy=RecoveryStrategy.ADVANCE, message="go on")


# --- signatures ----------------------------------------------------------------


def test_nearby_clicks_share_a_50px_target_bucket() -> None:
    near = pt.target_signature(AgentAction(action_type=ActionType.CLICK, x=101, y=149))
    also_near = pt.target_signature(AgentAction(action_type=ActionType.CLICK, x=149, y=100))
    far = pt.target_signature(AgentAction(action_type=ActionType.CLICK, x=150, y=100))

    assert near == also_near == "xy:100:100"
    assert far == "xy:150:100"


def test_an_element_id_wins_over_coordinates() -> None:
    action = AgentAction(action_type=ActionType.CLICK, target_element_id="ok", x=10, y=10)
    assert pt.target_signature(action) == "id:ok"
    assert pt.target_signature(AgentAction(action_type=ActionType.WAIT, wait_ms=500)) is None


def test_action_signature_ignores_case_and_surrounding_space_in_text() -> None:
    a = AgentAction(action_type=ActionType.TYPE, target_element_id="name", text="  Ada ")
    b = AgentAction(action_type=ActionType.TYPE, target_element_id="name", text="ada")
    assert pt.action_signature(a) == pt.action_signature(b) == "type|id:name|ada"
    assert pt.action_signature(AgentAction(action_type=ActionType.WAIT, wait_ms=500)) == "wait|no_target|500"


@pytest.mark.parametrize(
    ("subgoal", "expected"),
    [("Fill the Name field!", "fill_the_name_field"), ("  ", "unknown_subgoal"), (None, "unknown_subgoal"), ("---", "unknown_subgoal")],
)
def test_subgoal_signature_normalizes(subgoal, expected) -> None:
    assert pt.subgoal_signature(subgoal) == expected


def test_the_recent_window_keeps_the_last_six() -> None:
    entries: list[str] = []
    for i in range(10):
        entries = pt.append_window(entries, str(i))
    assert entries == ["4", "5", "6", "7", "8", "9"]


@pytest.mark.parametrize(
    ("actions", "loop"),
    [
        (["a", "b", "c", "a", "b", "c"], True),  # period 3
        (["x", "a", "b", "a", "b"], True),  # period 2
        (["a", "a", "a", "a"], False),  # one action repeated is caught elsewhere, not here
        (["a", "b", "c", "a", "b", "d"], False),
        (["a", "b", "a"], False),
    ],
)
def test_alternating_loops_of_period_two_and_three(actions, loop) -> None:
    assert pt.alternating_action_loop(actions) is loop


# --- the progress rule -----------------------------------------------------------


@pytest.mark.parametrize(
    ("success", "stop", "subgoal_changed", "ratio", "novel", "progress"),
    [
        (False, True, True, 0.5, True, False),  # a failed action is never progress
        (True, True, False, 0.0, False, True),  # the task's stop condition always is
        (True, False, True, 0.0, False, True),  # so is the model moving to a new subgoal
        (True, False, False, 0.5, True, True),  # screen changed and the action is new
        (True, False, False, 0.5, False, False),  # screen changed but the action repeats
        (True, False, False, SCREEN_CHANGE_THRESHOLD / 2, True, False),  # screen did not change
        (True, False, False, None, True, False),  # no screenshots: not counted as progress
    ],
)
def test_progress_needs_a_screen_change_and_a_new_action(success, stop, subgoal_changed, ratio, novel, progress) -> None:
    result = pt.meaningful_progress(
        _executed(_click(), success),
        _verification(stop),
        subgoal_changed=subgoal_changed,
        screen_change_ratio=ratio,
        is_novel_action=novel,
    )
    assert result is progress


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known gap with OPERON_PROGRESS_REVISIT off (the default): meaningful_progress's docstring says a "
        "repeated action that changes the screen (launch_app Notepad 5x) is not progress, but novelty is read "
        "from repeated_action_count, which every progress step clears, so each repeat looks new and counts. "
        "RFC 0003's flag closes it; see the revisit tests below."
    ),
)
def test_repeating_an_action_that_changes_the_screen_still_builds_a_no_progress_streak(monkeypatch) -> None:
    monkeypatch.delenv("OPERON_PROGRESS_REVISIT", raising=False)
    tracker = pt.ProgressTracker()
    state = _state()
    action = AgentAction(action_type=ActionType.LAUNCH_APP, text="notepad")
    decision = PolicyDecision(action=action, rationale="open it", confidence=0.8, active_subgoal="open notepad")

    traces = [
        tracker.update_progress_state(
            state=state, decision=decision, executed_action=_executed(action), verification=_verification(),
            recovery=_recovery(), step_index=i, screen_change_ratio=0.3,
        )
        for i in (1, 2, 3)
    ]

    # Step 1 counts (first time this subgoal is seen); 2 and 3 repeat the same launch.
    assert [t.progress_made for t in traces] == [True, False, False]
    assert state.progress_state.no_progress_streak == 2
    assert state.progress_state.last_screen_change_ratio == 0.3


# --- RFC 0003: OPERON_PROGRESS_REVISIT ----------------------------------------------

# Greys far apart: the screen diff compares grayscale with a 30-level per-pixel threshold.
WINDOW_A, WINDOW_B, PAGE_1, PAGE_2, PAGE_3 = (230, 230, 230), (40, 40, 40), (10, 10, 10), (120, 120, 120), (240, 240, 240)
_LAUNCH = AgentAction(action_type=ActionType.LAUNCH_APP, text="notepad")
_SCROLL = AgentAction(action_type=ActionType.SCROLL, x=600, y=500, scroll_amount=-300)


def _run(tracker, state, action, after_paths, *, subgoal="open notepad", stop_on=None, ratio=0.3):
    decision = PolicyDecision(action=action, rationale="r", confidence=0.8, active_subgoal=subgoal)
    traces = []
    for i, after in enumerate(after_paths, start=1):
        executed = _executed(action).model_copy(update={"artifact_path": after})
        traces.append(
            tracker.update_progress_state(
                state=state, decision=decision, executed_action=executed, verification=_verification(stop=i == stop_on),
                recovery=_recovery(), step_index=i, screen_change_ratio=ratio,
            )
        )
    return traces


@pytest.fixture
def revisit_on(monkeypatch):
    monkeypatch.setenv("OPERON_PROGRESS_REVISIT", "on")


@pytest.mark.parametrize(("value", "on"), [(None, False), ("off", False), ("on", True), ("TRUE", True), ("1", True), ("nope", False)])
def test_the_revisit_flag_is_off_unless_turned_on(monkeypatch, value, on) -> None:
    if value is None:
        monkeypatch.delenv("OPERON_PROGRESS_REVISIT", raising=False)
    else:
        monkeypatch.setenv("OPERON_PROGRESS_REVISIT", value)
    assert pt.progress_revisit_enabled() is on


def test_a_relaunch_that_lands_on_a_screen_it_already_produced_is_not_progress(tmp_path, revisit_on) -> None:
    # Each relaunch flips between two windows: the screen changes every step, but the
    # screens it lands on repeat (A, B, A, B), which is the runaway RFC 0003 targets.
    a, b = _screen(tmp_path, "a", WINDOW_A), _screen(tmp_path, "b", WINDOW_B)
    state = _state()

    traces = _run(pt.ProgressTracker(), state, _LAUNCH, [a, b, a, b])

    assert [t.progress_made for t in traces] == [True, True, False, False]
    # Step 3 is where the revisit overturns "progress"; by step 4 the relaunch is
    # already a counted repeat, so today's rule says no progress on its own.
    assert [t.revisited_screen for t in traces] == [False, False, True, False]
    assert state.progress_state.no_progress_streak == 2
    # ...so the existing repeat guard now blocks the next relaunch.
    blocked = pt.ProgressTracker().block_redundant_action(state, _LAUNCH, 5, LOG)
    assert blocked is not None and blocked.failure_category is FailureCategory.REPEATED_ACTION_WITHOUT_PROGRESS


def test_a_repeat_that_reaches_new_screens_keeps_counting(tmp_path, revisit_on) -> None:
    pages = [_screen(tmp_path, f"p{i}", c) for i, c in enumerate((PAGE_1, PAGE_2, PAGE_3))]
    state = _state()

    traces = _run(pt.ProgressTracker(), state, _SCROLL, pages, subgoal="scroll to the footer")

    assert [t.progress_made for t in traces] == [True, True, True]
    assert not any(t.revisited_screen for t in traces)
    assert state.progress_state.no_progress_streak == 0


def test_a_repeat_recovers_when_it_reaches_a_new_screen_after_a_revisit(tmp_path, revisit_on) -> None:
    # Scroll lands on A, B, back on A (a revisit, no progress), then on a new screen C:
    # C is progress again, so the repeat guard does not block the next scroll.
    a, b, c = (_screen(tmp_path, n, col) for n, col in (("a", PAGE_1), ("b", PAGE_2), ("c", PAGE_3)))
    state = _state()

    traces = _run(pt.ProgressTracker(), state, _SCROLL, [a, b, a, c], subgoal="scroll to the footer")

    assert [t.progress_made for t in traces] == [True, True, False, True]
    assert [t.revisited_screen for t in traces] == [False, False, True, False]
    assert state.progress_state.no_progress_streak == 0
    assert pt.ProgressTracker().block_redundant_action(state, _SCROLL, 5, LOG) is None


def test_a_counted_repeat_without_an_earlier_screenshot_keeps_todays_verdict(tmp_path, revisit_on) -> None:
    # Nothing to compare (earlier screenshots gone): a counted repeat is still not progress.
    a, b = _screen(tmp_path, "a", WINDOW_A), _screen(tmp_path, "b", WINDOW_B)
    state = _state()
    tracker = pt.ProgressTracker()
    _run(tracker, state, _LAUNCH, [a, b, a])
    Path(a).unlink()
    Path(b).unlink()

    [trace] = _run(tracker, state, _LAUNCH, [_screen(tmp_path, "c", PAGE_2)])

    assert not trace.progress_made


def test_a_missing_screenshot_never_counts_as_the_same_screen(tmp_path, revisit_on) -> None:
    a = _screen(tmp_path, "a", WINDOW_A)
    gone = str(tmp_path / "deleted.png")
    state = _state()

    # Earlier screen missing, then this screen missing: both fall back to today's rule.
    traces = _run(pt.ProgressTracker(), state, _LAUNCH, [gone, a, gone])

    assert [t.progress_made for t in traces] == [True, True, True]
    assert not any(t.revisited_screen for t in traces)


def test_the_stop_condition_still_counts_on_a_revisited_screen(tmp_path, revisit_on) -> None:
    a = _screen(tmp_path, "a", WINDOW_A)

    traces = _run(pt.ProgressTracker(), _state(), _LAUNCH, [a, a], stop_on=2)

    assert traces[1].progress_made and not traces[1].revisited_screen


def test_a_new_subgoal_still_counts_on_a_revisited_screen(tmp_path, revisit_on) -> None:
    a = _screen(tmp_path, "a", WINDOW_A)
    tracker, state = pt.ProgressTracker(), _state()
    _run(tracker, state, _LAUNCH, [a], subgoal="open notepad")

    [trace] = _run(tracker, state, _LAUNCH, [a], subgoal="open notepad for the answer")

    assert trace.progress_made and not trace.revisited_screen


def test_with_the_flag_off_nothing_is_remembered(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPERON_PROGRESS_REVISIT", raising=False)
    a, b = _screen(tmp_path, "a", WINDOW_A), _screen(tmp_path, "b", WINDOW_B)
    state = _state()

    traces = _run(pt.ProgressTracker(), state, _LAUNCH, [a, b, a, b])

    assert all(t.progress_made for t in traces)  # today's behaviour, the gap above
    assert state.progress_state.action_outcomes == {}


def test_remembered_outcomes_are_bounded() -> None:
    progress = ProgressState()
    for i in range(5):
        pt.ProgressTracker._remember_outcome(progress, "launch", f"shot{i}.png")
    assert progress.action_outcomes["launch"] == ["shot2.png", "shot3.png", "shot4.png"]

    for i in range(pt.REVISIT_MAX_ACTIONS + 5):
        pt.ProgressTracker._remember_outcome(progress, f"action{i}", "x.png")
    assert len(progress.action_outcomes) == pt.REVISIT_MAX_ACTIONS
    assert "launch" not in progress.action_outcomes  # the least recently used went first


# --- the stall reset ---------------------------------------------------------------


def test_a_stalled_screen_forces_a_subgoal_reset() -> None:
    state = _state(last_screen_change_ratio=0.001, no_progress_streak=2)

    blocked = pt.ProgressTracker().block_redundant_action(state, _click(), 3, LOG)

    assert blocked is not None and not blocked.success
    assert blocked.failure_category is FailureCategory.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS
    assert state.current_subgoal.startswith("Stalled — choose a completely different approach for: ")
    assert state.current_subgoal.endswith("fill name")


@pytest.mark.parametrize(
    ("ratio", "streak", "action"),
    [
        (0.001, 1, _click()),  # one quiet step is not a stall yet
        (0.05, 3, _click()),  # the screen is still moving
        (0.001, 3, AgentAction(action_type=ActionType.NAVIGATE, url="https://example.com")),  # not an input action
    ],
    ids=["first-quiet-step", "screen-moving", "navigate"],
)
def test_no_stall_reset_without_all_three_conditions(ratio, streak, action) -> None:
    state = _state(last_screen_change_ratio=ratio, no_progress_streak=streak)

    blocked = pt.ProgressTracker().block_redundant_action(state, action, 3, LOG)

    assert blocked is None or blocked.failure_category is not FailureCategory.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS
    assert state.current_subgoal == "fill name"


def test_a_target_hit_twice_without_progress_is_blocked_even_with_a_new_action() -> None:
    # Different text, same box: the action signature is new, the target is not.
    state = _state(repeated_target_count={"id:name": 2}, no_progress_streak=1, last_screen_change_ratio=0.5)
    action = AgentAction(action_type=ActionType.TYPE, target_element_id="name", text="something else")

    blocked = pt.ProgressTracker().block_redundant_action(state, action, 3, LOG)

    assert blocked is not None
    assert blocked.failure_category is FailureCategory.REPEATED_TARGET_WITHOUT_PROGRESS


def test_nothing_is_blocked_while_progress_is_being_made() -> None:
    state = _state(repeated_action_count={"click|id:submit|": 5}, repeated_target_count={"id:submit": 5}, no_progress_streak=0)
    assert pt.ProgressTracker().block_redundant_action(state, _click(), 3, LOG) is None


# --- loop detection and the stop guard ----------------------------------------------


def test_the_no_progress_ceiling_stops_the_run() -> None:
    state = _state(no_progress_streak=pt.ProgressTracker.MAX_NO_PROGRESS_STEPS)

    category, pattern = pt.ProgressTracker().detect_loop_failure(
        progress_state=state.progress_state, action_signature="click|id:new|", target_signature="id:new", failure_signature=None
    )

    assert category is FailureCategory.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS
    assert pattern == "no_meaningful_progress_threshold_reached"


def test_the_stop_guard_passes_recovery_through_when_no_loop_was_found() -> None:
    recovery = _recovery()
    trace = pt.ProgressTracker().update_progress_state(
        state=_state(), decision=PolicyDecision(action=_click(), rationale="r", confidence=0.9, active_subgoal="submit"),
        executed_action=_executed(_click()), verification=_verification(), recovery=recovery, step_index=1, screen_change_ratio=0.5,
    )

    assert trace.final_failure_category is None
    assert pt.ProgressTracker().apply_progress_stop_guard(recovery, trace) is recovery


def test_the_stop_guard_turns_a_loop_into_a_terminal_stop() -> None:
    state = _state(no_progress_streak=pt.ProgressTracker.MAX_NO_PROGRESS_STEPS - 1, latest_subgoal_signature="submit")
    trace = pt.ProgressTracker().update_progress_state(
        state=state, decision=PolicyDecision(action=_click(), rationale="r", confidence=0.9, active_subgoal="submit"),
        executed_action=_executed(_click(), success=False), verification=_verification(), recovery=_recovery(), step_index=4,
    )

    guarded = pt.ProgressTracker().apply_progress_stop_guard(_recovery(), trace)

    assert guarded.strategy is RecoveryStrategy.STOP and guarded.terminal and not guarded.recoverable
    assert guarded.stop_reason is StopReason.NO_MEANINGFUL_PROGRESS_ACROSS_STEPS


# --- repeated no-progress on one target -----------------------------------------------


def test_two_no_progress_attempts_on_the_same_target_become_a_failure() -> None:
    state = _state()
    state.action_history.append(_executed(_click("submit"), no_progress=True))

    merged = pt.apply_no_progress_detection(state, _executed(_click("submit"), no_progress=True))

    assert merged.success is False
    assert merged.failure_category is FailureCategory.EXECUTION_NO_PROGRESS
    assert merged.execution_trace.final_failure_category is FailureCategory.EXECUTION_NO_PROGRESS


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (_executed(_click("submit"), no_progress=True), _executed(_click("cancel"), no_progress=True)),  # other target
        (_executed(_click("submit"), no_progress=False), _executed(_click("submit"), no_progress=True)),  # first time
        (_executed(_click("submit"), no_progress=True), _executed(_click("submit"), no_progress=False)),  # it moved
        (None, _executed(_click("submit"), no_progress=True)),  # nothing before it
        (_executed(_click("submit"), no_progress=True), _executed(_click("submit"))),  # no trace
    ],
    ids=["other-target", "first-time", "progressed", "no-history", "no-trace"],
)
def test_no_progress_is_left_alone_unless_it_repeats_on_the_same_target(previous, current) -> None:
    state = _state()
    if previous is not None:
        state.action_history.append(previous)

    assert pt.apply_no_progress_detection(state, current) is current

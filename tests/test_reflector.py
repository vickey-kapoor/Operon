"""Tests for PostRunReflector — pattern detection and episode extraction (reflector.py).

Tier: SIMPLE (individual detectors) → MODERATE (reflect() integration) → COMPLEX (episode extraction)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.agent.reflector import PostRunReflector
from src.models.common import RunStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_dir(tmp_path: Path, run_id: str) -> Path:
    d = tmp_path / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_state(run_dir: Path, status: str = "succeeded", intent: str = "Fill out contact form") -> None:
    (run_dir / "state.json").write_text(
        json.dumps({"status": status, "intent": intent}), encoding="utf-8"
    )


def _write_steps(run_dir: Path, steps: list[dict]) -> None:
    lines = "\n".join(json.dumps(s) for s in steps)
    (run_dir / "run.jsonl").write_text(lines, encoding="utf-8")


def _click_step(step_idx: int, x: int = 400, y: int = 300) -> dict:
    return {
        "step_index": step_idx,
        "policy_decision": {
            "action": {"action_type": "click", "x": x, "y": y},
            "active_subgoal": "click_element",
        },
        "executed_action": {"failure_category": None},
        "perception": {"page_hint": "form_page", "visible_elements": []},
        "progress_state": {"no_progress_streak": 0},
    }


def _press_key_step(step_idx: int, key: str = "tab", subgoal: str = "navigate") -> dict:
    return {
        "step_index": step_idx,
        "policy_decision": {
            "action": {"action_type": "press_key", "key": key},
            "active_subgoal": subgoal,
        },
        "executed_action": {"failure_category": None},
        "perception": {"page_hint": "form_page", "visible_elements": []},
        "progress_state": {"no_progress_streak": 0},
    }


def _type_step(step_idx: int, text: str, target_id: str, subgoal: str = "type_name") -> dict:
    return {
        "step_index": step_idx,
        "policy_decision": {
            "action": {"action_type": "type", "text": text, "target_element_id": target_id},
            "active_subgoal": subgoal,
        },
        "executed_action": {"failure_category": None},
        "perception": {
            "page_hint": "form_page",
            "visible_elements": [{"element_id": target_id, "primary_name": f"field_{target_id}"}],
        },
        "progress_state": {"no_progress_streak": 0},
    }


def _no_progress_step(step_idx: int) -> dict:
    return {
        "step_index": step_idx,
        "policy_decision": {
            "action": {"action_type": "click", "x": 400, "y": 300},
            "active_subgoal": "stuck",
        },
        "executed_action": {"failure_category": None},
        "perception": {"page_hint": "form_page", "visible_elements": []},
        "progress_state": {"no_progress_streak": 2},
    }


def _mock_memory_store() -> MagicMock:
    store = MagicMock()
    store._append_record = MagicMock()
    store.save_episode = MagicMock()
    return store


# ===========================================================================
# SIMPLE: _detect_repeated_key_press
# ===========================================================================

def test_detect_repeated_key_no_streak() -> None:
    """Fewer than 3 consecutive presses produces no patterns."""
    steps = [
        _press_key_step(1, "tab"),
        _press_key_step(2, "enter"),
        _press_key_step(3, "tab"),
    ]
    patterns = PostRunReflector._detect_repeated_key_press(steps, "run-1")
    assert patterns == []


def test_detect_repeated_key_exactly_three_triggers_pattern() -> None:
    """Exactly 3 identical consecutive key presses produces a pattern."""
    steps = [_press_key_step(i, "tab") for i in range(1, 4)]
    patterns = PostRunReflector._detect_repeated_key_press(steps, "run-1")
    assert len(patterns) == 1
    assert "tab" in patterns[0].pattern_key


def test_detect_repeated_key_long_streak_single_pattern() -> None:
    """A streak of 8 identical keys yields exactly one pattern."""
    steps = [_press_key_step(i, "escape") for i in range(1, 9)]
    patterns = PostRunReflector._detect_repeated_key_press(steps, "run-1")
    assert len(patterns) == 1
    assert patterns[0].confidence > 0.5


def test_detect_repeated_key_final_streak_detected() -> None:
    """Streak at the end of the step list (no trailing different key) is detected."""
    steps = [_click_step(1)] + [_press_key_step(i, "ctrl") for i in range(2, 6)]
    patterns = PostRunReflector._detect_repeated_key_press(steps, "run-1")
    assert len(patterns) == 1
    assert "ctrl" in patterns[0].pattern_key


# ===========================================================================
# SIMPLE: _detect_taskbar_clicking
# ===========================================================================

def test_detect_taskbar_no_clicks_below_threshold() -> None:
    """Fewer than 3 taskbar clicks produces no pattern."""
    steps = [_click_step(i, x=400, y=1050) for i in range(1, 3)]
    patterns = PostRunReflector._detect_taskbar_clicking(steps, "run-1")
    assert patterns == []


def test_detect_taskbar_exactly_three_fires_pattern() -> None:
    """Exactly 3 clicks at y >= 1000 produces the taskbar pattern."""
    steps = [_click_step(i, x=200, y=1020) for i in range(1, 4)]
    patterns = PostRunReflector._detect_taskbar_clicking(steps, "run-1")
    assert len(patterns) == 1
    assert "taskbar" in patterns[0].pattern_key


def test_detect_taskbar_ignores_clicks_above_threshold() -> None:
    """Clicks above the taskbar Y threshold (y < 1000) are ignored."""
    steps = [_click_step(i, x=400, y=800) for i in range(1, 6)]
    patterns = PostRunReflector._detect_taskbar_clicking(steps, "run-1")
    assert patterns == []


def test_detect_taskbar_mixed_clicks_counts_correctly() -> None:
    """Only the taskbar-region clicks count toward the threshold."""
    steps = [
        _click_step(1, y=300),   # normal
        _click_step(2, y=1050),  # taskbar
        _click_step(3, y=400),   # normal
        _click_step(4, y=1010),  # taskbar
        _click_step(5, y=1030),  # taskbar
    ]
    patterns = PostRunReflector._detect_taskbar_clicking(steps, "run-1")
    assert len(patterns) == 1  # 3 taskbar clicks


# ===========================================================================
# SIMPLE: _detect_stuck_subgoal
# ===========================================================================

def test_detect_stuck_subgoal_no_repetition() -> None:
    """Each step on a different subgoal produces no patterns."""
    steps = [
        {**_click_step(i), "policy_decision": {**_click_step(i)["policy_decision"], "active_subgoal": f"goal_{i}"}}
        for i in range(1, 5)
    ]
    patterns = PostRunReflector._detect_stuck_subgoal(steps, "run-1")
    assert patterns == []


def test_detect_stuck_subgoal_four_or_more_fires() -> None:
    """The same subgoal for 4 steps produces a stuck-subgoal pattern."""
    stuck = "fill_name_field"
    steps = []
    for i in range(1, 5):
        s = _click_step(i)
        s["policy_decision"]["active_subgoal"] = stuck
        steps.append(s)
    patterns = PostRunReflector._detect_stuck_subgoal(steps, "run-1")
    assert len(patterns) == 1
    assert stuck in patterns[0].description


def test_detect_stuck_subgoal_exactly_three_does_not_fire() -> None:
    """Three consecutive same subgoal is below threshold (requires 4)."""
    stuck = "navigate"
    steps = []
    for i in range(1, 4):
        s = _click_step(i)
        s["policy_decision"]["active_subgoal"] = stuck
        steps.append(s)
    patterns = PostRunReflector._detect_stuck_subgoal(steps, "run-1")
    assert patterns == []


# ===========================================================================
# SIMPLE: _detect_no_screen_change_actions
# ===========================================================================

def test_detect_no_screen_change_no_streak() -> None:
    steps = [_click_step(i) for i in range(1, 4)]  # all have no_progress_streak=0
    patterns = PostRunReflector._detect_no_screen_change_actions(steps, "run-1")
    assert patterns == []


def test_detect_no_screen_change_three_fires_pattern() -> None:
    steps = [_no_progress_step(i) for i in range(1, 4)]
    patterns = PostRunReflector._detect_no_screen_change_actions(steps, "run-1")
    assert len(patterns) == 1
    assert "screen" in patterns[0].pattern_key


# ===========================================================================
# MODERATE: reflect() — missing/corrupt files
# ===========================================================================

def test_reflect_missing_run_returns_minimal_result(tmp_path: Path) -> None:
    """reflect() on a non-existent run ID returns a zero-step reflection."""
    store = _mock_memory_store()
    reflector = PostRunReflector(store, root_dir=tmp_path)
    result = reflector.reflect("does-not-exist")
    assert result.run_id == "does-not-exist"
    assert result.total_steps == 0
    assert result.patterns == []


def test_reflect_empty_steps_returns_no_patterns(tmp_path: Path) -> None:
    """A run with state.json but no steps produces no patterns."""
    run_id = "run-empty"
    d = _run_dir(tmp_path, run_id)
    _write_state(d)
    _write_steps(d, [])
    store = _mock_memory_store()
    reflector = PostRunReflector(store, root_dir=tmp_path)
    result = reflector.reflect(run_id)
    assert result.total_steps == 0
    assert result.patterns == []


def test_reflect_corrupt_state_json_handled(tmp_path: Path) -> None:
    """reflect() on a run with corrupted state.json does not raise."""
    run_id = "run-corrupt"
    d = _run_dir(tmp_path, run_id)
    (d / "state.json").write_text("{{not json", encoding="utf-8")
    _write_steps(d, [_click_step(1)])
    store = _mock_memory_store()
    reflector = PostRunReflector(store, root_dir=tmp_path)
    result = reflector.reflect(run_id)
    assert result.run_id == run_id


# ===========================================================================
# MODERATE: reflect() — pattern detection writes memory records
# ===========================================================================

def test_reflect_detects_repeated_key_and_writes_memory(tmp_path: Path) -> None:
    """reflect() on a run with 3+ repeated key presses writes a memory record."""
    run_id = "run-keyrepeat"
    d = _run_dir(tmp_path, run_id)
    _write_state(d, status="failed")
    steps = [_press_key_step(i, "escape") for i in range(1, 5)]
    _write_steps(d, steps)

    store = _mock_memory_store()
    reflector = PostRunReflector(store, root_dir=tmp_path)
    result = reflector.reflect(run_id)

    assert result.memories_generated >= 1
    store._append_record.assert_called()


def test_reflect_detects_taskbar_and_writes_memory(tmp_path: Path) -> None:
    run_id = "run-taskbar"
    d = _run_dir(tmp_path, run_id)
    _write_state(d, status="failed")
    steps = [_click_step(i, y=1020) for i in range(1, 5)]
    _write_steps(d, steps)

    store = _mock_memory_store()
    reflector = PostRunReflector(store, root_dir=tmp_path)
    result = reflector.reflect(run_id)

    assert result.memories_generated >= 1


def test_reflect_writes_reflection_artifact_to_disk(tmp_path: Path) -> None:
    """reflect() always writes reflection.json under the run directory."""
    run_id = "run-artifact"
    d = _run_dir(tmp_path, run_id)
    _write_state(d, status="succeeded")
    _write_steps(d, [_click_step(1), _click_step(2)])

    store = _mock_memory_store()
    reflector = PostRunReflector(store, root_dir=tmp_path)
    reflector.reflect(run_id)

    artifact = d / "reflection.json"
    assert artifact.exists(), "reflection.json must be written to the run directory"
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data["run_id"] == run_id


# ===========================================================================
# COMPLEX: _extract_episode (episode trajectory from successful runs)
# ===========================================================================

def test_extract_episode_skips_failed_steps() -> None:
    """Steps with a failure_category should be excluded from the episode."""
    failed_step = _type_step(1, "Alice", "name-field")
    failed_step["executed_action"]["failure_category"] = "execution_target_not_found"
    good_step = _type_step(2, "alice@example.com", "email-field")

    episode = PostRunReflector._extract_episode("run-1", "fill form", "auth_free_form", [failed_step, good_step])
    # Only 1 usable step → fewer than 2 → episode should be None
    assert episode is None


def test_extract_episode_requires_at_least_two_steps() -> None:
    """Episode extraction requires >= 2 usable steps."""
    single = [_type_step(1, "Alice", "name-field")]
    episode = PostRunReflector._extract_episode("run-1", "fill form", "auth_free_form", single)
    assert episode is None


def test_extract_episode_success_creates_episode_with_correct_fields() -> None:
    """A successful run with 3 steps produces a well-formed Episode."""
    steps = [
        _type_step(1, "Alice", "name-field", subgoal="fill_name"),
        _type_step(2, "alice@example.com", "email-field", subgoal="fill_email"),
        _click_step(3),
    ]
    episode = PostRunReflector._extract_episode("run-42", "fill contact form", "auth_free_form", steps)
    assert episode is not None
    assert episode.episode_id == "run-42"
    assert episode.normalized_intent == "fill contact form"
    assert len(episode.steps) >= 2


def test_extract_episode_skips_stop_and_wait_actions() -> None:
    """STOP and WAIT action types are excluded from episode steps."""
    steps = [
        _type_step(1, "Alice", "name-field"),
        {
            **_click_step(2),
            "policy_decision": {
                "action": {"action_type": "stop"},
                "active_subgoal": "done",
            },
        },
        _type_step(3, "alice@example.com", "email-field"),
    ]
    episode = PostRunReflector._extract_episode("run-1", "fill form", "auth_free_form", steps)
    if episode:
        action_types = [s.action_type.value for s in episode.steps]
        assert "stop" not in action_types


def test_reflect_saves_episode_on_successful_run(tmp_path: Path) -> None:
    """reflect() on a succeeded run extracts and saves an episode."""
    run_id = "run-success"
    d = _run_dir(tmp_path, run_id)
    _write_state(d, status="succeeded", intent="fill contact form")
    steps = [
        _type_step(1, "Alice", "name-input"),
        _type_step(2, "alice@example.com", "email-input"),
        _click_step(3),
    ]
    _write_steps(d, steps)

    store = _mock_memory_store()
    reflector = PostRunReflector(store, root_dir=tmp_path)
    result = reflector.reflect(run_id)

    assert result.success is True
    store.save_episode.assert_called_once()


def test_reflect_does_not_save_episode_on_failed_run(tmp_path: Path) -> None:
    """reflect() on a failed run should not attempt episode extraction."""
    run_id = "run-failed"
    d = _run_dir(tmp_path, run_id)
    _write_state(d, status="failed")
    _write_steps(d, [_click_step(1), _click_step(2), _click_step(3)])

    store = _mock_memory_store()
    reflector = PostRunReflector(store, root_dir=tmp_path)
    reflector.reflect(run_id)

    store.save_episode.assert_not_called()

"""Unit tests for WebArena evaluation logic and models."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.benchmarks.webarena import _extract_final_text
from src.benchmarks.webarena_eval import evaluate_task, string_match, url_match
from src.benchmarks.webarena_models import (
    WebArenaEval,
    WebArenaSummary,
    WebArenaTask,
    WebArenaTaskResult,
)
from src.models.policy import ActionType, AgentAction

# ---------------------------------------------------------------------------
# string_match
# ---------------------------------------------------------------------------

def test_string_match_all_present():
    assert string_match(["1991", "Python"], "Python was released in 1991.") is True


def test_string_match_case_insensitive():
    assert string_match(["ABC"], "The language succeeded abc.") is True


def test_string_match_missing_one():
    assert string_match(["1991", "Guido"], "Released in 1991.") is False


def test_string_match_empty_must_include():
    assert string_match([], "anything") is True


# ---------------------------------------------------------------------------
# url_match
# ---------------------------------------------------------------------------

def test_url_match_present():
    assert url_match("practice-automation.com", "https://practice-automation.com/success") is True


def test_url_match_absent():
    assert url_match("example.com", "https://other.com/page") is False


def test_url_match_case_insensitive():
    assert url_match("EXAMPLE.COM", "https://example.com/") is True


# ---------------------------------------------------------------------------
# evaluate_task
# ---------------------------------------------------------------------------

def _make_task(eval_types, reference_answers) -> WebArenaTask:
    return WebArenaTask(
        task_id="test_001",
        site="test",
        category="test",
        start_url="https://example.com",
        intent="Do something",
        eval=WebArenaEval(eval_types=eval_types, reference_answers=reference_answers),
    )


def test_evaluate_task_string_match_pass():
    task = _make_task(["string_match"], {"must_include": ["1991"]})
    result = evaluate_task(task, "https://example.com", "Python was released in 1991.")
    assert result == {"string_match": True}


def test_evaluate_task_string_match_fail():
    task = _make_task(["string_match"], {"must_include": ["1991"]})
    result = evaluate_task(task, "https://example.com", "No year mentioned.")
    assert result == {"string_match": False}


def test_evaluate_task_url_match_pass():
    task = _make_task(["url_match"], {"url_contains": "example.com"})
    result = evaluate_task(task, "https://example.com/success", "")
    assert result == {"url_match": True}


def test_evaluate_task_both_types_all_pass():
    task = _make_task(
        ["url_match", "string_match"],
        {"url_contains": "example.com", "must_include": ["Alice"]},
    )
    result = evaluate_task(task, "https://example.com/thanks", "Thanks, Alice!")
    assert result == {"url_match": True, "string_match": True}


def test_evaluate_task_both_types_one_fails():
    task = _make_task(
        ["url_match", "string_match"],
        {"url_contains": "example.com", "must_include": ["Alice"]},
    )
    result = evaluate_task(task, "https://other.com/thanks", "Thanks, Alice!")
    assert result["url_match"] is False
    assert result["string_match"] is True


def test_evaluate_task_unknown_type_fails():
    task = _make_task(["custom_eval"], {})
    result = evaluate_task(task, "https://x.com", "text")
    assert result == {"custom_eval": False}


def test_evaluate_task_must_include_single_string():
    """reference_answers.must_include may be a bare string, not a list."""
    task = _make_task(["string_match"], {"must_include": "1991"})
    result = evaluate_task(task, "", "Python released in 1991.")
    assert result == {"string_match": True}


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def test_load_webarena_tasks_json():
    tasks_path = Path("benchmarks/webarena_tasks.json")
    if not tasks_path.exists():
        pytest.skip("benchmarks/webarena_tasks.json not present")
    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    from src.benchmarks.webarena_models import WebArenaTask
    tasks = [WebArenaTask.model_validate(t) for t in raw]
    assert len(tasks) >= 1
    for t in tasks:
        assert t.task_id
        assert t.start_url.startswith("http")
        assert t.eval.eval_types


# ---------------------------------------------------------------------------
# WebArenaSummary
# ---------------------------------------------------------------------------

def test_summary_pass_rate():
    results = [
        WebArenaTaskResult(task_id="t1", run_id="r1", passed=True, duration_seconds=1.0),
        WebArenaTaskResult(task_id="t2", run_id="r2", passed=False, duration_seconds=2.0),
    ]
    summary = WebArenaSummary(
        total=2,
        passed=1,
        pass_rate=0.5,
        tasks=results,
    )
    assert summary.pass_rate == 0.5
    assert summary.total == 2


# ---------------------------------------------------------------------------
# _extract_final_text — answer extraction for the eval corpus
# ---------------------------------------------------------------------------

def _executed_action(action_type: ActionType, text: str | None = None) -> MagicMock:
    """Mock ExecutedAction with .action.action_type and .action.text attributes."""
    executed = MagicMock()
    executed.action = AgentAction(action_type=action_type, text=text)
    return executed


def _agent_state(*, action_history=None, observation_history=None) -> MagicMock:
    state = MagicMock()
    state.action_history = action_history or []
    state.observation_history = observation_history or []
    return state


def test_extract_final_text_returns_empty_for_none_state():
    assert _extract_final_text(None) == ""


def test_extract_final_text_returns_empty_for_empty_state():
    state = _agent_state()
    assert _extract_final_text(state) == ""


def test_extract_final_text_uses_stop_action_text_when_present():
    """The stop action's text is the agent's explicit answer; it must lead the corpus
    so the string_match evaluator can find it even if perception missed the value."""
    state = _agent_state(
        action_history=[_executed_action(ActionType.STOP, text="1991")],
        observation_history=[],
    )
    out = _extract_final_text(state)
    assert "1991" in out
    assert out.startswith("1991")


def test_extract_final_text_falls_back_to_perception_when_no_stop_text():
    """Tasks where the agent stopped without an explicit answer still get the
    perception summary + element labels as the search corpus."""
    perception = MagicMock()
    perception.summary = "Wikipedia article about Python"
    el = MagicMock()
    el.label = "Released"
    el.text = "1991"
    perception.visible_elements = [el]

    state = _agent_state(
        action_history=[_executed_action(ActionType.STOP)],  # no text
        observation_history=[perception],
    )
    out = _extract_final_text(state)
    assert "Wikipedia article about Python" in out
    assert "Released" in out
    assert "1991" in out


def test_extract_final_text_includes_both_stop_text_and_perception():
    """Stop.text leads, perception fallback follows — never lose either signal."""
    perception = MagicMock()
    perception.summary = "page summary"
    perception.visible_elements = []

    state = _agent_state(
        action_history=[_executed_action(ActionType.STOP, text="answer-from-stop")],
        observation_history=[perception],
    )
    out = _extract_final_text(state)
    assert "answer-from-stop" in out
    assert "page summary" in out
    # stop.text must come first so it's never starved out by long perception summaries
    assert out.index("answer-from-stop") < out.index("page summary")


def test_extract_final_text_ignores_non_stop_last_action():
    """Don't treat a stray text payload on a non-stop action (e.g. the last TYPE
    that filled an input) as the answer — only stop is the answer-commit signal."""
    perception = MagicMock()
    perception.summary = "page summary"
    perception.visible_elements = []

    state = _agent_state(
        action_history=[_executed_action(ActionType.TYPE, text="alice@test.com")],
        observation_history=[perception],
    )
    out = _extract_final_text(state)
    # The TYPE text must NOT appear — only stop.text is the answer signal
    assert "alice@test.com" not in out
    assert "page summary" in out

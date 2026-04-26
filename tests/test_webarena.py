"""Unit tests for WebArena evaluation logic and models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmarks.webarena_eval import evaluate_task, string_match, url_match
from src.benchmarks.webarena_models import (
    WebArenaEval,
    WebArenaSummary,
    WebArenaTask,
    WebArenaTaskResult,
)

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

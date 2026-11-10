"""Run-over-run score changes for a grid cell (KAN-60)."""

from __future__ import annotations

from evals.leaderboard.changes import (
    FIRST_RUN,
    change_markdown,
    flipped_tasks,
    run_change,
    score_range,
    task_pass_rates,
)
from evals.leaderboard.stats import IMPROVED, NO_CLEAR_CHANGE, REGRESSED


def _run(passed: int, total: int = 10) -> list[dict]:
    return [{"task": f"t{i}", "outcome": "success" if i < passed else "failure"} for i in range(total)]


def test_a_repeated_task_scores_its_share_of_passed_attempts() -> None:
    results = [{"task": "a", "outcome": "success"}, {"task": "a", "outcome": "false_success"}, {"task": "b", "outcome": "success"}]
    assert task_pass_rates(results) == {"a": 0.5, "b": 1.0}


def test_score_range_brackets_the_pass_rate() -> None:
    rng = score_range(_run(9))
    assert rng["score"] == 0.9 and rng["tasks"] == 10
    assert rng["low"] < 0.9 <= rng["high"] <= 1.0
    assert score_range([]) is None


def test_the_first_run_of_a_cell_has_nothing_to_compare_with() -> None:
    assert run_change(_run(9), None) == {"label": FIRST_RUN, "previous": None}


def test_one_or_two_flips_are_no_clear_change_but_the_tasks_are_listed() -> None:
    change = run_change(_run(10), _run(8), {"key": "old", "started_at": "20261109T000000Z", "success": 8, "tasks": 10})
    assert change["label"] == NO_CLEAR_CHANGE
    assert change["fixed"] == ["t8", "t9"] and change["broke"] == []
    assert change["difference"] == 0.2 and change["low"] <= 0 <= change["high"]
    assert change["previous"]["success"] == 8 and change["shared_tasks"] == 10


def test_four_tasks_flipping_the_same_way_is_a_clear_change_both_ways() -> None:
    up, down = run_change(_run(10), _run(6)), run_change(_run(6), _run(10))
    assert up["label"] == IMPROVED and up["low"] > 0
    assert down["label"] == REGRESSED and down["high"] < 0
    assert down["broke"] == ["t6", "t7", "t8", "t9"]


def test_flips_both_ways_cancel_out() -> None:
    new = _run(5)
    old = [{"task": f"t{i}", "outcome": "success" if i >= 5 else "failure"} for i in range(10)]
    change = run_change(new, old)
    assert change["label"] == NO_CLEAR_CHANGE and len(change["fixed"]) == 5 and len(change["broke"]) == 5


def test_only_shared_tasks_are_compared_or_flipped() -> None:
    assert flipped_tasks({"a": 1.0, "new": 0.0}, {"a": 0.0, "gone": 1.0}) == (["a"], [])
    change = run_change([{"task": "x", "outcome": "success"}], [{"task": "y", "outcome": "failure"}])
    assert change["label"] == NO_CLEAR_CHANGE and change["shared_tasks"] == 0


def test_the_pr_comment_has_a_row_per_cell_with_score_change_and_flips() -> None:
    summary = {"success": 9, "tasks": 10, "score_range": score_range(_run(9)), "change": run_change(_run(9), _run(10))}
    first = {"success": 10, "tasks": 10, "score_range": score_range(_run(10)), "change": run_change(_run(10), None)}
    text = change_markdown([
        {"cell": "gemini · Desktop · Full (desktop-core)", "summary": summary, "run_url": "https://gh/run/1"},
        {"cell": "gemini · Browser · Thin (core)", "summary": first},
    ], "https://dash")
    assert "| Cell | Passed | Score (95% range) | Change vs previous run | Flipped tasks |" in text
    assert "| [gemini · Desktop · Full (desktop-core)](https://gh/run/1) | 9/10 | 90% (range 70% to 100%) | ⚪ No clear change (-10 pts) | now fail: t9 |" in text
    assert "| gemini · Browser · Thin (core) | 10/10 | 100% (range 100% to 100%) | First run | – |" in text
    assert "https://dash" in text
    assert change_markdown([]) == ""


def test_a_flagged_run_says_why_next_to_its_count() -> None:
    summary = {"success": 3, "tasks": 3, "score_range": score_range(_run(3, 3)), "change": None}
    text = change_markdown([{"cell": "c", "summary": summary, "flagged": "stopped at the $3 cost cap after 3 of 10 tasks"}])
    assert "| c | 3/3 ⚠️ stopped at the $3 cost cap after 3 of 10 tasks | 100% (range 100% to 100%) | – | – |" in text

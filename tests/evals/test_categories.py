"""Benchmark categories, tags and core sets (evals/categories.py, KAN-72)."""

from __future__ import annotations

import pytest

from evals import categories, live_bench
from evals.app_tasks import AppTask
from evals.categories import CORE, core_tasks, is_safety, length_of, tags_of
from evals.safety_tasks import SafetyTask


@pytest.mark.parametrize("mode", ["browser", "desktop", "hybrid"])
def test_every_core_task_exists_once_and_is_graded_from_state(mode: str) -> None:
    tasks = core_tasks(mode)
    names = [t.name for t in tasks]
    assert names == list(CORE[mode]) and len(set(names)) == len(names)
    if mode == "browser":
        # No screenshot judge in model comparisons: every core task is graded from the apps.
        assert all(isinstance(t, (AppTask, SafetyTask)) for t in tasks)


def test_the_core_sets_are_about_ten_tasks_of_short_long_and_safety() -> None:
    browser = core_tasks("browser")
    assert len(browser) == 10
    assert sum(length_of(t) == "short" and not is_safety(t) for t in browser) == 5
    assert sum(length_of(t) == "long" for t in browser) == 2
    assert sum(is_safety(t) for t in browser) == 3
    desktop = core_tasks("desktop")
    assert len(desktop) == 10
    assert sum(length_of(t) == "long" for t in desktop) == 4  # the file tasks, 40+ steps
    hybrid = core_tasks("hybrid")  # KAN-73
    assert len(hybrid) == 10
    assert sum(length_of(t) == "short" and not is_safety(t) for t in hybrid) == 5
    assert sum(length_of(t) == "long" for t in hybrid) == 2
    assert sum(is_safety(t) for t in hybrid) == 3


@pytest.mark.parametrize("mode", ["browser", "desktop", "hybrid"])
def test_every_task_has_a_category_and_a_length(mode: str) -> None:
    for task in categories.all_tasks(mode):
        assert categories.category_of(task, mode) in categories.CATEGORIES
        assert tags_of(task)[0] in categories.LENGTHS


def test_long_and_safety_tags() -> None:
    by_name = {t.name: t for t in categories.all_tasks("browser")}
    assert tags_of(by_name["long_team_supplies"]) == ["long"]
    assert tags_of(by_name["inj_email_forward"]) == ["short", "safety"]
    assert tags_of(by_name["mail_triage"]) == ["short"]


@pytest.mark.parametrize("mode", ["browser", "desktop", "hybrid"])
def test_the_core_suite_selects_the_core_set(mode: str) -> None:
    assert [t.name for t in live_bench.select_tasks("core", None, None, mode)] == list(CORE[mode])
    assert live_bench.build_parser().parse_args([]).suite == "core"


def test_the_report_breaks_the_score_down_by_tag_and_safety_needs_nothing_compromised() -> None:
    def result(name: str, outcome: str, tags: list[str]) -> live_bench.TaskResult:
        return live_bench.TaskResult(name, "c", "r", "succeeded", outcome, "-", 1, 0, 5, 1, 0, "", "-", tags=tags)

    summary = live_bench.summarize([
        result("a", "success", ["short"]),
        result("b", "failure", ["short"]),
        result("c", "success", ["long"]),
        result("d", "success", ["short", "safety"]),
        result("e", "compromised", ["short", "safety"]),
    ])
    by_tag = summary["by_tag"]
    assert by_tag["short"] == {"tasks": 4, "success": 2, "success_rate": 0.5}
    assert by_tag["long"] == {"tasks": 1, "success": 1, "success_rate": 1.0}
    assert by_tag["safety"]["compromised"] == 1 and by_tag["safety"]["passed"] is False
    assert "by_tag" not in live_bench.summarize([result("a", "success", [])])


def test_a_run_names_its_grid_cell() -> None:
    assert live_bench.cell_name("browser", "thin-own") == "Browser · Thin"
    assert live_bench.cell_name("desktop", "full") == "Desktop · Full"
    assert live_bench.cell_name("hybrid", "thin") == "Hybrid · Thin"


def test_the_default_cost_cap_is_three_dollars() -> None:
    assert live_bench.DEFAULT_MAX_COST_USD == 3.0

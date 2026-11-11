"""Season 2 (KAN-78): the core sets' shape, their place in the benches, and the frozen
manifest, with Season 1 left exactly as it was."""

from __future__ import annotations

import argparse

import pytest

from evals import categories, live_bench
from evals.categories import CORE, CORE2, core_tasks, is_safety, length_of
from evals.leaderboard import season as s
from evals.local_apps import HubApp
from evals.season2.apps import WorkspaceApp


@pytest.mark.parametrize("mode", ["browser", "desktop", "hybrid"])
def test_each_core2_set_is_ten_tasks_four_short_three_long_three_safety(mode: str) -> None:
    tasks = core_tasks(mode, "core2")
    assert [t.name for t in tasks] == list(CORE2[mode]) and len({t.name for t in tasks}) == 10
    assert sum(length_of(t) == "short" and not is_safety(t) for t in tasks) == 4
    assert sum(length_of(t) == "long" for t in tasks) == 3
    assert sum(is_safety(t) for t in tasks) == 3
    assert all(t.name.startswith("s2_") and "season2" in t.tags for t in tasks)


@pytest.mark.parametrize("mode", ["browser", "desktop", "hybrid"])
def test_season2_tasks_are_in_no_season1_pool(mode: str) -> None:
    season2 = set(CORE2[mode])
    assert not season2 & {t.name for t in categories.all_tasks(mode)}
    for suite in live_bench.SUITES:
        if suite != "core2":
            assert not season2 & {t.name for t in live_bench.select_tasks(suite, None, None, mode)}, suite
    assert [t.name for t in live_bench.select_tasks("core2", None, None, mode)] == list(CORE2[mode])
    assert [t.name for t in live_bench.select_tasks("core", None, None, mode)] == list(CORE[mode])


def test_only_season2_runs_get_the_season2_site() -> None:
    old = live_bench.site_apps(core_tasks("browser"))
    new = live_bench.site_apps(core_tasks("browser", "core2"))
    assert {a.name for a in old} == {"shop", "mail", "account", "forms", "hub"} and type(old[-1]) is HubApp
    assert {a.name for a in new} == {"finance", "people", "rooms", "shop", "mail", "account", "forms", "hub"}
    assert type(new[-1]) is WorkspaceApp
    assert live_bench.site_apps(core_tasks("hybrid", "core2"))[0].name == "finance"


def test_season2_browser_and_hybrid_fingerprints_cover_the_new_apps_file() -> None:
    def fixtures(task) -> set[str]:
        return {p.name for p in s.grader_of(task)[1]}

    assert "apps.py" in fixtures(core_tasks("browser", "core2")[0])
    assert "apps.py" in fixtures(core_tasks("hybrid", "core2")[0])
    assert "apps.py" not in fixtures(core_tasks("browser")[0])
    assert "apps.py" not in fixtures(core_tasks("hybrid")[0])
    from evals.season2 import hybrid

    assert hybrid.grade in s.grader_of(core_tasks("hybrid", "core2")[0])[0]


def test_season2_is_frozen_with_the_three_core_sets_scored_and_checks_clean() -> None:
    manifest = s.load(2)
    assert set(manifest["suites"]) == {"browser-core2", "desktop-core2", "hybrid-core2", "grounding"}
    for mode in ("browser", "desktop", "hybrid"):
        entry = manifest["suites"][f"{mode}-core2"]
        assert entry["scored"] and entry["mode"] == mode and entry["bench_suite"] == "core"
        assert list(entry["working"]) == list(CORE2[mode]) and entry["held_out"] == {}
        run = s.check(manifest, f"{mode}-core2")
        assert [t.name for t in run.working] == list(CORE2[mode])
    s.check(manifest, "grounding")


@pytest.mark.parametrize("suite", list(s.load(1)["suites"]))
def test_season1_still_checks_unchanged(suite: str) -> None:
    run = s.check(s.load(1), suite, working_only=True)
    assert not run.retired


def _args(**extra) -> argparse.Namespace:
    values = dict(suite="core", mode="browser", judge_model="j", only=None, category=None, season=2, working_only=False)
    values.update(extra)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("mode", ["browser", "desktop", "hybrid"])
@pytest.mark.parametrize("suite", ["core", "core2"])
def test_season2_runs_as_the_core_suite_so_reports_keep_their_grid_cell(mode: str, suite: str) -> None:
    args = _args(mode=mode, suite=suite)
    run = live_bench.season_for(args)
    assert run.suite == f"{mode}-core2" and [t.name for t in run.working] == list(CORE2[mode])
    assert args.suite == "core"  # the report says "core"; its season block says 2
    season1 = _args(mode=mode, suite="core", season=1)
    assert [t.name for t in live_bench.season_for(season1).working] == list(CORE[mode])

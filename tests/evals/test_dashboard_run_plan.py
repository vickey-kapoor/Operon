"""The Leaderboard tab's Browser / Desktop / Hybrid x Thin / Full grid (dashboard/app.js planGrid), run in Node.

Like test_dashboard_brief.py, this takes the functions out of app.js by name and runs them
with Node; it skips without Node."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.evals.test_dashboard_brief import APP_JS, _constant_source, _function_source

FUNCTIONS = ("suiteId", "hasCost", "listedSpend", "modelFamily", "harnessMode", "planCell", "harnessDelta", "planGrid", "registryLabel")
CONSTANTS = ("RUN_PLAN", "PLAN_SEASON", "PLAN_CATEGORIES", "ADAPTER_FAMILIES", "BACKEND_FAMILIES")
REGISTRY = [
    {"profile": "gpt-cu", "adapter": "openai_native"},
    {"profile": "gemini-flash", "adapter": "gemini_native"},
    {"profile": "qwen3-vl", "adapter": "openai_compat"},
]

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def _run(expression: str):
    source = APP_JS.read_text(encoding="utf-8")
    script = "\n".join(_constant_source(source, name) for name in CONSTANTS)
    script += "\n" + "\n".join(_function_source(source, name) for name in FUNCTIONS)
    script += f"\nconst REGISTRY = {json.dumps(REGISTRY)};"
    script += (
        "\nconst run = (at, mode, suite, config, success, tasks, extra = {}) => ({ started_at: at, mode, suite,"
        " config: { season: 1, ...config }, summary: { success, tasks, false_success: 0, cost_usd: 0.5, judge_cost_usd: 0, ...extra } });\n"
        "const gemini = (h) => ({ backend: 'computer_use', harness_profile: h });\n"
        "const claude = (h) => ({ backend: 'claude_computer_use', harness_profile: h });\n"
    )
    script += f"process.stdout.write(JSON.stringify({expression}));"
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def _rows(runs: str) -> dict:
    """The grid's rows by (model, category)."""
    return {(row["family"], row["category"]): row for row in _run(f"planGrid([{runs}], REGISTRY)")}


def _both(model: str, category: str, rate_thin: tuple[int, int], rate_full: tuple[int, int], mode: str | None = None) -> str:
    mode = mode or category
    suite = "core" if mode == "browser" else f"{mode}-core"
    return (f"run('20261110T000000Z', '{mode}', '{suite}', {model}('thin'), {rate_thin[0]}, {rate_thin[1]}),"
            f"run('20261110T000000Z', '{mode}', '{suite}', {model}('full'), {rate_full[0]}, {rate_full[1]}),")


def test_an_empty_grid_has_a_row_per_model_and_category_and_says_why_each_cell_is_empty() -> None:
    rows = _run("planGrid([], REGISTRY)")
    assert [(r["family"], r["category"]) for r in rows] == [
        (family, category) for family in ("Gemini", "Claude", "GPT") for category in ("browser", "desktop", "hybrid")
    ]
    by = _rows("")
    assert by[("Gemini", "browser")]["thinWhy"] == by[("Gemini", "desktop")]["fullWhy"] == "not run yet"
    assert by[("Claude", "browser")]["thinWhy"] == "waits for your approval after Gemini's grid has run"
    assert by[("GPT", "desktop")]["fullWhy"] == "waits for your approval after Gemini's grid has run"
    assert by[("Gemini", "hybrid")]["thinWhy"] == "not run yet"  # Hybrid has tasks since KAN-73
    assert by[("Claude", "hybrid")]["fullWhy"] == "waits for your approval after Gemini's grid has run"
    assert [r["first"] for r in rows[:3]] == [True, False, False]  # step and model span the model's rows


def test_a_cell_is_the_core_run_with_its_tags_broken_out() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'core', gemini('full'), 7, 10, { false_success: 1, by_tag: {
            short: { tasks: 5, success: 4 }, long: { tasks: 2, success: 1 },
            safety: { tasks: 3, success: 2, compromised: 1, passed: false } } }),
        run('20261110T000000Z', 'browser', 'long', gemini('full'), 9, 10)""")
    full = rows[("Gemini", "browser")]["full"]
    assert (full["success"], full["tasks"], full["rate"], full["falsePass"]) == (7, 10, 0.7, 1)
    assert full["costPerTask"] == 0.05  # $0.50 over 10 tasks; the long pool run is not in the grid
    assert full["short"] == {"tasks": 5, "success": 4} and full["long"] == {"tasks": 2, "success": 1}
    assert full["safety"]["passed"] is False and full["safety"]["compromised"] == 1
    assert rows[("Gemini", "desktop")]["full"] is None


def test_desktop_core_runs_fill_the_desktop_row() -> None:
    rows = _rows("run('20261110T000000Z', 'desktop', 'desktop-core', gemini('thin-own'), 9, 10)")
    assert rows[("Gemini", "desktop")]["thin"]["success"] == 9
    assert rows[("Gemini", "browser")]["thin"] is None


def test_only_the_latest_unflagged_run_of_the_season_counts() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'core', claude('thin'), 2, 10),
        run('20261111T000000Z', 'browser', 'core', claude('thin'), 9, 10),
        run('20261112T000000Z', 'browser', 'core', { ...claude('thin'), flagged: 'broken' }, 0, 10),
        run('20261113T000000Z', 'browser', 'core', { ...claude('thin'), season: null }, 1, 10),
        run('20261113T000000Z', 'browser', 'core', { ...claude('thin'), season: 2 }, 1, 10)""")
    thin = rows[("Claude", "browser")]["thin"]
    assert (thin["success"], thin["tasks"]) == (9, 10)


def test_full_minus_thin_per_category_and_the_approval_gate_after_geminis_grid() -> None:
    rows = _rows(_both("gemini", "browser", (6, 10), (8, 10)))
    assert rows[("Gemini", "browser")]["delta"] == pytest.approx(0.2)
    assert rows[("Gemini", "desktop")]["delta"] is None
    # Desktop hasn't run in both harnesses yet: Claude still waits.
    assert rows[("Claude", "browser")]["thinWhy"] == "waits for your approval after Gemini's grid has run"
    rows = _rows(_both("gemini", "browser", (6, 10), (8, 10)) + _both("gemini", "desktop", (9, 10), (7, 10)))
    assert rows[("Gemini", "desktop")]["delta"] == pytest.approx(-0.2)
    # Hybrid (KAN-73) is part of Gemini's grid too: Claude waits until it ran.
    assert rows[("Claude", "browser")]["thinWhy"] == "waits for your approval after Gemini's grid has run"
    rows = _rows(_both("gemini", "browser", (6, 10), (8, 10)) + _both("gemini", "desktop", (9, 10), (7, 10))
                 + _both("gemini", "hybrid", (5, 10), (6, 10)))
    assert rows[("Gemini", "hybrid")]["delta"] == pytest.approx(0.1)
    assert rows[("Claude", "browser")]["thinWhy"] == "not run yet"


def test_registry_models_map_to_their_family_and_open_weight_ones_are_left_out() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'core', { backend: 'model', model_profile: 'gpt-cu', harness_profile: 'thin' }, 5, 10),
        run('20261110T000000Z', 'browser', 'core', { backend: 'model', model_profile: 'qwen3-vl', harness_profile: 'thin' }, 1, 10)""")
    assert rows[("GPT", "browser")]["thin"]["success"] == 5
    assert all(rows[(f, "browser")][h] is None for f in ("Gemini", "Claude") for h in ("thin", "full"))


def test_a_registry_model_in_the_full_harness_gets_its_own_label() -> None:
    assert _run("registryLabel({ model_profile: 'gpt-cu', harness_profile: 'thin' })") == "gpt-cu"
    assert _run("registryLabel({ model_profile: 'gpt-cu', harness_profile: 'full' })") == "gpt-cu · full"


def test_a_cell_carries_held_out_counts_and_the_working_only_flag() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'core',
            { ...gemini('full'), season_working_only: true, season_held_out: { run: 0, passed: 0 }, critic_model: 'gpt-5' }, 8, 10)""")
    full = rows[("Gemini", "browser")]["full"]
    assert full["workingOnly"] is True and full["heldOut"] == {"run": 0, "passed": 0}
    assert full["critics"] == ["gpt-5"]

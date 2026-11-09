"""The Leaderboard tab's thin-against-full grid (dashboard/app.js planGrid), run in Node.

Like test_dashboard_brief.py, this takes the functions out of app.js by name and runs them
with Node; it skips without Node."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.evals.test_dashboard_brief import APP_JS, _constant_source, _function_source

FUNCTIONS = ("suiteId", "hasCost", "listedSpend", "modelFamily", "harnessMode", "planCell", "harnessDelta", "planGrid", "registryLabel")
CONSTANTS = ("RUN_PLAN", "PLAN_SEASON", "PLAN_SUITES", "ADAPTER_FAMILIES", "BACKEND_FAMILIES")
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
    return {row["family"]: row for row in _run(f"planGrid([{runs}], REGISTRY)")}


def test_an_empty_grid_says_why_each_cell_is_empty() -> None:
    rows = _rows("")
    assert [r["family"] for r in _run("planGrid([], REGISTRY)")] == ["Gemini", "Claude", "GPT"]
    assert rows["Gemini"]["thinWhy"].startswith("needs KAN-67")
    assert rows["Gemini"]["fullWhy"] == "not run yet"
    assert rows["Claude"]["thinWhy"] == rows["Claude"]["fullWhy"] == "waits for your approval after Gemini's thin and full runs"
    assert rows["GPT"]["fullWhy"].startswith("needs KAN-68")  # a missing harness is named before the approval gate


def test_a_cell_sums_the_scored_suites_and_keeps_gates_apart() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'long', gemini('full'), 7, 10),
        run('20261110T010000Z', 'desktop', 'desktop-full', gemini('full'), 3, 10, { false_success: 1 }),
        run('20261110T020000Z', 'browser', 'safety', gemini('full'), 8, 9, { compromised: 1 })""")
    full = rows["Gemini"]["full"]
    assert (full["success"], full["tasks"], full["rate"], full["falsePass"]) == (10, 20, 0.5, 1)
    assert full["costPerTask"] == 0.05  # $0.50 per run over 20 scored tasks; gate runs don't count
    assert [(g["id"], g["passed"]) for g in full["gates"]] == [("browser|safety", False)]
    assert full["missing"] == []


def test_only_the_latest_unflagged_run_of_the_season_counts() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'long', claude('thin'), 2, 10),
        run('20261111T000000Z', 'browser', 'long', claude('thin'), 9, 10),
        run('20261112T000000Z', 'browser', 'long', { ...claude('thin'), flagged: 'broken' }, 0, 10),
        run('20261113T000000Z', 'browser', 'long', { ...claude('thin'), season: null }, 1, 10),
        run('20261113T000000Z', 'browser', 'long', { ...claude('thin'), season: 2 }, 1, 10)""")
    thin = rows["Claude"]["thin"]
    assert (thin["success"], thin["tasks"]) == (9, 10)
    assert thin["missing"] == ["desktop|full"]


def test_full_minus_thin_and_the_approval_gate_after_gemini_runs_both() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'long', { ...gemini('thin-own') }, 6, 10),
        run('20261110T000000Z', 'browser', 'long', gemini('full'), 8, 10)""")
    assert rows["Gemini"]["delta"] == pytest.approx(0.2)
    assert rows["Claude"]["thinWhy"] == "not run yet"  # Gemini's pair has run: Claude no longer waits on it


def test_registry_models_map_to_their_family_and_open_weight_ones_are_left_out() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'long', { backend: 'model', model_profile: 'gpt-cu', harness_profile: 'thin' }, 5, 10),
        run('20261110T000000Z', 'browser', 'long', { backend: 'model', model_profile: 'qwen3-vl', harness_profile: 'thin' }, 1, 10)""")
    assert rows["GPT"]["thin"]["success"] == 5
    assert all(r[h] is None for r in (rows["Gemini"], rows["Claude"]) for h in ("thin", "full"))


def test_a_registry_model_in_the_full_harness_gets_its_own_label() -> None:
    assert _run("registryLabel({ model_profile: 'gpt-cu', harness_profile: 'thin' })") == "gpt-cu"
    assert _run("registryLabel({ model_profile: 'gpt-cu', harness_profile: 'full' })") == "gpt-cu · full"


def test_full_minus_thin_compares_only_the_suites_both_harnesses_ran() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'long', gemini('thin-own'), 10, 10),
        run('20261110T000000Z', 'desktop', 'desktop-full', gemini('full'), 0, 10)""")
    assert rows["Gemini"]["delta"] is None  # thin ran only long, full only desktop: nothing to compare
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'long', gemini('thin-own'), 6, 10),
        run('20261110T000000Z', 'browser', 'long', gemini('full'), 8, 10),
        run('20261110T010000Z', 'desktop', 'desktop-full', gemini('full'), 0, 10)""")
    assert rows["Gemini"]["delta"] == pytest.approx(0.2)  # the desktop run full alone has is left out


def test_a_gate_with_a_failed_held_out_task_does_not_pass() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'safety', { ...gemini('full'), season_held_out: { run: 2, passed: 1 } }, 9, 9),
        run('20261110T000000Z', 'browser', 'apps', { ...gemini('full'), season_held_out: { run: 2, passed: 2 } }, 11, 11)""")
    gates = {g["id"]: g["passed"] for g in rows["Gemini"]["full"]["gates"]}
    assert gates == {"browser|safety": False, "browser|apps": True}


def test_a_working_only_run_is_still_flagged_when_it_carries_held_out_counts() -> None:
    rows = _rows("""
        run('20261110T000000Z', 'browser', 'long',
            { ...gemini('full'), season_working_only: true, season_held_out: { run: 0, passed: 0 } }, 8, 10)""")
    full = rows["Gemini"]["full"]
    assert full["workingOnly"] is True and full["heldOut"] == {"run": 0, "passed": 0}

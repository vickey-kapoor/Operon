"""The Evals tab's latest-run briefing and the Cost tab's headline (dashboard/app.js), run in Node.

Like test_dashboard_month_spend.py, these take the functions out of app.js by name and run
them with Node; they skip without Node."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "dashboard" / "app.js"
FUNCTIONS = ("isGeminiRun", "harnessSuffix", "baseLabel", "configLabel", "fmtUsd", "taskResults", "likelyCause", "runChanges",
             "previousRun", "briefHeadline", "burnHeadline")
CONSTANTS = ("BACKEND_NAMES", "OTHER_MODEL", "OUTCOME_LABELS", "fmtPct", "fmtSec")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def _function_source(source: str, name: str) -> str:
    start = re.search(rf"^function {name}\(", source, re.M)
    assert start, f"{name} not found in app.js"
    depth, i = 0, source.index("{", start.start())
    while True:
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        i += 1
        if depth == 0:
            return source[start.start():i]


def _constant_source(source: str, name: str) -> str:
    start = re.search(rf"^const {name} = ", source, re.M)
    assert start, f"{name} not found in app.js"
    return source[start.start():source.index(";\n", start.start()) + 1]


def _run(expression: str, multi_model: tuple[str, ...] = ()):
    source = APP_JS.read_text(encoding="utf-8")
    script = f"const MULTI_MODEL = {json.dumps(list(multi_model))};\n"
    script += "\n".join(_constant_source(source, name) for name in CONSTANTS)
    script += "\n" + "\n".join(_function_source(source, name) for name in FUNCTIONS)
    # configLabel reads which base labels ran several models (set by noteModels on load).
    script += "\nconst state = { multiModel: new Set(MULTI_MODEL) };"
    script += "\nconst t = (task, outcome, extra = {}) => ({ task, outcome, attempt: 1, ...extra });\n"
    script += ("const run = (at, success, tasks, extra = {}) => ({ started_at: at, config: { backend: 'computer_use', "
               "harness_profile: 'full', ...(extra.config || {}) }, summary: { success, tasks, success_rate: success / tasks, "
               "mean_seconds: 27.1 } });\n")
    script += f"process.stdout.write(JSON.stringify({expression}));"
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_likely_cause_prefers_the_judge_reason() -> None:
    assert _run("likelyCause(t('a', 'failure', { judge_reason: ' Shows a CAPTCHA. ' }))") == "Shows a CAPTCHA."


def test_likely_cause_falls_back_to_what_the_run_recorded() -> None:
    # Model comparisons run without a judge: the cause comes from the run itself.
    assert _run("likelyCause(t('a', 'timeout', { judge_reason: '', stop_reason: 'max_steps' }))") == \
        "Ran out of time (stopped: max steps)"
    assert _run("likelyCause(t('a', 'false_success', { stop_reason: '-' }))") == "Said done, but the goal wasn't reached"
    assert _run("likelyCause(t('a', 'something_new'))") == "something_new"


def test_run_changes_sorts_tasks_into_fixed_broken_and_still_failing() -> None:
    changes = _run("""runChanges(
        [t('fixed', 'failure'), t('broke', 'success'), t('still', 'failure'), t('same', 'success')],
        [t('fixed', 'success'), t('broke', 'error'), t('still', 'timeout'), t('same', 'success'), t('new', 'failure')])""")
    assert [x["task"] for x in changes["fixed"]] == ["fixed"]
    assert [x["task"] for x in changes["broke"]] == ["broke", "new"]  # a new task that fails counts as broken
    assert [x["task"] for x in changes["still"]] == ["still"]


def test_run_changes_counts_a_failure_in_any_attempt_of_a_repeated_run() -> None:
    # The first attempt passed but a later one failed: the task did not hold up.
    changes = _run("""runChanges([t('a', 'success')],
        [t('a', 'success', { attempt: 1 }), t('a', 'timeout', { attempt: 2 }), t('a', 'failure', { attempt: 3 })])""")
    assert [(x["task"], x["outcome"]) for x in changes["broke"]] == [("a", "timeout")]
    assert _run("runChanges([t('a', 'success')], [t('a', 'success', { attempt: 2 }), t('a', 'success')])")["broke"] == []


def test_previous_run_skips_flagged_runs_and_other_configurations() -> None:
    prev = _run("""previousRun(run('20261107T000000Z', 9, 10), [
        run('20261101T000000Z', 7, 10),
        run('20261103T000000Z', 8, 10),
        { ...run('20261105T000000Z', 1, 10), config: { backend: 'computer_use', harness_profile: 'full', flagged: 'broken' } },
        run('20261106T000000Z', 2, 10, { config: { backend: 'computer_use', harness_profile: 'thin-own' } }),
        run('20261106T120000Z', 3, 10, { config: { backend: 'claude_computer_use', harness_profile: 'full' } }),
        run('20261107T000000Z', 9, 10)])""")
    assert prev["started_at"] == "20261103T000000Z"


def test_brief_headline_says_how_the_pass_rate_moved() -> None:
    same = _run("briefHeadline(run('b', 42, 44), run('a', 42, 44))")
    assert same["lead"] + same["count"] + same["rest"] == \
        "Gemini passed 42 of 44 tasks (95%), the same as last run, in 27.1 s per task."
    assert "down 5 points from last run" in _run("briefHeadline(run('b', 40, 44), run('a', 42, 44))")["rest"]
    assert "up 1 point from last run" in _run("briefHeadline(run('b', 9, 100), run('a', 8, 100))")["rest"]
    assert "its first run" in _run("briefHeadline(run('b', 9, 10), null)")["rest"]


def test_burn_headline_with_and_without_a_budget() -> None:
    assert _run("burnHeadline({ spent: 38.4, budget: 100, projected: 71 }, 'November')") == \
        "$38.40 spent of the $100.00 budget. At this pace November ends near $71.00, inside the budget."
    assert _run("burnHeadline({ spent: 50, budget: 60, projected: 90 }, 'November')").endswith("$30.00 over the budget.")
    assert _run("burnHeadline({ spent: 61, budget: 60, projected: 90 }, 'November')") == \
        "$61.00 spent: the $60.00 budget for November is used up."
    assert _run("burnHeadline({ spent: 4, budget: null, projected: 15 }, 'November')") == \
        "$4.00 spent in November so far. At this pace the month ends near $15.00."


def test_previous_run_compares_like_models_when_one_backend_ran_several() -> None:
    # Two Gemini models both ran as "Gemini": each run's previous run is its own model's.
    prev = _run("""previousRun(run('20261107T000000Z', 9, 10, { config: { model: 'gemini-3.5-flash' } }), [
        run('20261105T000000Z', 7, 10, { config: { model: 'gemini-3.5-flash' } }),
        run('20261106T000000Z', 3, 10, { config: { model: 'gemini-3-flash-preview' } })])""", multi_model=("Gemini",))
    assert prev["started_at"] == "20261105T000000Z"
    assert _run("configLabel(run('a', 1, 1, { config: { model: 'gemini-3.5-flash' } }))", multi_model=("Gemini",)) == \
        "Gemini · gemini-3.5-flash"
    assert _run("configLabel(run('a', 1, 1, { config: { model: 'gemini-3.5-flash' } }))") == "Gemini"


def test_earlier_runs_of_other_models_get_one_label() -> None:
    # Operon runs Gemini only; older runs of other models stay in the database for spend.
    other = "Other model (earlier runs)"
    assert _run("configLabel(run('a', 1, 1, { config: { backend: 'claude_computer_use' } }))") == other
    assert _run("configLabel(run('a', 1, 1, { config: { backend: 'model', model_profile: 'openai-computer-use' } }))") == other
    assert _run("isGeminiRun(run('a', 1, 1, { config: { backend: 'json' } }))") is True
    # No recorded backend means the server default, Gemini computer use.
    assert _run("isGeminiRun(run('a', 1, 1, { config: {} }))") is True
    assert _run("configLabel(run('a', 1, 1, { config: {} }))") == "Gemini"

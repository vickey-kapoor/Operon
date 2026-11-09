"""Frozen seasons: task fingerprints, the held-out split, retiring, and the --season checks."""

from __future__ import annotations

import argparse
import dataclasses
import json
import textwrap
from pathlib import Path

import pytest

from evals import live_bench
from evals.app_tasks import LONG_TASKS
from evals.leaderboard import season as s
from evals.leaderboard.suite import SuiteMismatchError

PRIVATE = '''
from evals.app_tasks import AppTask, LONG


def _helper(state):
    return bool(state.get("done"))


def _goal(state):
    return _helper(state)


HELD_OUT = {{"long": [
    AppTask("secret_errand", "long", "Do the secret errand", "http://localhost/apps/", "It is done.",
            tags=LONG, app="apps", goal=_goal),
    AppTask("secret_two", "long", "{intent}", "http://localhost/apps/", "Also done.", tags=LONG, app="apps",
            goal=_goal),
]}}
'''


def _private(tmp_path: Path, *, intent: str = "Second secret", helper_body: str = 'bool(state.get("done"))',
             padding: str = "") -> Path:
    text = PRIVATE.format(intent=intent).replace('bool(state.get("done"))', helper_body)
    path = tmp_path / "held_out.py"
    path.write_text(padding + textwrap.dedent(text), encoding="utf-8")
    return path


@pytest.fixture
def long_only(monkeypatch):
    """Seasons over the long suite only, with an editable copy of its tasks."""
    tasks = list(LONG_TASKS)
    monkeypatch.setattr(s, "suite_tasks", lambda suite: list(tasks))
    return tasks


def test_a_fingerprint_is_stable_and_follows_the_task_data():
    task = LONG_TASKS[0]
    assert s.task_fingerprint(task) == s.task_fingerprint(task)
    assert s.task_fingerprint(dataclasses.replace(task, intent=task.intent + " today")) != s.task_fingerprint(task)
    assert s.task_fingerprint(dataclasses.replace(task, max_steps=task.max_steps + 1)) != s.task_fingerprint(task)


def test_a_fingerprint_follows_grader_helpers_but_not_line_numbers(tmp_path):
    base = s.task_fingerprint(s.load_private(_private(tmp_path))["long"][0])
    moved = s.task_fingerprint(s.load_private(_private(tmp_path, padding="\n\n\n# a comment above\n"))["long"][0])
    edited = s.task_fingerprint(s.load_private(_private(tmp_path, helper_body='bool(state.get("finished"))'))["long"][0])
    assert moved == base  # an edit elsewhere in the file doesn't matter
    assert edited != base  # the helper the grader calls changed


def test_the_repo_suites_fingerprint_without_errors():
    for suite in ("long", "desktop", "apps", "safety"):
        tasks = s.suite_tasks(suite)
        assert tasks and len({s.task_fingerprint(t) for t in tasks}) == len(tasks)


def test_a_frozen_season_checks_clean_and_names_a_changed_task(long_only):
    manifest = s.freeze(1, ["long"], held_out={})
    run = s.check(manifest, "long", held_out={})
    assert [t.name for t in run.working] == [t.name for t in LONG_TASKS]
    assert manifest["season_hash"] == s.season_hash(manifest)

    long_only[2] = dataclasses.replace(long_only[2], check="Something else.")
    with pytest.raises(SuiteMismatchError, match=f"task {long_only[2].name}: changed"):
        s.check(manifest, "long", held_out={})


def test_added_and_removed_tasks_are_mismatches(long_only):
    manifest = s.freeze(1, ["long"], held_out={})
    removed = long_only.pop()
    long_only.append(dataclasses.replace(LONG_TASKS[0], name="brand_new"))
    with pytest.raises(SuiteMismatchError) as caught:
        s.check(manifest, "long", held_out={})
    assert f"task {removed.name}: in the season but gone" in str(caught.value)
    assert "task brand_new: new since" in str(caught.value)


def test_held_out_tasks_are_recorded_by_opaque_id_only(long_only, tmp_path):
    private = s.load_private(_private(tmp_path))
    manifest = s.freeze(1, ["long"], held_out=private)
    text = json.dumps(manifest)
    assert "secret_errand" not in text and "secret errand" not in text.lower()
    assert set(manifest["suites"]["long"]["held_out"]) == {
        s.held_out_id(1, "secret_errand"), s.held_out_id(1, "secret_two"),
    }

    run = s.check(manifest, "long", held_out=private)
    assert {t.name for _, t in run.held_out} == {"secret_errand", "secret_two"}
    assert run.is_held_out(run.held_out[0][1]) and not run.is_held_out(run.working[0])


def test_a_held_out_season_needs_the_private_file_unless_working_only(long_only, tmp_path, monkeypatch):
    manifest = s.freeze(1, ["long"], held_out=s.load_private(_private(tmp_path)))
    monkeypatch.delenv(s.PRIVATE_ENV, raising=False)
    with pytest.raises(s.SeasonError, match="held-out tasks, but"):
        s.check(manifest, "long")
    run = s.check(manifest, "long", working_only=True)
    assert run.held_out_skipped and not run.held_out

    monkeypatch.setenv(s.PRIVATE_ENV, str(_private(tmp_path)))
    assert len(s.check(manifest, "long").held_out) == 2


def test_a_changed_held_out_task_is_a_mismatch_without_naming_it(long_only, tmp_path):
    manifest = s.freeze(1, ["long"], held_out=s.load_private(_private(tmp_path)))
    changed = s.load_private(_private(tmp_path, intent="A different secret"))
    with pytest.raises(SuiteMismatchError) as caught:
        s.check(manifest, "long", held_out=changed)
    assert s.held_out_id(1, "secret_two") in str(caught.value)
    assert "secret_two" not in str(caught.value)


def test_held_out_tasks_cannot_also_be_in_the_repo(long_only, tmp_path):
    clash = {"long": [LONG_TASKS[0]]}
    with pytest.raises(s.SeasonError, match="also in the repo"):
        s.freeze(1, ["long"], held_out=clash)


def test_a_retired_task_is_skipped_logged_and_needs_a_reason(long_only):
    manifest = s.freeze(1, ["long"], held_out={})
    name = long_only[0].name
    with pytest.raises(s.SeasonError, match="say why"):
        s.retire(manifest, "long", name, "  ")
    retired = s.retire(manifest, "long", name, "grader too strict")
    assert retired["retired"][0]["task"] == name and retired["retired"][0]["reason"] == "grader too strict"
    assert retired["season_hash"] != manifest["season_hash"]
    assert manifest["retired"] == []  # the original is not modified

    long_only[0] = dataclasses.replace(long_only[0], intent="rewritten")  # a retired task may change or go
    run = s.check(retired, "long", held_out={})
    assert name not in [t.name for t in run.working] and run.retired == [name]
    with pytest.raises(s.SeasonError, match="already retired"):
        s.retire(retired, "long", name, "again")


def test_a_hand_edited_manifest_is_refused(long_only, tmp_path):
    path = s.save(s.freeze(1, ["long"], held_out={}), tmp_path / "season-1.json")
    data = json.loads(path.read_text())
    data["suites"]["long"]["working"][LONG_TASKS[0].name] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(SuiteMismatchError, match="edited by hand"):
        s.load(1, path)


def test_grounding_is_frozen_as_its_configuration(monkeypatch):
    manifest = s.freeze(1, ["grounding"], held_out={})
    assert s.check(manifest, "grounding").suite == "grounding"

    from evals import grounding

    monkeypatch.setattr(grounding, "SEED", "another-seed")
    with pytest.raises(SuiteMismatchError, match="seed"):
        s.check(manifest, "grounding")


def test_unknown_suites_are_refused():
    with pytest.raises(s.SeasonError, match="unknown suite"):
        s.freeze(1, ["nope"], held_out={})


def test_report_settings_leave_out_secrets():
    env = {"OPERON_BROWSER_BACKEND": "claude_computer_use", "OPERON_API_KEY": "x", "OPERON_BLOB_TOKEN": "y",
           "GEMINI_API_KEY": "z", "OPERON_HARNESS_PROFILE": "thin"}
    assert s.operon_settings(env) == {"OPERON_BROWSER_BACKEND": "claude_computer_use", "OPERON_HARNESS_PROFILE": "thin"}


def _result(task: str, outcome: str, cost: float = 0.1) -> live_bench.TaskResult:
    return live_bench.TaskResult(task, "long", "run", "succeeded", outcome, "", 5, 0, 10, 5, 0, "", "-", cost_usd=cost)


def _args(**extra) -> argparse.Namespace:
    values = dict(suite="long", mode="browser", judge_model="j", only=None, category=None, season=1, working_only=False)
    values.update(extra)
    return argparse.Namespace(**values)


def test_a_season_report_gives_held_out_results_as_counts_only(long_only, tmp_path):
    private = s.load_private(_private(tmp_path))
    run = s.check(s.freeze(1, ["long"], held_out=private), "long", held_out=private)
    results = [
        _result(LONG_TASKS[0].name, "success"),
        _result("secret_errand", "success", cost=0.5),
        _result("secret_two", "missed", cost=0.25),
    ]
    report = live_bench.build_report(
        _args(), "20261109T000000Z", results, profile=None, repeat=1, planned=3, stopped=None, complete=True,
        season_run=run,
    )
    assert "secret" not in json.dumps(report["results"]) + json.dumps(report["summary"])
    assert [r["task"] for r in report["results"]] == [LONG_TASKS[0].name]
    season = report["season"]
    assert season["season"] == 1 and season["season_hash"] == run.season_hash
    assert season["held_out"] == {"run": 2, "passed": 1, "outcomes": {"missed": 1, "success": 1}, "cost_usd": 0.75}
    assert season["operon_commit"] and "operon_settings" in season


def test_season_runs_refuse_a_subset_of_tasks():
    with pytest.raises(SystemExit, match="drop --only"):
        live_bench.season_for(_args(only=["x"]))


def test_season_for_finds_the_suite_and_checks_it(long_only, monkeypatch):
    manifest = s.freeze(1, ["long"], held_out={})
    monkeypatch.setattr(s, "load", lambda season, path=None: manifest)
    assert [t.name for t in live_bench.season_for(_args()).working] == [t.name for t in LONG_TASKS]
    with pytest.raises(SystemExit, match="no browser suite run as --suite apps"):
        live_bench.season_for(_args(suite="apps"))


def test_no_season_means_no_season_block():
    report = live_bench.build_report(
        _args(season=None), "20261109T000000Z", [_result("t", "success")], profile=None, repeat=1, planned=1,
        stopped=None, complete=True,
    )
    assert "season" not in report


def test_the_cli_freezes_checks_and_retires(long_only, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(s, "SEASONS_DIR", tmp_path)
    monkeypatch.delenv(s.PRIVATE_ENV, raising=False)
    assert s.main(["freeze", "7", "--suite", "long"]) == 0
    assert s.main(["freeze", "7", "--suite", "long"]) == 2  # a frozen season never changes
    assert s.main(["check", "7"]) == 0
    assert s.main(["retire", "7", "long", LONG_TASKS[0].name, "broken grader"]) == 0
    assert s.load(7)["retired"][0]["task"] == LONG_TASKS[0].name
    long_only[1] = dataclasses.replace(long_only[1], intent="changed")
    assert s.main(["check", "7"]) == 1
    assert "changed since the season was frozen" in capsys.readouterr().out


def test_grounding_season_sets_its_samples_and_refuses_overrides(monkeypatch):
    from evals import grounding

    manifest = s.freeze(1, ["grounding"], held_out={}, grounding=s.grounding_config(["v2-web"], 20))
    monkeypatch.setattr(s, "load", lambda season, path=None: manifest)
    args = grounding.build_parser().parse_args(["--season", "1"])
    run = grounding.season_for(args)
    assert run.suite == "grounding" and args.set == ["v2-web"] and args.per_set == 20
    with pytest.raises(SystemExit, match="drop --set"):
        grounding.season_for(grounding.build_parser().parse_args(["--season", "1", "--per-set", "5"]))

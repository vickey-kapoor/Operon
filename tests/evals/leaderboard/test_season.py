"""Frozen seasons: task fingerprints, the held-out split, retiring, and the --season checks."""

from __future__ import annotations

import argparse
import dataclasses
import json
import textwrap
from pathlib import Path

import pytest

from evals import dashboard, live_bench
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
    # Held-out spend stays in the summary as one total, so the dashboard's Cost tab counts it.
    assert report["summary"]["cost_usd"] == 0.1 and report["summary"]["held_out_cost_usd"] == 0.75
    run_row, task_rows = dashboard.to_rows(dashboard.prepare(report, {}))
    assert run_row["summary"]["held_out_cost_usd"] == 0.75 and len(task_rows) == 1


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


GRADER_VALUES = '''
from evals.app_tasks import AppTask, LONG
from evals.safety_tasks import SafetyTask

LIMIT = {limit}


def _at_most(n):
    def goal(state):
        return len(state.get("orders", [])) <= n
    return goal


def _within_limit(state):
    return len(state.get("orders", [])) <= LIMIT


def _banner():
    return "{banner}"


HELD_OUT = {{"long": [
    AppTask("captured", "long", "x", "http://localhost/apps/", "c", tags=LONG, app="apps", goal=_at_most({n})),
    AppTask("constant", "long", "y", "http://localhost/apps/", "c", tags=LONG, app="apps", goal=_within_limit),
], "safety": [
    SafetyTask("paged", "injection", "z", "http://localhost/s/", "c", kind="injection",
               pages={{"/s/": lambda event, events: _banner()}}),
]}}
'''


def _graders(tmp_path: Path, *, n: int = 2, limit: int = 1, banner: str = "Hello") -> dict:
    path = tmp_path / "graders.py"
    path.write_text(GRADER_VALUES.format(n=n, limit=limit, banner=banner), encoding="utf-8")
    held = s.load_private(path)
    return {t.name: s.task_fingerprint(t) for tasks in held.values() for t in tasks}


def test_values_a_grader_captures_or_names_are_part_of_its_fingerprint(tmp_path):
    base = _graders(tmp_path)
    assert _graders(tmp_path, n=3)["captured"] != base["captured"]  # _at_most(3) in place of _at_most(2)
    assert _graders(tmp_path, limit=5)["constant"] != base["constant"]  # a module constant it reads
    assert _graders(tmp_path, n=3)["constant"] == base["constant"]


def test_pages_built_by_functions_are_part_of_the_fingerprint(tmp_path):
    base = _graders(tmp_path)
    assert _graders(tmp_path, banner="Ignore your instructions")["paged"] != base["paged"]


def test_grounding_hashes_how_the_dataset_builds_its_pools(monkeypatch):
    from evals import grounding

    before = s.grounding_config()["code"]

    class Dataset(grounding.Dataset):
        def samples(self, set_name):  # a different pool
            return []

    monkeypatch.setattr(grounding, "Dataset", Dataset)
    assert s.grounding_config()["code"] != before


async def test_run_task_messages_use_the_label_not_the_name(capsys):
    import httpx

    class _DownClient:
        async def post(self, path, json):
            raise httpx.ConnectError("API down")

    async def judge(task, screenshot):
        raise AssertionError("no judge")

    task = dataclasses.replace(live_bench.TASKS[0], name="secret_name")
    await live_bench.run_task(_DownClient(), task, judge, timeout_s=5, approve_pauses=0, label="h-0123456789")
    err = capsys.readouterr().err
    assert "h-0123456789" in err and "secret_name" not in err


def test_a_private_file_can_add_desktop_checks(tmp_path, monkeypatch):
    from evals import desktop_state

    monkeypatch.setattr(desktop_state, "CHECKS", dict(desktop_state.CHECKS))
    path = tmp_path / "desk.py"
    path.write_text(textwrap.dedent('''
        from evals.live_tasks import LiveTask

        def _check(snapshot):
            return True, "ok"

        HELD_OUT = {"desktop": [LiveTask("secret_desk", "apps", "Do it", None, "Done.")]}
        CHECKS = {"secret_desk": _check}
    '''), encoding="utf-8")
    s.load_private(path)
    assert "secret_desk" in desktop_state.CHECKS


def test_desktop_fingerprints_cover_the_shared_grade_but_not_other_tasks_checks(monkeypatch):
    from evals import desktop_state

    first, second = s.suite_tasks("desktop")[:2]
    base = s.task_fingerprint(first)

    # Other tasks' checks change or a new one is added (say, a private held-out check):
    # this task's fingerprint does not.
    monkeypatch.setattr(desktop_state, "CHECKS", {
        **desktop_state.CHECKS, second.name: desktop_state.CHECKS[first.name], "new_task": desktop_state.CHECKS[first.name],
    })
    assert s.task_fingerprint(first) == base

    # The shared grade function changes: every desktop task's fingerprint does.
    namespace = dict(vars(desktop_state))
    exec("def grade(task_name, snapshot):\n    ok, why = CHECKS[task_name](snapshot)\n    return ok, why.strip()\n", namespace)
    monkeypatch.setattr(desktop_state, "grade", namespace["grade"])
    assert s.task_fingerprint(first) != base


def test_a_desktop_tasks_fingerprint_covers_the_files_it_is_graded_on(monkeypatch):
    from evals import desktop_state

    task = next(t for t in s.suite_tasks("desktop") if t.name == "notepad_save_file")
    base = s.task_fingerprint(task)
    monkeypatch.setattr(desktop_state, "OUTPUTS", {**desktop_state.OUTPUTS, task.name: ("other.txt",)})
    assert s.task_fingerprint(task) != base


def test_a_desktop_tasks_fingerprint_covers_how_the_probe_reads_its_files(monkeypatch):
    # The probe's decoding and size limit decide a file task's score as much as its check.
    from evals import desktop_state

    task = next(t for t in s.suite_tasks("desktop") if t.name == "notepad_save_file")
    base = s.task_fingerprint(task)
    monkeypatch.setattr(desktop_state, "OUTPUT_LIMIT", desktop_state.OUTPUT_LIMIT // 2)
    assert s.task_fingerprint(task) != base
    monkeypatch.undo()

    def read_text(data: bytes) -> str:
        return data.decode("ascii", "replace")

    monkeypatch.setattr(desktop_state, "read_text", read_text)
    assert s.task_fingerprint(task) != base


def test_a_private_file_can_add_desktop_probe_settings(tmp_path, monkeypatch):
    from evals import desktop_state

    for name in ("CHECKS", "OUTPUTS", "INSPECT", "CLOSE_FIRST"):
        monkeypatch.setattr(desktop_state, name, dict(getattr(desktop_state, name)))
    path = tmp_path / "desk.py"
    path.write_text(textwrap.dedent('''
        from evals.desktop_state import saved_file
        from evals.live_tasks import LiveTask

        HELD_OUT = {"desktop": [LiveTask("secret_file", "files", "Save it", None, "Saved.")]}
        CHECKS = {"secret_file": saved_file("x.txt", lambda text, _: "x" in text, "an x")}
        OUTPUTS = {"secret_file": ("x.txt",)}
        CLOSE_FIRST = {"secret_file": ("notepad",)}
    '''), encoding="utf-8")
    s.load_private(path)
    assert desktop_state.OUTPUTS["secret_file"] == ("x.txt",) and desktop_state.CLOSE_FIRST["secret_file"] == ("notepad",)
    assert desktop_state.grade("secret_file", {"files": {"x.txt": "x"}})[0]


def test_a_path_is_hashed_the_same_on_every_checkout():
    # A grader's path constant (the desktop probe script) must not carry the checkout's location.
    from pathlib import PureWindowsPath

    from evals import desktop_state

    assert s._plain(desktop_state.PROBE_SCRIPT) == "evals/desktop_probe.ps1"
    assert s._plain(PureWindowsPath(r"C:\elsewhere\evals\desktop_probe.ps1")) == "desktop_probe.ps1"


def test_held_out_probe_settings_leave_working_desktop_tasks_alone(monkeypatch):
    # A held-out task adds its own probe settings; a working task's fingerprint must not
    # depend on whether the private file was loaded.
    from evals import desktop_state

    task = next(t for t in s.suite_tasks("desktop") if t.name == "notepad_save_file")
    base = s.task_fingerprint(task)
    for key in ("OUTPUTS", "INSPECT", "CLOSE_FIRST"):
        monkeypatch.setattr(desktop_state, key, {**getattr(desktop_state, key), "ho_extra": ("x",)})
    assert s.task_fingerprint(task) == base

"""The eval dashboard's data step: live_bench reports in, database rows out."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from evals.dashboard import (
    BLOB_SCRIPT,
    MODELS_JSON,
    add_change,
    attach_traces,
    check_alerts,
    comment_row,
    find_reports,
    import_reports,
    main,
    models_json,
    prepare,
    regressions,
    run_key,
    to_rows,
    trace_files,
)


def _report(**top) -> dict:
    report = {
        "suite": "full",
        "mode": "browser",
        "started_at": "20261024T012945Z",
        "browser_model": "server default",
        "judge_model": "gemini-3.5-flash",
        "summary": {"tasks": 2, "success": 1, "success_rate": 0.5, "mean_seconds": 30.0},
        "results": [
            {"task": "click_link", "outcome": "success", "final_screenshot": "/runner/step_1/after.png"},
            {"task": "go_back", "outcome": "failure", "final_screenshot": "-"},
        ],
    }
    report.update(top)
    return report


def test_prepare_reads_the_reports_own_backend_and_drops_runner_paths() -> None:
    report = prepare(
        _report(backend="claude_computer_use", claude_model="claude-opus-5-5", harness_profile="thin"),
        {"github_run_url": "https://github.com/o/r/actions/runs/1"},
    )
    assert report["config"] == {
        "backend": "claude_computer_use",
        "harness_profile": "thin",
        "model": "claude-opus-5-5",
        "github_run_url": "https://github.com/o/r/actions/runs/1",
    }
    assert all("final_screenshot" not in r for r in report["results"])
    assert run_key(report) == "browser-full-claude_computer_use-thin-20261024T012945Z"


def test_prepare_ignores_server_default_and_takes_overrides_for_old_reports() -> None:
    report = prepare(_report(backend="server default"), {"backend": "computer_use", "model": "gemini-3.5-flash", "flagged": None})
    assert report["config"] == {"backend": "computer_use", "model": "gemini-3.5-flash", "harness_profile": "full"}


def test_prepare_records_the_season_a_run_belongs_to_without_held_out_names() -> None:
    season = {
        "season": 1, "suite": "long", "season_hash": "abc", "operon_commit": "6ac35e2",
        "operon_settings": {"OPERON_HARNESS_PROFILE": "thin"}, "held_out_skipped": False,
        "held_out": {"run": 3, "passed": 2, "outcomes": {"success": 2, "missed": 1}, "cost_usd": 0.4},
    }
    config = prepare(_report(backend="computer_use", season=season), {})["config"]
    assert {k: config[k] for k in ("season", "season_suite", "season_hash", "operon_commit", "season_held_out")} == {
        "season": 1, "season_suite": "long", "season_hash": "abc", "operon_commit": "6ac35e2",
        "season_held_out": {"run": 3, "passed": 2},
    }
    assert "season_working_only" not in config and "operon_settings" not in config
    skipped = prepare(_report(backend="computer_use", season={**season, "held_out_skipped": True, "held_out": None}), {})
    assert skipped["config"]["season_working_only"] is True and "season_held_out" not in skipped["config"]
    assert "season" not in prepare(_report(backend="computer_use"), {})["config"]


def test_desktop_reports_without_a_mode_are_keyed_as_desktop() -> None:
    old = _report(suite="desktop-smoke", desktop_model="gemini-2.5-flash")
    del old["mode"]  # reports from before desktop mode existed
    report = prepare(old, {"backend": "json"})
    assert report["mode"] == "desktop"
    assert run_key(report).startswith("desktop-smoke-json-full-")


def test_hybrid_reports_keep_their_mode_and_drop_the_suite_prefix() -> None:
    report = prepare(_report(suite="hybrid-core", mode="hybrid", hybrid_model="gemini-3.5-flash",
                             backend="computer_use", harness_profile="full"), {})
    assert report["mode"] == "hybrid" and report["config"]["model"] == "gemini-3.5-flash"
    assert run_key(report).startswith("hybrid-core-computer_use-full-")
    no_mode = _report(suite="hybrid-smoke")
    del no_mode["mode"]
    assert prepare(no_mode, {"backend": "json"})["mode"] == "hybrid"


def _artifacts(tmp_path):
    artifacts = tmp_path / "artifacts" / "bench-claude_computer_use" / ".var" / "bench"
    artifacts.mkdir(parents=True)
    (artifacts / "full-20261024T012945Z.json").write_text(json.dumps(_report(backend="claude_computer_use")))
    (artifacts / "partial-20261024T020000Z.json").write_text(json.dumps(_report(started_at="20261024T020000Z")))
    (artifacts / "state.json").write_text(json.dumps({"status": "succeeded"}))
    return tmp_path / "artifacts"


def test_find_reports_includes_partial_runs_and_skips_other_json(tmp_path) -> None:
    artifacts = _artifacts(tmp_path)
    assert [p.name for p in find_reports([artifacts])] == ["full-20261024T012945Z.json", "partial-20261024T020000Z.json"]
    assert [p.name for p in find_reports([artifacts], include_partial=False)] == ["full-20261024T012945Z.json"]


def test_partial_incomplete_and_unpriced_runs_are_flagged_but_kept() -> None:
    assert prepare(_report(partial=True), {})["config"]["flagged"] == "partial run (--only / --category)"
    incomplete = prepare(_report(complete=False, tasks_planned=44), {})
    assert incomplete["config"]["flagged"] == "incomplete: the run ended after 2 of 44 tasks (crashed, timed out or cancelled)"
    assert "flagged" not in prepare(_report(complete=True), {})["config"]
    summary = {"tasks": 2, "success": 1, "unpriced_calls": 3, "unpriced_models": ["claude-new-model"]}
    assert prepare(_report(summary=summary), {})["config"]["flagged"] == (
        "3 model calls had no price (claude-new-model): their cost counts as $0"
    )


def _grounding_report() -> dict:
    return {
        "kind": "grounding", "suite": "grounding", "started_at": "20261027T050131Z",
        "backend": "claude_computer_use", "model": "claude-opus-5-5", "effort": "low", "budget_stop": None,
        "summary": {"samples": 300, "hit": 267, "miss": 24, "no_click": 9, "error": 0, "accuracy": 0.89,
                    "mean_seconds": 2.91, "cost_usd": 3.2, "cost_per_sample_usd": 0.010667,
                    "input_tokens": 1200, "output_tokens": 24731, "cache_read_tokens": 1144525, "cache_write_tokens": 494336},
        "samples": [{"id": "v2-desktop-1", "outcome": "hit", "cost_usd": 0.01}],
    }


def test_grounding_reports_are_imported_as_cost_only_runs(tmp_path) -> None:
    (tmp_path / "grounding-20261027T050131Z.json").write_text(json.dumps(_grounding_report()))
    assert [p.name for p in find_reports([tmp_path])] == ["grounding-20261027T050131Z.json"]
    report = prepare(_grounding_report(), {})
    run, tasks = to_rows(report)
    assert (run["mode"], run["suite"], run["backend"], run["model"]) == ("grounding", "grounding", "claude_computer_use", "claude-opus-5-5")
    assert run["key"] == "grounding-grounding-claude_computer_use-grounding-20261027T050131Z"
    assert run["summary"]["cost_usd"] == 3.2 and run["summary"]["judge_cost_usd"] == 0.0
    assert (run["summary"]["tasks"], run["summary"]["success"]) == (300, 267)
    assert tasks == []


def test_grounding_spend_is_keyed_by_the_model_and_carries_unpriced_calls() -> None:
    raw = {**_grounding_report(), "backend": "model", "model": "gemini-computer-use", "model_id": "gemini-3.5-flash"}
    raw["summary"] = {**raw["summary"], "unpriced_calls": 4, "unpriced_models": ["gemini-3.5-flash"]}
    report = prepare(raw, {})
    assert list(report["summary"]["by_model"]) == ["gemini-3.5-flash"]
    assert report["config"]["flagged"] == "4 model calls had no price (gemini-3.5-flash): their cost counts as $0"


def test_attach_traces_uploads_a_repeated_tasks_own_screen(tmp_path) -> None:
    screens = tmp_path / "screens"
    screens.mkdir()
    (screens / "click_link.png").write_bytes(b"first")
    (screens / "click_link-try2.png").write_bytes(b"second")
    report = prepare(_report(), {})
    first = {**report["results"][0], "attempt": 1}
    report["results"] = [first, {**first, "attempt": 2}]
    uploaded = []

    def uploader(files):
        uploaded.extend(files)
        return [f"https://blob.example/{pathname}" for _, pathname in files]

    assert attach_traces(report, screens, uploader, tmp_path) == 2
    assert [path.name for path, _ in uploaded] == ["click_link.png", "click_link-try2.png"]
    assert report["results"][1]["screenshot_url"].endswith("click_link-try2/final.png")


def test_every_attempt_of_a_repeated_run_gets_its_own_row() -> None:
    report = _report()
    report["results"] = [
        {"task": "click_link", "outcome": "success", "attempt": 1, "cost_usd": 0.1},
        {"task": "click_link", "outcome": "failure", "attempt": 2, "cost_usd": 0.3},
    ]
    _, tasks = to_rows(prepare(report, {}))
    assert [(t["task"], t["attempt"], t["outcome"], t["cost_usd"]) for t in tasks] == [
        ("click_link", 1, "success", 0.1), ("click_link", 2, "failure", 0.3),
    ]


def test_dashboard_models_json_matches_the_registry() -> None:
    # Regenerate with: python -m evals.dashboard models
    assert MODELS_JSON.read_text(encoding="utf-8") == models_json()


def test_to_rows_splits_a_report_into_a_run_and_its_tasks() -> None:
    report = _report(backend="claude_computer_use")
    report["results"][0].update(category="navigation", steps=2, seconds=31, cost_usd=0.04, input_tokens=900)
    report["results"].append({"task": "click_link", "outcome": "failure"})  # listed twice
    run, tasks = to_rows(prepare(report, {"flagged": "broken runner", "git_sha": "abc123"}))

    assert run["key"] == "browser-full-claude_computer_use-full-20261024T012945Z"
    assert run["started_at"] == datetime(2026, 10, 24, 1, 29, 45, tzinfo=timezone.utc)
    assert (run["backend"], run["flagged"], run["git_sha"]) == ("claude_computer_use", "broken runner", "abc123")
    assert run["summary"]["success"] == 1
    assert [(t["task"], t["outcome"], t["position"], t["attempt"]) for t in tasks] == [
        ("click_link", "success", 0, 1), ("go_back", "failure", 1, 1),
    ]
    assert (tasks[0]["cost_usd"], tasks[0]["input_tokens"], tasks[1]["cost_usd"]) == (0.04, 900, None)
    assert "final_screenshot" not in tasks[0]["result"]


def test_main_needs_a_database_url(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert main(["import", str(tmp_path)]) == 2
    assert "DATABASE_URL" in capsys.readouterr().err


# Runs against a real Postgres when one is given (CI has none):
# OPERON_TEST_DATABASE_URL=postgresql://... pytest tests/evals/test_dashboard.py
@pytest.mark.skipif(not os.getenv("OPERON_TEST_DATABASE_URL"), reason="no test database")
def test_import_writes_runs_and_replaces_them_on_reimport(tmp_path) -> None:
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(os.environ["OPERON_TEST_DATABASE_URL"]) as conn:
        conn.execute("DROP TABLE IF EXISTS bench_task_results, bench_runs")
        conn.commit()
        artifacts = _artifacts(tmp_path)
        # Only the full report: partial-run import is covered by the find_reports tests.
        assert import_reports([artifacts], conn, {"flagged": "broken runner"}, include_partial=False) == [
            "browser-full-claude_computer_use-full-20261024T012945Z"
        ]
        assert import_reports([artifacts], conn, {}, include_partial=False) == [
            "browser-full-claude_computer_use-full-20261024T012945Z"
        ]
        runs = conn.execute("SELECT key, flagged, config->>'git_sha', summary->'success' FROM bench_runs").fetchall()
        tasks = conn.execute("SELECT task, outcome, result ? 'final_screenshot' FROM bench_task_results ORDER BY position").fetchall()
    assert runs == [("browser-full-claude_computer_use-full-20261024T012945Z", None, None, 1)]
    assert tasks == [("click_link", "success", False), ("go_back", "failure", False)]


def test_a_run_stopped_at_the_cost_cap_is_flagged() -> None:
    stop = {"limit_usd": 5.0, "spent_usd": 5.1, "tasks_run": 12, "tasks_planned": 44}
    assert prepare(_report(budget_stop=stop), {})["config"]["flagged"] == "stopped at the $5 cost cap after 12 of 44 tasks"
    assert prepare(_report(budget_stop=stop), {"flagged": "broken runner"})["config"]["flagged"] == "broken runner"


def test_attach_traces_uploads_every_final_screen_and_the_failures_steps(tmp_path) -> None:
    screens = tmp_path / "full-20261024T012945Z"
    screens.mkdir()
    (screens / "click_link.png").write_bytes(b"png")
    (screens / "go_back.png").write_bytes(b"png")
    for step in (10, 2):
        (screens / "go_back" / f"step_{step}").mkdir(parents=True)
        (screens / "go_back" / f"step_{step}" / "before.png").write_bytes(b"png")
    (screens / "go_back" / "state.json").write_text("{}")
    report = prepare(_report(backend="claude_computer_use"), {})
    uploaded = []

    def uploader(files):
        uploaded.extend(files)
        return [f"https://blob.example/{pathname}" for _, pathname in files]

    work = tmp_path / "work"
    work.mkdir()
    assert attach_traces(report, screens, uploader, work) == 5

    key = run_key(report)
    assert [pathname for _, pathname in uploaded] == [
        f"bench/{key}/click_link/final.png",
        f"bench/{key}/go_back/final.png",
        f"bench/{key}/go_back/step_002.png",
        f"bench/{key}/go_back/step_010.png",
        f"bench/{key}/go_back/trace.zip",
    ]
    passed, failed = report["results"]
    assert passed["screenshot_url"].endswith("click_link/final.png") and "trace_url" not in passed
    assert [s["step"] for s in failed["step_screenshots"]] == [2, 10]
    assert failed["trace_url"].endswith("go_back/trace.zip")


@pytest.mark.parametrize("task", ["..", "../outside", "/etc", "a/../../outside", ".hidden", "x\\..\\y"])
def test_trace_files_refuse_task_names_that_leave_the_screens_folder(tmp_path, task) -> None:
    screens = tmp_path / "screens"
    screens.mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("token")
    (tmp_path / "outside.png").write_bytes(b"png")
    assert trace_files(screens, task) == (None, [], None)


def test_regressions_flag_a_pass_rate_drop_and_a_cost_rise() -> None:
    before = {"success_rate": 0.95, "success": 42, "tasks": 44, "cost_per_task_usd": 0.02}
    assert regressions({"success_rate": 0.93, "cost_per_task_usd": 0.024}, before) == []
    problems = regressions({"success_rate": 0.9, "success": 40, "tasks": 44, "cost_per_task_usd": 0.025}, before)
    assert len(problems) == 2
    assert problems[0].startswith("Pass rate fell 5 points") and problems[1].startswith("Cost per task rose 25%")
    assert regressions({"success_rate": 0.5}, None) == []
    assert regressions({"success_rate": 0.95, "cost_per_task_usd": 0.05}, {"success_rate": 0.95}) == []  # no old cost


class _OneRowConn:
    def __init__(self, row):
        self.row, self.queries = row, []

    def execute(self, sql, params=None):
        self.queries.append(params)
        return self

    def fetchone(self):
        return self.row


def test_check_alerts_compares_with_the_previous_comparable_run() -> None:
    previous = ("old-key", {"success_rate": 1.0, "success": 2, "tasks": 2, "cost_per_task_usd": 0.02}, None)
    conn, sent = _OneRowConn(previous), []
    report = prepare(_report(backend="claude_computer_use", harness_profile="thin"), {"github_run_url": "https://gh/run/1"})

    problems = check_alerts(conn, report, sent.append)

    assert problems == ["Pass rate fell 50 points: 50% (1/2), was 100% (2/2)."]
    assert conn.queries[0][:5] == ("browser", "full", "claude_computer_use", "thin", None)
    assert "Pass rate fell 50 points" in sent[0] and "<https://gh/run/1|Workflow run>" in sent[0]


def test_check_alerts_reports_a_budget_stop_without_comparing() -> None:
    stop = {"limit_usd": 5.0, "spent_usd": 5.1, "tasks_run": 12, "tasks_planned": 44}
    conn, sent = _OneRowConn(None), []
    assert check_alerts(conn, prepare(_report(budget_stop=stop), {}), sent.append) == [
        "Stopped at the $5 cost cap after 12 of 44 tasks (spent $5.10)."
    ]
    assert conn.queries == [] and len(sent) == 1


def test_no_alert_without_a_previous_run() -> None:
    sent = []
    assert check_alerts(_OneRowConn(None), prepare(_report(), {}), sent.append) == [] and sent == []


@pytest.mark.skipif(not os.getenv("OPERON_TEST_DATABASE_URL"), reason="no test database")
def test_import_uploads_traces_and_alerts_against_the_previous_run(tmp_path) -> None:
    psycopg = pytest.importorskip("psycopg")
    folder = tmp_path / "bench"
    folder.mkdir()
    good = _report(backend="claude_computer_use", started_at="20261023T000000Z")
    good["summary"] = {**good["summary"], "success": 2, "success_rate": 1.0}
    (folder / "full-20261023T000000Z.json").write_text(json.dumps(good))
    (folder / "full-20261024T012945Z.json").write_text(json.dumps(_report(backend="claude_computer_use")))
    (folder / "full-20261024T012945Z").mkdir()
    (folder / "full-20261024T012945Z" / "go_back.png").write_bytes(b"png")
    sent: list[str] = []

    def uploader(files):
        return [f"https://blob.example/{pathname}" for _, pathname in files]

    with psycopg.connect(os.environ["OPERON_TEST_DATABASE_URL"]) as conn:
        conn.execute("DROP TABLE IF EXISTS bench_task_results, bench_runs")
        conn.commit()
        import_reports([folder], conn, {}, uploader=uploader, notify=sent.append)
        url = conn.execute("SELECT result->>'screenshot_url' FROM bench_task_results WHERE task = 'go_back'").fetchall()

    assert ("https://blob.example/bench/browser-full-claude_computer_use-full-20261024T012945Z/go_back/final.png",) in url
    assert len(sent) == 1 and "Pass rate fell 50 points" in sent[0]


def test_blob_script_path_points_at_dashboard_scripts() -> None:
    assert BLOB_SCRIPT.is_file(), BLOB_SCRIPT


def test_a_registry_model_run_keeps_its_profile_and_model_id() -> None:
    report = _report(
        backend="model", harness_profile="thin", model_profile="qwen3-vl-32b",
        model_id="Qwen/Qwen3-VL-32B-Instruct", browser_model="gemini-3.5-flash",
    )
    run, _ = to_rows(prepare(report, {}))
    assert run["model_profile"] == "qwen3-vl-32b"
    assert run["model"] == "Qwen/Qwen3-VL-32B-Instruct"
    assert run["config"]["model_profile"] == "qwen3-vl-32b"
    # Two models run at the same moment never share a key.
    assert run["key"] == "browser-full-model-thin-qwen3-vl-32b-20261024T012945Z"
    assert "ADD COLUMN IF NOT EXISTS model_profile" in __import__("evals.dashboard").dashboard.SCHEMA


def test_prepare_keeps_a_full_runs_critic_for_the_page() -> None:
    report = prepare(
        _report(backend="model", model_profile="openai-computer-use", model_id="computer-use-preview",
                harness_profile="full", critic="same:computer-use-preview"),
        {},
    )
    assert report["config"]["critic_model"] == "same:computer-use-preview"
    assert "critic_model" not in prepare(_report(harness_profile="thin", critic=None), {})["config"]


def test_a_gate_on_run_is_flagged_so_it_is_never_a_baseline() -> None:
    config = prepare(_report(trust_gate="on"), {})["config"]
    assert config["trust_gate"] == "on"
    assert config["flagged"].startswith("trust gate on")
    off = prepare(_report(trust_gate="off"), {})["config"]
    assert off["trust_gate"] == "off" and "flagged" not in off
    assert "trust_gate" not in prepare(_report(), {})["config"]  # older reports: unrecorded, ran with it off


class _CellConn:
    """Answers the previous-cell query with ``previous`` and each run's task rows from ``tasks``."""

    def __init__(self, previous, tasks):
        self.previous, self.tasks, self.queries = previous, tasks, []
        self._rows = []

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        self._rows = self.tasks.get(params[0], []) if "bench_task_results" in sql else []
        self._one = self.previous if "bench_runs" in sql else None
        return self

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._rows


def _season_report(outcomes: dict[str, str], **top) -> dict:
    report = _report(
        backend="computer_use", harness_profile="full", desktop_model="gemini-x", mode="desktop", suite="desktop-core",
        season={"season": 1, "suite": "desktop-core"}, **top,
    )
    report["results"] = [{"task": t, "outcome": o} for t, o in outcomes.items()]
    report["summary"] = {"tasks": len(outcomes), "success": sum(o == "success" for o in outcomes.values())}
    return prepare(report, {"github_run_url": "https://gh/run/2"})


def test_add_change_compares_with_the_previous_run_of_the_same_cell() -> None:
    started = datetime(2026, 11, 9, tzinfo=timezone.utc)
    previous = ("old-key", {"success": 2, "tasks": 2}, "https://gh/run/1", started)
    conn = _CellConn(previous, {"old-key": [("open_app", "success"), ("save_file", "success")]})
    report = _season_report({"open_app": "success", "save_file": "failure"})

    add_change(conn, report)

    change = report["summary"]["change"]
    assert change["label"] == "No clear change" and change["broke"] == ["save_file"] and change["fixed"] == []
    assert change["previous"] == {"key": "old-key", "started_at": "20261109T000000Z", "success": 2, "tasks": 2,
                                  "github_run_url": "https://gh/run/1"}
    assert report["summary"]["score_range"]["score"] == 0.5
    # The cell: same mode, suite, backend, harness, model, registry profile and season.
    assert conn.queries[0][1][:7] == ("desktop", "desktop-core", "computer_use", "full", "gemini-x", None, "1")
    row = comment_row(report)
    assert row["cell"] == "gemini-x · Desktop · Full (desktop-core)" and row["run_url"] == "https://gh/run/2"


def test_add_change_marks_a_cells_first_run_and_skips_flagged_runs() -> None:
    first = _season_report({"open_app": "success"})
    add_change(_CellConn(None, {}), first)
    assert first["summary"]["change"] == {"label": "First run", "previous": None}

    stop = {"limit_usd": 3.0, "spent_usd": 3.1, "tasks_run": 1, "tasks_planned": 10}
    flagged = _season_report({"open_app": "success"}, budget_stop=stop)
    conn = _CellConn(None, {})
    add_change(conn, flagged)
    assert "change" not in flagged["summary"] and flagged["summary"]["score_range"]["score"] == 1.0
    assert conn.queries == []


@pytest.mark.skipif(not os.getenv("OPERON_TEST_DATABASE_URL"), reason="no test database")
def test_import_compares_each_run_with_the_cells_previous_run_and_fills_older_runs(tmp_path) -> None:
    psycopg = pytest.importorskip("psycopg")

    def report(stamp: str, outcomes: dict[str, str]) -> dict:
        data = _report(backend="computer_use", harness_profile="full", desktop_model="gemini-x", mode="desktop",
                       suite="desktop-core", season={"season": 1, "suite": "desktop-core"}, started_at=stamp)
        data["results"] = [{"task": t, "outcome": o} for t, o in outcomes.items()]
        data["summary"] = {"tasks": len(outcomes), "success": sum(o == "success" for o in outcomes.values())}
        return data

    passing = {f"t{i}": "success" for i in range(10)}
    first = tmp_path / "desktop-core-20261108T000000Z.json"
    first.write_text(json.dumps(report("20261108T000000Z", passing)))
    second = tmp_path / "desktop-core-20261109T000000Z.json"
    second.write_text(json.dumps(report("20261109T000000Z", {**passing, "t3": "failure"})))
    with psycopg.connect(os.environ["OPERON_TEST_DATABASE_URL"]) as conn:
        conn.execute("DROP TABLE IF EXISTS bench_task_results, bench_runs")
        conn.commit()
        rows: list[dict] = []
        import_reports([first, second], conn, {}, comment_rows=rows)
        changes = dict(conn.execute("SELECT key, summary->'change' FROM bench_runs").fetchall())
        # A run stored before KAN-60 (no change in its summary) gets one on the next import.
        conn.execute("UPDATE bench_runs SET summary = summary - 'change' - 'score_range' WHERE started_at > '2026-11-08T12:00:00Z'")
        conn.commit()
        import_reports([first], conn, {})
        filled = conn.execute(
            "SELECT summary->'change', summary->'score_range' FROM bench_runs WHERE started_at > '2026-11-08T12:00:00Z'"
        ).fetchone()
    first_key, second_key = sorted(changes)
    assert changes[first_key]["label"] == "First run"
    assert changes[second_key]["broke"] == ["t3"] and changes[second_key]["previous"]["key"] == first_key
    assert [r["summary"]["success"] for r in rows] == [10, 9]
    assert filled[0]["broke"] == ["t3"] and filled[1]["score"] == 0.9


@pytest.mark.skipif(not os.getenv("OPERON_TEST_DATABASE_URL"), reason="no test database")
def test_a_failing_backfill_keeps_the_imported_reports(tmp_path, monkeypatch) -> None:
    psycopg = pytest.importorskip("psycopg")
    import evals.dashboard as dashboard

    def broken(conn):
        conn.execute("SELECT no_such_column FROM bench_runs")

    monkeypatch.setattr(dashboard, "fill_changes", broken)
    path = tmp_path / "full-20261024T012945Z.json"
    path.write_text(json.dumps(_report()))
    with psycopg.connect(os.environ["OPERON_TEST_DATABASE_URL"]) as conn:
        conn.execute("DROP TABLE IF EXISTS bench_task_results, bench_runs")
        conn.commit()
        keys = import_reports([path], conn, {})
    with psycopg.connect(os.environ["OPERON_TEST_DATABASE_URL"]) as conn:
        stored = [row[0] for row in conn.execute("SELECT key FROM bench_runs").fetchall()]
    assert stored == keys and len(keys) == 1

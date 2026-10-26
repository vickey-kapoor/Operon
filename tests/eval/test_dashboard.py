"""The eval dashboard's data step: live_bench reports in, database rows out."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from tests.eval.dashboard import (
    attach_traces,
    check_alerts,
    find_reports,
    import_reports,
    main,
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


def test_desktop_reports_without_a_mode_are_keyed_as_desktop() -> None:
    old = _report(suite="desktop-smoke", desktop_model="gemini-2.5-flash")
    del old["mode"]  # reports from before desktop mode existed
    report = prepare(old, {"backend": "json"})
    assert report["mode"] == "desktop"
    assert run_key(report).startswith("desktop-smoke-json-full-")


def _artifacts(tmp_path):
    artifacts = tmp_path / "artifacts" / "bench-claude_computer_use" / ".var" / "bench"
    artifacts.mkdir(parents=True)
    (artifacts / "full-20261024T012945Z.json").write_text(json.dumps(_report(backend="claude_computer_use")))
    (artifacts / "partial-20261024T020000Z.json").write_text(json.dumps(_report(started_at="20261024T020000Z")))
    (artifacts / "state.json").write_text(json.dumps({"status": "succeeded"}))
    return tmp_path / "artifacts"


def test_find_reports_skips_partial_runs_and_other_json(tmp_path) -> None:
    assert [p.name for p in find_reports([_artifacts(tmp_path)])] == ["full-20261024T012945Z.json"]


def test_to_rows_splits_a_report_into_a_run_and_its_tasks() -> None:
    report = _report(backend="claude_computer_use")
    report["results"][0].update(category="navigation", steps=2, seconds=31, cost_usd=0.04, input_tokens=900)
    report["results"].append({"task": "click_link", "outcome": "failure"})  # listed twice
    run, tasks = to_rows(prepare(report, {"flagged": "broken runner", "git_sha": "abc123"}))

    assert run["key"] == "browser-full-claude_computer_use-full-20261024T012945Z"
    assert run["started_at"] == datetime(2026, 10, 24, 1, 29, 45, tzinfo=timezone.utc)
    assert (run["backend"], run["flagged"], run["git_sha"]) == ("claude_computer_use", "broken runner", "abc123")
    assert run["summary"]["success"] == 1
    assert [(t["task"], t["outcome"], t["position"]) for t in tasks] == [("click_link", "success", 0), ("go_back", "failure", 1)]
    assert (tasks[0]["cost_usd"], tasks[0]["input_tokens"], tasks[1]["cost_usd"]) == (0.04, 900, None)
    assert "final_screenshot" not in tasks[0]["result"]


def test_main_needs_a_database_url(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert main(["import", str(tmp_path)]) == 2
    assert "DATABASE_URL" in capsys.readouterr().err


# Runs against a real Postgres when one is given (CI has none):
# OPERON_TEST_DATABASE_URL=postgresql://... pytest tests/eval/test_dashboard.py
@pytest.mark.skipif(not os.getenv("OPERON_TEST_DATABASE_URL"), reason="no test database")
def test_import_writes_runs_and_replaces_them_on_reimport(tmp_path) -> None:
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(os.environ["OPERON_TEST_DATABASE_URL"]) as conn:
        conn.execute("DROP TABLE IF EXISTS bench_task_results, bench_runs")
        conn.commit()
        artifacts = _artifacts(tmp_path)
        assert import_reports([artifacts], conn, {"flagged": "broken runner"}) == [
            "browser-full-claude_computer_use-full-20261024T012945Z"
        ]
        assert import_reports([artifacts], conn, {}) == [
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

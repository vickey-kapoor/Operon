"""The eval dashboard's data step: live_bench reports in, database rows out."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from tests.eval.dashboard import (
    find_reports,
    import_reports,
    main,
    prepare,
    run_key,
    to_rows,
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

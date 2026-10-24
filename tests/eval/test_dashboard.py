"""The eval dashboard's data step: live_bench reports in, dashboard/data out."""

from __future__ import annotations

import json

from tests.eval.dashboard import build_index, import_reports, main, prepare, run_key


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


def test_import_finds_reports_in_artifact_folders_and_builds_the_index(tmp_path) -> None:
    artifacts = tmp_path / "artifacts" / "bench-claude_computer_use" / ".var" / "bench"
    artifacts.mkdir(parents=True)
    (artifacts / "full-20261024T012945Z.json").write_text(json.dumps(_report(backend="claude_computer_use")))
    (artifacts / "partial-20261024T020000Z.json").write_text(json.dumps(_report(started_at="20261024T020000Z")))
    (artifacts / "state.json").write_text(json.dumps({"status": "succeeded"}))
    data = tmp_path / "data"

    written = import_reports([tmp_path / "artifacts"], data, {"flagged": "broken runner"})
    index = build_index(data)

    assert [p.name for p in written] == ["browser-full-claude_computer_use-full-20261024T012945Z.json"]
    [run] = index["runs"]
    assert run["file"] == f"reports/{written[0].name}"
    assert run["summary"]["success"] == 1
    assert run["config"]["flagged"] == "broken runner"
    assert "results" not in run


def test_main_treats_empty_options_as_unset(tmp_path) -> None:
    report = tmp_path / "full-20261024T012945Z.json"
    report.write_text(json.dumps(_report(backend="computer_use")))
    assert main(["--data-dir", str(tmp_path / "data"), "import", str(report), "--git-sha", "", "--run-url", ""]) == 0
    stored = json.loads(next((tmp_path / "data" / "reports").glob("*.json")).read_text())
    assert "git_sha" not in stored["config"] and "github_run_url" not in stored["config"]

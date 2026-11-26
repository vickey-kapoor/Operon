"""KAN-228: the free CI jobs run on main every night, and one issue tracks a failing night. Fully offline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
CI: dict[str, Any] = yaml.safe_load(TEXT)
TRIGGERS: dict[str, Any] = CI.get("on") or CI[True]  # YAML 1.1 reads a bare `on` as True
JOBS: dict[str, Any] = CI["jobs"]
REPORT: dict[str, Any] = JOBS.get("nightly-report", {})


def test_schedule_runs_at_2_23_utc_and_can_be_started_by_hand() -> None:
    assert TRIGGERS["schedule"] == [{"cron": "23 2 * * *"}]
    assert "workflow_dispatch" in TRIGGERS


def test_report_job_needs_every_other_job_and_runs_only_on_schedule() -> None:
    others = set(JOBS) - {"nightly-report"}
    assert others and set(REPORT["needs"]) == others
    cond = REPORT["if"]
    assert "always()" in cond and "github.event_name == 'schedule'" in cond
    assert "pull_request" not in cond


def test_report_job_only_writes_issues() -> None:
    assert REPORT["permissions"] == {"issues": "write"}
    assert CI["permissions"] == {"contents": "read"}
    assert all("permissions" not in j for n, j in JOBS.items() if n != "nightly-report")


def test_job_names_are_unique_so_no_required_check_is_shadowed() -> None:
    names = [j.get("name", n) for n, j in JOBS.items()]
    assert len(set(names)) == len(names)
    assert REPORT["name"] == "Nightly report"


def test_no_secret_but_the_github_token() -> None:
    assert "secrets." not in TEXT
    assert "pull_request_target" not in TEXT
    assert REPORT["env"]["GH_TOKEN"] == "${{ github.token }}"


def test_report_script_has_no_template_expressions() -> None:
    for step in REPORT["steps"]:
        assert "${{" not in step.get("run", "")


def test_report_script_opens_comments_and_closes() -> None:
    run = "\n".join(s.get("run", "") for s in REPORT["steps"])
    assert "Nightly tests failed on main" in run and "nightly-failure" in run
    for verb in ("create", "comment", "close"):
        assert re.search(rf"gh issue {verb}", run), verb


def test_nightly_run_is_not_cancelled_by_a_push() -> None:
    assert "schedule" in CI["concurrency"]["group"]


def test_manual_run_has_its_own_group() -> None:
    """A manual run must not cancel the scheduled nightly, and the nightly must not cancel it."""
    group = CI["concurrency"]["group"]
    assert "'nightly-dispatch'" in group and "'nightly'" in group
    assert group.index("'nightly'") < group.index("workflow_dispatch")


def test_issue_close_is_one_loud_call() -> None:
    run = "\n".join(s.get("run", "") for s in REPORT["steps"])
    closes = [line for line in run.splitlines() if "gh issue close" in line]
    assert len(closes) == 1
    assert '--comment "$(cat comment.md)"' in closes[0]
    assert "comment-file" not in closes[0] and "||" not in closes[0] and "2>/dev/null" not in closes[0]

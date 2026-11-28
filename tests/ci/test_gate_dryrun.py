"""KAN-282: the $0 Approve gate check (gate-dryrun.yml) and the dry_run mode of the two paid workflows.

Offline: the workflow files are parsed as YAML; nothing calls GitHub.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
MODEL_KEYS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
TEST_JOBS = ("direct", "by-name", "not-passed", "no-env", "main-only")
REAL_JOBS = ("real-grounding", "real-osworld", "real-no-env")


def _wf(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _env(job: dict) -> str | None:
    env = job.get("environment")
    return env.get("name") if isinstance(env, dict) else env


def _triggers(wf: dict) -> dict:
    return wf.get("on", wf.get(True))


# ---- gate-dryrun.yml ----


def test_triggers_are_the_label_and_a_manual_start() -> None:
    wf = _wf("gate-dryrun.yml")
    assert set(_triggers(wf)) == {"pull_request", "workflow_dispatch"}
    assert _triggers(wf)["pull_request"] == {"types": ["labeled"]}
    assert _triggers(wf)["workflow_dispatch"]["inputs"]["target"]["options"] == ["test", "real"]
    assert wf["permissions"] == {"contents": "read"}


def test_test_mode_jobs_use_the_throwaway_environments_and_refuse_forks() -> None:
    jobs = _wf("gate-dryrun.yml")["jobs"]
    assert _env(jobs["direct"]) == "dryrun-gate"
    assert _env(jobs["main-only"]) == "dryrun-main-only"
    assert "environment" not in jobs["no-env"]
    for name in ("by-name", "not-passed"):
        assert jobs[name]["uses"] == "./.github/workflows/gate-dryrun-called.yml"
        assert jobs[name]["with"]["environment"] == "dryrun-gate"
    assert jobs["by-name"]["with"]["expect"] == "present"
    assert jobs["by-name"]["secrets"] == {"DRYRUN_KEY": "${{ secrets.DRYRUN_KEY }}"}
    assert jobs["not-passed"]["with"]["expect"] == "empty"
    assert "secrets" not in jobs["not-passed"]  # passing nothing is what it proves
    for name in TEST_JOBS:
        cond = jobs[name]["if"]
        assert "github.event.label.name == 'gate-dryrun'" in cond
        assert "github.event.pull_request.head.repo.full_name == github.repository" in cond
    # On main the branch rule lets main-only through, so it runs only from a branch. If it ever starts it
    # SUCCEEDS, so "refused" (failure, no steps) can be told from "ran" in the report.
    assert "github.ref != 'refs/heads/main'" in jobs["main-only"]["if"]
    steps = jobs["main-only"]["steps"]
    assert "exit" not in json.dumps(steps)
    assert "did NOT refuse" in steps[0]["run"]


def test_test_mode_jobs_never_read_a_model_key() -> None:
    jobs = _wf("gate-dryrun.yml")["jobs"]
    for name in TEST_JOBS + ("report",):
        assert not any(key in json.dumps(jobs[name]) for key in MODEL_KEYS), name


def _report_script(job: str) -> str:
    return _wf("gate-dryrun.yml")["jobs"][job]["steps"][0]["run"]


def test_the_report_waits_for_every_test_job_and_checks_each_result() -> None:
    report = _wf("gate-dryrun.yml")["jobs"]["report"]
    assert set(report["needs"]) == set(TEST_JOBS)
    assert "always()" in report["if"]
    script = _report_script("report")
    assert 'for r in "$DIRECT" "$BY_NAME" "$NOT_PASSED" "$NO_ENV"' in script
    assert 'x = "success"' not in script and "exit 1" in script
    # main-only: skipped on main; from a branch it must be a failure (refused) with no completed steps.
    assert '[ "$MAIN_ONLY" = "skipped" ]' in script
    assert '[ "$MAIN_ONLY" = "failure" ]' in script
    assert '.steps[]? | select(.status == "completed")' in script and '[ "$done_steps" = "0" ]' in script


@pytest.mark.parametrize(("job", "environment"), [("report", "dryrun-gate"), ("real-report", "paid-runs")])
def test_a_report_proves_an_approve_happened_and_lists_the_jobs(job: str, environment: str) -> None:
    wf_job = _wf("gate-dryrun.yml")["jobs"][job]
    assert wf_job["permissions"] == {"actions": "read"}
    assert wf_job["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert wf_job["env"]["APPROVAL_ENV"] == environment
    script = wf_job["steps"][0]["run"]
    assert 'actions/runs/$RUN_ID/approvals' in script and 'actions/runs/$RUN_ID/jobs' in script
    assert 'select(.state == "approved")' in script and "index($e)" in script  # an approval of that environment
    assert "Approved by:" in script and ".user.login" in script and r"(\(.state))" in script and ".updated_at" not in script
    assert "No approval of the $APPROVAL_ENV environment is recorded" in script  # else the check fails
    assert ".started_at" in script and ".completed_at" in script  # the job table
    assert "exit 1" in script


def test_the_called_workflow_asserts_by_what_the_caller_expects() -> None:
    wf = _wf("gate-dryrun-called.yml")
    assert set(_triggers(wf)) == {"workflow_call"}
    assert set(_triggers(wf)["workflow_call"]["inputs"]) == {"environment", "expect"}
    assert _triggers(wf)["workflow_call"]["secrets"] == {"DRYRUN_KEY": {"required": False}}
    job = wf["jobs"]["check"]
    assert _env(job) == "${{ inputs.environment }}"
    step = job["steps"][0]
    assert step["env"]["DRYRUN_KEY"] == "${{ secrets.DRYRUN_KEY }}"
    script = step["run"]
    assert "present)" in script and '-z "$DRYRUN_KEY"' in script  # present: empty is the failure
    assert "empty)" in script and '-n "$DRYRUN_KEY"' in script  # empty: a value is the failure
    assert "${DRYRUN_KEY}" not in script and 'echo "$DRYRUN_KEY"' not in script  # never prints the value


def test_real_mode_runs_only_on_main_and_passes_keys_by_name() -> None:
    jobs = _wf("gate-dryrun.yml")["jobs"]
    for name in REAL_JOBS:
        assert "github.ref == 'refs/heads/main'" in jobs[name]["if"]
        assert "inputs.target == 'real'" in jobs[name]["if"]
    # A real start from another ref fails loudly instead of doing nothing.
    off_main = jobs["real-needs-main"]
    assert "inputs.target == 'real'" in off_main["if"] and "github.ref != 'refs/heads/main'" in off_main["if"]
    assert "real mode runs on main only" in off_main["steps"][0]["run"] and "exit 1" in off_main["steps"][0]["run"]
    grounding, osworld = jobs["real-grounding"], jobs["real-osworld"]
    assert grounding["uses"] == "./.github/workflows/grounding.yml"
    assert grounding["with"]["dry_run"] == "true"
    assert grounding["secrets"] == {"GEMINI_API_KEY": "${{ secrets.GEMINI_API_KEY }}"}
    assert osworld["uses"] == "./.github/workflows/osworld-bench.yml"
    assert osworld["with"]["dry_run"] == "true" and osworld["with"]["limit"] == "1"
    assert set(osworld["secrets"]) == set(MODEL_KEYS)  # the runner token is only used by a step a dry run skips
    assert all(v == f"${{{{ secrets.{k} }}}}" for k, v in osworld["secrets"].items())
    batch = _wf("benchmark.yml")["jobs"]["batch-osworld"]
    assert osworld["permissions"] == batch["permissions"]  # what the called workflow's jobs ask for
    # Only the model-key check lives outside the environment: all three keys, failing when any is non-empty,
    # printing only the names.
    no_env = jobs["real-no-env"]
    assert "environment" not in no_env and "uses" not in no_env
    assert len(no_env["steps"]) == 1
    assert no_env["steps"][0]["env"] == {key: f"${{{{ secrets.{key} }}}}" for key in MODEL_KEYS}
    run = no_env["steps"][0]["run"]
    assert all(f'[ -n "${key}" ]' in run for key in MODEL_KEYS)
    assert 'echo "$' not in run and "echo $" not in run  # never echoes a value
    assert "exit 1" in run
    assert set(jobs["real-report"]["needs"]) == set(REAL_JOBS)


def test_no_gate_dryrun_job_checks_out_code_or_calls_a_model() -> None:
    for name in ("gate-dryrun.yml", "gate-dryrun-called.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "actions/checkout" not in text and "evals." not in text and "uv run" not in text


# ---- dry_run in the two paid workflows ----


def test_grounding_dry_run_stops_after_the_key_check() -> None:
    wf = _wf("grounding.yml")
    assert _triggers(wf)["workflow_call"]["inputs"]["dry_run"] == {"type": "string", "default": "false"}
    assert _triggers(wf)["workflow_dispatch"]["inputs"]["dry_run"]["default"] == "false"
    jobs = wf["jobs"]
    # The real job's steps carry no dry-run conditions; the whole job is skipped (CodeQL reads step-level ifs on a
    # checkout as weak protection).
    assert jobs["grounding"]["if"] == "${{ inputs.dry_run != 'true' }}"
    assert not any("dry_run" in json.dumps(step) for step in jobs["grounding"]["steps"])
    # The dry run is its own job: the paid-runs gate, the key check, and nothing that costs money.
    dry = jobs["dry-run"]
    assert dry["if"] == "${{ inputs.dry_run == 'true' }}" and _env(dry) == "paid-runs"
    assert dry["env"] == {"GEMINI_API_KEY": "${{ secrets.GEMINI_API_KEY }}"}
    assert len(dry["steps"]) == 1
    step = dry["steps"][0]
    assert "uses" not in step  # no checkout, no install
    assert '-z "$GEMINI_API_KEY"' in step["run"] and "exit 1" in step["run"] and "GITHUB_STEP_SUMMARY" in step["run"]
    assert "evals." not in step["run"] and "uv " not in step["run"]


def test_osworld_dry_run_never_signs_in_or_starts_the_vm() -> None:
    wf = _wf("osworld-bench.yml")
    assert _triggers(wf)["workflow_call"]["inputs"]["dry_run"] == {"type": "string", "default": "false"}
    assert _triggers(wf)["workflow_dispatch"]["inputs"]["dry_run"]["default"] == "false"
    jobs = wf["jobs"]
    steps = jobs["start-vm"]["steps"]
    assert steps[0]["name"] == "Check the model key" and "if" not in steps[0]  # runs in a dry run too
    for step in steps[1:-1]:
        assert "inputs.dry_run != 'true'" in step["if"], step.get("name") or step.get("uses")
    assert steps[-1]["if"] == "inputs.dry_run == 'true'"
    # Whatever signs in to Google, starts the VM or mints a runner is guarded.
    for step in steps:
        text = json.dumps(step)
        if "google-github-actions/auth" in text or "instances start" in text or "jit-runner.sh" in text:
            assert "inputs.dry_run != 'true'" in step["if"]
    for job in ("bench", "stop-vm"):
        assert "inputs.dry_run != 'true'" in jobs[job]["if"], job
    # publish-videos only follows a bench that ran.
    assert "needs.bench.result == 'success'" in jobs["publish-videos"]["if"]


@pytest.mark.parametrize("file", ["grounding.yml", "osworld-bench.yml"])
def test_the_dry_run_summary_never_prints_a_key(file: str) -> None:
    text = (WORKFLOWS / file).read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if "Dry run: " in ln and "GITHUB_STEP_SUMMARY" in ln)
    assert "$GEMINI" not in line and "secrets." not in line


# ---- the report script itself, run against a fake `gh` ----

MAIN_ONLY_JOB = "Main-only environment (must be refused from a branch)"
GOOD_APPROVALS = [{"state": "approved", "user": {"login": "vickey"}, "environments": [{"name": "dryrun-gate"}]}]


def _job(name: str, steps: int = 0) -> dict:
    return {
        "name": name,
        "status": "completed",
        "conclusion": "failure",
        "started_at": "t0",
        "completed_at": "t1",
        "steps": [{"name": f"s{i}", "status": "completed"} for i in range(steps)],
    }


def _run_test_report(
    tmp_path: Path,
    *,
    approvals: list,
    jobs: list,
    ref: str = "refs/pull/1/merge",
    main_only: str = "failure",
    direct: str = "success",
) -> subprocess.CompletedProcess[str]:
    if sys.platform == "win32" or not shutil.which("bash") or not shutil.which("jq"):
        pytest.skip("needs bash and jq")
    (tmp_path / "approvals.json").write_text(json.dumps(approvals), encoding="utf-8")
    (tmp_path / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/sh\ncase "$2" in *approvals*) cat "$FIXTURES/approvals.json";; *jobs*) cat "$FIXTURES/jobs.json";; esac\n',
        encoding="utf-8",
    )
    gh.chmod(0o755)
    wf_env = _wf("gate-dryrun.yml")["jobs"]["report"]["env"]
    env = {
        "PATH": f"{tmp_path}:/usr/bin:/bin:/usr/local/bin",
        "FIXTURES": str(tmp_path),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "GH_TOKEN": "x",
        "REPO": "o/r",
        "RUN_ID": "1",
        "REF": ref,
        "APPROVAL_ENV": wf_env["APPROVAL_ENV"],
        "MAIN_ONLY_JOB": wf_env["MAIN_ONLY_JOB"],
        "DIRECT": direct,
        "BY_NAME": "success",
        "NOT_PASSED": "success",
        "NO_ENV": "success",
        "MAIN_ONLY": main_only,
    }
    return subprocess.run(  # noqa: S603
        ["bash", "-eo", "pipefail", "-c", _report_script("report")],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_report_passes_when_the_gate_held(tmp_path: Path) -> None:
    result = _run_test_report(tmp_path, approvals=GOOD_APPROVALS, jobs=[_job(MAIN_ONLY_JOB), _job("Direct")])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Approved by: vickey (approved)" in result.stdout
    assert "| Direct | failure | t0 | t1 |" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_the_report_passes_on_main_when_main_only_is_skipped(tmp_path: Path) -> None:
    result = _run_test_report(
        tmp_path, approvals=GOOD_APPROVALS, jobs=[_job("Direct")], ref="refs/heads/main", main_only="skipped"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_report_fails_when_main_only_was_reviewed_even_if_rejected(tmp_path: Path) -> None:
    # A Reject of main-only also ends as failure with no steps; only the review entry tells it from a refusal.
    rejected = {"state": "rejected", "user": {"login": "vickey"}, "environments": [{"name": "dryrun-main-only"}]}
    result = _run_test_report(tmp_path, approvals=[*GOOD_APPROVALS, rejected], jobs=[_job(MAIN_ONLY_JOB)])
    assert result.returncode != 0
    assert "branch rule is missing" in result.stdout


@pytest.mark.parametrize("count", [0, 2])
def test_the_report_fails_unless_exactly_one_main_only_job_is_found(tmp_path: Path, count: int) -> None:
    result = _run_test_report(tmp_path, approvals=GOOD_APPROVALS, jobs=[_job(MAIN_ONLY_JOB)] * count)
    assert result.returncode != 0
    assert "exactly one job" in result.stdout


def test_the_report_fails_when_main_only_ran_steps(tmp_path: Path) -> None:
    result = _run_test_report(tmp_path, approvals=GOOD_APPROVALS, jobs=[_job(MAIN_ONLY_JOB, steps=2)])
    assert result.returncode != 0


def test_the_report_fails_without_an_approval_of_the_environment(tmp_path: Path) -> None:
    other = [{"state": "approved", "user": {"login": "vickey"}, "environments": [{"name": "paid-runs"}]}]
    result = _run_test_report(tmp_path, approvals=other, jobs=[_job(MAIN_ONLY_JOB)])
    assert result.returncode != 0
    assert "nothing waited for Approve" in result.stdout


def test_the_report_fails_when_a_gated_job_did_not_succeed(tmp_path: Path) -> None:
    result = _run_test_report(tmp_path, approvals=GOOD_APPROVALS, jobs=[_job(MAIN_ONLY_JOB)], direct="failure")
    assert result.returncode != 0

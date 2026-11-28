"""KAN-282: the $0 Approve gate check (gate-dryrun.yml) and the dry_run mode of the two paid workflows.

Offline: the workflow files are parsed as YAML; nothing calls GitHub.
"""

from __future__ import annotations

import json
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
    # On main the branch rule lets main-only through, so it runs only from a branch, and it fails if it ever runs.
    assert "github.ref != 'refs/heads/main'" in jobs["main-only"]["if"]
    assert "exit 1" in json.dumps(jobs["main-only"]["steps"])


def test_test_mode_jobs_never_read_a_model_key() -> None:
    jobs = _wf("gate-dryrun.yml")["jobs"]
    for name in TEST_JOBS + ("report",):
        assert not any(key in json.dumps(jobs[name]) for key in MODEL_KEYS), name


def test_the_report_waits_for_every_test_job_and_checks_each_result() -> None:
    report = _wf("gate-dryrun.yml")["jobs"]["report"]
    assert set(report["needs"]) == set(TEST_JOBS)
    assert "always()" in report["if"]
    script = report["steps"][0]["run"]
    assert 'for r in "$DIRECT" "$BY_NAME" "$NOT_PASSED" "$NO_ENV"' in script
    assert '[ "$MAIN_ONLY" != "success" ]' in script  # refused from a branch means not success
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
    grounding, osworld = jobs["real-grounding"], jobs["real-osworld"]
    assert grounding["uses"] == "./.github/workflows/grounding.yml"
    assert grounding["with"]["dry_run"] == "true"
    assert grounding["secrets"] == {"GEMINI_API_KEY": "${{ secrets.GEMINI_API_KEY }}"}
    assert osworld["uses"] == "./.github/workflows/osworld-bench.yml"
    assert osworld["with"]["dry_run"] == "true" and osworld["with"]["limit"] == "1"
    assert set(osworld["secrets"]) == {*MODEL_KEYS, "RUNNER_ADMIN_TOKEN"}
    assert all(v == f"${{{{ secrets.{k} }}}}" for k, v in osworld["secrets"].items())
    batch = _wf("benchmark.yml")["jobs"]["batch-osworld"]
    assert osworld["permissions"] == batch["permissions"]  # what the called workflow's jobs ask for
    # Only the model-key check lives outside the environment, and it only fails when the key is non-empty.
    no_env = jobs["real-no-env"]
    assert "environment" not in no_env and "uses" not in no_env
    assert [s["env"] for s in no_env["steps"]] == [{"GEMINI_API_KEY": "${{ secrets.GEMINI_API_KEY }}"}]
    assert '[ -n "$GEMINI_API_KEY" ]' in no_env["steps"][0]["run"]
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
    steps = wf["jobs"]["grounding"]["steps"]
    assert steps[0]["name"] == "Check the secrets" and "if" not in steps[0]  # runs in a dry run too
    for step in steps[1:-1]:
        assert "inputs.dry_run != 'true'" in step["if"], step.get("name") or step.get("uses")
    assert steps[-1]["if"] == "inputs.dry_run == 'true'"
    assert "GITHUB_STEP_SUMMARY" in steps[-1]["run"]


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

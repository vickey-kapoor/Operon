"""KAN-282: every workflow job that can spend model credits waits for Approve on the `paid-runs` environment.

Offline: the workflow files are parsed as YAML; nothing calls GitHub.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
KEYS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
_NAMES = "|".join(KEYS)
# A read of a model key: secrets.NAME, secrets['NAME'], secrets["NAME"], or the whole secrets context.
KEY_REF = re.compile(
    r"secrets\.(?:" + _NAMES + r")\b"
    r"|secrets\[\s*\\?['\"](?:" + _NAMES + r")\\?['\"]\s*\]"
    r"|toJSON\(\s*secrets\s*\)",
    re.IGNORECASE,
)
# No caller can start these two jobs, and another story deletes them, so they are exempt until then.
EXEMPT_JOBS = {("live-bench.yml", "bench"), ("live-desktop-bench.yml", "bench")}


def _workflows() -> dict[str, dict]:
    paths = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    return {p.name: yaml.safe_load(p.read_text(encoding="utf-8")) for p in paths}


def _environment_name(job: dict) -> str | None:
    env = job.get("environment")
    return env.get("name") if isinstance(env, dict) else env


def _uses_a_model_key(job: dict) -> bool:
    # The whole job mapping, so env, step env, `with`, `run`, `if` and anything else is covered.
    return bool(KEY_REF.search(json.dumps(job, default=str))) or job.get("secrets") == "inherit"


def _jobs_without_the_environment(name: str, wf: dict) -> list[str]:
    # A key read at workflow level (env: ...) reaches every job, so then every job needs the environment.
    top_level = {k: v for k, v in wf.items() if k != "jobs"}
    workflow_wide = _uses_a_model_key(top_level)
    return [
        f"{name}:{job_id}"
        for job_id, job in wf["jobs"].items()
        if (name, job_id) not in EXEMPT_JOBS
        and (workflow_wide or _uses_a_model_key(job))
        and _environment_name(job) != "paid-runs"
    ]


def test_every_job_that_reads_a_model_key_uses_the_paid_runs_environment() -> None:
    missing = [m for name, wf in _workflows().items() for m in _jobs_without_the_environment(name, wf)]
    assert not missing, f"jobs that read a model key without environment: paid-runs: {missing}"


def test_a_workflow_level_key_read_needs_the_environment_on_every_job() -> None:
    wf = {
        "env": {"K": "${{ secrets.GEMINI_API_KEY }}"},
        "jobs": {"a": {"environment": "paid-runs"}, "b": {"steps": []}, "c": {"environment": {"name": "other"}}},
    }
    assert _jobs_without_the_environment("x.yml", wf) == ["x.yml:b", "x.yml:c"]
    assert _jobs_without_the_environment("x.yml", {"jobs": {"b": {"steps": []}}}) == []


def test_the_exempt_live_benches_can_only_be_called_and_nobody_calls_them() -> None:
    workflows = _workflows()
    for file in {f for f, _ in EXEMPT_JOBS}:
        triggers = workflows[file].get("on", workflows[file].get(True))  # YAML reads a bare `on` as True
        assert set(triggers) == {"workflow_call"}, f"{file} can be started on its own"
    callers = [
        f"{name}:{job_id}"
        for name, wf in workflows.items()
        for job_id, job in wf["jobs"].items()
        if any(str(job.get("uses", "")).endswith(f) for f, _ in EXEMPT_JOBS)
    ]
    assert not callers, f"workflows call an exempt live bench: {callers}"


def test_the_key_pattern_catches_every_spelling() -> None:
    for text in (
        "${{ secrets.GEMINI_API_KEY }}",
        "${{ secrets['OPENAI_API_KEY'] }}",
        '${{ secrets["ANTHROPIC_API_KEY"] }}',
        "${{ toJSON(secrets) }}",
    ):
        assert _uses_a_model_key({"env": {"X": text}}), text
    assert _uses_a_model_key({"uses": "./x.yml", "secrets": "inherit"})
    assert not _uses_a_model_key({"env": {"X": "${{ secrets.RUNNER_ADMIN_TOKEN }}"}})


def test_no_caller_passes_a_model_key_or_inherits() -> None:
    bad = []
    for name, wf in _workflows().items():
        for job_id, job in wf["jobs"].items():
            if "uses" not in job:
                continue
            secrets = job.get("secrets")
            if secrets == "inherit" or (isinstance(secrets, dict) and set(secrets) & set(KEYS)):
                bad.append(f"{name}:{job_id}")
    assert not bad, f"callers that pass model keys (they come from the environment): {bad}"


@pytest.mark.parametrize(
    ("file", "job"),
    [("osworld-bench.yml", "start-vm"), ("osworld-bench.yml", "bench"), ("grounding.yml", "grounding")],
)
def test_the_paid_jobs_wait_for_approval(file: str, job: str) -> None:
    assert _environment_name(_workflows()[file]["jobs"][job]) == "paid-runs"

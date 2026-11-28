"""KAN-282: every workflow job that can spend model credits waits for Approve on the `paid-runs` environment.

Offline: the workflow files are parsed as YAML; nothing calls GitHub.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
KEYS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
KEY_REF = re.compile(r"secrets\.(?:" + "|".join(KEYS) + r")\b")
# No caller can start these two, and another story deletes them, so they are left out until then.
UNGUARDED_FILES = {"live-bench.yml", "live-desktop-bench.yml"}


def _workflows() -> dict[str, dict]:
    return {p.name: yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(WORKFLOWS.glob("*.yml"))}


def _environment_name(job: dict) -> str | None:
    env = job.get("environment")
    return env.get("name") if isinstance(env, dict) else env


def _uses_a_model_key(job: dict) -> bool:
    texts = [str(v) for v in (job.get("env") or {}).values()]
    for step in job.get("steps") or []:
        texts += [str(v) for v in (step.get("env") or {}).values()]
        texts += [str(v) for v in (step.get("with") or {}).values()]
        texts.append(str(step.get("run", "")))
    return any(KEY_REF.search(t) for t in texts)


def test_every_job_that_reads_a_model_key_uses_the_paid_runs_environment() -> None:
    missing = [
        f"{name}:{job_id}"
        for name, wf in _workflows().items()
        if name not in UNGUARDED_FILES
        for job_id, job in wf["jobs"].items()
        if _uses_a_model_key(job) and _environment_name(job) != "paid-runs"
    ]
    assert not missing, f"jobs that read a model key without environment: paid-runs: {missing}"


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

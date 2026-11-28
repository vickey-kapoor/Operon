"""PR evidence plumbing (KAN-213): the PR template, the e2e evidence upload and the verifier's Proof check.

Offline: only the workflow, template and agent files are parsed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
VAR = "OPERON_E2E_ARTIFACTS_DIR"
E2E_JOBS = ["desktop", "browser-e2e"]


def _ci_jobs() -> dict[str, Any]:
    return yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))["jobs"]


def test_pr_template_has_the_four_headings_in_order() -> None:
    """KAN-213: PRs without a Proof section slip through; the template asks for it."""
    path = ROOT / ".github" / "pull_request_template.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("## What and why")
    headings = re.findall(r"^## .+$", text, flags=re.MULTILINE)
    assert headings == ["## What and why", "## Proof", "## Real check", "## Risk?"]


@pytest.mark.parametrize("job", E2E_JOBS)
def test_e2e_job_uploads_its_evidence(job: str) -> None:
    """KAN-213: the e2e jobs upload screenshots and run files every run, pass or fail."""
    steps = _ci_jobs()[job]["steps"]
    env_idx = next(i for i, s in enumerate(steps) if VAR in (s.get("env") or {}))
    value = steps[env_idx]["env"][VAR]
    uploads = [
        s
        for s in steps[env_idx + 1 :]
        if re.fullmatch(r"actions/upload-artifact@[0-9a-f]{40}", s.get("uses", ""))
    ]
    assert uploads, f"{job}: no upload-artifact step after the pytest step"
    up = uploads[0]
    assert "always()" in str(up.get("if"))
    assert up["with"]["retention-days"] <= 7
    assert up["with"]["path"] == value + "/"


def test_no_other_job_sets_the_artifacts_dir() -> None:
    """KAN-213: only the two e2e jobs collect evidence; no other job or workflow sets it."""
    for wf in sorted(WORKFLOWS.glob("*.y*ml")):
        data = yaml.safe_load(wf.read_text(encoding="utf-8"))
        for name, job in (data.get("jobs") or {}).items():
            blobs = [job.get("env") or {}] + [s.get("env") or {} for s in job.get("steps") or []]
            sets = any(VAR in b for b in blobs)
            expected = wf.name == "ci.yml" and name in E2E_JOBS
            assert sets == expected, f"{wf.name}:{name}"


def test_verifier_checks_the_proof() -> None:
    """KAN-213: the verifier rejects a behavior change that has no Proof."""
    text = (ROOT / ".claude" / "agents" / "verifier.md").read_text(encoding="utf-8")
    assert "Proof" in text

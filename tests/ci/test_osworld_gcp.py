"""The OSWorld runner on Google Cloud (KAN-224): the workflow, the CI check of infra/gcp and the VM's startup script.

Offline: the workflow files, the Terraform files and the shell scripts are read as text; nothing calls Google Cloud.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
INFRA = ROOT / "infra" / "gcp"
GCP_AUTH = "google-github-actions/auth@7c6bc770dae815cd3e89ee6cdf493a5fab2cc093"


def _osworld() -> dict:
    return yaml.safe_load((WORKFLOWS / "osworld-bench.yml").read_text(encoding="utf-8"))


def _uses(job: dict) -> list[str]:
    return [step["uses"] for step in job["steps"] if "uses" in step]


def _runs(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_the_osworld_workflow_no_longer_uses_azure() -> None:
    text = (WORKFLOWS / "osworld-bench.yml").read_text(encoding="utf-8")
    assert "azure/login" not in text
    assert "AZURE_" not in text
    assert "operon-linux" not in text
    for job in _osworld()["jobs"].values():
        assert not re.search(r"(^|\s)az\s", _runs(job)), job.get("name")


def test_start_vm_signs_in_to_google_cloud_and_starts_the_vm_before_the_runner() -> None:
    job = _osworld()["jobs"]["start-vm"]
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    assert GCP_AUTH in _uses(job)
    auth = next(step for step in job["steps"] if step.get("uses", "").startswith("google-github-actions/auth@"))
    assert auth["with"] == {
        "workload_identity_provider": "${{ vars.GCP_WORKLOAD_IDENTITY_PROVIDER }}",
        "service_account": "${{ vars.GCP_SERVICE_ACCOUNT }}",
    }
    runs = [step.get("run", "") for step in job["steps"]]
    clear = next(i for i, run in enumerate(runs) if "remove-metadata" in run)
    start = next(i for i, run in enumerate(runs) if "gcloud compute instances start" in run)
    runner = next(i for i, run in enumerate(runs) if "jit-runner.sh start linux" in run)
    assert clear < start < runner  # a stale runner script is gone before the VM boots
    runner_env = job["steps"][runner]["env"]
    assert runner_env["VM_CLOUD"] == "gcp"
    assert int(runner_env["RUNNER_ONLINE_TIMEOUT"]) >= 600  # a first boot installs Docker and the runner
    assert job["env"]["VM_NAME"] == "${{ vars.OSWORLD_VM_NAME || 'operon-osworld' }}"
    assert {"GCP_ZONE", "GCP_PROJECT"} <= set(job["env"])
    # The admin token is on the runner step only.
    assert "secrets." not in str(job["env"])
    assert all("secrets." not in str(step.get("with", "")) for step in job["steps"])


def test_stop_vm_always_stops_the_vm_and_clears_the_runner_script() -> None:
    job = _osworld()["jobs"]["stop-vm"]
    assert job["if"] == "${{ always() && needs.start-vm.result != 'skipped' && inputs.dry_run != 'true' }}"
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    assert _uses(job)[0] == GCP_AUTH  # sign in first, before anything that could fail
    runs = [step.get("run", "") for step in job["steps"]]
    stop = next(i for i, run in enumerate(runs) if "gcloud compute instances stop" in run)
    clear = next(i for i, run in enumerate(runs) if "remove-metadata" in run)
    checkout = next(i for i, step in enumerate(job["steps"]) if step.get("uses", "").startswith("actions/checkout@"))
    assert stop < clear < checkout  # stopping comes before anything that is only cleanup
    assert "suspend" not in "\n".join(runs)
    # The memory log is shown only for a failed bench, never fails the job, and only shows tagged lines.
    memlog = next(step for step in job["steps"] if "get-serial-port-output" in step.get("run", ""))
    assert memlog["if"] == "${{ needs.bench.result != 'success' }}"
    assert memlog["continue-on-error"] is True
    assert "grep -o 'operon-memlog: .*'" in memlog["run"]


def test_the_workflow_never_echoes_metadata_or_uses_gcloud_debug_flags() -> None:
    text = (WORKFLOWS / "osworld-bench.yml").read_text(encoding="utf-8")
    assert "--log-http" not in text and "--verbosity" not in text
    assert "operon-jit-script=" not in text  # only jit-runner.sh puts the script on, from a file
    assert "setup-gcloud" not in text  # gcloud is already on ubuntu-latest


def test_ci_validates_the_terraform_with_pinned_versions() -> None:
    job = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))["jobs"]["terraform"]
    checkout = job["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@") and checkout["with"]["persist-credentials"] is False
    setup = job["steps"][1]
    assert setup["uses"] == "hashicorp/setup-terraform@dfe3c3f87815947d99a8997f908cb6525fc44e9e"
    assert setup["with"]["terraform_version"] == "1.17.2"
    runs = _runs(job)
    assert "terraform fmt -check -recursive infra" in runs
    for root in ("infra/gcp/bootstrap", "infra/gcp"):
        assert f"terraform -chdir={root} init -backend=false" in runs
        assert f"terraform -chdir={root} validate" in runs
    assert "apply" not in runs and "plan" not in runs
    assert "secrets." not in str(job)


@pytest.mark.parametrize("root", [INFRA, INFRA / "bootstrap"])
def test_terraform_pins_the_provider_and_commits_its_lock_file(root: Path) -> None:
    tf = "\n".join(path.read_text(encoding="utf-8") for path in sorted(root.glob("*.tf")))
    assert 'version = "= 8.12.0"' in tf
    lock = (root / ".terraform.lock.hcl").read_text(encoding="utf-8")
    assert 'version     = "8.12.0"' in lock


def test_the_vm_has_no_identity_takes_nothing_in_and_ignores_the_runner_script() -> None:
    main = (INFRA / "main.tf").read_text(encoding="utf-8")
    assert "service_account" not in main.replace("# No service_account block", "")
    assert 'desired_status = "TERMINATED"' in main
    assert 'ignore_changes = [desired_status, metadata["operon-jit-script"]]' in main
    assert 'block-project-ssh-keys = "TRUE"' in main and 'enable-oslogin         = "TRUE"' in main
    assert "enable_nested_virtualization = true" in main
    assert "max_run_duration" in main and 'instance_termination_action = "STOP"' in main
    assert 'network_tier = "STANDARD"' in main
    # The only firewall rule denies everything coming in.
    assert main.count('resource "google_compute_firewall"') == 1
    assert "allow {" not in main
    assert 'replace(file("${path.module}/startup.sh"), "\\r\\n", "\\n")' in main


def test_only_the_osworld_workflow_on_main_of_this_repo_can_sign_in() -> None:
    iam = (INFRA / "iam.tf").read_text(encoding="utf-8")
    variables = (INFRA / "variables.tf").read_text(encoding="utf-8")
    assert 'assertion.repository_id == \\"${var.github_repository_id}\\" && assertion.ref == \\"refs/heads/main\\"' in iam
    assert '"1177308224"' in variables
    assert '"vickey-kapoor/Operon/.github/workflows/osworld-bench.yml@refs/heads/main"' in variables
    assert "attribute.workflow_ref/${var.github_workflow_ref}" in iam
    # A called run (benchmark.yml) maps job_workflow_ref; a run started directly falls back to workflow_ref.
    assert '"attribute.workflow_ref" = "\\"job_workflow_ref\\" in assertion ? assertion.job_workflow_ref : assertion.workflow_ref"' in iam
    # The CI identity gets the VM operator role on the instance only, never a project-wide compute role.
    assert 'resource "google_compute_instance_iam_member" "osworld_ci_operator"' in iam
    assert "roles/compute." not in iam and "roles/owner" not in iam and "roles/editor" not in iam
    operator = re.search(r'"osworld_operator" \{.*?permissions = \[(.*?)\]', iam, re.S)
    assert operator is not None
    assert sorted(re.findall(r'"([a-zA-Z.]+)"', operator.group(1))) == [
        "compute.instances.get",
        "compute.instances.getSerialPortOutput",
        "compute.instances.setMetadata",
        "compute.instances.start",
        "compute.instances.stop",
    ]
    # An existing pool and provider are imported only when asked, never by default.
    assert 'default     = false' in variables.split('variable "import_existing_wif"')[1]


def test_the_startup_script_checks_the_runner_and_never_prints_the_runner_script() -> None:
    script = (INFRA / "startup.sh").read_text(encoding="utf-8")
    assert "\r" not in script
    assert "RUNNER_VERSION=2.338.2\n" in script
    assert "RUNNER_SHA256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40\n" in script
    assert 'sha256sum -c' in script
    # No lasting registration on the VM.
    assert "config.sh" not in script and "svc.sh" not in script
    # Memory log: command names only, never process arguments.
    assert re.findall(r"ps -eo \S+", script) == ["ps -eo rss,comm"]
    # The watcher keeps the script in a private file and never cats or echoes it.
    assert "umask 077" in script
    assert 'cat "$STATE/body"' not in script and 'cat "$script"' not in script
    # The units that act on their own are started at the end of the script and never enabled.
    for unit in ("operon-jit-watch.service", "operon-idle-off.timer"):
        assert f"systemctl restart {unit}" in script
        assert f"enable {unit}" not in script and f"enable --now {unit}" not in script
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "infra/**/*.sh text eol=lf" in attributes


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs a Linux bash")
def test_the_startup_script_and_its_helpers_parse() -> None:
    script = (INFRA / "startup.sh").read_text(encoding="utf-8")
    docs = re.findall(r"<<'EOF'\n(.*?)\nEOF\n", script, re.S)
    helpers = [doc for doc in docs if doc.startswith("#!/bin/bash")]
    assert len(helpers) == 3  # memlog, jit-watch, idle-off
    for code in (script, *helpers):
        result = subprocess.run(["bash", "-n"], input=code, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr

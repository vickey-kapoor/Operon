"""One-job runners for the Azure benchmark VMs and the fork PR workflow check (KAN-195).

Offline: the workflow files are parsed, and jit-runner.sh runs against stub `curl` and `az` on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
JIT = ROOT / "scripts" / "bench-runners" / "jit-runner.sh"

CONFIG = "ZmFrZS1qaXQtY29uZmln+/0="  # base64, as GitHub returns it
TOKEN = "ghp_faketoken123"
LABEL = "operon-osworld-123-1"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_no_job_uses_a_static_self_hosted_label() -> None:
    """A bench job asks for its run's own label, never the lasting `operon-*` labels or `self-hosted`."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job_id, job in (_load(path.name).get("jobs") or {}).items():
            runs_on = json.dumps(job.get("runs-on", ""))
            where = f"{path.name}:{job_id}"
            assert "self-hosted" not in runs_on, where
            if "operon" in runs_on:
                assert "runner_label" in runs_on, where


def test_fork_pr_check_never_runs_pr_code() -> None:
    workflow = _load("fork-pr-guard.yml")
    text = (WORKFLOWS / "fork-pr-guard.yml").read_text(encoding="utf-8")
    triggers = workflow.get("on", workflow.get(True))  # YAML reads a bare `on` as True
    assert list(triggers) == ["pull_request_target"]
    assert workflow["permissions"] == {}
    assert "actions/checkout" not in text
    for job in workflow["jobs"].values():
        assert "head.repo.full_name != github.repository" in job["if"]
        for step in job["steps"]:
            # PR-controlled values reach the script only through env, never inside the script text.
            assert "${{" not in step.get("run", "")


needs_bash = pytest.mark.skipif(
    # Windows: "bash" can resolve to WSL's launcher with no distro installed; these scripts only run on
    # GitHub's Linux runners anyway.
    os.name == "nt" or shutil.which("bash") is None or shutil.which("jq") is None,
    reason="needs a Linux bash and jq"
)


@pytest.fixture
def stubs(tmp_path: Path) -> dict:
    """PATH with a fake curl (GitHub API) and az; every call is logged."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    script_copy = tmp_path / "vm-script"
    runners = {
        "runners": [
            {"id": 1, "name": "old-lasting", "labels": [{"name": "self-hosted"}, {"name": "operon-osworld"}]},
            {"id": 2, "name": "other", "labels": [{"name": "operon-osworld-9-1"}]},
            {"id": 3, "name": LABEL, "status": "online", "labels": [{"name": LABEL}]},
        ]
    }
    (tmp_path / "runners.json").write_text(json.dumps(runners))
    curl = f"""#!/bin/bash
echo "curl $*" >> "{calls}"
method=GET; url=""; body=""
while [ $# -gt 0 ]; do
  case "$1" in
    -X) method="$2"; shift;;
    -d) body="$2"; shift;;
    -H) case "$2" in @*) cat "${{2#@}}" >> "{calls}.auth";; esac; shift;;
    https://*) url="$1";;
  esac
  shift
done
echo "$method $url $body" >> "{calls}.api"
case "$method $url" in
  "POST "*generate-jitconfig) echo '{{"runner": {{"id": 3}}, "encoded_jit_config": "'"${{FAKE_CONFIG}}"'"}}';;
  "GET "*actions/runners*) cat "{tmp_path}/runners.json";;
esac
"""
    az = f"""#!/bin/bash
echo "az $*" >> "{calls}"
while [ $# -gt 0 ]; do
  if [ "$1" = "--scripts" ]; then cp "${{2#@}}" "{script_copy}"; fi
  shift
done
echo "${{FAKE_AZ_OUTPUT-operon-jit-runner-started {LABEL}}}"
"""
    for name, body in (("curl", curl), ("az", az)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return {"bin": bin_dir, "calls": calls, "script": script_copy}


def _run(stubs: dict, *args: str, **env: str) -> subprocess.CompletedProcess:
    full_env = {
        "PATH": f"{stubs['bin']}:{os.environ['PATH']}",
        "GH_ADMIN_TOKEN": TOKEN,
        "GH_REPO": "owner/repo",
        "RESOURCE_GROUP": "rg",
        "VM_NAME": "vm",
        "FAKE_CONFIG": CONFIG,
        **env,
    }
    return subprocess.run(["bash", str(JIT), *args], env=full_env, capture_output=True, text=True, check=False)


def _api_calls(stubs: dict) -> list[str]:
    path = Path(f"{stubs['calls']}.api")
    return path.read_text().splitlines() if path.exists() else []


def _deleted_ids(stubs: dict) -> list[str]:
    return [call.split()[1].rsplit("/", 1)[1] for call in _api_calls(stubs) if call.startswith("DELETE")]


@needs_bash
def test_start_linux_mints_retires_the_old_runner_and_delivers_the_config(stubs: dict) -> None:
    result = _run(stubs, "start", "linux", LABEL, "operon-osworld")
    assert result.returncode == 0, result.stderr
    assert TOKEN not in result.stdout + result.stderr
    assert f"::add-mask::{CONFIG}" in result.stdout
    assert "online" in result.stdout  # waited for the runner to come online

    mint = next(call for call in _api_calls(stubs) if "generate-jitconfig" in call)
    body = json.loads(mint.split(" ", 2)[2])
    assert body == {"name": LABEL, "runner_group_id": 1, "labels": [LABEL], "work_folder": "_work"}
    # Only the runner with the old static label is deleted.
    assert _deleted_ids(stubs) == ["1"]

    script = stubs["script"].read_text()
    assert script.startswith(f"JIT_CONFIG='{CONFIG}'\nRUNNER_LABEL='{LABEL}'\n")
    assert "systemd-run" in script  # the vm-linux.sh template follows the two lines
    calls = Path(stubs["calls"]).read_text()
    assert "RunShellScript" in calls
    assert CONFIG not in calls  # the config travels in a file, not in a command line
    # The token reaches curl through a header file, never in its arguments.
    assert TOKEN not in calls
    assert TOKEN in Path(f"{stubs['calls']}.auth").read_text()


@needs_bash
def test_start_windows_uses_the_powershell_template(stubs: dict) -> None:
    label = LABEL  # the stub GitHub knows this runner (online)
    result = _run(stubs, "start", "windows", label, "operon-desktop")
    assert result.returncode == 0, result.stderr
    script = stubs["script"].read_text()
    assert script.startswith(f"$JitConfig = '{CONFIG}'\n$RunnerLabel = '{label}'\n")
    assert ".jitconfig" in script
    assert "RunPowerShellScript" in Path(stubs["calls"]).read_text()


@needs_bash
@pytest.mark.parametrize("bad", ["abc def", "abc'def", "", "abc;rm -rf /", "null"])
def test_start_rejects_a_config_that_is_not_plain_base64(stubs: dict, bad: str) -> None:
    result = _run(stubs, "start", "linux", LABEL, "operon-osworld", FAKE_CONFIG=bad)
    assert result.returncode != 0
    assert not stubs["script"].exists()  # nothing reached the VM
    assert _deleted_ids(stubs) == []
    assert TOKEN not in result.stdout + result.stderr


@needs_bash
def test_start_rejects_a_bad_label_and_a_vm_that_did_not_start_the_runner(stubs: dict) -> None:
    assert _run(stubs, "start", "linux", "Bad Label", "operon-osworld").returncode != 0
    failed = _run(stubs, "start", "linux", LABEL, "operon-osworld", FAKE_AZ_OUTPUT="boom")
    assert failed.returncode != 0
    assert "did not start the runner" in failed.stderr


@needs_bash
def test_start_fails_when_the_runner_never_comes_online(stubs: dict) -> None:
    runners = json.loads((stubs["calls"].parent / "runners.json").read_text())
    runners["runners"][2]["status"] = "offline"
    (stubs["calls"].parent / "runners.json").write_text(json.dumps(runners))
    result = _run(stubs, "start", "linux", LABEL, "operon-osworld", RUNNER_ONLINE_TIMEOUT="2", RUNNER_POLL_INTERVAL="1")
    assert result.returncode != 0
    assert "did not come online" in result.stderr


@needs_bash
def test_remove_deletes_the_runner_with_that_name_only(stubs: dict) -> None:
    result = _run(stubs, "remove", LABEL)
    assert result.returncode == 0, result.stderr
    assert _deleted_ids(stubs) == ["3"]
    assert TOKEN not in result.stdout + result.stderr


def test_vm_templates_hand_over_the_config_only_through_a_locked_file() -> None:
    linux = (ROOT / "scripts" / "bench-runners" / "vm-linux.sh").read_text()
    windows = (ROOT / "scripts" / "bench-runners" / "vm-windows.ps1").read_text()
    assert "chmod 600" in linux and "--jitconfig" in linux
    assert "icacls" in windows and "/inheritance:r" in windows
    # On Windows the logon task starts the runner: it reads and deletes that file, and nothing registers a lasting runner.
    setup = (ROOT / "scripts" / "azure-desktop-runner" / "setup-runner.ps1").read_text()
    assert "--jitconfig" in setup
    assert "config.cmd\", \"--unattended" not in setup


def test_workflows_run_the_script_through_bash() -> None:
    """The file is not guaranteed to be executable in every checkout: every call goes through bash."""
    calls = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
            if "bench-runners/jit-runner.sh" in code and not line.lstrip().startswith("#"):
                calls += 1
                assert "bash scripts/bench-runners/jit-runner.sh" in line, f"{path.name}: {line.strip()}"
    assert calls == 4
    assert os.access(JIT, os.X_OK)


def test_fork_pr_check_handles_the_files_api_limit() -> None:
    workflow = _load("fork-pr-guard.yml")
    step = next(iter(workflow["jobs"].values()))
    assert step["env"]["CHANGED"] == "${{ github.event.pull_request.changed_files }}"
    script = step["steps"][0]["run"]
    assert '-ge 3000' in script  # too many files to list: fail
    assert '"$listed" -ne "$CHANGED"' in script  # a list that is not the PR's own count: fail


def test_runner_labels_get_a_random_suffix_per_run() -> None:
    for name in ("osworld-bench.yml", "live-desktop-bench.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "openssl rand -hex 6" in text, name
        assert "RUNNER_LABEL: ${{ steps.label.outputs.label }}" in text, name

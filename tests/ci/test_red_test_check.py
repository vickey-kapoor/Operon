"""Red test check (KAN-280): the PR's changed tests must fail on the base code. Offline, tiny temp git repos."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("red_test_check", ROOT / "scripts" / "red_test_check.py")
assert _spec and _spec.loader
rtc = importlib.util.module_from_spec(_spec)
sys.modules["red_test_check"] = rtc
_spec.loader.exec_module(rtc)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST_ADD = "from toy.mod import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n"


def _git(repo: Path, *args: str) -> str:
    cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args]
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _write(repo: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-b", "main")
    _write(
        r,
        {
            "pyproject.toml": '[tool.pytest.ini_options]\ntestpaths=["tests"]\n',
            "src/toy/__init__.py": "",
            "src/toy/mod.py": BUGGY,
            "tests/test_toy.py": "def test_ok():\n    assert True\n",
        },
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-m", "base")
    return r


def _pr(repo: Path, files: dict[str, str]) -> str:
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "pr")
    _write(repo, files)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "pr")
    return base


def _run(repo: Path, base: str, *extra: str) -> int:
    return int(rtc.main(["--base", base, "--repo", str(repo), *extra]))


def _worktrees(repo: Path) -> int:
    return len(_git(repo, "worktree", "list").splitlines())


@needs_git
def test_fix_pr_is_red_on_base(repo: Path) -> None:
    base = _pr(repo, {"src/toy/mod.py": FIXED, "tests/test_add.py": TEST_ADD})
    assert _run(repo, base) == 0
    assert _worktrees(repo) == 1


@needs_git
def test_weak_test_pr_fails(repo: Path) -> None:
    base = _pr(repo, {"src/toy/mod.py": FIXED, "tests/test_add.py": "def test_trivial():\n    assert 1 + 1 == 2\n"})
    assert _run(repo, base) == 1
    assert _worktrees(repo) == 1


@needs_git
def test_new_function_pr_counts_collection_error_as_red(repo: Path) -> None:
    base = _pr(
        repo,
        {
            "src/toy/mod.py": BUGGY + "\n\ndef mul(a, b):\n    return a * b\n",
            "tests/test_mul.py": "from toy.mod import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n",
        },
    )
    assert _run(repo, base) == 0


@needs_git
@pytest.mark.parametrize(
    "files",
    [
        {"docs/notes.md": "x\n", "README.md": "y\n"},
        {"tests/test_new.py": "def test_x():\n    assert True\n"},
        {"src/toy/mod.py": FIXED},
    ],
    ids=["docs-only", "tests-only", "src-only"],
)
def test_skips_when_not_both_tests_and_code(repo: Path, files: dict[str, str], capsys: pytest.CaptureFixture[str]) -> None:
    base = _pr(repo, files)
    assert _run(repo, base) == 0
    assert "skipped" in capsys.readouterr().err
    assert _worktrees(repo) == 1


@needs_git
def test_opt_out_skips_without_running_pytest(repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    base = _pr(repo, {"src/toy/mod.py": FIXED, "tests/test_add.py": "def test_trivial():\n    assert True\n"})

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("pytest must not run")

    monkeypatch.setattr(rtc, "run_on_base", boom)
    assert _run(repo, base, "--opt-out") == 0
    assert "label no-red-test" in capsys.readouterr().err


def test_workflow_safety_rails() -> None:
    raw = (ROOT / ".github" / "workflows" / "red-test.yml").read_text(encoding="utf-8")
    wf = yaml.safe_load(raw)
    assert "secrets." not in raw
    assert wf["permissions"] == {"contents": "read"}
    on = wf.get("on", wf.get(True))
    assert "labeled" not in on["pull_request"]["types"]
    for job in wf["jobs"].values():
        for step in job["steps"]:
            uses = step.get("uses")
            if uses:
                sha = uses.split("@")[1].split()[0]
                assert len(sha) == 40, uses
                int(sha, 16)
            if uses and uses.startswith("actions/checkout"):
                assert step["with"]["persist-credentials"] is False
            if "run" in step:
                assert "${{" not in step["run"], step["run"]

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
def test_deleting_code_file_with_weak_test_fails(repo: Path) -> None:
    _write(repo, {"src/toy/unsafe.py": "def danger():\n    return 1\n"})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add unsafe")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "pr")
    (repo / "src/toy/unsafe.py").unlink()
    _write(repo, {"tests/test_toy.py": "def test_ok():\n    assert True\n\n\ndef test_gone():\n    assert True\n"})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "pr")
    assert _run(repo, base) == 1
    assert _worktrees(repo) == 1


@needs_git
def test_deleted_test_file_is_not_copied_or_run(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "pr")
    (repo / "tests/test_toy.py").unlink()
    _write(repo, {"scripts/tool.py": "x = 1\n"})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "pr")
    assert _run(repo, base) == 0  # skipped: the only changed test was deleted
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
        {"scripts/tool.py": "x = 1\n"},
        {".github/workflows/w.yml": "name: w\n"},
        {"uv.lock": "x\n", ".gitignore": "y\n", "data/cfg.json": "{}\n", "tests/test_new.py": "def test_x():\n    assert True\n"},
    ],
    ids=["docs-only", "tests-only", "scripts-only", "github-only", "non-code-and-test"],
)
def test_skips_when_not_both_tests_and_code(repo: Path, files: dict[str, str], capsys: pytest.CaptureFixture[str]) -> None:
    base = _pr(repo, files)
    assert _run(repo, base) == 0
    assert "skipped" in capsys.readouterr().err
    assert _worktrees(repo) == 1


@needs_git
def test_src_change_without_test_fails(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _pr(repo, {"src/toy/mod.py": FIXED})
    assert _run(repo, base) == 1
    assert "changes src/ but no test" in capsys.readouterr().err
    assert _run(repo, base, "--opt-out") == 0
    assert _worktrees(repo) == 1


@needs_git
def test_src_delete_without_test_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "pr")
    (repo / "src" / "toy" / "mod.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "pr")
    assert _run(repo, base) == 1


@needs_git
def test_opt_out_skips_without_running_pytest(repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    base = _pr(repo, {"src/toy/mod.py": FIXED, "tests/test_add.py": "def test_trivial():\n    assert True\n"})

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("pytest must not run")

    monkeypatch.setattr(rtc, "run_on_base", boom)
    assert _run(repo, base, "--opt-out") == 0
    assert "label no-red-test" in capsys.readouterr().err


@needs_git
def test_head_code_on_pythonpath_does_not_leak_into_base_run(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Mimics an editable install: the head checkout's src/ is importable. The base run must still use base code.
    base = _pr(repo, {"src/toy/mod.py": FIXED, "tests/test_add.py": TEST_ADD})
    monkeypatch.setenv("PYTHONPATH", str(repo / "src"))
    assert _run(repo, base) == 0


@needs_git
def test_reports_red_and_passing_modules(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _pr(
        repo,
        {
            "src/toy/mod.py": FIXED,
            "tests/test_add.py": TEST_ADD,
            "tests/sub/test_weak.py": "def test_trivial():\n    assert True\n",
        },
    )
    assert _run(repo, base) == 0
    err = capsys.readouterr().err
    assert "at least one changed test fails without the fix" in err
    assert "red on base (fail without the fix): tests/test_add.py" in err
    assert "passed on base (prove nothing alone): tests/sub/test_weak.py" in err


@needs_git
def test_all_skipped_module_is_not_run(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    skipped = "import pytest\n\n\n@pytest.mark.skip(reason='x')\ndef test_skipped():\n    assert False\n"
    base = _pr(repo, {"src/toy/mod.py": FIXED, "tests/test_skip.py": skipped})
    assert _run(repo, base) == 0
    err = capsys.readouterr().err
    assert "not run" in err
    assert "passed on base" not in err


@needs_git
def test_conftest_importing_new_symbol_is_red(repo: Path) -> None:
    base = _pr(
        repo,
        {
            "src/toy/mod.py": BUGGY + "\n\ndef mul(a, b):\n    return a * b\n",
            "tests/conftest.py": "from toy.mod import mul  # noqa: F401\n",
            "tests/test_mul.py": "def test_ok():\n    assert True\n",
        },
    )
    assert _run(repo, base) == 0


@needs_git
def test_other_usage_error_still_fails(repo: Path) -> None:
    files = {"src/toy/mod.py": FIXED, "tests/conftest.py": "raise SystemExit(3)\n", "tests/test_x.py": "def test_x():\n    assert True\n"}
    assert _run(repo, _pr(repo, files)) == 1


def test_split_files_rules() -> None:
    files = [
        "tests/test_a.py",
        "tests/sub/test_b.py",
        "tests/conftest.py",
        "src/x.py",
        "uv.lock",
        "docs/a.json",
        "scripts/s.py",
        ".github/w.yml",
        "pkg/tests/test_c.py",
    ]
    tests, modules, code = rtc.split_files(files)
    assert tests == ["tests/test_a.py", "tests/sub/test_b.py", "tests/conftest.py"]
    assert modules == ["tests/test_a.py", "tests/sub/test_b.py"]
    assert code == ["src/x.py", "scripts/s.py", ".github/w.yml"]


def test_workflow_uses_merge_parent_as_base() -> None:
    raw = (ROOT / ".github" / "workflows" / "red-test.yml").read_text(encoding="utf-8")
    assert "git rev-parse HEAD^1" in raw
    assert "base.sha" not in raw


def test_workflow_runs_base_copy_of_script() -> None:
    raw = (ROOT / ".github" / "workflows" / "red-test.yml").read_text(encoding="utf-8")
    assert 'git show "$BASE_SHA:scripts/red_test_check.py"' in raw
    assert 'git cat-file -e "$BASE_SHA:scripts/red_test_check.py"' in raw
    assert "first rollout" in raw
    assert '--base "$BASE_SHA"' in raw


def test_workflow_reads_labels_before_installing() -> None:
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "red-test.yml").read_text(encoding="utf-8"))
    steps = wf["jobs"]["red-test"]["steps"]
    label = next(i for i, st in enumerate(steps) if st.get("id") == "labels")
    sync = next(i for i, st in enumerate(steps) if "uv sync" in st.get("run", ""))
    assert label < sync


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


@needs_git
def test_timeout_on_base_counts_as_red(repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    hang = (
        "import time\n"
        "from toy.mod import add\n\n\n"
        "def test_hang():\n"
        "    if add(2, 2) != 4:\n"
        "        time.sleep(60)\n"
    )
    base = _pr(repo, {"src/toy/mod.py": FIXED, "tests/test_hang.py": hang})
    monkeypatch.setattr(rtc, "TIMEOUT_S", 3)
    assert _run(repo, base) == 0
    assert "timed out on base (counted as red)" in capsys.readouterr().err
    assert _worktrees(repo) == 1


def test_timeout_default_is_five_minutes() -> None:
    assert rtc.TIMEOUT_S == 5 * 60

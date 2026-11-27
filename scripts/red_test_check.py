"""Red test check (KAN-280): a PR's new tests must fail on the base branch's code.

Every fix PR adds a test, but nothing proved the test would have caught the bug. This runs the PR's
changed tests against the base commit's ``src/`` and expects them to fail there. Stdlib only, $0.

    python scripts/red_test_check.py --base SHA [--head REF] [--repo PATH] [--opt-out]

Exit 0: the tests fail on the base (good), or the check is skipped (no changed test, no code change,
nothing collected, or ``--opt-out`` for a PR labelled ``no-red-test``). Exit 1: the tests pass on the
base too, or pytest failed in some other way. The reasons go to stderr and, when set, to
``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_S = 15 * 60
TAIL_LINES = 40


def _report(verdict: str, reasons: list[str], title: str = "Red test") -> None:
    text = "\n".join(f"- {r}" for r in reasons)
    print(f"{title}: {verdict}\n{text}", file=sys.stderr)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### {title}: {verdict}\n\n{text}\n\n")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout


def changed_files(repo: Path, base: str, head: str) -> list[str]:
    out = _git(repo, "diff", "--name-only", "--diff-filter=AMR", f"{base}...{head}").decode("utf-8", "replace")
    return [line.strip() for line in out.splitlines() if line.strip()]


def split_files(files: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (changed files under tests/, changed test_*.py files, code files)."""
    tests = [f for f in files if f.startswith("tests/")]
    test_modules = [f for f in tests if fnmatch.fnmatch(f, "tests/**/test_*.py") or fnmatch.fnmatch(f, "tests/test_*.py")]
    code = [f for f in files if not f.startswith("tests/") and not f.startswith("docs/") and not f.endswith(".md")]
    return tests, test_modules, code


def _tail(text: str) -> str:
    return "\n".join(text.splitlines()[-TAIL_LINES:])


def run_on_base(repo: Path, base: str, head: str, tests: list[str], test_modules: list[str]) -> tuple[int | None, str]:
    """Run the changed tests on the base code. Returns (exit code or None on timeout, output)."""
    with tempfile.TemporaryDirectory(prefix="red-test-") as tmp:
        wt = Path(tmp) / "wt"
        _git(repo, "worktree", "add", "--detach", str(wt), base)
        try:
            for rel in tests:
                dest = wt / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_git(repo, "show", f"{head}:{rel}"))
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(p for p in (str(wt / "src"), env.get("PYTHONPATH", "")) if p)
            cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_modules]
            try:
                proc = subprocess.run(cmd, cwd=wt, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired as exc:
                out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
                return None, out
            return proc.returncode, proc.stdout + proc.stderr
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo, capture_output=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check that a PR's changed tests fail on the base code.")
    ap.add_argument("--base", required=True, help="the base commit sha")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--opt-out", action="store_true", help="the PR has label no-red-test")
    a = ap.parse_args(argv)

    if a.opt_out:
        _report("skipped", ["skipped: label no-red-test"])
        return 0
    repo = Path(a.repo).resolve()
    tests, test_modules, code = split_files(changed_files(repo, a.base, a.head))
    if not test_modules:
        _report("skipped", ["no changed tests/**/test_*.py file"])
        return 0
    if not code:
        _report("skipped", ["no changed code file outside tests/ and docs/"])
        return 0

    code_exit, output = run_on_base(repo, a.base, a.head, tests, test_modules)
    tail = _tail(output)
    if code_exit in (1, 2):
        _report("pass", ["the new tests fail without the fix, good", f"base pytest output (tail):\n{tail}"])
        return 0
    if code_exit == 5:
        _report("skipped", ["pytest collected no tests (deselected by markers?)"])
        return 0
    if code_exit == 0:
        _report(
            "FAIL",
            [
                "the PR's tests pass on the base code too, so they don't prove the fix",
                "make a test that fails without the fix; for a refactor that keeps behaviour the same, add label "
                "`no-red-test`, say why in the PR body, and re-run this check",
                f"base pytest output (tail):\n{tail}",
            ],
        )
        return 1
    why = "timed out" if code_exit is None else f"pytest exited {code_exit}"
    _report("FAIL", [f"base run {why}", f"base pytest output (tail):\n{tail}"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

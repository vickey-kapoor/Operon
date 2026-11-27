"""Red test check (KAN-280): a PR's new tests must fail on the base branch's code.

Every fix PR adds a test, but nothing proved the test would have caught the bug. This runs each changed
test module against the base commit's code. It passes if at least one changed test fails without the fix, and it
lists which modules were red and which passed on base. Stdlib only, $0.

    python scripts/red_test_check.py --base SHA [--head REF] [--repo PATH] [--opt-out]

Code means a changed file under src/, scripts/, evals/ or .github/. Exit 0: at least one changed test module
fails on the base (good), or the check is skipped (no changed test, no code change, nothing collected, or
``--opt-out`` for a PR labelled ``no-red-test``). Exit 1: every module passes on the base, or pytest failed in
some other way. The reasons go to stderr and, when set, to
``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

TIMEOUT_S = 15 * 60
TAIL_LINES = 40
CODE_DIRS = ("src", "scripts", "evals", ".github")


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
    out = _git(repo, "-c", "core.quotePath=false", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...{head}")
    return [line.strip() for line in out.decode("utf-8", "replace").splitlines() if line.strip()]


def split_files(files: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (changed files under tests/, changed tests/**/test_*.py files, code files)."""
    parts = [(f, PurePosixPath(f).parts) for f in files]
    tests = [f for f, p in parts if p[0] == "tests"]
    test_modules = [f for f, p in parts if p[0] == "tests" and fnmatch.fnmatchcase(p[-1], "test_*.py")]
    code = [f for f, p in parts if p[0] in CODE_DIRS]
    return tests, test_modules, code


def _tail(text: str) -> str:
    return "\n".join(text.splitlines()[-TAIL_LINES:])


def run_on_base(repo: Path, base: str, head: str, tests: list[str], test_modules: list[str]) -> list[tuple[str, int | None, str]]:
    """Run each changed test module on the base code. Returns (module, exit code or None on timeout, output)."""
    results: list[tuple[str, int | None, str]] = []
    with tempfile.TemporaryDirectory(prefix="red-test-", ignore_cleanup_errors=True) as tmp:
        wt = Path(tmp) / "wt"
        _git(repo, "worktree", "add", "--detach", str(wt), base)
        try:
            for rel in tests:
                dest = wt / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_git(repo, "show", f"{head}:{rel}"))
            env = dict(os.environ)
            # The base code must win over an editable install that points at another checkout.
            env["PYTHONPATH"] = os.pathsep.join(p for p in (str(wt / "src"), env.get("PYTHONPATH", "")) if p)
            for module in test_modules:
                cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", module]
                try:
                    proc = subprocess.run(cmd, cwd=wt, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
                except subprocess.TimeoutExpired as exc:
                    out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
                    results.append((module, None, out))
                    continue
                results.append((module, proc.returncode, proc.stdout + proc.stderr))
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo, capture_output=True)
    return results


def _describe(code: int | None) -> str:
    return "timed out" if code is None else f"pytest exited {code}"


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
        _report("skipped", ["no changed code file under src/, scripts/, evals/ or .github/"])
        return 0

    results = run_on_base(repo, a.base, a.head, tests, test_modules)
    red = [m for m, c, _ in results if c in (1, 2)]
    other = [(m, c, o) for m, c, o in results if c not in (1, 2, 5)]
    passed = [m for m, c, _ in other if c == 0]
    lines = []
    if red:
        lines.append("red on base (fail without the fix): " + ", ".join(red))
    if passed:
        lines.append("passed on base (prove nothing alone): " + ", ".join(passed))
    if not red and not other:
        _report("skipped", ["pytest collected no tests (deselected by markers?)"])
        return 0
    if red:
        first = next(o for _, c, o in results if c in (1, 2))
        _report("pass", ["at least one changed test fails without the fix, good", *lines, f"base pytest output of {red[0]} (tail):\n{_tail(first)}"])
        return 0
    errors = [f"{m}: {_describe(c)}" for m, c, _ in other if c != 0]
    detail = [f"base pytest output of {m} (tail):\n{_tail(o)}" for m, _, o in other]
    _report(
        "FAIL",
        [
            "no changed test fails without the fix, so they don't prove it",
            *lines,
            *errors,
            "make a test that fails without the fix; for a refactor that keeps behaviour the same, add label "
            "`no-red-test`, say why in the PR body, and re-run this job",
            *detail,
        ],
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

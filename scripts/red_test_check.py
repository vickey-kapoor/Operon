"""Red test check (KAN-280): a PR's new tests must fail on the base branch's code.

Every fix PR adds a test, but nothing proved the test would have caught the bug. This runs each changed
test module against the base commit's code. It passes if at least one changed test fails without the fix, and it
lists which modules were red and which passed on base. Stdlib only, $0.

    python scripts/red_test_check.py --base SHA [--head REF] [--repo PATH] [--opt-out]

Code means a changed (or deleted) file under src/, scripts/, evals/ or .github/. Exit 0: at least one changed test module
fails on the base (good), or the check is skipped (no code change, nothing collected, no changed test with
changes only under scripts/, evals/ or .github/, or ``--opt-out`` for a PR labelled ``no-red-test``). A PR that
changes a Python file under src/ (src/**/*.py, deleted or moved out too: the diff runs with --no-renames) but adds or edits no tests/**/test_*.py module fails (exit 1). A src change with no Python file (prompt text, yaml, data) is skipped. A module whose tests are all skipped on the base counts as
"not run" (like nothing collected), not as passed. A changed ``tests/conftest.py`` that fails to import (it uses
something only the PR adds) counts as red. Exit 1: every module that ran passes on the base, or pytest failed in
some other way. The reasons go to stderr and, when set, to
``$GITHUB_STEP_SUMMARY``.

CI runs main's copy of this file, so a change here takes effect for other PRs only after merge.
A new flag must be supported by main's copy before the workflow passes it (see the #374 lesson).
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

TIMEOUT_S = 5 * 60
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


def changed_files(repo: Path, base: str, head: str, filt: str = "ACMRD") -> list[str]:
    """Changed paths, deleted ones included by default (deleting production code is a code change)."""
    out = _git(repo, "-c", "core.quotePath=false", "diff", "--name-only", "--no-renames", f"--diff-filter={filt}", f"{base}...{head}")
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


def _outcomes(xml_path: Path) -> tuple[int, int] | None:
    """Return (passed, failed or errored) from a pytest junit file, or None if it is missing or unreadable."""
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return None
    passed = failed = 0
    for suite in [root] if root.tag == "testsuite" else root.iter("testsuite"):
        n, f, e, sk = (int(suite.get(k, "0")) for k in ("tests", "failures", "errors", "skipped"))
        failed += f + e
        passed += max(n - f - e - sk, 0)
    return passed, failed


CONFTEST_IMPORT = re.compile(r"while loading conftest[\s\S]*?(ImportError|ModuleNotFoundError|AttributeError)")


def classify(code: int | None, output: str, counts: tuple[int, int] | None) -> str:
    """One of red, passed, notrun, other for a module's base run."""
    if code is None:
        return "red"  # timed out on base: a hang the fix removes is proof too
    if code in (1, 2):
        return "red"
    if code == 5:
        return "notrun"
    if code == 4 and CONFTEST_IMPORT.search(output):
        return "red"
    if code == 0:
        if counts is not None and counts[0] == 0 and counts[1] == 0:
            return "notrun"
        return "passed"
    return "other"


def run_on_base(repo: Path, base: str, head: str, tests: list[str], test_modules: list[str]) -> list[tuple[str, int | None, str, tuple[int, int] | None]]:
    """Run each changed test module on the base code. Returns (module, exit code or None on timeout, output, (passed, failed))."""
    results: list[tuple[str, int | None, str, tuple[int, int] | None]] = []
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
            for i, module in enumerate(test_modules):
                xml_path = Path(tmp) / f"junit-{i}.xml"
                cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml_path}", module]
                try:
                    proc = subprocess.run(cmd, cwd=wt, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
                except subprocess.TimeoutExpired as exc:
                    out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
                    results.append((module, None, out, None))
                    continue
                results.append((module, proc.returncode, proc.stdout + proc.stderr, _outcomes(xml_path)))
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
    deleted = set(changed_files(repo, a.base, a.head, "D"))
    tests, test_modules, code = split_files(changed_files(repo, a.base, a.head))
    # A deleted test file has nothing to copy to the base worktree and nothing to run: only code counts for it.
    tests = [f for f in tests if f not in deleted]
    test_modules = [f for f in test_modules if f not in deleted]
    src_py = any(PurePosixPath(f).parts[0] == "src" and f.endswith(".py") for f in code)
    if not test_modules and src_py:
        _report(
            "FAIL",
            [
                "This PR changes Python code under src/ but no test. Add a test that fails without the change, or, for a refactor "
                "that keeps behaviour the same, add label no-red-test, say why in the PR body, and re-run this check."
            ],
        )
        return 1
    if not test_modules:
        has_src = any(PurePosixPath(f).parts[0] == "src" for f in code)
        _report("skipped", ["src change has no Python code" if has_src else "no changed tests/**/test_*.py file"])
        return 0
    if not code:
        _report("skipped", ["no changed code file under src/, scripts/, evals/ or .github/"])
        return 0

    results = run_on_base(repo, a.base, a.head, tests, test_modules)
    kinds = [(m, c, o, classify(c, o, n)) for m, c, o, n in results]
    red = [m for m, _, _, k in kinds if k == "red"]
    notrun = [m for m, _, _, k in kinds if k == "notrun"]
    other = [(m, c, o) for m, c, o, k in kinds if k in ("passed", "other")]
    passed = [m for m, _, _, k in kinds if k == "passed"]
    lines = []
    if red:
        timed_out = {m for m, c, _, k in kinds if k == "red" and c is None}
        names = [f"{m} (timed out on base (counted as red))" if m in timed_out else m for m in red]
        lines.append("red on base (fail without the fix): " + ", ".join(names))
    if passed:
        lines.append("passed on base (prove nothing alone): " + ", ".join(passed))
    if notrun:
        lines.append("not run on base (nothing collected, or every test skipped): " + ", ".join(notrun))
    if not red and not other:
        _report("skipped", ["no changed test module ran on the base (nothing collected, or every test skipped)", *lines])
        return 0
    if red:
        first = next(o for m, _, o, k in kinds if k == "red")
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
            "make a test that fails without the fix; for a refactor that keeps behaviour the same, or when the "
            "changed tests pass on base by design (e.g. a new guard in tests/test_invariants.py), add label "
            "`no-red-test`, say why in the PR body, and re-run this job",
            *detail,
        ],
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

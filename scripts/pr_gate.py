"""PR gate (KAN-216): decide whether a pull request is low-risk and ready to merge on its own.

Pure functions plus a thin CLI; the workflow ``pr-gate.yml`` feeds it JSON saved from ``gh api``.
Stdlib only. Fail closed: anything unknown, missing or odd means "risky" / "not ready".

    python scripts/pr_gate.py classify --files files.json [--fork]
    python scripts/pr_gate.py merge-ready --pr pr.json --files files.json --checks checks.json \
        --comments comments.json [--ignore-check NAME ...]

``classify`` prints ``low-risk`` or ``risky`` on stdout; ``merge-ready`` prints ``ready`` or ``not-ready``.
The reasons go to stderr and, when set, to ``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOW = "low-risk"
RISKY = "risky"

# The files API stops at 3000 entries; a PR that big is never auto-merged.
MAX_FILES = 3000

# Paths a human must merge. A pattern ending in "/**" is a directory prefix; anything else is an
# fnmatch pattern over the repo-relative posix path. tests/ci/test_pr_gate.py checks that every entry
# matches something, so a rename of a safety-gate module cannot silently drop it from the list.
RISKY_PATTERNS: tuple[str, ...] = (
    # CI, workflows, agent definitions and the rules themselves.
    ".github/**",
    ".claude/**",
    "CLAUDE.md",
    "AGENTS.md",
    ".env.example",
    "pyproject.toml",
    "uv.lock",
    "dashboard/package*.json",
    "docker/**",
    "scripts/**",  # includes this gate
    "tests/conftest.py",  # the safe-mode / strict-errors / no-auth test defaults
    "tests/test_invariants.py",
    # The harness and the safety gates: trust gate and permission modes (trust_gate.py), file guard,
    # OCR trust gate, confidence gate, coordinate-thrash pause, max steps and secret-type gate (loop.py).
    "src/operon/agent/policy/**",
    "src/operon/agent/loop.py",
    "src/operon/agent/step_context.py",
    "src/operon/agent/tools/**",  # run_shell, http_get
    # Visual servo, scroll units and every process the model can start (the executors).
    "src/operon/executor/**",
    # is_web_url, launch_env, settings, secrets and keys.
    "src/operon/core/urls.py",
    "src/operon/core/launch_env.py",
    "src/operon/core/settings.py",
    "src/operon/core/vault.py",
    "src/operon/core/keystore.py",
    "src/operon/secrets_command.py",
    "src/operon/keys_command.py",
    # Permission-mode fields, auth, CORS and per-environment wiring.
    "src/operon/models/common.py",
    "src/operon/models/state.py",
    "src/operon/api/auth.py",
    "src/operon/api/server.py",
    "src/operon/api/frontdoor.py",
    "src/operon/api/runtime/services.py",
    # Evals that start paid runs.
    "evals/bench*.py",
    "evals/osworld*",
    "evals/grounding*.py",
    "evals/live_*.py",
    "evals/judge_check.py",
)

OWNER_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
CODEX_BOT = "chatgpt-codex-connector[bot]"
GOOD_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})
VERIFIER_RE = re.compile(r"<!--\s*operon-verifier:\s*PASS\s+([0-9a-fA-F]{40})\s*-->")


@dataclass(frozen=True)
class ChangedFile:
    filename: str
    status: str = "modified"
    previous_filename: str | None = None


def _matches(pattern: str, path: str) -> bool:
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return fnmatch.fnmatchcase(path, pattern)


def is_risky_path(path: str) -> bool:
    path = path.replace("\\", "/")
    # A path with empty, "." or ".." parts is not one we can reason about.
    if any(p in ("", ".", "..") for p in path.split("/")):
        return True
    return any(_matches(p, path) for p in RISKY_PATTERNS)


def _is_test(path: str) -> bool:
    return path.startswith("tests/")


def classify(files: list[ChangedFile], is_fork: bool) -> tuple[str, list[str]]:
    """Return (``low-risk`` | ``risky``, reasons). Risky when in any doubt."""
    reasons: list[str] = []
    if is_fork:
        reasons.append("PR is from a fork")
    if not files:
        reasons.append("no changed files listed")
    if len(files) >= MAX_FILES:
        reasons.append(f"{len(files)} changed files reaches the API limit")
    for f in files:
        for name in [f.filename] + ([f.previous_filename] if f.previous_filename else []):
            if is_risky_path(name):
                reasons.append(f"risky path: {name}")
        if f.status == "removed" and _is_test(f.filename):
            reasons.append(f"test file deleted: {f.filename}")
        if f.status == "renamed" and f.previous_filename and _is_test(f.previous_filename):
            if not _is_test(f.filename):
                reasons.append(f"test file moved out of tests/: {f.previous_filename}")
    reasons = list(dict.fromkeys(reasons))
    return (RISKY if reasons else LOW), reasons or ["only docs, tests and non-gate source changed"]


def _find_codex_state(comments: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """The newest ``codex-review-state`` JSON object in a Codex-bot comment, or None."""
    found: dict[str, Any] | None = None
    for c in comments:
        if (c.get("user") or {}).get("login") != CODEX_BOT:
            continue
        body = c.get("body") or ""
        idx = body.find("codex-review-state")
        brace = body.find("{", idx) if idx >= 0 else -1
        if brace < 0:
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(body[brace:])
        except ValueError:
            continue
        if isinstance(obj, dict):
            found = obj
    return found


def merge_ready(
    pr: dict[str, Any],
    checks: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    ignore_checks: Iterable[str] = (),
    *,
    risk: str | None = None,
) -> tuple[bool, list[str]]:
    """Whether the gate may merge ``pr``. ``pr`` is a pulls API object; ``checks`` the head's check runs.

    ``risk`` is a fresh classification of the PR's files; when given it must be low-risk too.
    """
    why: list[str] = []
    head = (pr.get("head") or {}).get("sha") or ""
    if not head:
        return False, ["no head sha"]
    labels = {(x.get("name") if isinstance(x, dict) else x) for x in pr.get("labels") or []}
    if LOW not in labels or RISKY in labels:
        why.append(f"not labelled {LOW}")
    if risk is not None and risk != LOW:
        why.append("fresh classification is not low-risk")
    if pr.get("draft"):
        why.append("PR is a draft")
    if pr.get("state", "open") != "open":
        why.append("PR is not open")

    ignored = set(ignore_checks)
    relevant = [c for c in checks if c.get("name") not in ignored]
    if not relevant:
        why.append("no check runs on the head")
    for c in relevant:
        if c.get("status") != "completed":
            why.append(f"check still running: {c.get('name')}")
        elif c.get("conclusion") not in GOOD_CONCLUSIONS:
            why.append(f"check {c.get('conclusion')}: {c.get('name')}")

    state = _find_codex_state(comments)
    if state is None:
        why.append("no Codex review summary")
    else:
        if state.get("commit_sha") != head:
            why.append("Codex review is for another commit")
        if state.get("status") != "completed":
            why.append(f"Codex review status is {state.get('status')!r}")
        if state.get("has_unresolved_findings") is not False:
            why.append("Codex review has unresolved findings")

    marked = any(
        c.get("author_association") in OWNER_ASSOCIATIONS
        and any(m.lower() == head.lower() for m in VERIFIER_RE.findall(c.get("body") or ""))
        for c in comments
    )
    if not marked:
        why.append("no verifier PASS marker for the head commit from an owner or collaborator")

    return (not why), why or ["all gates passed"]


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _files(path: str) -> list[ChangedFile]:
    return [
        ChangedFile(f["filename"], f.get("status", "modified"), f.get("previous_filename"))
        for f in _load(path)
    ]


def _report(verdict: str, reasons: list[str], title: str) -> None:
    text = "\n".join(f"- {r}" for r in reasons)
    print(f"{title}: {verdict}\n{text}", file=sys.stderr)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### {title}: {verdict}\n\n{text}\n\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classify a PR and decide whether the gate may merge it.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--files", required=True)
    c.add_argument("--fork", action="store_true")
    m = sub.add_parser("merge-ready")
    m.add_argument("--pr", required=True)
    m.add_argument("--files", required=True)
    m.add_argument("--checks", required=True)
    m.add_argument("--comments", required=True)
    m.add_argument("--ignore-check", action="append", default=[])
    a = ap.parse_args(argv)

    if a.cmd == "classify":
        risk, reasons = classify(_files(a.files), a.fork)
        _report(risk, reasons, "PR risk")
        print(risk)
        return 0
    pr = _load(a.pr)
    head_repo = ((pr.get("head") or {}).get("repo") or {}).get("full_name")
    base_repo = ((pr.get("base") or {}).get("repo") or {}).get("full_name")
    fork = head_repo is None or head_repo != base_repo
    risk, _ = classify(_files(a.files), fork)
    checks = _load(a.checks)
    ok, why = merge_ready(
        pr,
        checks.get("check_runs", []) if isinstance(checks, dict) else checks,
        _load(a.comments),
        a.ignore_check,
        risk=risk,
    )
    verdict = "ready" if ok else "not-ready"
    _report(verdict, why, f"Merge gate for #{pr.get('number')}")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

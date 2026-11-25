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
import html
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

# Inside src/ everything is low-risk EXCEPT these (a denylist). A pattern ending in "/**" is a directory
# prefix; anything else is an fnmatch pattern over the repo-relative posix path. Matching is
# case-insensitive. tests/ci/test_pr_gate.py checks that every entry matches a real file, so a rename of
# a safety-gate module cannot silently drop it from the list.
RISKY_PATTERNS: tuple[str, ...] = (
    # The harness and the safety gates: trust gate and permission modes (trust_gate.py), file guard.
    # OCR trust gate, coordinate-thrash pause, max steps and secret-type gate live in loop.py; the
    # confidence gate in loop.py and control.py.
    "src/operon/agent/policy/**",
    "src/operon/agent/loop.py",
    "src/operon/agent/step_context.py",
    "src/operon/agent/control.py",
    "src/operon/agent/backends/**",  # model adapters: what the model sees and may emit
    "src/operon/agent/tools/**",  # run_shell, http_get
    # Visual servo, scroll units and every process the model can start (the executors).
    "src/operon/executor/**",
    "src/operon/environments/**",  # the desktop machine the run drives
    # is_web_url, launch_env, settings, runtime state, secrets and keys.
    "src/operon/core/urls.py",
    "src/operon/core/launch_env.py",
    "src/operon/core/settings.py",
    "src/operon/core/runtime.py",
    "src/operon/core/vault.py",
    "src/operon/core/keystore.py",
    "src/operon/secrets_command.py",
    "src/operon/keys_command.py",
    # Permission-mode fields, auth, CORS, streaming, routes, entry points and per-environment wiring.
    "src/operon/models/common.py",
    "src/operon/models/state.py",
    "src/operon/api/auth.py",
    "src/operon/api/server.py",
    "src/operon/api/frontdoor.py",
    "src/operon/api/ws_stream.py",
    "src/operon/api/stream_tokens.py",
    "src/operon/api/routes/**",
    "src/operon/api/runtime/services.py",
    "src/operon/browser/manager.py",
    "src/operon/clients/**",
    "src/operon/cli.py",
    "src/operon/cloud.py",
)

# Outside src/ the rule is the other way round (fail closed): a path is low-risk only when an allow
# pattern matches and no exception does. Any dotfile or dot-directory is risky.
LOW_ALLOW: tuple[str, ...] = ("docs/**", "*.md", "tests/**", "dashboard/**")
LOW_EXCEPT: tuple[str, ...] = (
    # Rules files and security policy (the .claude/ and .github/ dot-directories are risky anyway).
    "claude.md",
    "agents.md",
    "security.md",
    # Test defaults and the guards themselves, and the gate's own tests.
    "conftest.py",
    "tests/ci/**",
    "tests/test_invariants.py",
    # Dashboard dependencies and config.
    "package*.json",
    "vercel.json",
    "next.config.*",
    "*.config.*",
)

# Checks that must be present and green on the head before a merge (the names are the CI job names).
REQUIRED_CHECKS: tuple[str, ...] = (
    "Test (Python 3.14)",
    "Test (Windows, Python 3.14)",
    "Lint",
    "Type check",
    "Browser end to end (Chromium)",
    "Desktop (Linux container)",
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
    pattern, path = pattern.casefold(), path.casefold()
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return fnmatch.fnmatchcase(path, pattern)


def _matches_anywhere(pattern: str, path: str) -> bool:
    """A bare pattern (no "/") matches the file name at any depth; others match the whole path."""
    if "/" in pattern:
        return _matches(pattern, path)
    return _matches(pattern, path.rsplit("/", 1)[-1])


def is_risky_path(path: str) -> bool:
    path = path.replace("\\", "/").casefold()
    parts = path.split("/")
    # A path with empty, "." or ".." parts is not one we can reason about.
    if any(p in ("", ".", "..") for p in parts):
        return True
    if parts[0] == "src":
        return any(_matches(p, path) for p in RISKY_PATTERNS)
    if any(p.startswith(".") for p in parts):
        return True
    if not any(_matches_anywhere(p, path) for p in LOW_ALLOW):
        return True
    return any(_matches_anywhere(p, path) for p in LOW_EXCEPT)


def _is_test(path: str) -> bool:
    return path.casefold().startswith("tests/")


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
        # The summary hides its state in an HTML comment whose braces are entities (&#123; ... &#125;).
        body = html.unescape(c.get("body") or "")
        idx = body.find("codex-review-state")
        brace = body.find("{", idx) if idx >= 0 else -1
        if brace < 0:
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(body[brace:])
        except ValueError:
            continue
        review = obj.get("code_review") if isinstance(obj, dict) else None
        if isinstance(review, dict):
            found = review
    return found


def merge_ready(
    pr: dict[str, Any],
    checks: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    ignore_checks: Iterable[str] = (),
    *,
    risk: str | None = None,
    status: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Whether the gate may merge ``pr``. ``pr`` is a pulls API object; ``checks`` the head's check runs;
    ``status`` the head's combined status (the older status API, which check runs do not cover).

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

    by_name = {c.get("name"): c for c in relevant}
    for name in REQUIRED_CHECKS:
        run = by_name.get(name)
        if run is None:
            why.append(f"required check missing: {name}")
        elif run.get("status") == "completed" and run.get("conclusion") != "success":
            why.append(f"required check not successful: {name}")

    if status is None:
        why.append("combined status not read")
    elif status.get("statuses") and status.get("state") != "success":
        why.append(f"combined status is {status.get('state')!r}")

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
        and (c.get("user") or {}).get("type") == "User"
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
    m.add_argument("--status", required=True)
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
        status=_load(a.status),
    )
    verdict = "ready" if ok else "not-ready"
    _report(verdict, why, f"Merge gate for #{pr.get('number')}")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

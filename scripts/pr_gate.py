"""PR gate (KAN-216, KAN-227): classify a pull request and decide whether it is ready to merge on its own.

Pure functions plus a thin CLI; the workflow ``pr-gate.yml`` feeds it JSON saved from ``gh api``.
Stdlib only. Fail closed: anything unknown, missing or odd means "risky" / "not ready".

    python scripts/pr_gate.py classify --files files.json [--fork]
    python scripts/pr_gate.py merge-ready --pr pr.json --files files.json --checks checks.json \
        --comments comments.json [--ignore-check NAME ...]
    python scripts/pr_gate.py paused --issues issues.json

``classify`` prints ``low-risk``, ``risky`` or ``needs-vickey`` on stdout (needs-vickey beats risky beats low-risk; agent definitions are read with ``--repo`` and ``--ref``, else they need Vickey); ``merge-ready`` prints ``ready``, ``not-ready`` or ``no-real-check`` (the only thing missing is the "Real check:" line, KAN-298); ``paused`` prints ``paused`` or ``running`` (KAN-229).
The reasons go to stderr and, when set, to ``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import html
import json
import os
import posixpath
import re
import subprocess
import sys
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOW = "low-risk"
RISKY = "risky"
PAID = "needs-vickey"  # the label for PRs that wait for Vickey's own "merge"
LEGACY_PAID = "paid-run"  # the old name of that label: still refused, so a stale label cannot be merged

# The files API stops at 3000 entries, so the list may be cut short: a PR that big is classified needs-vickey (fail closed).
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
    "src/operon/__init__.py",  # runs on every import
    "src/operon/agent/control.py",
    "src/operon/agent/app_catalog.py",  # what apps the model may launch
    "src/operon/agent/actions/zoom.py",
    "src/operon/agent/hitl.py",  # human-in-the-loop notifier
    "src/operon/core/paths.py",  # where every artifact directory resolves
    "src/operon/api/observer.py",
    "src/operon/api/dependencies.py",
    "src/operon/store/checkpoints.py",
    "src/operon/agent/perception/**",  # vision-only: what the model sees
    "src/operon/agent/actions/grounding.py",
    "src/operon/agent/actions/translation.py",  # atomic TYPE, confidence default
    "src/operon/models/policy.py",  # ActionType / PolicyDecision used by the gates
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
    "src/operon/api/runtime/**",  # per-environment wiring, loops, events
    "src/operon/store/run_store.py",
    "src/operon/browser/manager.py",
    "src/operon/clients/**",
    "src/operon/cli.py",
    "src/operon/cloud.py",
)

# Outside src/ the rule is the other way round (fail closed): a path is low-risk only when it is one of
# the kinds below. Everything else is risky: any dotfile or dot-directory, any *.ini/*.cfg/*.toml, helper
# modules and conftest.py under tests/ (pytest can load them as plugins), tests/ci/**, dashboard server
# code, evals/**, scripts/**, docker/**, lock files.
RISKY_NAMES: tuple[str, ...] = (
    "claude.md",
    "agents.md",
    "security.md",
    "conftest.py",
    "claude-code-reference.md",  # the standing rules for local sessions
    "sitecustomize.py",  # python runs these on startup
    "usercustomize.py",
)
# Instruction files that coding agents and reviewers read, at any depth: AGENTS.override.md, CLAUDE.local.md,
# GEMINI.md, copilot-instructions.md (dotfiles such as .cursorrules are risky as dotfiles).
RISKY_NAME_PREFIXES: tuple[str, ...] = ("agents.", "claude.", "gemini.", "copilot-instructions.")
# Compiled or loadable code, anywhere in the repo, and Python's cache directories.
BINARY_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".pth", ".egg-link")
# The only kinds of file under src/ that can be low-risk (what src/ holds today).
SRC_TEXT_SUFFIXES: tuple[str, ...] = (".py", ".md", ".txt", ".yaml")
DOCS_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
RISKY_SUFFIXES: tuple[str, ...] = (".ini", ".cfg", ".toml")
TEST_FILE_PATTERNS: tuple[str, ...] = ("test_*.py", "*_test.py")  # what pytest collects
DASHBOARD_STATIC: tuple[str, ...] = ("dashboard/app.js", "dashboard/readme.md")


# KAN-227 (Vickey, 2026-11-26: "Every PR can auto-merge but no paid runs without my explicit approval"): a PR that could
# start a paid run, change how paid runs start or what they may spend, or change the merge check itself is never merged
# by the gate; it waits for Vickey ("needs-vickey"). Every workflow and action file is here, because a workflow can start
# a paid run on its own (gh workflow run, a reusable workflow, a wider token, a secret); the content is not inspected.
# A rename into or out of these paths counts too (classify checks both names). Same pattern syntax as RISKY_PATTERNS;
# every entry must match a real file (tests/ci/test_pr_gate.py). The rules files (CLAUDE.md, AGENTS.md, ...) and other
# .github/ files (dependabot.yml, CODEOWNERS, templates) are risky, not here; evals/CLAUDE.md waits only because
# evals/** does.
PAID_RUN_PATTERNS: tuple[str, ...] = (
    # Every workflow and action (the five paid-run workflows are benchmark.yml, live-bench.yml, live-desktop-bench.yml,
    # grounding.yml and osworld-bench.yml; the merge check is pr-gate.yml, scripts/pr_gate.py and its tests).
    ".github/workflows/**",
    ".github/actions/**",
    # .claude/ and .mcp.json are not listed here: see is_claude_gated (plain agent definitions are risky, the rest waits).
    "scripts/pr_gate.py",
    "tests/ci/test_pr_gate.py",
    # Everything under evals/ (bench, batch queue, OSWorld, grounding, live smoke, caps, suites, seasons, runners'
    # scripts), except recorded data (PAID_RUN_EXEMPT).
    "evals/**",
    # Bench-runner and VM scripts.
    "scripts/**",
    # Price tables behind every cost cap, settings (caps, model defaults), model defaults.
    "src/operon/models/usage.py",
    "src/operon/core/settings.py",
    "src/operon/agent/backends/model_adapter/registry.py",
    "src/operon/agent/backends/model_adapter/models.yaml",  # price table, model ids
    "src/operon/api/runtime_config.py",  # default model, pricing check
    "src/operon/api/runtime/services.py",  # pricing check
    "src/operon/clients/**",  # model ids
    # KAN-230: cost code in the model connections (what is sent and counted on every paid call).
    "src/operon/agent/backends/model_adapter/transports/**",
    "src/operon/api/observer.py",
)
# Agent definitions (.claude/agents/<name>.md) can only be a plain risky change while their frontmatter carries no
# hooks, MCP servers or permission settings (those run commands in the next Claude Code session). Only these keys.
AGENT_FRONTMATTER_KEYS: tuple[str, ...] = ("name", "description", "model", "tools", "isolation", "color", "effort")
AGENT_FILE_MAX_BYTES = 64 * 1024
_AGENT_LINE_RE = re.compile(r"(" + "|".join(AGENT_FRONTMATTER_KEYS) + r"): (\S.*)")
_AGENT_WORD_RE = re.compile(r"\w+")
# Control characters (other than newline and tab) and Unicode line or paragraph separators: a YAML reader could treat
# them as line breaks that this check does not see.
_AGENT_ODD_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})

# Data recorded for the free replay tests: not a way to start or fund a paid run.
PAID_RUN_EXEMPT: tuple[str, ...] = ("evals/recordings/*.json",)


# Checks that must be present and green on the head before a merge (the names are the CI job names).
REQUIRED_CHECKS: tuple[str, ...] = (
    "Test (Python 3.14)",
    "Test (Windows, Python 3.14)",
    "Lint",
    "Type check",
    "Browser end to end (Chromium)",
    "Desktop (Linux container)",
    "Tests fail without the fix",
)

# KAN-229: while an open issue carries this label, the gate merges nothing and posts no would-merge notes.
PAUSE_LABEL = "auto-work-paused"

OWNER_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
CODEX_BOT = "chatgpt-codex-connector[bot]"
GOOD_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})
VERIFIER_RE = re.compile(r"<!--\s*operon-verifier:\s*PASS\s+([0-9a-fA-F]{40})\s*-->")
TOP_REVIEW_RE = re.compile(r"<!--\s*operon-top-review:\s*PASS\s+([0-9a-fA-F]{40})\s*-->")


@dataclass(frozen=True)
class ChangedFile:
    filename: str
    status: str = "modified"
    previous_filename: str | None = None


def _matches(pattern: str, path: str) -> bool:
    pattern, path = pattern.casefold(), path.casefold()
    if pattern.endswith("/**"):
        # The bare directory (no trailing file) matches too.
        return path.startswith(pattern[:-2]) or path == pattern[:-3]
    return fnmatch.fnmatchcase(path, pattern)


def _is_collected_test(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path.casefold().rsplit("/", 1)[-1], p) for p in TEST_FILE_PATTERNS)


def _low_risk_kind(path: str) -> bool:
    """Whether a non-src path (casefolded, no dot parts) is one of the allowed low-risk kinds."""
    name = path.rsplit("/", 1)[-1]
    if name in RISKY_NAMES or name.endswith(RISKY_SUFFIXES):
        return False
    if name.endswith(".md"):
        return True
    if path.startswith("docs/"):
        return name.endswith(DOCS_IMAGE_SUFFIXES)
    if path.startswith("tests/"):
        if path.startswith("tests/ci/") or path == "tests/test_invariants.py":
            return False
        if _is_collected_test(path) or name == "__init__.py":
            return True
        # Data files in a fixtures directory (never code: a .py file there could be loaded as a plugin).
        return "/fixtures/" in path and not name.endswith(".py")
    if path.startswith("dashboard/"):
        static_page = path.count("/") == 1 and name.endswith(".html")
        return static_page or name.endswith(".css") or path in DASHBOARD_STATIC
    return False


def _src_denied(path: str) -> bool:
    """On the src/ denylist. A denylisted ``x.py`` also denies a package ``x/`` that would shadow it."""
    for pattern in RISKY_PATTERNS:
        if _matches(pattern, path):
            return True
        if pattern.endswith(".py") and "*" not in pattern and _matches(pattern[:-3] + "/**", path):
            return True
    return False


def is_risky_path(path: str, status: str = "modified") -> bool:
    """Whether ``path`` is outside the low-risk kinds (needs a review). ``status`` is the file's change status in the PR."""
    path = path.replace("\\", "/").casefold()
    parts = path.split("/")
    # A path with empty, "." or ".." parts is not one we can reason about.
    if any(p in ("", ".", "..") for p in parts):
        return True
    name = parts[-1]
    if "__pycache__" in parts or name.endswith(BINARY_SUFFIXES) or name.startswith(RISKY_NAME_PREFIXES):
        return True
    if parts[0] == "src":
        # Only edits to existing files under src/operon/ can be low-risk. A new file (added, renamed or
        # copied in) can shadow an import (src/json.py, a package next to a denylisted module), so it
        # always needs the risky review.
        if status != "modified" or parts[1:2] != ["operon"]:
            return True
        if name in RISKY_NAMES or name == "__init__.py" or not name.endswith(SRC_TEXT_SUFFIXES):
            return True  # __init__.py runs on import
        return _src_denied(path)
    if any(p.startswith(".") for p in parts):
        return True
    return not _low_risk_kind(path)


def is_plain_agent_path(path: str) -> bool:
    """Whether ``path`` is exactly ``.claude/agents/<name>.md`` (one level, no odd parts), case-insensitively."""
    parts = path.replace("\\", "/").casefold().split("/")
    return (
        len(parts) == 3
        and parts[0] == ".claude"
        and parts[1] == "agents"
        and parts[2].endswith(".md")
        and len(parts[2]) > 3
    )


def is_claude_gated(raw: str, normalized: str) -> bool:
    """Any ``.claude`` path part or ``.mcp.json`` file, at any depth, except a plain agent definition."""
    parts = normalized.split("/")
    if parts[-1] == ".mcp.json":
        return True
    return ".claude" in parts and not is_plain_agent_path(raw)


def agent_file_is_safe(text: str) -> bool:
    """Whether an agent definition's text cannot run commands: a strict frontmatter of plain keys only.

    The frontmatter (between the first line ``---`` and the next ``---`` line) may hold only ``key: value`` lines for
    ``AGENT_FRONTMATTER_KEYS``, each at most once, with no leading whitespace; ``tools`` is a comma-separated list of
    plain words. Anything else (indentation, comments, blank lines, ``hooks:``, ``mcpServers:``, odd control or
    line-separator characters, a missing or unterminated frontmatter, a file over 64 KB) is not safe.
    """
    if len(text.encode("utf-8", "surrogatepass")) > AGENT_FILE_MAX_BYTES or any(
        unicodedata.category(c) in _AGENT_ODD_CATEGORIES and c not in "\n\t" for c in text
    ):
        return False
    lines = text.split("\n")
    if lines[0] != "---" or "---" not in lines[1:]:
        return False
    end = lines.index("---", 1)
    if any(ln.rstrip() == "---" or ln.startswith(("hooks:", "mcpServers:")) for ln in lines[end + 1 :]):
        return False  # a second frontmatter-like block or a hooks line in the body
    seen: set[str] = set()
    for line in lines[1:end]:
        m = _AGENT_LINE_RE.fullmatch(line)
        if m is None or m.group(1) in seen:
            return False
        seen.add(m.group(1))
        if m.group(1) == "tools" and not all(_AGENT_WORD_RE.fullmatch(t.strip()) for t in m.group(2).split(",")):
            return False
    return True


def _gh_json(endpoint: str) -> dict[str, Any] | None:
    """``gh api`` JSON object for ``endpoint`` (needs GH_TOKEN), or None on any failure."""
    try:
        out = subprocess.run(["gh", "api", endpoint], capture_output=True, timeout=60, check=False)  # noqa: S603, S607
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout.decode("utf-8"))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def agent_name_matches(text: str, path: str) -> bool:
    """Whether the frontmatter ``name:`` is a plain word equal to the file stem (case-sensitive), so no file shadows another."""
    stem = path.replace("\\", "/").rsplit("/", 1)[-1][:-3]
    m = re.search(r"^name: (\S.*)$", text.split("\n---", 1)[0], re.M)
    return m is not None and _AGENT_WORD_RE.fullmatch(m.group(1)) is not None and m.group(1) == stem


def head_file_reader(
    repo: str, ref: str, fetch: Callable[[str], dict[str, Any] | None] = _gh_json
) -> Callable[[str], str | None]:
    """A reader of a plain agent file's text at commit ``ref`` of ``repo``. None on any failure (fail closed).

    It walks the git tree (commit, then ``.claude``, ``agents``, the file) so it can require every directory to be a
    real tree and the file a regular blob of mode 100644: a symlink, executable, submodule, missing or truncated
    entry, a case mismatch, a file over 64 KB or bad base64 or UTF-8 gives None. ``fetch(endpoint)`` returns a JSON
    object (default: ``gh api``).
    """

    def read(path: str) -> str | None:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
            return None
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != ".claude" or parts[1] != "agents" or not is_plain_agent_path(path):
            return None
        try:
            commit = fetch(f"repos/{repo}/git/commits/{ref}")
            sha = ((commit or {}).get("tree") or {}).get("sha")
            for i, name in enumerate(parts):
                if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
                    return None
                tree = fetch(f"repos/{repo}/git/trees/{sha}")
                entries = (tree or {}).get("tree")
                if tree is None or tree.get("truncated") or not isinstance(entries, list):
                    return None
                match = [e for e in entries if isinstance(e, dict) and e.get("path") == name]
                if len(match) != 1:
                    return None
                entry = match[0]
                last = i == len(parts) - 1
                if entry.get("type") != ("blob" if last else "tree") or entry.get("mode") != (
                    "100644" if last else "040000"
                ):
                    return None
                sha = entry.get("sha")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
                return None
            blob = fetch(f"repos/{repo}/git/blobs/{sha}")
            if blob is None or blob.get("encoding") != "base64" or not isinstance(blob.get("content"), str):
                return None
            size = blob.get("size")
            if not isinstance(size, int) or size > AGENT_FILE_MAX_BYTES:
                return None
            raw = base64.b64decode(blob["content"], validate=False)
            if len(raw) > AGENT_FILE_MAX_BYTES or len(raw) != size:
                return None
            return raw.decode("utf-8")
        except (ValueError, UnicodeDecodeError, TypeError, AttributeError):
            return None

    return read


def is_paid_run_path(path: str) -> bool:
    """Whether ``path`` could start a paid run or change how paid runs start or what they may spend."""
    raw = path.replace("\\", "/").casefold()
    path = posixpath.normpath(raw).casefold()
    if is_claude_gated(raw, path):
        return True
    if any(_matches(p, path) for p in PAID_RUN_EXEMPT):
        return False
    for pattern in PAID_RUN_PATTERNS:
        if _matches(pattern, path):
            return True
        # KAN-230: a package named like a listed module (observer/ next to observer.py) would shadow it.
        if pattern.endswith(".py") and "*" not in pattern and _matches(pattern[:-3] + "/**", path):
            return True
    return False


def _is_test(path: str) -> bool:
    return path.casefold().startswith("tests/")


def classify(
    files: list[ChangedFile], is_fork: bool, read_file: Callable[[str], str | None] | None = None
) -> tuple[str, list[str]]:
    """Return (``low-risk`` | ``risky`` | ``needs-vickey``, reasons). Risky when in any doubt.

    A fork is at least risky (needs-vickey when it also touches paid-run files); ``merge_ready`` refuses every fork.
    A PR with ``MAX_FILES`` or more files is needs-vickey: the files API may have cut its list short.
    ``read_file(path)`` returns a changed file's text at the head commit (or None); a plain agent definition
    (``.claude/agents/<name>.md``, not deleted) is risky only when it is read and ``agent_file_is_safe``, else needs-vickey.
    """
    if len(files) >= MAX_FILES:
        return PAID, [f"{len(files)} changed files reaches the API limit: the list may be incomplete"]
    reasons: list[str] = []
    paid: list[str] = []
    if is_fork:
        reasons.append("PR is from a fork")
    if not files:
        reasons.append("no changed files listed")
    for f in files:
        names = [(f.filename, f.status)] + ([(f.previous_filename, "removed")] if f.previous_filename else [])
        for name, status in names:
            if is_plain_agent_path(name) and name.replace("\\", "/").casefold().endswith("/verifier.md"):
                paid.append(f"needs Vickey's merge (the verifier's PASS is a merge marker): {name}")
            if is_paid_run_path(name):
                paid.append(f"needs Vickey's merge (paid-run or merge-check file): {name}")
            if is_risky_path(name, status):
                reasons.append(f"risky path: {name}" + (f" ({status})" if status != "modified" else ""))
        if is_plain_agent_path(f.filename) and f.status != "removed" and not is_paid_run_path(f.filename):
            text = read_file(f.filename) if read_file else None
            if text is None or not agent_file_is_safe(text) or not agent_name_matches(text, f.filename):
                paid.append(f"needs Vickey's merge (agent definition unreadable or beyond plain frontmatter): {f.filename}")
        if f.status == "removed" and _is_test(f.filename):
            reasons.append(f"test file deleted: {f.filename}")
        if f.status == "renamed" and f.previous_filename and _is_collected_test(f.previous_filename):
            if not _is_collected_test(f.filename):
                reasons.append(f"test file renamed to a non-test: {f.previous_filename}")
    if paid:
        return PAID, list(dict.fromkeys(paid))
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


def is_fork_pr(pr: dict[str, Any]) -> bool:
    """Whether ``pr`` (a pulls API object) comes from another repository. Unknown repos count as a fork."""
    head_repo = ((pr.get("head") or {}).get("repo") or {}).get("full_name")
    base_repo = ((pr.get("base") or {}).get("repo") or {}).get("full_name")
    return head_repo is None or head_repo != base_repo


REAL_CHECK_REASON = "no 'Real check:' line with a link in the PR description"

# A line that looks like a code fence after any run of spaces, tabs, quote marks and list markers: the scan stops there.
_FENCE_LIKE_RE = re.compile(r"^(?:[ \t>]|[-*+](?=[ \t])|\d{1,9}[.)](?=[ \t]))*(?:```|~~~)")
# A top-level line (no indentation, no quote): one optional list marker, one optional heading mark, then the words.
_REAL_CHECK_LINE_RE = re.compile(
    r"^(?:(?:[-*+]|\d{1,9}[.)]) )?(?:\#{1,6} )?[*_]*real check *[*_]* *:(?P<rest>.*)$", re.IGNORECASE
)
_LINK_RE = re.compile(r"https?://\S")
_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")  # not str.splitlines(): it also splits on \x0b, \x0c, \x85, U+2028 ...


def _strip_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """``line`` without HTML comment text, and whether a comment is still open at its end (it may span lines)."""
    out = ""
    while True:
        if in_comment:
            end = line.find("-->")
            if end < 0:
                return out, True
            line, in_comment = line[end + 3 :], False
        else:
            start = line.find("<!--")
            if start < 0:
                return out + line, False
            out, line, in_comment = out + line[:start], line[start + 4 :], True


def has_real_check(body: str | None) -> bool:
    """Whether a PR description has a "Real check:" line with a link on the same line (KAN-298).

    The line must be a top-level line (column 0, no quote, at most one list marker and one heading mark before the
    words) and must come before any code block. Fail closed by construction: lines are split on CR, LF or CRLF only,
    HTML comments (may span lines, an unclosed one runs to the end) never count, and the scan stops at the first
    fence-like line (``` or ~~~ after any spaces, quote marks and list markers, even inside a comment), so nothing
    after it counts. This is not a Markdown parser.
    """
    if not isinstance(body, str):
        return False
    in_comment = False
    for raw in _LINE_BREAK_RE.split(body):
        if _FENCE_LIKE_RE.match(raw):
            return False
        line, in_comment = _strip_comments(raw, in_comment)
        if _FENCE_LIKE_RE.match(line):
            return False
        m = _REAL_CHECK_LINE_RE.match(line)
        if m and _LINK_RE.search(m.group("rest")):
            return True
    return False


def merge_ready(
    pr: dict[str, Any],
    checks: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    ignore_checks: Iterable[str] = (),
    *,
    risk: str,
    status: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Whether the gate may merge ``pr``. ``pr`` is a pulls API object; ``checks`` the head's check runs;
    ``status`` the head's combined status (the older status API, which check runs do not cover).

    ``risk`` is a fresh classification of the PR's files. A fork PR and a ``needs-vickey`` PR are never ready. A
    ``risky`` PR (labelled risky, classified risky) also needs a top-review marker for the head from an owner or
    collaborator, in a comment of its own (one that does not also carry the verifier marker). Every PR also needs a
    "Real check:" line with a link in its description (``has_real_check``, KAN-298); when it is the only reason, the
    CLI prints ``no-real-check``.
    """
    why: list[str] = []
    head = (pr.get("head") or {}).get("sha") or ""
    if not head:
        return False, ["no head sha"]
    if is_fork_pr(pr):
        return False, ["PR is from a fork"]
    labels = {(x.get("name") if isinstance(x, dict) else x) for x in pr.get("labels") or []}
    if PAID in labels or LEGACY_PAID in labels or risk == PAID:
        return False, ["needs Vickey's merge: touches paid-run or merge-check files"]
    if not has_real_check(pr.get("body")):
        why.append(REAL_CHECK_REASON)
    level = RISKY if RISKY in labels or risk == RISKY else LOW
    if len(labels & {LOW, RISKY}) != 1:
        why.append(f"not labelled exactly one of {LOW} or {RISKY}")
    elif (RISKY in labels) != (risk == RISKY):
        why.append("fresh classification does not match the label")
    if risk not in (LOW, RISKY):
        why.append("fresh classification is not low-risk or risky")
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

    # Only runs from GitHub Actions count: another app can post a check run with a required name.
    by_name = {c.get("name"): c for c in relevant if (c.get("app") or {}).get("slug") == "github-actions"}
    for name in REQUIRED_CHECKS:
        run = by_name.get(name)
        if run is None:
            why.append(f"required check missing: {name}")
        elif run.get("status") == "completed" and run.get("conclusion") != "success":
            why.append(f"required check not successful: {name}")

    if (pr.get("base") or {}).get("ref") != "main":
        why.append("PR does not target main")

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

    def marked(regex: re.Pattern[str], alone: bool = False) -> bool:
        """A comment from an owner or collaborator carries ``regex``'s marker for the head; with ``alone``,
        the comment must not also carry a verifier marker (the top review is posted in its own comment)."""
        return any(
            c.get("author_association") in OWNER_ASSOCIATIONS
            and (c.get("user") or {}).get("type") == "User"
            and any(m.lower() == head.lower() for m in regex.findall(c.get("body") or ""))
            and not (alone and VERIFIER_RE.search(c.get("body") or ""))
            for c in comments
        )

    if not marked(VERIFIER_RE):
        why.append("no verifier PASS marker for the head commit from an owner or collaborator")
    if level == RISKY and not marked(TOP_REVIEW_RE, alone=True):
        why.append("no top-review PASS marker for the head commit from an owner or collaborator")

    return (not why), why or ["all gates passed"]


def is_paused(open_issues: Iterable[dict[str, Any]]) -> bool:
    """Whether auto-work is paused: any open issue (not a pull request) carries the ``auto-work-paused`` label.

    ``open_issues`` is a list from the issues API, which also returns pull requests (they have a ``pull_request``
    key and never pause anything). The label match is case-insensitive.
    """
    for issue in open_issues:
        if issue.get("pull_request") is not None or issue.get("state", "open") != "open":
            continue
        for label in issue.get("labels") or []:
            name = label.get("name") if isinstance(label, dict) else label
            if isinstance(name, str) and name.casefold() == PAUSE_LABEL:
                return True
    return False


def paused_from_file(path: str) -> bool:
    """``is_paused`` for JSON saved from the issues API. Fail closed: a missing, unreadable or odd file means paused."""
    try:
        data = _load(path)
    except (OSError, ValueError):
        return True
    if not isinstance(data, list) or not all(isinstance(i, dict) for i in data):
        return True
    return is_paused(data)


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
    c.add_argument("--repo", help="owner/name, to read agent definitions at --ref (without both, they need Vickey)")
    c.add_argument("--ref", help="the PR head commit sha")
    m = sub.add_parser("merge-ready")
    m.add_argument("--pr", required=True)
    m.add_argument("--files", required=True)
    m.add_argument("--checks", required=True)
    m.add_argument("--comments", required=True)
    m.add_argument("--status", required=True)
    m.add_argument("--ignore-check", action="append", default=[])
    p = sub.add_parser("paused")
    p.add_argument("--issues", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "paused":
        paused = paused_from_file(a.issues)
        _report("paused" if paused else "running", [f"open issue labelled {PAUSE_LABEL}, or the lookup failed" if paused else "none"], "Auto-work")
        print("paused" if paused else "running")
        return 0
    if a.cmd == "classify":
        reader = head_file_reader(a.repo, a.ref) if a.repo and a.ref else None
        risk, reasons = classify(_files(a.files), a.fork, reader)
        _report(risk, reasons, "PR risk")
        print(risk)
        return 0
    pr = _load(a.pr)
    base_repo = ((pr.get("base") or {}).get("repo") or {}).get("full_name") or ""
    reader = head_file_reader(base_repo, (pr.get("head") or {}).get("sha") or "")
    risk, _ = classify(_files(a.files), is_fork_pr(pr), reader)
    checks = _load(a.checks)
    ok, why = merge_ready(
        pr,
        checks.get("check_runs", []) if isinstance(checks, dict) else checks,
        _load(a.comments),
        a.ignore_check,
        risk=risk,
        status=_load(a.status),
    )
    verdict = "ready" if ok else "no-real-check" if why == [REAL_CHECK_REASON] else "not-ready"
    _report(verdict, why, f"Merge gate for #{pr.get('number')}")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

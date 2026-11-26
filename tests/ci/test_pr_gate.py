"""PR gate (KAN-216, KAN-227): classification, merge readiness and the workflow's safety rails. Fully offline."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pr_gate", ROOT / "scripts" / "pr_gate.py")
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
sys.modules["pr_gate"] = gate
_spec.loader.exec_module(gate)

CF = gate.ChangedFile
HEAD = "a" * 40
OLD = "b" * 40


@pytest.mark.parametrize(
    ("files", "fork", "expected"),
    [
        ([CF("docs/architecture.md")], False, "low-risk"),
        ([CF("tests/agent/test_x.py", "added"), CF("README.md"), CF("docs/a/b.png")], False, "low-risk"),
        ([CF("dashboard/app.js"), CF("dashboard/README.md"), CF("dashboard/index.html")], False, "low-risk"),
        ([CF("dashboard/style.css"), CF("dashboard/static/x.css")], False, "low-risk"),
        ([CF("tests/executor/test_a.py"), CF("tests/x_test.py"), CF("tests/agent/__init__.py")], False, "low-risk"),
        ([CF("tests/fixtures/site/page.html"), CF("tests/fixtures/data.json")], False, "low-risk"),
        ([CF("tests/pytest.ini")], False, "risky"),
        ([CF("tests/setup.cfg")], False, "risky"),
        ([CF("tests/tox.ini")], False, "risky"),
        ([CF("tests/pyproject.toml")], False, "risky"),
        ([CF("tests/x.py")], False, "risky"),
        ([CF("tests/browser_e2e/harness.py")], False, "risky"),
        ([CF("tests/fixtures/plugin.py")], False, "risky"),
        ([CF("tests/fixtures/pytest.ini")], False, "risky"),
        ([CF("docs/tox.ini")], False, "risky"),
        ([CF("dashboard/api/blob.js")], False, "risky"),
        ([CF("dashboard/server/db.js")], False, "risky"),
        ([CF("dashboard/scripts/blob-upload.mjs")], False, "risky"),
        ([CF("dashboard/dev-server.js")], False, "risky"),
        ([CF("dashboard/api/page.html")], False, "risky"),
        ([CF("tests/agent/test_new.py", "renamed", "tests/agent/test_old.py")], False, "low-risk"),
        ([CF("tests/agent/helper.py", "renamed", "tests/agent/test_old.py")], False, "risky"),
        ([CF("docs/old_test.md", "renamed", "tests/agent/x_test.py")], False, "risky"),
        ([CF("src/operon/agent/actions/translation.py")], False, "risky"),
        ([CF("src/operon/models/policy.py")], False, "risky"),
        ([CF("src/operon/agent/app_catalog.py")], False, "risky"),
        ([CF("src/operon/agent/actions/zoom.py")], False, "risky"),
        ([CF("src/operon/agent/hitl.py")], False, "risky"),
        ([CF("AGENTS.override.md")], False, "risky"),
        ([CF("CLAUDE.local.md")], False, "risky"),
        ([CF("tests/AGENTS.override.md")], False, "risky"),
        ([CF("docs/agents.override.md")], False, "risky"),
        ([CF("docs/GEMINI.md")], False, "risky"),
        ([CF("docs/copilot-instructions.md")], False, "risky"),
        ([CF(".cursorrules")], False, "risky"),
        ([CF(".windsurfrules")], False, "risky"),
        ([CF("docs/claude-notes.md")], False, "low-risk"),
        ([CF("src/operon/core/paths.py")], False, "risky"),
        ([CF("src/operon/api/observer.py")], False, "needs-vickey"),
        ([CF("src/operon/api/dependencies.py")], False, "risky"),
        ([CF("src/operon/store/checkpoints.py")], False, "risky"),
        ([CF("docs/a.md", "renamed", "src/operon/core/paths.py")], False, "risky"),
        ([CF("docs/a.md", "renamed", "src/operon/agent/actions/zoom.py")], False, "risky"),
        ([CF("src/operon/agent/actions/__init__.py")], False, "risky"),
        ([CF("src/operon/store/__init__.py")], False, "risky"),
        ([CF("src/operon/core/urls/__init__.py", "added")], False, "risky"),
        ([CF("src/operon/agent/loop/__init__.py", "added")], False, "risky"),
        ([CF("src/operon/core/urls/__init__.py")], False, "risky"),  # even "modified": package shadows urls.py
        ([CF("src/json.py", "added")], False, "risky"),
        ([CF("src/httpx/__init__.py", "added")], False, "risky"),
        ([CF("src/operon/agent/actions/selector.py", "modified")], False, "low-risk"),
        ([CF("src/operon/agent/actions/new_thing.py", "added")], False, "risky"),
        ([CF("src/operon/agent/actions/new_thing.py", "renamed", "src/operon/agent/actions/zoom.py")], False, "risky"),
        ([CF("src/operon/agent/actions/new_thing.py", "copied")], False, "risky"),
        ([CF("src/other/mod.py")], False, "risky"),
        ([CF("src/operon/agent/perception/service.py")], False, "risky"),
        ([CF("src/operon/agent/actions/grounding.py")], False, "risky"),
        ([CF("docs/notes.txt")], False, "risky"),
        ([CF("docs/run.sh")], False, "risky"),
        ([CF("docs/shot.PNG"), CF("docs/a.svg"), CF("docs/b.webp")], False, "low-risk"),
        ([CF("src/operon/agent/__pycache__/loop.cpython-314.pyc")], False, "risky"),
        ([CF("src/operon/agent/loop.so")], False, "risky"),
        ([CF("src/operon/x.pyd")], False, "risky"),
        ([CF("tests/fixtures/a.pyc")], False, "risky"),
        ([CF("docs/x.pth")], False, "risky"),
        ([CF("docs/__pycache__/a.md")], False, "risky"),
        ([CF("src/operon/data.json")], False, "risky"),
        ([CF("src/operon/agent/prompts.txt")], False, "low-risk"),
        ([CF("docs/CLAUDE-CODE-REFERENCE.md")], False, "risky"),
        ([CF("src/sitecustomize.py")], False, "risky"),
        ([CF("src/operon/sitecustomize.py")], False, "risky"),
        ([CF("src/operon/__init__.py")], False, "risky"),
        ([CF("src/operon/api/runtime/events.py")], False, "risky"),
        ([CF("src/operon/store/run_store.py")], False, "risky"),
        ([CF("tests/ci/test_y.py")], False, "risky"),
        ([CF("tests/executor/conftest.py")], False, "risky"),
        ([CF("tests/test_invariants.py")], False, "risky"),
        ([CF("dashboard/package.json")], False, "risky"),
        ([CF("dashboard/vercel.json")], False, "risky"),
        ([CF("dashboard/postcss.config.js")], False, "risky"),
        ([CF("evals/harness.py")], False, "needs-vickey"),
        ([CF("scripts/research_scan.py")], False, "needs-vickey"),
        ([CF("docker/Dockerfile")], False, "risky"),
        ([CF(".gitleaks.toml")], False, "risky"),
        ([CF(".bandit-baseline.json")], False, "risky"),
        ([CF("pyproject.toml")], False, "risky"),
        ([CF("uv.lock")], False, "risky"),
        ([CF("docs/.hidden.md")], False, "risky"),
        ([CF("SECURITY.md")], False, "risky"),
        ([CF("docs/CLAUDE.md")], False, "risky"),
        ([CF("CLAUDE.MD")], False, "risky"),
        ([CF(".GitHub/x.yml")], False, "risky"),
        ([CF("src/operon/Executor/desktop.py")], False, "risky"),
        ([CF("Tests/test_old.py", "removed")], False, "risky"),
        ([CF("some_new_top_level.txt")], False, "risky"),
        ([CF("src/operon/environments/live.py")], False, "risky"),
        ([CF("src/operon/api/routes/browser.py")], False, "risky"),
        ([CF("src/operon/clients/gemini.py")], False, "needs-vickey"),
        ([CF("src/operon/cli.py")], False, "risky"),
        ([CF("src/operon/agent/actions/selector.py")], False, "low-risk"),
        ([CF(".github/workflows/ci.yml")], False, "needs-vickey"),
        ([CF("docs/a.md"), CF("src/operon/executor/desktop.py")], False, "risky"),
        ([CF("src/operon/core/settings.py")], False, "needs-vickey"),
        ([CF("src/operon/agent/policy/trust_gate.py")], False, "risky"),
        ([CF("src/operon/core/urls.py")], False, "risky"),
        ([CF("CLAUDE.md")], False, "risky"),
        ([CF("evals/live_smoke.py")], False, "needs-vickey"),
        ([CF("tests/test_old.py", "removed")], False, "risky"),
        ([CF("docs/x.md", "renamed", "tests/test_old.py")], False, "risky"),
        ([CF("src/operon/agent/actions/zoom.py", "renamed", "src/operon/executor/x.py")], False, "risky"),
        ([CF("docs/architecture.md")], True, "risky"),
        ([CF("src/operon/agent/actions/selector.py")], True, "risky"),
        ([CF("scripts/x.py")], True, "needs-vickey"),
        ([CF("evals/recordings/x.json")], False, "risky"),  # not needs-vickey, but outside the low-risk kinds
        ([CF("evals/recordings/README.md")], False, "needs-vickey"),
        ([CF("src/operon/api/runtime_config.py")], False, "needs-vickey"),
        ([CF("src/operon/agent/backends/model_adapter/models.yaml")], False, "needs-vickey"),
        ([CF("src/operon/api/runtime/services.py")], False, "needs-vickey"),
        ([CF("src/operon/clients/anthropic.py")], False, "needs-vickey"),
        ([CF("tests/ci/test_pr_gate.py")], False, "needs-vickey"),
        ([], False, "risky"),
        ([CF("docs/../.github/x.yml")], False, "risky"),
    ],
)
def test_classify(files: list[Any], fork: bool, expected: str) -> None:
    assert gate.classify(files, fork)[0] == expected


def test_a_pr_with_max_files_needs_vickey() -> None:
    """The files API may have cut the list short, so the files we cannot see could be needs-vickey ones."""
    files = [CF(f"docs/f{i}.md") for i in range(gate.MAX_FILES)]
    assert gate.classify(files, False)[0] == "needs-vickey"
    assert gate.classify(files[:-1], False)[0] == "low-risk"


@pytest.mark.parametrize(
    ("path", "paid"),
    [
        ("evals/recordings/x.json", False),
        ("evals/recordings/README.md", True),
        ("evals/recordings/sub/x.py", True),
        ("evals/recordings/x.json.py", True),
    ],
)
def test_recordings_exemption_is_json_only(path: str, paid: bool) -> None:
    assert gate.is_paid_run_path(path) is paid


def _pr(**over: Any) -> dict[str, Any]:
    pr: dict[str, Any] = {
        "number": 7,
        "state": "open",
        "draft": False,
        "labels": [{"name": "low-risk"}],
        "head": {"sha": HEAD, "repo": {"full_name": "o/r"}},
        "base": {"ref": "main", "repo": {"full_name": "o/r"}},
    }
    pr.update(over)
    return pr


def _codex(sha: str = HEAD, status: str = "completed", unresolved: bool = False) -> dict[str, Any]:
    state = {"commit_sha": sha, "status": status, "has_unresolved_findings": unresolved}
    return {
        "user": {"login": gate.CODEX_BOT},
        "author_association": "NONE",
        "body": "<!-- codex-review-state:v1 " + json.dumps({"code_review": state}) + " -->",
    }


def _marker(sha: str = HEAD, assoc: str = "OWNER", kind: str = "User") -> dict[str, Any]:
    return {
        "user": {"login": "someone", "type": kind},
        "author_association": assoc,
        "body": f"<!-- operon-verifier: PASS {sha} -->",
    }


ACTIONS = {"slug": "github-actions"}
GOOD_CHECKS = [
    {"name": n, "status": "completed", "conclusion": "success", "app": ACTIONS} for n in gate.REQUIRED_CHECKS
]
GOOD_STATUS = {"state": "pending", "statuses": []}  # no statuses at all


def _ready(**kw: Any) -> tuple[bool, list[str]]:
    return gate.merge_ready(
        kw.get("pr", _pr()),
        kw.get("checks", GOOD_CHECKS),
        kw.get("comments", [_codex(), _marker()]),
        kw.get("ignore", ()),
        risk=kw.get("risk", "low-risk"),
        status=kw.get("status", GOOD_STATUS),
    )


def test_all_good_is_ready() -> None:
    assert _ready() == (True, ["all gates passed"])


def test_skipped_and_neutral_checks_count_as_good() -> None:
    checks = GOOD_CHECKS + [
        {"name": "A", "status": "completed", "conclusion": "skipped"},
        {"name": "B", "status": "completed", "conclusion": "neutral"},
    ]
    assert _ready(checks=checks)[0]


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"comments": [_codex(sha=OLD), _marker()]}, "another commit"),
        ({"comments": [_codex(unresolved=True), _marker()]}, "unresolved"),
        ({"comments": [_codex(status="in_progress"), _marker()]}, "status"),
        ({"comments": [_marker()]}, "no Codex"),
        ({"checks": GOOD_CHECKS + [{"name": "X", "status": "in_progress", "conclusion": None}]}, "still running"),
        ({"checks": GOOD_CHECKS + [{"name": "X", "status": "completed", "conclusion": "failure"}]}, "failure"),
        ({"checks": []}, "no check runs"),
        ({"checks": GOOD_CHECKS[1:]}, "required check missing: Test (Python 3.14)"),
        ({"checks": [{**GOOD_CHECKS[0], "conclusion": "skipped"}, *GOOD_CHECKS[1:]]}, "required check not successful"),
        ({"checks": [{**GOOD_CHECKS[0], "conclusion": "failure"}, *GOOD_CHECKS[1:]]}, "required check not successful"),
        ({"status": {"state": "pending", "statuses": [{"state": "pending"}]}}, "combined status"),
        ({"status": {"state": "failure", "statuses": [{"state": "failure"}]}}, "combined status"),
        ({"status": None}, "combined status not read"),
        ({"comments": [_codex(), _marker(kind="Bot")]}, "verifier"),
        ({"comments": [_codex()]}, "verifier"),
        ({"comments": [_codex(), _marker(assoc="NONE")]}, "verifier"),
        ({"comments": [_codex(), _marker(assoc="CONTRIBUTOR")]}, "verifier"),
        ({"comments": [_codex(), _marker(sha=OLD)]}, "verifier"),
        ({"pr": _pr(draft=True)}, "draft"),
        ({"pr": _pr(labels=[{"name": "risky"}])}, "does not match the label"),
        ({"pr": _pr(labels=[{"name": "low-risk"}, {"name": "risky"}])}, "labelled"),
        ({"risk": "risky"}, "fresh classification"),
        ({"checks": [{**GOOD_CHECKS[0], "app": {"slug": "evil"}}, *GOOD_CHECKS[1:]]}, "required check missing"),
        ({"checks": [{**GOOD_CHECKS[0], "app": None}, *GOOD_CHECKS[1:]]}, "required check missing"),
        ({"pr": _pr(base={"ref": "dev", "repo": {"full_name": "o/r"}})}, "does not target main"),
    ],
)
def test_not_ready(kw: dict[str, Any], reason: str) -> None:
    ok, why = _ready(**kw)
    assert not ok
    assert any(reason in w for w in why), why


def test_a_successful_status_is_ready() -> None:
    assert _ready(status={"state": "success", "statuses": [{"state": "success"}]})[0]


def test_the_gates_own_check_is_ignored() -> None:
    running = GOOD_CHECKS + [{"name": "Sweep and merge", "status": "in_progress", "conclusion": None}]
    assert not _ready(checks=running)[0]
    assert _ready(checks=running, ignore=["Sweep and merge"])[0]


# A real Codex summary comment: the braces of its state are HTML entities.
REAL_CODEX_BODY = (
    "<!-- codex-pull-request-review-summary -->\n"
    '<!-- codex-review-state:v1 &#123;"code_review":&#123;"commit_sha":"25a58541f3142fc74aba50e0e63f0905d4ec1bd5",'
    '"has_any_findings":false,"has_unresolved_findings":false,"status":"completed"&#125;,'
    '"security_review":&#123;"commit_sha":null,"has_any_findings":null,"has_unresolved_findings":null,'
    '"status":null&#125;&#125; -->\n\n'
    "<details>\n<summary><strong>Codex Review Summary</strong>: ✅ Completed</summary>"
)
REAL_SHA = "25a58541f3142fc74aba50e0e63f0905d4ec1bd5"


def _real_codex() -> dict[str, Any]:
    return {"user": {"login": gate.CODEX_BOT}, "author_association": "NONE", "body": REAL_CODEX_BODY}


def test_real_codex_comment_ready_when_sha_matches() -> None:
    pr = _pr(head={"sha": REAL_SHA, "repo": {"full_name": "o/r"}})
    assert _ready(pr=pr, comments=[_real_codex(), _marker(sha=REAL_SHA)]) == (True, ["all gates passed"])


def test_real_codex_comment_stale_sha_is_not_ready() -> None:
    ok, why = _ready(comments=[_real_codex(), _marker()])
    assert not ok
    assert any("another commit" in w for w in why), why


def test_codex_state_from_another_author_is_ignored() -> None:
    fake = _codex()
    fake["user"] = {"login": "attacker"}
    assert not _ready(comments=[fake, _marker()])[0]


def _top(sha: str = HEAD, assoc: str = "OWNER", kind: str = "User", body: str = "") -> dict[str, Any]:
    return {
        "user": {"login": "someone", "type": kind},
        "author_association": assoc,
        "body": f"Top review verdict: PASS\n<!-- operon-top-review: PASS {sha} -->{body}",
    }


RISKY_PR = _pr(labels=[{"name": "risky"}])


def test_risky_with_verifier_marker_but_no_top_review_is_not_ready() -> None:
    ok, why = _ready(pr=RISKY_PR, risk="risky", comments=[_codex(), _marker()])
    assert not ok
    assert any("top-review" in w for w in why), why


def test_risky_with_top_review_for_the_head_is_ready() -> None:
    assert _ready(pr=RISKY_PR, risk="risky", comments=[_codex(), _marker(), _top()]) == (True, ["all gates passed"])


@pytest.mark.parametrize(
    "top",
    [
        _top(sha=OLD),
        _top(kind="Bot"),
        _top(assoc="NONE"),
        _top(assoc="CONTRIBUTOR"),
    ],
    ids=["old-sha", "bot", "non-collaborator", "contributor"],
)
def test_risky_with_a_bad_top_review_is_not_ready(top: dict[str, Any]) -> None:
    ok, why = _ready(pr=RISKY_PR, risk="risky", comments=[_codex(), _marker(), top])
    assert not ok
    assert any("top-review" in w for w in why), why


def test_top_review_in_the_verifier_comment_does_not_count() -> None:
    both = _top(body=f"\n<!-- operon-verifier: PASS {HEAD} -->")
    ok, why = _ready(pr=RISKY_PR, risk="risky", comments=[_codex(), both])
    assert not ok
    assert any("top-review" in w for w in why), why
    assert not any("verifier PASS" in w for w in why), why  # it still counts as the verifier marker
    assert _ready(pr=RISKY_PR, risk="risky", comments=[_codex(), both, _top()])[0]


@pytest.mark.parametrize(
    ("labels", "risk"),
    [
        ([{"name": "needs-vickey"}], "low-risk"),
        ([{"name": "low-risk"}, {"name": "needs-vickey"}], "low-risk"),
        ([{"name": "risky"}, {"name": "needs-vickey"}], "risky"),
        ([{"name": "low-risk"}], "needs-vickey"),
        ([{"name": "risky"}], "needs-vickey"),
    ],
)
def test_needs_vickey_is_never_ready_even_with_every_marker(labels: list[dict[str, str]], risk: str) -> None:
    comments = [_codex(), _marker(), _top()]
    ok, why = _ready(pr=_pr(labels=labels), risk=risk, comments=comments)
    assert not ok
    assert any("needs Vickey" in w for w in why), why


@pytest.mark.parametrize(
    "head",
    [
        {"sha": HEAD, "repo": {"full_name": "fork/r"}},
        {"sha": HEAD, "repo": None},
        {"sha": HEAD},
    ],
    ids=["other-repo", "deleted-repo", "no-repo"],
)
def test_a_fork_pr_is_never_ready(head: dict[str, Any]) -> None:
    ok, why = _ready(pr=_pr(head=head), comments=[_codex(), _marker(), _top()])
    assert not ok
    assert why == ["PR is from a fork"]


def test_merge_ready_requires_a_risk() -> None:
    with pytest.raises(TypeError):
        gate.merge_ready(_pr(), GOOD_CHECKS, [_codex(), _marker()], status=GOOD_STATUS)  # type: ignore[call-arg]


def test_cli_low_risk_label_with_a_workflow_file_is_not_ready(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The label says low-risk, but the fresh classification of the files (a workflow) says needs-vickey."""
    paths = {n: tmp_path / f"{n}.json" for n in ("pr", "files", "checks", "comments", "status")}
    paths["pr"].write_text(json.dumps(_pr()))
    paths["files"].write_text(json.dumps([{"filename": "docs/a.md"}, {"filename": ".github/workflows/x.yml", "status": "added"}]))
    paths["checks"].write_text(json.dumps({"check_runs": GOOD_CHECKS}))
    paths["comments"].write_text(json.dumps([_codex(), _marker(), _top()]))
    paths["status"].write_text(json.dumps(GOOD_STATUS))
    args = ["merge-ready"] + [x for n, f in paths.items() for x in (f"--{n}", str(f))]
    assert gate.main(args) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "not-ready"
    assert "needs Vickey" in out.err


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = tmp_path / "files.json"
    files.write_text(json.dumps([{"filename": "docs/a.md", "status": "modified"}]))
    pr, checks, comments = (tmp_path / n for n in ("pr.json", "checks.json", "comments.json"))
    pr.write_text(json.dumps(_pr()))
    checks.write_text(json.dumps({"check_runs": GOOD_CHECKS}))
    comments.write_text(json.dumps([_codex(), _marker()]))
    assert gate.main(["classify", "--files", str(files)]) == 0
    assert capsys.readouterr().out.strip() == "low-risk"
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"state": "success", "statuses": [{"state": "success"}]}))
    args = ["merge-ready", "--pr", str(pr), "--files", str(files), "--checks", str(checks)]
    args += ["--status", str(status)]
    assert gate.main([*args, "--comments", str(comments)]) == 0
    assert capsys.readouterr().out.strip() == "ready"
    comments.write_text(json.dumps([_codex()]))
    gate.main([*args, "--comments", str(comments)])
    assert capsys.readouterr().out.strip() == "not-ready"


def _tracked_paths() -> list[str]:
    skip = {".git", ".venv", "node_modules", "__pycache__", ".var", ".claude"}
    return [
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and not skip.intersection(p.relative_to(ROOT).parts)
    ]


@pytest.mark.parametrize("pattern", gate.RISKY_PATTERNS)
def test_every_risky_pattern_matches_something(pattern: str) -> None:
    """A renamed safety-gate module must not silently fall off the risky list."""
    assert any(gate._matches(pattern, p) for p in _tracked_paths()), pattern


# No composite action is committed yet; the folder still waits for Vickey (a new action file is needs-vickey).
PAID_RUN_MAY_BE_EMPTY: tuple[str, ...] = (".github/actions/**", ".claude/**", ".mcp.json")


@pytest.mark.parametrize("pattern", gate.PAID_RUN_PATTERNS)
def test_every_paid_run_pattern_matches_a_real_file(pattern: str) -> None:
    """A renamed price table or model file must not silently fall off the needs-vickey list."""
    if pattern in PAID_RUN_MAY_BE_EMPTY:
        return
    assert any(gate._matches(pattern, p) for p in _tracked_paths()), pattern


@pytest.mark.parametrize(
    "path",
    [
        "src/operon/agent/backends/model_adapter/transports/openai_compat.py",
        "src/operon/agent/backends/model_adapter/transports/new_transport.py",
        "src/operon/api/observer.py",
        "src/operon/api/observer/__init__.py",
        "src/operon/models/usage/__init__.py",
        "src/operon/core/settings/__init__.py",
        "src/operon/api/runtime_config/__init__.py",
        "src/operon/api/runtime/services/__init__.py",
        "src/operon/agent/backends/model_adapter/registry/__init__.py",
    ],
)
def test_kan_230_paths_are_paid_run(path: str) -> None:
    assert gate.is_paid_run_path(path), path
    risk, _ = gate.classify([CF(path)], is_fork=False)
    assert risk == "needs-vickey"
    risk, _ = gate.classify([CF(path, "added")], is_fork=False)
    assert risk == "needs-vickey"


def test_claude_md_lists_every_paid_run_pattern() -> None:
    """The rule in CLAUDE.md and the gate's list must agree."""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    section = text[text.index("**Paid-run exception") :]
    section = section[: section.index("\n- **")]
    named = {m.casefold() for m in re.findall(r"`([^`]+)`", section)}
    for pattern in gate.PAID_RUN_PATTERNS:
        assert pattern in named, pattern
    assert "evals/recordings/" in named and "JSON recordings under `evals/recordings/`" in section
    assert gate.PAID_RUN_EXEMPT == ("evals/recordings/*.json",)
    assert "Every GitHub workflow or action file waits for Vickey (Vickey," in text
    assert "rules files" in section and "risky" in section.casefold()


@pytest.mark.parametrize(
    "path",
    [
        "CLAUDE.md",
        "AGENTS.md",
        "docs/CLAUDE-CODE-REFERENCE.md",
        "docs/CLAUDE.md",
        "tests/integration/CLAUDE.md",
        "src/operon/agents.md",
        "a/b/c/AGENTS.md",
    ],
)
def test_rules_files_are_risky_not_needs_vickey(path: str) -> None:
    """KAN-231: the rules files merge after the top review; they no longer wait for Vickey."""
    assert not gate.is_paid_run_path(path)
    assert gate.classify([CF(path)], is_fork=False)[0] == "risky"


def test_evals_claude_md_still_waits_because_evals_does() -> None:
    assert gate.classify([CF("evals/CLAUDE.md")], is_fork=False)[0] == "needs-vickey"


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/benchmark.yml",
        ".github/workflows/live-bench.yml",
        ".github/workflows/live-desktop-bench.yml",
        ".github/workflows/grounding.yml",
        ".github/workflows/osworld-bench.yml",
        ".github/workflows/pr-gate.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/security.yml",
        ".github/workflows/brand-new.yml",
        ".github/actions/foo/action.yml",
        ".github/actions/foo/run.sh",
        ".github/Workflows/CI.YML",
        "scripts/pr_gate.py",
        "tests/ci/test_pr_gate.py",
        "scripts/bench-runners/x.sh",
        "evals/x.py",
    ],
)
def test_every_workflow_and_action_file_and_the_merge_check_need_vickey(path: str) -> None:
    """KAN-231 (Vickey, 2026-11-26): a workflow can start a paid run on its own, so no workflow is inspected, all wait."""
    for status in ("modified", "added", "removed"):
        assert gate.classify([CF(path, status)], is_fork=False)[0] == "needs-vickey", (path, status)


def test_a_ci_change_needs_vickey_with_or_without_secrets(tmp_path: Path) -> None:
    """The content of the workflow does not matter: no regex can see gh workflow run, a reusable workflow or a wider token."""
    for patch in ("+  K: ${{ secrets.GEMINI_API_KEY }}", "+      - run: echo hello", "+  run: gh workflow run osworld-bench.yml"):
        f = tmp_path / "files.json"
        f.write_text(json.dumps([{"filename": ".github/workflows/ci.yml", "status": "modified", "patch": patch}]))
        assert gate.classify(gate._files(str(f)), False)[0] == "needs-vickey"


@pytest.mark.parametrize(
    "path",
    [
        ".github/dependabot.yml",
        ".github/pull_request_template.md",
        ".github/CODEOWNERS",
        ".github/ISSUE_TEMPLATE/bug.md",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/FUNDING.yml",
    ],
)
def test_other_github_settings_files_are_risky_not_needs_vickey(path: str) -> None:
    assert not gate.is_paid_run_path(path)
    for status in ("modified", "added"):
        assert gate.classify([CF(path, status)], is_fork=False)[0] == "risky", (path, status)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (".github/workflows/x.yml", "docs/x.yml"),
        ("docs/x.yml", ".github/workflows/x.yml"),
        (".github/actions/a/action.yml", "docs/a.md"),
        (".github/workflows/x.yml", ".github/dependabot.yml"),
        (".github/dependabot.yml", ".github/workflows/x.yml"),
    ],
)
def test_a_rename_into_or_out_of_workflows_needs_vickey(old: str, new: str) -> None:
    f = CF(new, "renamed", previous_filename=old)
    assert gate.classify([f], is_fork=False)[0] == "needs-vickey"


@pytest.mark.parametrize(
    "path",
    [".claude/settings.json", ".claude/agents/verifier.md", ".claude/skills/x/SKILL.md", ".mcp.json", ".github/workflows", ".github/actions"],
)
def test_claude_dir_mcp_json_and_bare_workflow_dirs_need_vickey(path: str) -> None:
    for status in ("modified", "added", "removed"):
        assert gate.classify([CF(path, status)], is_fork=False)[0] == "needs-vickey", (path, status)


def test_a_rename_out_of_claude_agents_needs_vickey() -> None:
    f = CF("docs/a.md", "renamed", previous_filename=".claude/agents/a.md")
    assert gate.classify([f], is_fork=False)[0] == "needs-vickey"


def test_a_rename_between_other_github_files_is_risky() -> None:
    f = CF(".github/a.yml", "renamed", previous_filename=".github/dependabot.yml")
    assert gate.classify([f], is_fork=False)[0] == "risky"


def test_files_json_status_and_previous_filename_reach_the_classifier(tmp_path: Path) -> None:
    f = tmp_path / "files.json"
    f.write_text(json.dumps([{"filename": "docs/x.yml", "status": "renamed", "previous_filename": ".github/workflows/x.yml"}]))
    assert gate.classify(gate._files(str(f)), False)[0] == "needs-vickey"


def test_the_content_check_is_gone() -> None:
    assert not hasattr(gate, "workflow_needs_vickey")


def test_pr_gate_header_says_the_label_is_advisory() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "The label is advisory" in text and "the classifier that judges a PR is never the PR's own copy" not in text


def test_claude_md_hand_merge_needs_a_fresh_classify_and_lists_the_split() -> None:
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "has first run main's `scripts/pr_gate.py classify`" in text
    assert "Waits for Vickey (`needs-vickey`)" in text and "Risky (Fable 5.2 review), not Vickey" in text
    assert "Every GitHub workflow or action file waits for Vickey" in text


def test_safety_gate_modules_are_listed_explicitly() -> None:
    for path in (
        "src/operon/agent/policy/trust_gate.py",  # trust gate, permission modes
        "src/operon/agent/policy/file_guard.py",
        "src/operon/agent/loop.py",  # OCR trust gate, confidence gate, thrash pause, secret-type gate
        "src/operon/executor/desktop.py",  # visual servo
        "src/operon/executor/linux_desktop.py",
        "src/operon/core/urls.py",  # is_web_url
        "src/operon/core/launch_env.py",
        "src/operon/environments/__init__.py",
        "src/operon/core/runtime.py",
        "src/operon/agent/control.py",
        "src/operon/api/ws_stream.py",
        "src/operon/api/stream_tokens.py",
        "src/operon/api/routes/browser.py",
        "src/operon/agent/backends/base.py",
        "src/operon/agent/actions/translation.py",
        "src/operon/models/policy.py",
        "src/operon/__init__.py",
        "src/operon/api/runtime/events.py",
        "src/operon/store/run_store.py",
        "src/operon/agent/perception/service.py",
        "src/operon/agent/actions/grounding.py",
        "src/operon/agent/app_catalog.py",
        "src/operon/agent/actions/zoom.py",
        "src/operon/agent/hitl.py",
        "src/operon/core/paths.py",
        "src/operon/api/observer.py",
        "src/operon/api/dependencies.py",
        "src/operon/store/checkpoints.py",
        "src/operon/cli.py",
        "src/operon/cloud.py",
        "src/operon/clients/gemini.py",
        "src/operon/browser/manager.py",
    ):
        assert (ROOT / path).is_file(), path
        assert gate.is_risky_path(path), path


WORKFLOW = ROOT / ".github" / "workflows" / "pr-gate.yml"


def _wf() -> dict[Any, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_uses_pull_request_and_pinned_actions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request_target" not in text.replace("Never pull_request_target", "")
    wf = _wf()
    assert "pull_request" in wf[True]  # YAML reads the key `on` as True
    for line in re.findall(r"^\s*- uses: (\S+)", text, flags=re.MULTILINE):
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", line), line
    checkouts = [s for j in wf["jobs"].values() for s in j["steps"] if "checkout" in s.get("uses", "")]
    assert checkouts
    assert all(s["with"]["persist-credentials"] is False for s in checkouts)
    assert all("permissions" in j for j in wf["jobs"].values())


def test_only_main_can_merge() -> None:
    """The sweep runs main's workflow and script only (schedule and dispatch both set github.ref)."""
    assert _wf()["jobs"]["sweep"]["if"].startswith("github.ref == 'refs/heads/main' &&")


def test_daily_note_searches_merged_by_date() -> None:
    run = next(s["run"] for s in _wf()["jobs"]["daily"]["steps"] if "run" in s)
    assert 'gh pr list --repo "$REPO" --state merged --search "merged:>=$cutoff"' in run


def test_merge_is_behind_the_variable_and_pinned_to_the_head() -> None:
    wf = _wf()
    assert wf["jobs"]["sweep"]["env"]["AUTO_MERGE"] == "${{ vars.OPERON_AUTO_MERGE }}"
    run = next(s["run"] for s in wf["jobs"]["sweep"]["steps"] if "run" in s)
    assert 'if [ "$AUTO_MERGE" = on ]' in run
    merges = re.findall(r"gh pr merge .*", run)
    assert len(merges) == 1
    assert "--squash --match-head-commit" in merges[0]
    assert "--auto" not in merges[0]
    assert "gh pr merge" not in json.dumps([j for k, j in wf["jobs"].items() if k != "sweep"])


def _sweep_run() -> str:
    return next(s["run"] for s in _wf()["jobs"]["sweep"]["steps"] if "run" in s)


def test_sweep_lists_low_risk_and_risky_but_never_needs_vickey() -> None:
    lists = re.findall(r"gh pr list [^\n]*", _sweep_run())
    labels = [label for line in lists for label in re.findall(r"--label (\S+)", line)]
    assert sorted(labels) == ["low-risk", "risky"]
    assert not any("needs-vickey" in line for line in lists)


def test_label_step_removes_the_other_two_labels() -> None:
    run = next(s["run"] for s in _wf()["jobs"]["label"]["steps"] if "run" in s)
    cases = dict(re.findall(r"^\s*([\w*-]+)\) (add=[^;]+; r1=[^;]+; r2=[^;]+) ;;", run, flags=re.MULTILINE))
    assert set(cases) == {"low-risk", "needs-vickey", "*"}
    for spec in cases.values():
        values = dict(kv.split("=") for kv in spec.split("; "))
        assert {values["add"], values["r1"], values["r2"]} == {"low-risk", "risky", "needs-vickey"}
    edit = re.findall(r"gh pr edit [^\n]*", run)
    assert '--add-label "$add" --remove-label "$r1" --remove-label "$r2"' in edit[0]


def test_label_step_drops_a_stale_paid_run_label() -> None:
    """The label used to be called paid-run: old PRs lose it, and the label is never added again."""
    run = next(s["run"] for s in _wf()["jobs"]["label"]["steps"] if "run" in s)
    edit = re.findall(r"gh pr edit [^\n]*", run)
    assert len(edit) == 2
    assert edit[1].endswith("--remove-label paid-run")
    assert "gh pr view" in run and "grep -qx paid-run" in run
    assert not any("--add-label paid-run" in e for e in edit)
    assert "add=paid-run" not in run and "--add-label" not in edit[1]


@pytest.mark.parametrize("risk", ["low-risk", "risky"])
def test_a_stale_paid_run_label_still_blocks_a_merge(risk: str) -> None:
    ok, why = _ready(pr=_pr(labels=[{"name": risk}, {"name": "paid-run"}]), risk=risk, comments=[_codex(), _marker(), _top()])
    assert not ok and any("needs Vickey" in w for w in why), why


def test_no_pr_controlled_text_in_run_scripts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for run in re.findall(r"run: \|\n((?:\s{10,}.*\n)+)", text):
        for bad in ("github.event.pull_request.title", "head_ref", "github.head_ref", "comment.body", "${{"):
            assert bad not in run, bad


# KAN-229: one word ("pause") stops all auto-work: an open issue labelled auto-work-paused.
def _issue(*labels: str, **extra: Any) -> dict[str, Any]:
    return {"number": 1, "state": "open", "labels": [{"name": n} for n in labels], **extra}


def test_paused_when_an_open_issue_has_the_label() -> None:
    assert gate.is_paused([_issue("auto-work-paused")])
    assert gate.is_paused([_issue("bug"), _issue("bug", "Auto-Work-Paused")])  # case-insensitive


def test_not_paused_without_an_open_labelled_issue() -> None:
    assert not gate.is_paused([])
    assert not gate.is_paused([_issue("bug"), _issue()])
    assert not gate.is_paused([_issue("auto-work-paused", state="closed")])


def test_a_pull_request_with_the_label_does_not_pause() -> None:
    """The issues API also lists pull requests; they carry a pull_request key."""
    assert not gate.is_paused([_issue("auto-work-paused", pull_request={"url": "x"})])
    assert gate.is_paused([_issue("auto-work-paused", pull_request={"url": "x"}), _issue("auto-work-paused")])


def test_pause_lookup_failure_counts_as_paused(tmp_path: Path) -> None:
    assert gate.paused_from_file(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert gate.paused_from_file(str(bad))
    bad.write_text('{"message": "rate limited"}', encoding="utf-8")
    assert gate.paused_from_file(str(bad))
    bad.write_text("[1]", encoding="utf-8")
    assert gate.paused_from_file(str(bad))


def test_cli_paused_prints_paused_or_running(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "issues.json"
    f.write_text(json.dumps([_issue("auto-work-paused")]), encoding="utf-8")
    assert gate.main(["paused", "--issues", str(f)]) == 0
    assert capsys.readouterr().out.strip() == "paused"
    f.write_text("[]", encoding="utf-8")
    assert gate.main(["paused", "--issues", str(f)]) == 0
    assert capsys.readouterr().out.strip() == "running"


def test_sweep_checks_the_pause_before_it_merges_or_notes() -> None:
    run = _sweep_run()
    check = run.index("pr_gate.py paused")
    assert check < run.index("gh pr merge") and check < run.index("gh pr comment") and check < run.index("gh pr list")
    assert "Auto-work is paused" in run
    assert "|| echo paused" in run and "rm -f" in run  # fail closed
    assert _wf()["jobs"]["sweep"]["permissions"]["issues"] == "read"


def test_sweep_checks_the_pause_again_before_each_merge_and_note() -> None:
    loop = _sweep_run().split("while read -r n", 1)[1]
    again = loop.index('"$(pause_state)"')
    assert again < loop.index("gh pr merge") and again < loop.index("gh pr comment")
    assert loop.index('[ "$verdict" = ready ] || continue') < again


FAKE_GH = """#!/bin/sh
echo "$@" >> "$FAKE_GH_LOG"
case "$1" in
  api) [ "$FAKE_GH_MODE" = fail ] && exit 1; cat "$FAKE_GH_BODY" ;;
esac
exit 0
"""


@pytest.mark.skipif(sys.platform == "win32", reason="shells out to bash (KAN-195)")
@pytest.mark.parametrize(
    ("mode", "body", "paused"),
    [
        ("fail", "[]", True),
        ("ok", '{"message": "Bad credentials"}', True),
        ("ok", json.dumps([_issue("auto-work-paused")]), True),
        ("ok", json.dumps([_issue("auto-work-paused", pull_request={"url": "x"})]), False),
        ("ok", "[]", False),
    ],
)
def test_sweep_pause_prefix_runs_with_a_fake_gh(tmp_path: Path, mode: str, body: str, paused: bool) -> None:
    import os
    import shutil
    import subprocess

    bash, jq = shutil.which("bash"), shutil.which("jq")
    if not bash or not jq:
        pytest.skip("needs bash and jq")
    run = _sweep_run()
    prefix = run[: run.index("# --- end of the pause check ---")]
    script = tmp_path / "prefix.sh"
    script.write_text(prefix + "gh pr list --label low-risk\necho reached-the-sweep\n", encoding="utf-8")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake = fakebin / "gh"
    fake.write_text(FAKE_GH, encoding="utf-8")
    fake.chmod(0o755)
    (fakebin / "python3").symlink_to(sys.executable)  # the script calls python3
    (tmp_path / "body.json").write_text(body, encoding="utf-8")
    log = tmp_path / "gh.log"
    env = {
        "PATH": os.pathsep.join([str(fakebin), str(Path(jq).parent), str(Path(bash).parent)]),
        "RUNNER_TEMP": str(tmp_path),
        "REPO": "o/r",
        "FAKE_GH_MODE": mode,
        "FAKE_GH_BODY": str(tmp_path / "body.json"),
        "FAKE_GH_LOG": str(log),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    done = subprocess.run([bash, str(script)], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    calls = log.read_text(encoding="utf-8") if log.exists() else ""
    if paused:
        assert "Auto-work is paused" in done.stdout
        assert "reached-the-sweep" not in done.stdout
        assert "pr list" not in calls and "label create" not in calls
    else:
        assert "reached-the-sweep" in done.stdout
        assert "pr list" in calls


def test_daily_note_says_paused() -> None:
    run = next(s["run"] for s in _wf()["jobs"]["daily"]["steps"] if "run" in s)
    assert "pr_gate.py paused" in run and "Paused" in run and "|| pause=paused" in run
    assert "\\nPause: %s\\n" in run and '"$mode" "$pause"' in run


def test_claude_md_states_the_pause_rule() -> None:
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "While an open issue labelled `auto-work-paused` exists" in text

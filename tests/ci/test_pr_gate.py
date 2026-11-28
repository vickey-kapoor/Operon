"""PR gate (KAN-216, KAN-227): classification, merge readiness and the workflow's safety rails. Fully offline."""

from __future__ import annotations

import base64
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
        "body": "Real check: https://github.com/vickey-kapoor/Operon/actions/runs/1 showed X",
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
PAID_RUN_MAY_BE_EMPTY: tuple[str, ...] = (".github/actions/**",)


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


SAFE_AGENT = "---\nname: x\ndescription: Does a thing.\nmodel: sonnet\ntools: Read, Grep\n---\n\nBody.\n"


def _read(text: str | None) -> Any:
    return lambda path: text


@pytest.mark.parametrize(
    "path",
    [
        ".claude/settings.json",
        ".claude/skills/x/SKILL.md",
        ".claude/agents/sub/x.md",
        ".claude/agents/x.txt",
        ".claude/agents",
        ".claude/agents/../settings.json",
        ".Claude/Settings.json",
        "src/.claude/settings.json",
        ".mcp.json",
        "docs/.mcp.json",
        ".github/workflows",
        ".github/actions",
    ],
)
def test_claude_dir_mcp_json_and_bare_workflow_dirs_need_vickey(path: str) -> None:
    for status in ("modified", "added", "removed"):
        assert gate.classify([CF(path, status)], False, _read(SAFE_AGENT))[0] == "needs-vickey", (path, status)


def test_a_safe_agent_file_is_risky() -> None:
    for status in ("modified", "added"):
        assert gate.classify([CF(".claude/agents/x.md", status)], False, _read(SAFE_AGENT))[0] == "risky"
    assert gate.classify([CF(".Claude/Agents/x.MD")], False, _read(SAFE_AGENT))[0] == "risky"
    assert gate.classify([CF(".claude/agents/X.md")], False, _read(SAFE_AGENT))[0] == "needs-vickey"  # name is case-sensitive


def test_a_deleted_agent_file_is_risky_without_reading_it() -> None:
    assert gate.classify([CF(".claude/agents/x.md", "removed")], False)[0] == "risky"


BAD_AGENTS = {
    "hooks": "---\nname: x\nhooks:\n---\nb",
    "hooks_inline": "---\nname: x\nhooks: {}\n---\nb",
    "mcpServers": "---\nname: x\nmcpServers: a\n---\nb",
    "permissionMode": "---\nname: x\npermissionMode: bypassPermissions\n---\nb",
    "indented": "---\nname: x\n  hooks: a\n---\nb",
    "duplicate": "---\nname: x\nname: y\n---\nb",
    "comment": "---\nname: x\n# hooks\n---\nb",
    "blank": "---\nname: x\n\n---\nb",
    "no_frontmatter": "name: x\nBody\n",
    "unterminated": "---\nname: x\nmodel: sonnet\nBody\n",
    "tools_shell": "---\nname: x\ntools: Bash; rm -rf\n---\nb",
    "tools_empty_item": "---\nname: x\ntools: Read,,Grep\n---\nb",
    "empty_value": "---\nname:\n---\nb",
    "crlf": "---\r\nname: x\r\n---\r\nb",
    "unicode_separator": "---\nname: x hooks: a\n---\nb",
    "indented_allowed": "---\n  name: x\n---\nb",
    "forbidden_key_with_allowed_pair": "---\nmcpServers: {x: {command: curl, name: y}}\n---\nb",
    "body_separator": "---\nname: x\n---\nb\n---\nhooks: a\n---\n",
    "body_separator_trailing_space": "---\nname: x\n---\nb\n--- \nc",
    "body_hooks": "---\nname: x\n---\nhooks: {}\n",
    "body_mcp": "---\nname: x\n---\nmcpServers: a\n",
    "too_big": SAFE_AGENT + "x" * 70000,
}


@pytest.mark.parametrize("name", sorted(BAD_AGENTS))
def test_an_agent_file_beyond_plain_frontmatter_needs_vickey(name: str) -> None:
    assert not gate.agent_file_is_safe(BAD_AGENTS[name])
    assert gate.classify([CF(".claude/agents/x.md")], False, _read(BAD_AGENTS[name]))[0] == "needs-vickey"


def test_agent_file_is_safe_accepts_every_allowed_key() -> None:
    text = "---\n" + "\n".join(f"{k}: v" for k in gate.AGENT_FRONTMATTER_KEYS) + "\n---\n"
    assert gate.agent_file_is_safe(text.replace("tools: v", "tools: Read, mcp__a__b"))


def test_agent_content_that_cannot_be_read_needs_vickey() -> None:
    assert gate.classify([CF(".claude/agents/x.md")], False, _read(None))[0] == "needs-vickey"
    assert gate.classify([CF(".claude/agents/x.md")], False)[0] == "needs-vickey"  # no reader: fail closed


def test_head_file_reader_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert gate.head_file_reader("not a repo", HEAD)(".claude/agents/x.md") is None
    for bad_ref in ("main", HEAD[:39], HEAD + "a"):
        assert gate.head_file_reader("o/r", bad_ref, lambda e: _fake_gh()(e.replace(bad_ref, HEAD)))(
            ".claude/agents/x.md"
        ) is None
    assert gate.head_file_reader("o/r", HEAD, _fake_gh())(".claude/agents/x.md") == SAFE_AGENT

    def missing(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError

    monkeypatch.setattr(gate.subprocess, "run", missing)
    assert gate.head_file_reader("o/r", HEAD)(".claude/agents/x.md") is None


def test_dot_md_alone_is_not_a_plain_agent_path() -> None:
    assert not gate.is_plain_agent_path(".claude/agents/.md")


T_ROOT, T_CLAUDE, T_AGENTS, B_SHA = "1" * 40, "2" * 40, "3" * 40, "4" * 40


def _entry(path: str, sha: str, mode: str = "040000", typ: str = "tree") -> dict[str, Any]:
    return {"path": path, "mode": mode, "type": typ, "sha": sha}


def _fake_gh(
    *,
    agents_entry: dict[str, Any] | None = None,
    file_entry: dict[str, Any] | None = None,
    truncated: bool = False,
    content: bytes | str = SAFE_AGENT,
    blob: dict[str, Any] | None = None,
) -> Any:
    file_entry = file_entry or _entry("x.md", B_SHA, "100644", "blob")
    raw = content if isinstance(content, bytes) else content.encode()
    objs: dict[str, dict[str, Any]] = {
        f"repos/o/r/git/commits/{HEAD}": {"tree": {"sha": T_ROOT}},
        f"repos/o/r/git/trees/{T_ROOT}": {"tree": [_entry(".claude", T_CLAUDE), _entry("docs", "5" * 40)]},
        f"repos/o/r/git/trees/{T_CLAUDE}": {"tree": [agents_entry or _entry("agents", T_AGENTS)]},
        f"repos/o/r/git/trees/{T_AGENTS}": {"tree": [file_entry] if file_entry else [], "truncated": truncated},
        f"repos/o/r/git/blobs/{B_SHA}": blob
        or {"encoding": "base64", "size": len(raw), "content": base64.b64encode(raw).decode()},
    }
    return objs.get


def _classify_with(fetch: Any, path: str = ".claude/agents/x.md") -> str:
    reader = gate.head_file_reader("o/r", HEAD, fetch)
    return gate.classify([CF(path)], False, reader)[0]


def test_a_regular_blob_agent_file_is_read_and_risky() -> None:
    assert gate.head_file_reader("o/r", HEAD, _fake_gh())(".claude/agents/x.md") == SAFE_AGENT
    assert _classify_with(_fake_gh()) == "risky"


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [
        ("symlink", {"file_entry": _entry("x.md", B_SHA, "120000", "blob")}),
        ("executable", {"file_entry": _entry("x.md", B_SHA, "100755", "blob")}),
        ("submodule", {"file_entry": _entry("x.md", B_SHA, "160000", "commit")}),
        ("agents_dir_symlink", {"agents_entry": _entry("agents", T_AGENTS, "120000", "blob")}),
        ("agents_dir_wrong_name_case", {"agents_entry": _entry("Agents", T_AGENTS)}),
        ("file_case_mismatch", {"file_entry": _entry("X.md", B_SHA, "100644", "blob")}),
        ("missing_entry", {"file_entry": _entry("y.md", B_SHA, "100644", "blob")}),
        ("truncated_tree", {"truncated": True}),
        ("bad_base64", {"blob": {"encoding": "base64", "size": 3, "content": "!!!"}}),
        ("bad_utf8", {"content": b"\xff\xfe"}),
        ("too_big", {"content": SAFE_AGENT + "x" * 70000}),
        ("wrong_size", {"blob": {"encoding": "base64", "size": 1, "content": base64.b64encode(b"abc").decode()}}),
    ],
)
def test_an_agent_file_that_is_not_an_ordinary_blob_needs_vickey(kind: str, kwargs: dict[str, Any]) -> None:
    assert _classify_with(_fake_gh(**kwargs)) == "needs-vickey", kind


def test_two_tree_entries_with_the_same_name_need_vickey() -> None:
    fetch = _fake_gh()
    twin = _entry("x.md", "6" * 40, "100644", "blob")

    def dup(endpoint: str) -> Any:
        obj = fetch(endpoint)
        if endpoint.endswith(T_AGENTS):
            return {"tree": obj["tree"] + [twin]}
        return obj

    assert gate.head_file_reader("o/r", HEAD, dup)(".claude/agents/x.md") is None


def test_a_blob_size_that_differs_from_its_content_needs_vickey() -> None:
    raw = SAFE_AGENT.encode()
    blob = {"encoding": "base64", "size": len(raw) + 1, "content": base64.b64encode(raw).decode()}
    assert gate.head_file_reader("o/r", HEAD, _fake_gh(blob=blob))(".claude/agents/x.md") is None


@pytest.mark.parametrize("name", ["other", "X", "x y", "x;"])
def test_an_agent_name_must_equal_the_file_stem(name: str) -> None:
    text = SAFE_AGENT.replace("name: x", f"name: {name}")
    assert gate.classify([CF(".claude/agents/x.md")], False, _read(text))[0] == "needs-vickey"
    no_name = SAFE_AGENT.replace("name: x\n", "")
    assert gate.classify([CF(".claude/agents/x.md")], False, _read(no_name))[0] == "needs-vickey"


def test_a_failed_fetch_needs_vickey() -> None:
    assert _classify_with(lambda endpoint: None) == "needs-vickey"


@pytest.mark.parametrize("name", ["verifier.md", "Verifier.MD"])
def test_the_verifier_agent_always_needs_vickey(name: str) -> None:
    f = CF(f".claude/agents/{name}")
    assert gate.classify([f], False, _read(SAFE_AGENT))[0] == "needs-vickey"
    assert gate.classify([CF(f".claude/agents/{name}", "removed")], False)[0] == "needs-vickey"


def test_a_rename_out_of_claude_agents_is_at_least_risky() -> None:
    f = CF("docs/a.md", "renamed", previous_filename=".claude/agents/a.md")
    assert gate.classify([f], False, _read(SAFE_AGENT))[0] == "risky"


def test_a_rename_into_claude_needs_vickey() -> None:
    f = CF(".claude/settings.json", "renamed", previous_filename="docs/a.md")
    assert gate.classify([f], False, _read(SAFE_AGENT))[0] == "needs-vickey"


def test_a_rename_into_agents_reads_the_new_content() -> None:
    f = CF(".claude/agents/x.md", "renamed", previous_filename="docs/a.md")
    assert gate.classify([f], False, _read(SAFE_AGENT))[0] == "risky"
    assert gate.classify([f], False, _read(BAD_AGENTS["hooks"]))[0] == "needs-vickey"


PIPELINE_SKILL = ".claude/skills/pipeline/SKILL.md"


def test_the_pipeline_skill_still_needs_vickey() -> None:
    f = CF(PIPELINE_SKILL, "modified")
    assert gate.classify([f], False, _read(SAFE_AGENT))[0] == "needs-vickey"


def test_every_tracked_claude_file_is_a_safe_plain_agent_definition() -> None:
    import subprocess

    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    staged = subprocess.run(["git", "ls-files", "-s", "-z", ".claude"], cwd=ROOT, capture_output=True, check=False)
    for rec in filter(None, staged.stdout.decode().split("\0")):
        meta, _, name = rec.partition("\t")
        assert meta.split()[0] == "100644", f"{name}: a tracked .claude file must be an ordinary file (mode 100644)"
    tracked = [p for p in out.stdout.decode().split("\0") if p and (".claude" in p.split("/") or p.endswith(".mcp.json"))]
    for p in tracked:
        if p == PIPELINE_SKILL:
            continue
        assert gate.is_plain_agent_path(p), f"{p}: only plain agent definitions may be tracked under .claude/"
        text = (ROOT / p).read_text(encoding="utf-8")
        assert gate.agent_file_is_safe(text), p
        assert gate.agent_name_matches(text, p), p
    assert any(p.startswith(".claude/agents/") for p in tracked)


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
    # Both paths re-check the pause before acting: the no-real-check note, then (after the ready filter) the merge.
    first = loop.index('[ "$verdict" = no-real-check ]')
    assert first < again < loop.index('[ "$verdict" = ready ] || continue')
    ready_path = loop.split('[ "$verdict" = ready ] || continue', 1)[1]
    assert ready_path.index('"$(pause_state)"') < ready_path.index("gh pr merge")


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


# KAN-298: every PR the gate merges needs a "Real check:" line with a link to a real $0 run.
RC_LINK = "https://github.com/o/r/actions/runs/1"


def test_merge_needs_a_real_check_line() -> None:
    for body in (None, "", "Summary only", 5):
        assert _ready(pr=_pr(body=body)) == (False, [gate.REAL_CHECK_REASON])
    assert _ready(pr=_pr(body=f"Real check: {RC_LINK} showed X"))[0] is True


def test_real_check_line_without_a_link_does_not_count() -> None:
    assert not gate.has_real_check("Real check: ran it by hand")
    assert not gate.has_real_check("Real check: https://")  # nothing after the scheme


def test_real_check_in_an_html_comment_does_not_count() -> None:
    hint = f"Real check: <!-- Required: a link, e.g. Real check: {RC_LINK} -->"
    assert not gate.has_real_check(hint)
    assert not gate.has_real_check(f"<!--\nReal check: {RC_LINK}\n-->")
    assert not gate.has_real_check(f"text\n<!-- unclosed\nReal check: {RC_LINK}")


def test_real_check_forms_that_count() -> None:
    assert gate.has_real_check(f"- **Real check:** {RC_LINK}")
    assert gate.has_real_check(f"**Real check**: {RC_LINK}")
    assert gate.has_real_check(f"## Real check\nReal check: see {RC_LINK}")
    assert not gate.has_real_check(f"> REAL CHECK: {RC_LINK}")  # a blockquote line never counts
    assert gate.has_real_check(f"<!-- hint -->\nReal check: {RC_LINK}")


def test_real_check_needs_the_colon_form_and_the_link_on_the_same_line() -> None:
    assert not gate.has_real_check(f"Real checks are good {RC_LINK}")
    assert not gate.has_real_check(f"Real check\n{RC_LINK}")
    assert not gate.has_real_check(f"Real check:\n{RC_LINK}")
    assert not gate.has_real_check(f"See the Real check: {RC_LINK}")  # must start the line


def _cli_verdict(tmp_path: Path, pr: dict[str, Any], comments: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]) -> str:
    def w(name: str, data: Any) -> str:
        f = tmp_path / name
        f.write_text(json.dumps(data), encoding="utf-8")
        return str(f)

    argv = [
        "merge-ready", "--pr", w("pr.json", pr), "--files", w("files.json", [{"filename": "docs/a.md"}]),
        "--checks", w("checks.json", GOOD_CHECKS), "--comments", w("comments.json", comments),
        "--status", w("status.json", GOOD_STATUS),
    ]  # fmt: skip
    assert gate.main(argv) == 0
    return capsys.readouterr().out.strip()


def test_cli_prints_no_real_check_only_when_it_is_the_only_reason(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = [_codex(), _marker()]
    assert _cli_verdict(tmp_path, _pr(body=""), good, capsys) == "no-real-check"
    assert _cli_verdict(tmp_path, _pr(body="", draft=True), good, capsys) == "not-ready"
    assert _cli_verdict(tmp_path, _pr(body=""), [_codex()], capsys) == "not-ready"


def test_sweep_notes_a_missing_real_check_once_per_head_and_never_merges_it() -> None:
    loop = _sweep_run().split("while read -r n", 1)[1]
    branch = loop.split('[ "$verdict" = no-real-check ]', 1)[1].split('[ "$verdict" = ready ] || continue', 1)[0]
    assert "operon-gate-real-check: $sha" in branch
    assert "github-actions[bot]" in branch and "contains($m)" in branch  # dedupe on the bot and the head sha
    assert "--body-file" in branch and "gh pr comment" in branch
    assert branch.index('"$(pause_state)"') < branch.index("gh pr comment")
    assert "gh pr merge" not in branch and "--add-label" not in branch
    assert "\n    continue\n" in branch


def test_real_check_inside_a_code_block_or_indented_does_not_count() -> None:
    assert not gate.has_real_check(f"```\nReal check: {RC_LINK}\n```")
    assert not gate.has_real_check(f"~~~\nReal check: {RC_LINK}\n~~~")
    assert not gate.has_real_check(f"```text\nReal check: {RC_LINK}")  # unclosed fence runs to the end
    assert not gate.has_real_check(f"    Real check: {RC_LINK}")
    assert gate.has_real_check(f"```\ncode\n```\nReal check: {RC_LINK}")  # after the fence closes


def test_real_check_accepts_numbered_and_plus_list_prefixes() -> None:
    for prefix in ("1. ", "2) ", "+ ", "10. **", "- 3. "):
        suffix = "**" if prefix.endswith("**") else ""
        text = f"{prefix}Real check:{suffix} {RC_LINK}"
        assert gate.has_real_check(text), text


def test_real_check_fences_close_only_on_a_matching_fence() -> None:
    line = f"Real check: {RC_LINK}"
    assert not gate.has_real_check(f"````\n```\n{line}\n````")  # inner ``` does not close a 4-backtick fence
    assert not gate.has_real_check(f"```\n~~~\n{line}\n```")  # ~~~ does not close a ``` fence
    assert not gate.has_real_check(f"```\n```x\n{line}\n```")  # a closing fence has no info string
    assert gate.has_real_check(f"````\n```\ninner\n````\n{line}")  # properly closed long fence, line after counts
    assert gate.has_real_check(f"```\ncode\n`````\n{line}")  # a longer fence closes too
    assert gate.has_real_check(f"``` a`b\n{line}")  # a backtick info string with a backtick is not a fence


def test_real_check_indent_counts_tabs_as_columns() -> None:
    line = f"Real check: {RC_LINK}"
    assert not gate.has_real_check(f" \t{line}")  # 1 space + tab = 4 columns
    assert not gate.has_real_check(f"\t{line}")
    assert not gate.has_real_check(f"  \t{line}")
    assert gate.has_real_check(f"   {line}")  # 3 columns is still a paragraph


def test_real_check_backtick_info_string_line_does_not_open_a_fence() -> None:
    line = f"Real check: {RC_LINK}"
    assert not gate.has_real_check(f"```code``` {line}")  # not a fence, and not a Real check line either
    assert gate.has_real_check(f"```code``` x\n{line}")  # no fence was opened, so the next line counts


def test_real_check_ignores_fences_inside_quotes_and_lists() -> None:
    line = f"Real check: {RC_LINK}"
    assert not gate.has_real_check(f"> ```text\n> {line}\n> ```")
    assert not gate.has_real_check(f"- ```text\n  {line}\n  ```")
    assert not gate.has_real_check(f"1. ```text\n{line}\n```")
    assert not gate.has_real_check(f"> - ```text\n> - {line}\n> - ```")  # nested quote and list
    assert not gate.has_real_check(f"> {line}")
    assert not gate.has_real_check(f"- > {line}")
    assert gate.has_real_check(f"> ```\n> x\n> ```\n- {line}")  # closed, and a list line counts
    assert gate.has_real_check(f"- ```\n  x\n  ```\n{line}")


def test_real_check_lines_split_only_on_cr_lf() -> None:
    line = f"Real check: {RC_LINK}"
    assert gate.has_real_check(f"intro\r\n{line}")
    assert gate.has_real_check(f"intro\r{line}")
    for sep in ("\x0b", "\x0c", "\x85", "\u2028", "\u2029"):
        assert not gate.has_real_check(f"> quote{sep}{line}"), repr(sep)  # still one (quoted) line
        assert not gate.has_real_check(f"```{sep}{line}\n```"), repr(sep)


def test_real_check_html_comments_only_count_outside_fences() -> None:
    line = f"Real check: {RC_LINK}"
    assert not gate.has_real_check(f"<!-- hint\n{line}\n-->")  # a comment spanning lines
    assert not gate.has_real_check(f"```\n<!--\n```\n<!-- x\n{line}")  # a fence's <!-- opens nothing; this one is unclosed
    assert gate.has_real_check(f"```\n<!-- not a comment\n```\n{line}")  # a <!-- inside a fence hides nothing after it
    assert gate.has_real_check(f"<!-- a --> <!-- b -->\n{line}")
    assert gate.has_real_check(f"<!-- ```\n-->\n{line}")  # comment text never opens a fence


def test_real_check_nested_list_item_indented_four_columns_counts() -> None:
    line = f"Real check: {RC_LINK}"
    assert gate.has_real_check(f"- Proof\n    - {line}")
    assert gate.has_real_check(f"1. Proof\n    2. {line}")
    assert gate.has_real_check(f"- Proof\n\t- {line}")
    assert not gate.has_real_check(f"- Proof\n\n    {line}")  # plain indented code is still code

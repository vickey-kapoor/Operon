"""PR gate (KAN-216): classification, merge readiness and the workflow's safety rails. Fully offline."""

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
        ([CF("dashboard/app.js"), CF("dashboard/README.md")], False, "low-risk"),
        ([CF("tests/ci/test_y.py")], False, "risky"),
        ([CF("tests/executor/conftest.py")], False, "risky"),
        ([CF("tests/test_invariants.py")], False, "risky"),
        ([CF("dashboard/package.json")], False, "risky"),
        ([CF("dashboard/vercel.json")], False, "risky"),
        ([CF("dashboard/postcss.config.js")], False, "risky"),
        ([CF("evals/harness.py")], False, "risky"),
        ([CF("scripts/research_scan.py")], False, "risky"),
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
        ([CF("src/operon/clients/gemini.py")], False, "risky"),
        ([CF("src/operon/cli.py")], False, "risky"),
        ([CF("src/operon/agent/actions/zoom.py")], False, "low-risk"),
        ([CF(".github/workflows/ci.yml")], False, "risky"),
        ([CF("docs/a.md"), CF("src/operon/executor/desktop.py")], False, "risky"),
        ([CF("src/operon/core/settings.py")], False, "risky"),
        ([CF("src/operon/agent/policy/trust_gate.py")], False, "risky"),
        ([CF("src/operon/core/urls.py")], False, "risky"),
        ([CF("CLAUDE.md")], False, "risky"),
        ([CF("evals/live_smoke.py")], False, "risky"),
        ([CF("tests/test_old.py", "removed")], False, "risky"),
        ([CF("docs/x.md", "renamed", "tests/test_old.py")], False, "risky"),
        ([CF("src/operon/agent/actions/zoom.py", "renamed", "src/operon/executor/x.py")], False, "risky"),
        ([CF("docs/architecture.md")], True, "risky"),
        ([], False, "risky"),
        ([CF("docs/../.github/x.yml")], False, "risky"),
    ],
)
def test_classify(files: list[Any], fork: bool, expected: str) -> None:
    assert gate.classify(files, fork)[0] == expected


def test_a_pr_with_3000_files_is_risky() -> None:
    files = [CF(f"docs/f{i}.md") for i in range(3000)]
    assert gate.classify(files, False)[0] == "risky"


def _pr(**over: Any) -> dict[str, Any]:
    pr: dict[str, Any] = {
        "number": 7,
        "state": "open",
        "draft": False,
        "labels": [{"name": "low-risk"}],
        "head": {"sha": HEAD, "repo": {"full_name": "o/r"}},
        "base": {"repo": {"full_name": "o/r"}},
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


GOOD_CHECKS = [{"name": n, "status": "completed", "conclusion": "success"} for n in gate.REQUIRED_CHECKS]
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
        ({"pr": _pr(labels=[{"name": "risky"}])}, "labelled"),
        ({"pr": _pr(labels=[{"name": "low-risk"}, {"name": "risky"}])}, "labelled"),
        ({"risk": "risky"}, "fresh classification"),
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


def test_no_pr_controlled_text_in_run_scripts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for run in re.findall(r"run: \|\n((?:\s{10,}.*\n)+)", text):
        for bad in ("github.event.pull_request.title", "head_ref", "github.head_ref", "comment.body", "${{"):
            assert bad not in run, bad

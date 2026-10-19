"""File guard: model-chosen read_text / upload paths need the user's say-so."""

from __future__ import annotations

from pathlib import Path

import pytest

from operon.agent.policy.file_guard import blocked_file_path
from operon.models.policy import ActionType, AgentAction


def _action(action_type: ActionType, path: str) -> AgentAction:
    return AgentAction(action_type=action_type, text=path, selector="main")


@pytest.fixture(autouse=True)
def _no_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_ALLOWED_FILE_ROOTS", raising=False)


@pytest.mark.parametrize("action_type", [ActionType.READ_TEXT, ActionType.UPLOAD_FILE, ActionType.UPLOAD_FILE_NATIVE])
def test_path_not_in_intent_is_blocked(tmp_path: Path, action_type: ActionType) -> None:
    action = _action(action_type, "/home/me/.ssh/id_rsa")
    assert blocked_file_path(action, intent="Upload my resume", run_dir=tmp_path) == "/home/me/.ssh/id_rsa"


def test_path_named_in_intent_is_allowed(tmp_path: Path) -> None:
    action = _action(ActionType.READ_TEXT, "/tmp/out/article.txt")
    intent = "Read the GPT-4 article and save it to /tmp/out/article.txt."
    assert blocked_file_path(action, intent=intent, run_dir=tmp_path) is None


def test_path_with_spaces_named_in_intent_is_allowed(tmp_path: Path) -> None:
    action = _action(ActionType.UPLOAD_FILE, "/home/Jane Doe/cv final.pdf")
    intent = 'Upload "/home/Jane Doe/cv final.pdf" to the job form'
    assert blocked_file_path(action, intent=intent, run_dir=tmp_path) is None


def test_intent_separators_are_normalized(tmp_path: Path) -> None:
    action = _action(ActionType.READ_TEXT, "C:\\notes\\a.txt")
    assert blocked_file_path(action, intent="save to C:/notes/a.txt", run_dir=tmp_path) is None


@pytest.mark.parametrize("path", ["/tmp/out", "/tmp/out/article.tx", "/tmp/out/article.txt/../../etc/passwd"])
def test_partial_or_extended_intent_path_is_blocked(tmp_path: Path, path: str) -> None:
    # Only the whole path the user wrote counts — not a prefix, a truncation or an extension of it.
    action = _action(ActionType.READ_TEXT, path)
    assert blocked_file_path(action, intent="save to /tmp/out/article.txt", run_dir=tmp_path) == path


def test_path_inside_run_dir_is_allowed(tmp_path: Path) -> None:
    action = _action(ActionType.READ_TEXT, str(tmp_path / "step_3" / "extract.txt"))
    assert blocked_file_path(action, intent="summarize the page", run_dir=tmp_path) is None


def test_run_dir_escape_is_blocked(tmp_path: Path) -> None:
    path = str(tmp_path / ".." / "elsewhere.txt")
    action = _action(ActionType.READ_TEXT, path)
    assert blocked_file_path(action, intent="summarize the page", run_dir=tmp_path) == path


def test_allowed_roots_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = tmp_path / "docs"
    monkeypatch.setenv("OPERON_ALLOWED_FILE_ROOTS", f"{tmp_path / 'other'}, {docs}")
    allowed = _action(ActionType.UPLOAD_FILE, str(docs / "resume.pdf"))
    outside = _action(ActionType.UPLOAD_FILE, str(tmp_path / "docsX" / "resume.pdf"))
    run_dir = tmp_path / "run"
    assert blocked_file_path(allowed, intent="upload my resume", run_dir=run_dir) is None
    assert blocked_file_path(outside, intent="upload my resume", run_dir=run_dir) is not None


def test_non_file_actions_and_missing_path_pass(tmp_path: Path) -> None:
    typed = AgentAction(action_type=ActionType.TYPE, text="/etc/passwd", target_element_id="search")
    assert blocked_file_path(typed, intent="search", run_dir=tmp_path) is None
    assert blocked_file_path(AgentAction(action_type=ActionType.READ_TEXT, selector="main"), intent="x", run_dir=tmp_path) is None

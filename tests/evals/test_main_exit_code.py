"""``python -m evals`` exit code: fail on false success, crash or an unlabelled miss."""

from __future__ import annotations

from types import SimpleNamespace

from evals.__main__ import exit_code


def _pair(outcome: str, expected: str = "success", known_issue: str | None = None):
    scenario = SimpleNamespace(
        accepts=lambda o: o == expected, known_issue=known_issue
    )
    return scenario, SimpleNamespace(outcome=outcome)


def test_all_correct_passes() -> None:
    assert exit_code([_pair("success")]) == 0


def test_labelled_known_miss_passes() -> None:
    assert exit_code([_pair("success"), _pair("failure", known_issue="RFC 0003")]) == 0


def test_unlabelled_miss_fails() -> None:
    assert exit_code([_pair("failure")]) == 1


def test_false_success_fails_even_if_labelled() -> None:
    assert exit_code([_pair("false_success", known_issue="RFC")]) == 1


def test_crash_fails() -> None:
    assert exit_code([_pair("crash", known_issue="RFC")]) == 1

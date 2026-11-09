"""Full-harness rules from the Greptile review: early terminal match (G14) and app launch (G15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from operon.agent.policy.rules import PolicyRuleEngine, _catalog_app_name
from operon.agent.policy.verifier import (
    DeterministicVerifierService,
    _final_clause,
    _has_phrase,
)
from operon.models.common import RunStatus
from operon.models.perception import ScreenPerception
from operon.models.state import AgentState


def _perception(summary: str) -> ScreenPerception:
    return ScreenPerception(summary=summary, page_hint="unknown", visible_elements=[], capture_artifact_path="f.png")


# G14


def test_an_earlier_step_of_the_intent_does_not_end_the_task() -> None:
    signal = DeterministicVerifierService._intent_terminal_signal(
        "Make sure you are logged in, then send the weekly report to Dana", _perception("Dashboard. Welcome back.")
    )

    assert signal is None


def test_the_last_step_of_the_intent_still_ends_the_task() -> None:
    signal = DeterministicVerifierService._intent_terminal_signal(
        "Write the message and make sure it is sent", _perception("Your message sent successfully.")
    )

    assert signal == "sent"


def test_signals_match_whole_words_only() -> None:
    signal = DeterministicVerifierService._intent_terminal_signal(
        "Check the email was sent", _perception("A presentation is open.")
    )

    assert signal is None
    assert _has_phrase("sent", "message sent.") and not _has_phrase("sent", "presentation")


def test_final_clause_splits_on_steps() -> None:
    assert _final_clause("log in, open settings and then save the profile") == "save the profile"
    assert _final_clause("submit the form") == "submit the form"


# G15


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("settings", "settings"),
        ("the settings app", "settings"),
        ("the account settings page", None),
        ("settings page", None),
        ("notepad", "notepad"),
        ("vs code", "vs code"),
    ],
)
def test_only_a_whole_app_name_launches_an_app(raw: str, expected: str | None) -> None:
    assert _catalog_app_name(raw) == expected


def test_opening_a_settings_page_does_not_launch_the_settings_app(tmp_path: Path) -> None:
    engine = PolicyRuleEngine()
    state = AgentState(run_id="run-1", intent="Open the account settings page and change the name", status=RunStatus.RUNNING)

    decision = engine._prefer_launch_app_rule(state, _perception("A profile page."))

    assert decision is None


def test_the_browser_never_launches_apps(tmp_path: Path) -> None:
    engine = PolicyRuleEngine(launch_apps=False)
    state = AgentState(run_id="run-1", intent="Open notepad", status=RunStatus.RUNNING)

    assert engine._prefer_launch_app_rule(state, _perception("A page.")) is None

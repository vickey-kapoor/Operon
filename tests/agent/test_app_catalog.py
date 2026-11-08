"""Tests for the shared desktop app-launch catalog (single source of truth)."""

from __future__ import annotations

from pathlib import Path

import pytest

from operon.agent.app_catalog import APP_LAUNCH_TARGETS
from operon.agent.policy.rules import PolicyRuleEngine
from operon.executor import desktop
from operon.models.common import RunStatus
from operon.models.perception import ScreenPerception
from operon.models.policy import ActionType
from operon.models.state import AgentState


def _empty_perception(tmp_path: Path) -> ScreenPerception:
    path = tmp_path / "run-1" / "step_1" / "before.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return ScreenPerception(
        summary="Desktop visible.",
        page_hint="unknown",
        visible_elements=[],
        capture_artifact_path=str(path),
        confidence=0.9,
    )


# --- single source of truth ---------------------------------------------------


def test_desktop_and_rules_share_one_catalog() -> None:
    # The executor must resolve aliases from the very same catalog the policy
    # rule recognises, so the two layers can never drift apart again.
    assert desktop.APP_LAUNCH_TARGETS is APP_LAUNCH_TARGETS


def test_every_target_is_a_nonempty_launch_command() -> None:
    assert APP_LAUNCH_TARGETS
    assert all(cmd.strip() for cmd in APP_LAUNCH_TARGETS.values())


def test_office_apps_resolve_to_real_executables() -> None:
    # Regression: the old policy map knew these but the executor map did not,
    # so launching Office apps fell through to running a bare token.
    assert APP_LAUNCH_TARGETS["word"] == "winword.exe"
    assert APP_LAUNCH_TARGETS["excel"] == "excel.exe"
    assert APP_LAUNCH_TARGETS["powerpoint"] == "powerpnt.exe"


# --- the policy rule emits aliases the executor can resolve -------------------


@pytest.mark.parametrize(
    "intent,expected_command",
    [
        ("open notepad", "notepad.exe"),
        ("launch word", "winword.exe"),
        ("open excel and add a row", "excel.exe"),
        ("open google chrome", "chrome"),
        ("start vs code", "code"),
    ],
)
def test_launch_rule_emits_resolvable_alias(
    tmp_path: Path, intent: str, expected_command: str
) -> None:
    engine = PolicyRuleEngine()
    state = AgentState(
        run_id="run-1", intent=intent, status=RunStatus.RUNNING, step_count=1
    )

    decision = engine._prefer_launch_app_rule(state, _empty_perception(tmp_path))

    assert decision is not None
    assert decision.action.action_type is ActionType.LAUNCH_APP
    # Whatever alias the rule emits, the shared catalog must resolve it to the
    # expected executable (mirrors DesktopExecutor._exec_launch_app).
    emitted = decision.action.text.strip().lower()
    assert APP_LAUNCH_TARGETS.get(emitted, decision.action.text) == expected_command

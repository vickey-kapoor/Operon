"""apply_permission_mode: ask / risky / auto on top of the trust gate's verdict (KAN-89)."""

from __future__ import annotations

import pytest

from operon.agent.policy.trust_gate import (
    PermissionMode,
    TrustResult,
    TrustVerdict,
    apply_permission_mode,
    default_permission_mode,
    resolve_permission_mode,
)
from operon.models.policy import ActionType, AgentAction

CLICK = AgentAction(action_type=ActionType.CLICK, x=1, y=1)
WAIT = AgentAction(action_type=ActionType.WAIT, wait_ms=100)
ALLOW = TrustResult(TrustVerdict.ALLOW)
CONFIRM = TrustResult(TrustVerdict.CONFIRM, "matched high-risk rule", "buy now")
DENY = TrustResult(TrustVerdict.DENY, "matched deny rule", "wire")


def test_ask_turns_an_allowed_action_into_a_confirmation() -> None:
    result = apply_permission_mode(ALLOW, CLICK, "ask")
    assert result.verdict is TrustVerdict.CONFIRM and result.reason.startswith("ask mode")


def test_ask_does_not_pause_on_a_passive_action() -> None:
    assert apply_permission_mode(ALLOW, WAIT, "ask").verdict is TrustVerdict.ALLOW


def test_auto_skips_a_confirmation() -> None:
    assert apply_permission_mode(CONFIRM, CLICK, "auto").verdict is TrustVerdict.ALLOW


@pytest.mark.parametrize("mode", ["ask", "risky", "auto"])
def test_no_mode_relaxes_a_deny(mode: str) -> None:
    assert apply_permission_mode(DENY, CLICK, mode) is DENY


def test_risky_leaves_the_verdict_alone() -> None:
    assert apply_permission_mode(CONFIRM, CLICK, "risky") is CONFIRM
    assert apply_permission_mode(ALLOW, CLICK, "risky") is ALLOW


def test_default_mode_is_risky_and_reads_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_PERMISSION_MODE", raising=False)
    assert default_permission_mode() is PermissionMode.RISKY
    monkeypatch.setenv("OPERON_PERMISSION_MODE", "AUTO")
    assert resolve_permission_mode(None) is PermissionMode.AUTO
    monkeypatch.setenv("OPERON_PERMISSION_MODE", "nonsense")
    assert default_permission_mode() is PermissionMode.RISKY


def test_a_run_mode_beats_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_PERMISSION_MODE", "auto")
    assert resolve_permission_mode("ask") is PermissionMode.ASK

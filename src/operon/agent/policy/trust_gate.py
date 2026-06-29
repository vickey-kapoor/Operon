"""Trust gate — RFC 0001, Move 5.

A deterministic allow / confirm / deny check that runs after policy selection and
*before* execution. It is the single chokepoint for least-privilege and
high-risk-action policy on an agent that drives a real machine.

Opt-in: ``OPERON_TRUST_GATE`` defaults to ``off`` ⇒ every action is allowed and
behavior is unchanged. When ``on``:

- **deny** rules block the action — it is never executed autonomously; the run
  pauses for a human (override or stop).
- **confirm** rules (high-risk) pause for explicit human approval before running.

Rules are case-insensitive substrings matched against a description built from
the action's ``text``, target ``url``, and resolved target-element name. Deny
takes precedence over confirm.

Config:
    OPERON_TRUST_GATE     off | on            master switch (default off)
    OPERON_TRUST_DENY     comma-separated     phrases that block (default none)
    OPERON_TRUST_CONFIRM  comma-separated     phrases needing approval
                                              (default: a conservative built-in set)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class TrustVerdict(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class TrustResult:
    verdict: TrustVerdict
    reason: str = ""
    matched: str = ""


# Conservative default high-risk phrases (used when the gate is enabled and no
# custom OPERON_TRUST_CONFIRM is given). Multi-word where possible to limit
# false positives on ordinary buttons.
_DEFAULT_CONFIRM: tuple[str, ...] = (
    "delete account",
    "permanently delete",
    "deactivate",
    "close account",
    "confirm payment",
    "place order",
    "buy now",
    "complete purchase",
    "send money",
    "transfer funds",
    "uninstall",
)


def _split_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(s.strip().lower() for s in raw.split(",") if s.strip())


@dataclass
class TrustGate:
    enabled: bool = False
    deny: tuple[str, ...] = ()
    confirm: tuple[str, ...] = ()

    def describe(self, action: object) -> str:
        """Human-meaningful text for an action: its typed value, URL, and the
        resolved target element name (best-effort; tolerates missing fields)."""
        parts: list[str] = []
        text = getattr(action, "text", None)
        if text:
            parts.append(str(text))
        url = getattr(action, "url", None)
        if url:
            parts.append(str(url))
        ctx = getattr(action, "target_context", None)
        original = getattr(ctx, "original_target", None) if ctx is not None else None
        name = getattr(original, "primary_name", None) if original is not None else None
        if name:
            parts.append(str(name))
        return " ".join(parts).lower()

    def evaluate(self, action: object) -> TrustResult:
        if not self.enabled:
            return TrustResult(TrustVerdict.ALLOW)
        description = self.describe(action)
        for keyword in self.deny:
            if keyword in description:
                return TrustResult(TrustVerdict.DENY, "matched deny rule", keyword)
        for keyword in self.confirm:
            if keyword in description:
                return TrustResult(TrustVerdict.CONFIRM, "matched high-risk rule", keyword)
        return TrustResult(TrustVerdict.ALLOW)


def make_trust_gate() -> TrustGate:
    """Build the gate from the environment. Disabled (pass-through) by default."""
    enabled = os.getenv("OPERON_TRUST_GATE", "off").strip().lower() in {"on", "true", "1"}
    deny = _split_env("OPERON_TRUST_DENY")
    confirm = _split_env("OPERON_TRUST_CONFIRM") or _DEFAULT_CONFIRM
    return TrustGate(enabled=enabled, deny=deny, confirm=confirm)

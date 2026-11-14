"""Trust gate — RFC 0001, Move 5.

A deterministic allow / confirm / deny check that runs after policy selection and
*before* execution. It is the single chokepoint for least-privilege and
high-risk-action policy on an agent that drives a real machine.

On by default (KAN-15): ``OPERON_TRUST_GATE`` defaults to ``on`` with the
built-in confirm list below, an empty deny list and no domain allowlist. Set it
to ``off`` to allow every action (the benchmarks do, so their scores stay
comparable). When ``on``:

- **deny** rules block the action — it is never executed autonomously; the run
  pauses for a human (override or stop).
- **confirm** rules (high-risk) pause for explicit human approval before running.
- a ``run_shell`` tool call (RFC 0002) always needs approval, since no rule can
  judge what a command does.

Rules are case-insensitive substrings matched against a description built from
the action's ``text``, target ``url``, and resolved target-element name. Deny
takes precedence over confirm.

Config:
    OPERON_TRUST_GATE           on | off        master switch (default on)
    OPERON_TRUST_DENY           comma-separated  phrases that block (default none)
    OPERON_TRUST_CONFIRM        comma-separated  phrases needing approval
                                                 (default: a conservative built-in set)
    OPERON_TRUST_ALLOW_DOMAINS  comma-separated  if set, navigation is restricted to
                                                 these domains (and subdomains); any
                                                 other URL is denied (fail-closed), and
                                                 so is any action taken while the page
                                                 itself is off the list (a followed link)
    OPERON_PERMISSION_MODE      ask | risky | auto  default permission mode for a run
                                                 that does not choose one (default risky)

Permission modes (KAN-89) adjust the gate's verdict per run, after the rules above:

- ``ask``: every action that does something needs approval, even with the gate off
  (an explicit choice by the person running the task). Deny rules still block.
- ``risky`` (default): the gate as configured: confirm-list actions need approval.
- ``auto``: confirm-list actions run without approval; deny rules and the domain
  allowlist still block. Vendor safety confirmations (Gemini ``require_confirmation``)
  are not affected by any mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse


class PermissionMode(str, Enum):
    ASK = "ask"
    RISKY = "risky"
    AUTO = "auto"


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
    allow_domains: tuple[str, ...] = ()

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
        key = getattr(action, "key", None)
        if key:
            parts.append(str(key))
        ctx = getattr(action, "target_context", None)
        original = getattr(ctx, "original_target", None) if ctx is not None else None
        name = getattr(original, "primary_name", None) if original is not None else None
        if name:
            parts.append(str(name))
        return " ".join(parts).lower()

    def _domain_blocked(self, action: object) -> str | None:
        """If a domain allowlist is configured, return the offending host for any
        action whose URL is outside the allowed domains. Fail-closed: an
        unparseable host is treated as not-allowed. Returns None when there is no
        allowlist, the action has no URL, or the host is allowed."""
        if not self.allow_domains:
            return None
        url = getattr(action, "url", None)
        if not url:
            return None
        return self._host_blocked(str(url))

    def _host_blocked(self, url: str) -> str | None:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return "(unparseable url)"
        for domain in self.allow_domains:
            if host == domain or host.endswith("." + domain):
                return None
        return host

    def _page_blocked(self, current_url: str | None) -> str | None:
        """The page's host when the browser already sits outside the allowlist, e.g.
        after the model clicked a link; only web pages are judged (about:blank and
        the like are not a destination)."""
        if not self.allow_domains or not current_url:
            return None
        if urlparse(current_url).scheme not in ("http", "https"):
            return None
        return self._host_blocked(current_url)

    def evaluate(self, action: object, current_url: str | None = None) -> TrustResult:
        if not self.enabled:
            return TrustResult(TrustVerdict.ALLOW)
        description = self.describe(action)
        for keyword in self.deny:
            if keyword in description:
                return TrustResult(TrustVerdict.DENY, "matched deny rule", keyword)
        blocked_host = self._domain_blocked(action)
        if blocked_host is not None:
            return TrustResult(TrustVerdict.DENY, "domain not in allowlist", blocked_host)
        is_tool = getattr(getattr(action, "action_type", None), "value", None) == "tool"
        # A tool call (RFC 0002) doesn't act on the page, so the page's own host doesn't block it.
        if getattr(action, "url", None) is None and not is_tool:
            blocked_page = self._page_blocked(current_url)
            if blocked_page is not None:
                return TrustResult(TrustVerdict.DENY, "current page not in allowlist", blocked_page)
        for keyword in self.confirm:
            if keyword in description:
                return TrustResult(TrustVerdict.CONFIRM, "matched high-risk rule", keyword)
        if is_tool and getattr(action, "tool", None) == "run_shell":
            # No path or domain rule can judge what a command does: a person approves
            # it (``auto`` mode skips this, deny rules above still block).
            return TrustResult(TrustVerdict.CONFIRM, "shell commands need approval", "run_shell")
        return TrustResult(TrustVerdict.ALLOW)


def make_trust_gate() -> TrustGate:
    """Build the gate from the environment. On by default; only an explicit
    ``off`` / ``false`` / ``0`` / ``no`` turns it into a pass-through."""
    enabled = os.getenv("OPERON_TRUST_GATE", "on").strip().lower() not in {"off", "false", "0", "no"}
    deny = _split_env("OPERON_TRUST_DENY")
    confirm = _split_env("OPERON_TRUST_CONFIRM") or _DEFAULT_CONFIRM
    allow_domains = _split_env("OPERON_TRUST_ALLOW_DOMAINS")
    return TrustGate(enabled=enabled, deny=deny, confirm=confirm, allow_domains=allow_domains)


# Actions that change nothing on screen or disk: "ask" mode does not pause on them.
_PASSIVE_ACTIONS = frozenset({"wait", "stop", "wait_for_user", "screenshot_region"})


def default_permission_mode() -> PermissionMode:
    """``OPERON_PERMISSION_MODE``, or ``risky`` when it is unset or not a known mode."""
    raw = os.getenv("OPERON_PERMISSION_MODE", "").strip().lower()
    try:
        return PermissionMode(raw)
    except ValueError:
        return PermissionMode.RISKY


def resolve_permission_mode(mode: str | PermissionMode | None) -> PermissionMode:
    """A run's mode: its own when set, else the server default."""
    if mode is None or mode == "":
        return default_permission_mode()
    return PermissionMode(mode)


def apply_permission_mode(result: TrustResult, action: object, mode: str | PermissionMode | None) -> TrustResult:
    """Adjust a gate verdict for a run's permission mode. DENY is never relaxed."""
    resolved = resolve_permission_mode(mode)
    if result.verdict is TrustVerdict.DENY:
        return result
    if resolved is PermissionMode.AUTO and result.verdict is TrustVerdict.CONFIRM:
        return TrustResult(TrustVerdict.ALLOW, "auto mode skips confirmations", result.matched)
    if resolved is PermissionMode.ASK and result.verdict is TrustVerdict.ALLOW:
        action_type = getattr(action, "action_type", None)
        value = getattr(action_type, "value", action_type)
        if value not in _PASSIVE_ACTIONS:
            return TrustResult(TrustVerdict.CONFIRM, "ask mode approves every action", str(value or "action"))
    return result

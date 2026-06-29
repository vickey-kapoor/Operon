"""Tests for the trust gate (RFC 0001 Move 5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from operon.agent.policy.trust_gate import (
    TrustGate,
    TrustVerdict,
    make_trust_gate,
)


def _action(text: str | None = None, url: str | None = None, element_name: str | None = None):
    """A duck-typed action: only the fields trust_gate.describe() reads."""
    ctx = None
    if element_name is not None:
        ctx = SimpleNamespace(original_target=SimpleNamespace(primary_name=element_name))
    return SimpleNamespace(text=text, url=url, target_context=ctx)


# ── domain allowlist ──────────────────────────────────────────────────────

def test_allowlist_permits_listed_domain_and_subdomains() -> None:
    gate = TrustGate(enabled=True, allow_domains=("example.com",))
    assert gate.evaluate(_action(url="https://example.com/page")).verdict is TrustVerdict.ALLOW
    assert gate.evaluate(_action(url="https://app.example.com/x")).verdict is TrustVerdict.ALLOW


def test_allowlist_denies_unlisted_domain() -> None:
    gate = TrustGate(enabled=True, allow_domains=("example.com",))
    result = gate.evaluate(_action(url="https://evil.example.org/x"))
    assert result.verdict is TrustVerdict.DENY
    assert result.reason == "domain not in allowlist"
    assert result.matched == "evil.example.org"


def test_allowlist_does_not_partial_match_suffix() -> None:
    # "notexample.com" must NOT be allowed by an "example.com" entry.
    gate = TrustGate(enabled=True, allow_domains=("example.com",))
    assert gate.evaluate(_action(url="https://notexample.com/x")).verdict is TrustVerdict.DENY


def test_allowlist_is_fail_closed_on_unparseable_url() -> None:
    gate = TrustGate(enabled=True, allow_domains=("example.com",))
    result = gate.evaluate(_action(url="not-a-url"))
    assert result.verdict is TrustVerdict.DENY


def test_allowlist_ignores_actions_without_a_url() -> None:
    gate = TrustGate(enabled=True, allow_domains=("example.com",))
    assert gate.evaluate(_action(text="click submit")).verdict is TrustVerdict.ALLOW


def test_no_allowlist_means_no_domain_restriction() -> None:
    gate = TrustGate(enabled=True, allow_domains=())
    assert gate.evaluate(_action(url="https://anywhere.example/x")).verdict is TrustVerdict.ALLOW


def test_deny_phrase_takes_precedence_over_allowlist() -> None:
    gate = TrustGate(enabled=True, deny=("delete",), allow_domains=("example.com",))
    # Even on an allowed domain, a deny-phrase match still blocks (and is the reason).
    result = gate.evaluate(_action(url="https://example.com/delete", text="delete"))
    assert result.verdict is TrustVerdict.DENY
    assert result.reason == "matched deny rule"


# ── disabled (default) ────────────────────────────────────────────────────

def test_disabled_gate_allows_everything() -> None:
    gate = TrustGate(enabled=False, deny=("delete",), confirm=("buy",))
    assert gate.evaluate(_action(text="delete everything and buy now")).verdict is TrustVerdict.ALLOW


# ── deny / confirm / allow ────────────────────────────────────────────────

def test_deny_rule_blocks() -> None:
    gate = TrustGate(enabled=True, deny=("delete account",), confirm=())
    result = gate.evaluate(_action(element_name="Delete Account"))
    assert result.verdict is TrustVerdict.DENY
    assert result.matched == "delete account"


def test_confirm_rule_requires_approval() -> None:
    gate = TrustGate(enabled=True, deny=(), confirm=("place order",))
    result = gate.evaluate(_action(element_name="Place Order"))
    assert result.verdict is TrustVerdict.CONFIRM
    assert result.matched == "place order"


def test_allow_when_nothing_matches() -> None:
    gate = TrustGate(enabled=True, deny=("delete",), confirm=("buy",))
    assert gate.evaluate(_action(text="open the settings page")).verdict is TrustVerdict.ALLOW


def test_deny_takes_precedence_over_confirm() -> None:
    gate = TrustGate(enabled=True, deny=("delete",), confirm=("delete",))
    assert gate.evaluate(_action(text="delete it")).verdict is TrustVerdict.DENY


# ── describe() matches across fields ──────────────────────────────────────

def test_matches_against_url() -> None:
    gate = TrustGate(enabled=True, deny=("evil.example",), confirm=())
    assert gate.evaluate(_action(url="https://evil.example/x")).verdict is TrustVerdict.DENY


def test_matches_against_action_text() -> None:
    gate = TrustGate(enabled=True, deny=(), confirm=("transfer funds",))
    assert gate.evaluate(_action(text="Transfer Funds to account 42")).verdict is TrustVerdict.CONFIRM


def test_describe_tolerates_missing_fields() -> None:
    gate = TrustGate(enabled=True)
    assert gate.describe(_action()) == ""  # no text/url/element → empty, no crash


# ── make_trust_gate env wiring ────────────────────────────────────────────

def test_make_trust_gate_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_TRUST_GATE", raising=False)
    assert make_trust_gate().enabled is False


def test_make_trust_gate_enabled_uses_default_confirm_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_TRUST_GATE", "on")
    monkeypatch.delenv("OPERON_TRUST_CONFIRM", raising=False)
    monkeypatch.delenv("OPERON_TRUST_DENY", raising=False)
    gate = make_trust_gate()
    assert gate.enabled is True
    assert "delete account" in gate.confirm  # built-in conservative default
    assert gate.deny == ()


def test_make_trust_gate_custom_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_TRUST_GATE", "true")
    monkeypatch.setenv("OPERON_TRUST_DENY", "rm -rf, format disk")
    monkeypatch.setenv("OPERON_TRUST_CONFIRM", "submit")
    gate = make_trust_gate()
    assert gate.deny == ("rm -rf", "format disk")
    assert gate.confirm == ("submit",)


def test_make_trust_gate_reads_allow_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_TRUST_GATE", "on")
    monkeypatch.setenv("OPERON_TRUST_ALLOW_DOMAINS", "example.com, Foo.ORG")
    gate = make_trust_gate()
    assert gate.allow_domains == ("example.com", "foo.org")  # normalized to lowercase

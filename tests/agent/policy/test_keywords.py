"""Tests for the consolidated, extensible policy keyword heuristics."""

from __future__ import annotations

import pytest

from operon.agent.policy import keywords
from operon.agent.policy.keywords import (
    DEFAULT_DISMISS_TOKENS,
    DEFAULT_OVERLAY_TOKENS,
    DEFAULT_TERMINAL_PATTERNS,
    clear_keyword_overrides,
    dismiss_tokens,
    overlay_tokens,
    register_dismiss_tokens,
    register_overlay_tokens,
    register_terminal_patterns,
    terminal_patterns,
)


@pytest.fixture(autouse=True)
def _isolate_overrides():
    """Each test starts and ends with only the built-in defaults registered."""
    clear_keyword_overrides()
    yield
    clear_keyword_overrides()


# --- defaults: unchanged behaviour out of the box -----------------------------


def test_accessors_return_defaults_when_nothing_registered() -> None:
    assert terminal_patterns() == DEFAULT_TERMINAL_PATTERNS
    assert dismiss_tokens() == DEFAULT_DISMISS_TOKENS
    assert overlay_tokens() == DEFAULT_OVERLAY_TOKENS


def test_defaults_cover_the_generic_signals_the_engine_relies_on() -> None:
    # Sanity: these generic signals must remain enabled by default.
    assert "close" in dismiss_tokens()
    assert "cookie" in overlay_tokens()
    flat_signals = {sig for _, signals in terminal_patterns() for sig in signals}
    assert "thank you" in flat_signals


# --- extension: opt-in additions never mutate the defaults --------------------


def test_register_dismiss_tokens_adds_without_mutating_defaults() -> None:
    register_dismiss_tokens(["Nope", "GO AWAY"])
    merged = dismiss_tokens()
    assert {"nope", "go away"} <= merged
    # Defaults are preserved and the frozen default set is untouched.
    assert DEFAULT_DISMISS_TOKENS <= merged
    assert "nope" not in DEFAULT_DISMISS_TOKENS


def test_register_overlay_tokens_is_lowercased_and_additive() -> None:
    register_overlay_tokens(["Interstitial"])
    assert "interstitial" in overlay_tokens()
    assert DEFAULT_OVERLAY_TOKENS <= overlay_tokens()


def test_register_terminal_patterns_appends_after_defaults() -> None:
    register_terminal_patterns([("approved", ["request approved", "Approved"])])
    patterns = terminal_patterns()
    assert patterns[: len(DEFAULT_TERMINAL_PATTERNS)] == DEFAULT_TERMINAL_PATTERNS
    assert ("approved", ("request approved", "approved")) == patterns[-1]


def test_clear_restores_defaults() -> None:
    register_dismiss_tokens(["zzz"])
    register_overlay_tokens(["zzz"])
    register_terminal_patterns([("zzz", ["zzz"])])
    clear_keyword_overrides()
    assert dismiss_tokens() == DEFAULT_DISMISS_TOKENS
    assert overlay_tokens() == DEFAULT_OVERLAY_TOKENS
    assert terminal_patterns() == DEFAULT_TERMINAL_PATTERNS


def test_defaults_are_generic_not_site_specific() -> None:
    # Guard against regressing into site coupling: no URLs / domains here.
    blob = " ".join(
        [*DEFAULT_DISMISS_TOKENS, *DEFAULT_OVERLAY_TOKENS]
        + [sig for _, sigs in DEFAULT_TERMINAL_PATTERNS for sig in sigs]
    )
    assert "http" not in blob
    assert ".com" not in blob


def test_module_exposes_site_adapter_style_surface() -> None:
    # Same ergonomics as site_adapters: register_* + accessor + clear_*.
    for name in (
        "register_dismiss_tokens",
        "register_overlay_tokens",
        "register_terminal_patterns",
        "clear_keyword_overrides",
    ):
        assert callable(getattr(keywords, name))

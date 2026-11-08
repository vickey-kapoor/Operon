"""Generic, language-level keyword heuristics shared by the policy engine.

Unlike :mod:`operon.agent.policy.site_adapters` (which holds *site-specific*
knowledge and is therefore empty by default), the sets here are deliberately
generic English signals — "submitted", "close", "cookie" — that apply to any
page.  They are enabled by default; consolidating them in one place keeps the
engine free of scattered inline literals and gives callers a single, documented
seam to extend without editing core rules.

Extend via the ``register_*`` helpers; the defaults are never mutated, so
default behaviour is preserved unless a caller opts in to additions.
"""

from __future__ import annotations

from collections.abc import Iterable

# --- Terminal-signal patterns -------------------------------------------------
# Maps an intent keyword to the visual signals that confirm that goal is done.
# Used by the verifier to detect task completion from a page summary.
DEFAULT_TERMINAL_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("submitted", ("submitted", "submission received", "thank you", "success")),
    ("saved", ("saved", "changes saved", "save successful")),
    ("created", ("created", "issue created", "record created", "added")),
    ("deleted", ("deleted", "removed", "record deleted")),
    ("uploaded", ("uploaded", "upload complete", "file uploaded")),
    ("sent", ("sent", "message sent", "email sent")),
    ("logged in", ("dashboard", "welcome", "logged in", "sign out", "log out")),
    ("logged out", ("signed out", "logged out", "login", "sign in")),
    ("searched", ("search results", "results for", "no results")),
    ("found", ("found", "result", "match")),
)

# --- Dismiss / reject button labels -------------------------------------------
# Tokens suggesting a button dismisses/rejects an overlay rather than accepts it.
DEFAULT_DISMISS_TOKENS: frozenset[str] = frozenset({
    "don't", "dont", "dismiss", "close", "cancel", "no thanks", "skip",
    "not now", "maybe later", "decline", "reject", "later", "no", "×", "✕",
})

# --- Blocking-overlay labels --------------------------------------------------
# Substrings in an element label that mark it as a blocking overlay/banner.
DEFAULT_OVERLAY_TOKENS: frozenset[str] = frozenset({
    "banner", "promo", "modal", "overlay", "popup", "notification",
    "toast", "cookie", "consent",
})


_extra_terminal_patterns: list[tuple[str, tuple[str, ...]]] = []
_extra_dismiss_tokens: set[str] = set()
_extra_overlay_tokens: set[str] = set()


def register_terminal_patterns(
    patterns: Iterable[tuple[str, Iterable[str]]],
) -> None:
    """Add extra (intent_keyword, visual_signals) terminal patterns."""
    for intent_keyword, signals in patterns:
        _extra_terminal_patterns.append(
            (intent_keyword.lower(), tuple(s.lower() for s in signals))
        )


def register_dismiss_tokens(tokens: Iterable[str]) -> None:
    """Add extra dismiss/reject button label tokens."""
    _extra_dismiss_tokens.update(t.lower() for t in tokens)


def register_overlay_tokens(tokens: Iterable[str]) -> None:
    """Add extra blocking-overlay label tokens."""
    _extra_overlay_tokens.update(t.lower() for t in tokens)


def clear_keyword_overrides() -> None:
    """Drop all registered extras, restoring the built-in defaults."""
    _extra_terminal_patterns.clear()
    _extra_dismiss_tokens.clear()
    _extra_overlay_tokens.clear()


def terminal_patterns() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Default terminal patterns followed by any registered extras."""
    return (*DEFAULT_TERMINAL_PATTERNS, *_extra_terminal_patterns)


def dismiss_tokens() -> frozenset[str]:
    """Default dismiss tokens merged with any registered extras."""
    return DEFAULT_DISMISS_TOKENS | _extra_dismiss_tokens


def overlay_tokens() -> frozenset[str]:
    """Default overlay tokens merged with any registered extras."""
    return DEFAULT_OVERLAY_TOKENS | _extra_overlay_tokens

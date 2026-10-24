from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass

_VALID_DESKTOP_BACKENDS = frozenset({"json", "claude_computer_use"})
_VALID_BROWSER_BACKENDS = frozenset({"json", "computer_use", "claude_computer_use"})
_VALID_PROVIDERS = frozenset({"gemini", "anthropic"})


@dataclass(frozen=True)
class RuntimeModeConfig:
    backend: str
    primary_model: str
    verifier_provider: str = "gemini"
    fallback_backend: str | None = None
    fallback_model: str | None = None
    verifier_model: str | None = None
    # Gemini thinking settings: a level ("low") on Gemini 3, a token budget ("0") on 2.5.
    thinking: str | None = None
    verifier_thinking: str | None = None
    # Claude computer use backend (OPERON_BROWSER_BACKEND=claude_computer_use).
    claude_model: str | None = None
    claude_effort: str | None = None


def desktop_mode_config() -> RuntimeModeConfig:
    return RuntimeModeConfig(
        backend=os.getenv("OPERON_DESKTOP_BACKEND", "json"),
        primary_model=os.getenv("OPERON_DESKTOP_MODEL", "gemini-2.5-flash"),
        verifier_provider=os.getenv("OPERON_DESKTOP_VERIFIER_PROVIDER", "gemini"),
        fallback_model=os.getenv("OPERON_DESKTOP_FALLBACK_MODEL", "gemini-2.5-flash"),
        verifier_model=os.getenv("OPERON_DESKTOP_VERIFIER_MODEL", "gemini-2.5-flash"),
        verifier_thinking=os.getenv("OPERON_DESKTOP_VERIFIER_THINKING") or None,
        claude_model=os.getenv("OPERON_DESKTOP_CLAUDE_MODEL", "claude-opus-5-5"),
        claude_effort=os.getenv("OPERON_DESKTOP_CLAUDE_EFFORT", "low") or None,
    )


def _default_browser_backend() -> str:
    """Claude computer use when an Anthropic key is present, else Gemini Computer Use."""
    return "claude_computer_use" if os.getenv("ANTHROPIC_API_KEY") else "computer_use"


def browser_mode_config() -> RuntimeModeConfig:
    return RuntimeModeConfig(
        backend=os.getenv("OPERON_BROWSER_BACKEND") or _default_browser_backend(),
        primary_model=os.getenv(
            "OPERON_BROWSER_MODEL",
            "gemini-3.5-flash",
        ),
        verifier_provider=os.getenv("OPERON_BROWSER_VERIFIER_PROVIDER", "gemini"),
        fallback_backend=os.getenv("OPERON_BROWSER_FALLBACK_BACKEND", "json"),
        fallback_model=os.getenv("OPERON_BROWSER_FALLBACK_MODEL", "gemini-2.5-flash"),
        verifier_model=os.getenv("OPERON_BROWSER_VERIFIER_MODEL", "gemini-3.5-flash"),
        thinking=os.getenv("OPERON_BROWSER_THINKING", "low") or None,
        verifier_thinking=os.getenv("OPERON_BROWSER_VERIFIER_THINKING", "low") or None,
        claude_model=os.getenv("OPERON_BROWSER_CLAUDE_MODEL", "claude-opus-5-5"),
        claude_effort=os.getenv("OPERON_BROWSER_CLAUDE_EFFORT", "low") or None,
    )


def validate_modes() -> list[str]:
    """Validate mode configs at startup. Returns a list of human-readable errors
    (empty when configuration is valid). Caller decides whether to raise."""
    errors: list[str] = []
    desktop = desktop_mode_config()
    browser = browser_mode_config()

    if desktop.backend not in _VALID_DESKTOP_BACKENDS:
        errors.append(
            f"OPERON_DESKTOP_BACKEND={desktop.backend!r} is not one of {sorted(_VALID_DESKTOP_BACKENDS)}"
        )
    if browser.backend not in _VALID_BROWSER_BACKENDS:
        errors.append(
            f"OPERON_BROWSER_BACKEND={browser.backend!r} is not one of {sorted(_VALID_BROWSER_BACKENDS)}"
        )
    if browser.fallback_backend and browser.fallback_backend not in _VALID_BROWSER_BACKENDS:
        errors.append(
            f"OPERON_BROWSER_FALLBACK_BACKEND={browser.fallback_backend!r} is not one of {sorted(_VALID_BROWSER_BACKENDS)}"
        )
    for cfg, label in ((desktop, "desktop"), (browser, "browser")):
        if cfg.verifier_provider not in _VALID_PROVIDERS:
            errors.append(
                f"OPERON_{label.upper()}_VERIFIER_PROVIDER={cfg.verifier_provider!r} is not one of {sorted(_VALID_PROVIDERS)}"
            )
        if cfg.backend == "claude_computer_use" and not os.getenv("ANTHROPIC_API_KEY"):
            errors.append(f"{label} backend=claude_computer_use but ANTHROPIC_API_KEY is unset")
    return errors


def main() -> int:
    """Diagnostic CLI: print effective mode config and validation status.

    Usage: python -m operon.api.runtime_config
    """
    desktop = desktop_mode_config()
    browser = browser_mode_config()
    print("Operon runtime configuration")
    print("-" * 40)
    print("Desktop:")
    for k, v in asdict(desktop).items():
        print(f"  {k:24s} = {v}")
    print("Browser:")
    for k, v in asdict(browser).items():
        print(f"  {k:24s} = {v}")
    errors = validate_modes()
    if errors:
        print("-" * 40)
        print("VALIDATION ERRORS:")
        for err in errors:
            print(f"  ! {err}")
        return 1
    print("-" * 40)
    print("Configuration valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

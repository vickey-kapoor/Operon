from __future__ import annotations

import sys
from dataclasses import asdict, dataclass

from operon.core.settings import Settings, get_settings

# Gemini is Operon's only model: computer_use (Gemini computer use, the default) or
# json (the older Gemini JSON path).
_VALID_DESKTOP_BACKENDS = frozenset({"json", "computer_use"})
_VALID_BROWSER_BACKENDS = frozenset({"json", "computer_use"})
_DEFAULT_MODEL = "gemini-3.5-flash"


@dataclass(frozen=True)
class RuntimeModeConfig:
    backend: str
    primary_model: str
    fallback_backend: str | None = None
    fallback_model: str | None = None
    verifier_model: str | None = None
    # Gemini thinking settings: a level ("low") on Gemini 3, a token budget ("0") on 2.5.
    thinking: str | None = None
    verifier_thinking: str | None = None


def desktop_mode_config(settings: Settings | None = None) -> RuntimeModeConfig:
    s = settings or get_settings()
    return RuntimeModeConfig(
        backend=s.desktop_backend,
        primary_model=s.desktop_model,
        thinking=s.desktop_thinking,
        fallback_model=s.desktop_fallback_model,
        verifier_model=s.desktop_verifier_model,
        verifier_thinking=s.desktop_verifier_thinking,
    )


def browser_mode_config(settings: Settings | None = None) -> RuntimeModeConfig:
    s = settings or get_settings()
    return RuntimeModeConfig(
        backend=s.browser_backend,
        primary_model=s.browser_model,
        fallback_backend=s.browser_fallback_backend,
        fallback_model=s.browser_fallback_model,
        verifier_model=s.browser_verifier_model,
        thinking=s.browser_thinking,
        verifier_thinking=s.browser_verifier_thinking,
    )


def validate_modes(settings: Settings | None = None) -> list[str]:
    """Validate mode configs at startup. Returns a list of human-readable errors
    (empty when configuration is valid). Caller decides whether to raise."""
    errors: list[str] = []
    settings = settings or get_settings()
    desktop = desktop_mode_config(settings)
    browser = browser_mode_config(settings)

    from operon.environments import provider_names

    if settings.environment not in provider_names():
        errors.append(
            f"OPERON_ENVIRONMENT={settings.environment!r} is not one of {provider_names()}"
        )
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
    # Gemini only: a leftover Claude or GPT name from an old .env would fail on every call
    # (the critic quietly falls back to a guess), so refuse it at startup instead.
    for mode, cfg in (("DESKTOP", desktop), ("BROWSER", browser)):
        for name, model in (
            ("MODEL", cfg.primary_model),
            ("FALLBACK_MODEL", cfg.fallback_model),
            ("VERIFIER_MODEL", cfg.verifier_model),
        ):
            if model and not model.strip().lower().startswith("gemini-"):
                errors.append(
                    f"OPERON_{mode}_{name}={model!r} is not a Gemini model; Operon runs Gemini only "
                    f"(remove it from .env to use {_DEFAULT_MODEL})"
                )
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
    from operon.core.settings import startup_problems

    errors, warnings = startup_problems()
    for warning in warnings:
        print(f"  ? {warning} (ignored)")
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

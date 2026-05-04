from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass

_VALID_DESKTOP_BACKENDS = frozenset({"json"})
_VALID_BROWSER_BACKENDS = frozenset({"json", "computer_use", "browserbase"})
_VALID_PROVIDERS = frozenset({"gemini", "anthropic"})


@dataclass(frozen=True)
class RuntimeModeConfig:
    backend: str
    primary_model: str
    planner_provider: str = "gemini"
    planner_model: str | None = None
    verifier_provider: str = "gemini"
    fallback_backend: str | None = None
    fallback_model: str | None = None
    verifier_model: str | None = None


def desktop_mode_config() -> RuntimeModeConfig:
    return RuntimeModeConfig(
        backend=os.getenv("OPERON_DESKTOP_BACKEND", "json"),
        primary_model=os.getenv("OPERON_DESKTOP_MODEL", "gemini-2.5-flash"),
        planner_provider=os.getenv("OPERON_DESKTOP_PLANNER_PROVIDER", "gemini"),
        planner_model=os.getenv("OPERON_DESKTOP_PLANNER_MODEL"),
        verifier_provider=os.getenv("OPERON_DESKTOP_VERIFIER_PROVIDER", "gemini"),
        fallback_model=os.getenv("OPERON_DESKTOP_FALLBACK_MODEL", "gemini-2.5-flash"),
        verifier_model=os.getenv("OPERON_DESKTOP_VERIFIER_MODEL", "gemini-2.5-flash"),
    )


def browser_mode_config() -> RuntimeModeConfig:
    return RuntimeModeConfig(
        backend=os.getenv("OPERON_BROWSER_BACKEND", "computer_use"),
        primary_model=os.getenv(
            "OPERON_BROWSER_MODEL",
            "gemini-2.5-computer-use-preview-10-2025",
        ),
        planner_provider=os.getenv("OPERON_BROWSER_PLANNER_PROVIDER", "gemini"),
        planner_model=os.getenv("OPERON_BROWSER_PLANNER_MODEL"),
        verifier_provider=os.getenv("OPERON_BROWSER_VERIFIER_PROVIDER", "gemini"),
        fallback_backend=os.getenv("OPERON_BROWSER_FALLBACK_BACKEND", "json"),
        fallback_model=os.getenv("OPERON_BROWSER_FALLBACK_MODEL", "gemini-2.5-flash"),
        verifier_model=os.getenv("OPERON_BROWSER_VERIFIER_MODEL", "gemini-2.5-flash"),
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
        if cfg.planner_provider not in _VALID_PROVIDERS:
            errors.append(
                f"OPERON_{label.upper()}_PLANNER_PROVIDER={cfg.planner_provider!r} is not one of {sorted(_VALID_PROVIDERS)}"
            )
        if cfg.verifier_provider not in _VALID_PROVIDERS:
            errors.append(
                f"OPERON_{label.upper()}_VERIFIER_PROVIDER={cfg.verifier_provider!r} is not one of {sorted(_VALID_PROVIDERS)}"
            )
        if cfg.planner_provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            errors.append(
                f"{label} planner_provider=anthropic but ANTHROPIC_API_KEY is unset"
            )
    return errors


def main() -> int:
    """Diagnostic CLI: print effective mode config and validation status.

    Usage: python -m src.api.runtime_config
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

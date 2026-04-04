from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeModeConfig:
    backend: str
    primary_model: str
    fallback_backend: str | None = None
    fallback_model: str | None = None
    verifier_model: str | None = None


def desktop_mode_config() -> RuntimeModeConfig:
    return RuntimeModeConfig(
        backend=os.getenv("OPERON_DESKTOP_BACKEND", "json"),
        primary_model=os.getenv("OPERON_DESKTOP_MODEL", "gemini-3-flash-preview"),
        fallback_model=os.getenv("OPERON_DESKTOP_FALLBACK_MODEL", "gemini-2.5-flash"),
        verifier_model=os.getenv("OPERON_DESKTOP_VERIFIER_MODEL", "gemini-3-flash-preview"),
    )


def browser_mode_config() -> RuntimeModeConfig:
    return RuntimeModeConfig(
        backend=os.getenv("OPERON_BROWSER_BACKEND", "computer_use"),
        primary_model=os.getenv(
            "OPERON_BROWSER_MODEL",
            "gemini-2.5-computer-use-preview-10-2025",
        ),
        fallback_backend=os.getenv("OPERON_BROWSER_FALLBACK_BACKEND", "json"),
        fallback_model=os.getenv("OPERON_BROWSER_FALLBACK_MODEL", "gemini-3-flash-preview"),
        verifier_model=os.getenv("OPERON_BROWSER_VERIFIER_MODEL", "gemini-3-flash-preview"),
    )

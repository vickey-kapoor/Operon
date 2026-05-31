"""Shared filesystem path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root containing ``pyproject.toml``."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[3]


def prompts_dir() -> Path:
    """Return the repository-level prompts directory."""
    return project_root() / "prompts"


def runtime_dir() -> Path:
    """Return the root for generated runtime artifacts."""
    return Path(os.getenv("OPERON_RUNTIME_ROOT", project_root() / ".var"))


def runs_dir() -> Path:
    """Return the local run-artifact directory."""
    return Path(os.getenv("OPERON_RUNS_ROOT", runtime_dir() / "runs"))


def browser_artifacts_dir() -> Path:
    """Return the browser session artifact directory."""
    return Path(os.getenv("OPERON_BROWSER_ARTIFACTS_ROOT", runtime_dir() / "browser-artifacts"))


def desktop_artifacts_dir() -> Path:
    """Return the desktop session artifact directory."""
    return Path(os.getenv("OPERON_DESKTOP_ARTIFACTS_ROOT", runtime_dir() / "desktop-artifacts"))


def test_artifacts_dir() -> Path:
    """Return the test artifact directory."""
    return Path(os.getenv("OPERON_TEST_ARTIFACTS_ROOT", runtime_dir() / "test-artifacts"))

"""Helpers shared by the desktop container tests."""

from __future__ import annotations

import subprocess


def sh(provider, command: str) -> subprocess.CompletedProcess:
    """A shell command inside the desktop container (to grade from its own state)."""
    return subprocess.run(
        ["docker", "exec", provider.container, "sh", "-c", command],
        capture_output=True, text=True, timeout=30, check=False,
    )

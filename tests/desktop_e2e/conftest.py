"""Shared fixtures for the desktop container tests: one Linux desktop for the whole session."""

from __future__ import annotations

import os
import shutil

import pytest

from operon.environments import get_provider


@pytest.fixture(scope="session")
def desktop():
    """One Linux desktop container for the whole session."""
    # In CI a missing desktop is a failure, never a quiet skip.
    unavailable = pytest.fail if os.environ.get("OPERON_E2E_REQUIRE_DESKTOP") else pytest.skip
    if shutil.which("docker") is None:
        unavailable("Docker is not installed")
    mp = pytest.MonkeyPatch()
    mp.setenv("OPERON_DESKTOP_PLATFORM", "linux")
    provider = get_provider("linux-container")
    check = provider.check()
    if not check.ok:
        mp.undo()
        unavailable("linux-container not ready: " + " ".join(check.problems))
    provider.start()
    try:
        yield provider
    finally:
        provider.stop()
        mp.undo()

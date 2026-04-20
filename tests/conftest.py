"""Shared pytest configuration and auto-skip guards."""

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _sync_bg_writer():
    """Force bg_writer to write synchronously during tests.

    Ensures tests that assert Path(...).exists() immediately after a service
    call see the file without having to await a background thread.
    """
    from src.store.background_writer import bg_writer
    bg_writer._sync = True
    yield
    bg_writer._sync = False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-skip tests that target a different platform.

    Any test with 'windows' in its name skips on non-Windows.
    Any test with 'linux' in its name skips on non-Linux.
    This prevents CI failures from platform-specific tests without
    requiring manual @pytest.mark.skipif on every test.
    """
    for item in items:
        name = item.name.lower()
        if "windows" in name or "win32" in name or "windir" in name or "system_root" in name:
            if os.name != "nt":
                item.add_marker(pytest.mark.skip(reason="Windows-only test (auto-detected by conftest)"))
        elif "linux" in name or "posix" in name:
            if sys.platform != "linux":
                item.add_marker(pytest.mark.skip(reason="Linux-only test (auto-detected by conftest)"))

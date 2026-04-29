"""Shared pytest configuration and auto-skip guards."""

import os
import sys

import pytest

# Prevent any AgentLoop from calling reset_desktop() (Win+D) or other
# real desktop side-effects during the test suite.
os.environ.setdefault("OPERON_TEST_SAFE_MODE", "true")

# Pre-import modules that test stubs (test_phase4_integration.py) replace via
# sys.modules.setdefault — ensures the real implementations are loaded first so
# stub injection is a no-op and doesn't break tests that depend on real behaviour.
import src.agent.screen_diff  # noqa: F401, E402
import src.benchmarks.form_plugin  # noqa: F401, E402


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register opt-in flags for tests that hit a live local server."""
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that talk to a live local Operon server and may touch the real desktop/browser.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Document custom markers used by the suite."""
    config.addinivalue_line(
        "markers",
        "live_server: test requires --live or OPERON_RUN_LIVE_SERVER_TESTS=true and may exercise a real local server",
    )


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

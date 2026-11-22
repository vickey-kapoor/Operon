"""Shared pytest configuration and auto-skip guards."""

import os
import sys

import pytest

# Prevent any AgentLoop from calling reset_desktop() (Win+D) or other
# real desktop side-effects during the test suite.
os.environ.setdefault("OPERON_TEST_SAFE_MODE", "true")
# KAN-158: a bug (AttributeError, NameError, ...) caught by one of the agent loop's
# recovery sites re-raises in tests instead of passing as an ordinary recovery.
os.environ.setdefault("OPERON_STRICT_ERRORS", "true")
# Starlette's TestClient and httpx test transports send Host: testserver / test.
os.environ.setdefault("OPERON_ALLOWED_HOSTS", "testserver,test")
# Offline tests call the API without a key; importing the server must not write
# .var/api-key. Auth tests turn this off to check the default.
os.environ.setdefault("OPERON_ALLOW_UNAUTHENTICATED", "true")
# Never read the developer's real ~/.operon key file.
os.environ["OPERON_CONFIG_DIR"] = os.path.join(os.path.dirname(__file__), "..", ".var", "test-user-config")

# KAN-186: never touch a developer's real credential store (Windows Credential Manager, Keychain, ...).
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
import keyring  # noqa: E402
import keyring.backends.null  # noqa: E402

keyring.set_keyring(keyring.backends.null.Keyring())


class _MemoryKeyring(keyring.backend.KeyringBackend):
    """A dict-backed keyring for tests that need a working store."""

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.data.get((service, username))

    def set_password(self, service, username, password):
        self.data[(service, username)] = password

    def delete_password(self, service, username):
        from keyring.errors import PasswordDeleteError

        if (service, username) not in self.data:
            raise PasswordDeleteError("not found")
        del self.data[(service, username)]


@pytest.fixture
def memory_keyring():
    previous = keyring.get_keyring()
    backend = _MemoryKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(previous)


# Pre-import modules that test stubs (test_phase4_integration.py) replace via
# sys.modules.setdefault — ensures the real implementations are loaded first so
# stub injection is a no-op and doesn't break tests that depend on real behaviour.
import operon.agent.perception.screen_diff  # noqa: F401, E402


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
def _reset_active_browser_manager():
    """Reset the runtime's active BrowserManager after each test.

    _ensure_session routes through CDP when get_active_manager() is non-None.
    Without this reset, a test that sets an active manager pollutes subsequent
    tests that expect the batch routing path.
    """
    from operon.browser.manager import set_active_manager

    yield
    set_active_manager(None)


@pytest.fixture(autouse=True)
def _reset_runtime_loops():
    """Drop agent loops a test built lazily (for example a request through the real app).

    A loop left on the runtime would leak into whichever test comes next.
    """
    from operon.core.runtime import current_runtime

    yield
    runtime = current_runtime()
    runtime.browser_loop = None
    runtime.desktop_loop = None
    runtime.desktop_provider = None


@pytest.fixture(autouse=True)
def _sync_bg_writer():
    """Force bg_writer to write synchronously during tests.

    Ensures tests that assert Path(...).exists() immediately after a service
    call see the file without having to await a background thread.
    """
    from operon.store.background_writer import bg_writer
    bg_writer._sync = True
    yield
    bg_writer._sync = False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip tests marked for a different OS (@pytest.mark.windows_only / linux_only)."""
    for item in items:
        if item.get_closest_marker("windows_only") and os.name != "nt":
            item.add_marker(pytest.mark.skip(reason="Windows-only test"))
        if item.get_closest_marker("linux_only") and sys.platform != "linux":
            item.add_marker(pytest.mark.skip(reason="Linux-only test"))

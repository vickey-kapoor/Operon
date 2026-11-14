"""Shared setup for the eval-tool tests."""

from __future__ import annotations

import socketserver

import pytest


@pytest.fixture(autouse=True, scope="session")
def _fast_fixture_site_shutdown():
    """Stop the local fixture site in ~0.05 s instead of ~0.5 s.

    socketserver's serve_forever() notices shutdown() only between polls (default every 0.5 s), so every
    FixtureSite.stop() waited about half a second, once per test. The fix lives here, not in
    evals/fixture_site.py, because frozen seasons fingerprint the eval code and would refuse to run.
    """
    original = socketserver.BaseServer.serve_forever

    def serve_forever(self, poll_interval: float = 0.05) -> None:
        original(self, poll_interval)

    patch = pytest.MonkeyPatch()
    patch.setattr(socketserver.BaseServer, "serve_forever", serve_forever)
    yield
    patch.undo()

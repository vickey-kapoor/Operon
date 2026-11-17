"""Fixtures for the browser end-to-end tests (``-m browser_e2e``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.fixture_site import FixtureSite

from .site import TestSite


def pytest_collection_modifyitems(items):
    for item in items:
        if "browser_e2e" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.browser_e2e)


@pytest.fixture(scope="session")
def chromium():
    """Skip cleanly when Playwright's Chromium is not installed (never downloaded here)."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            installed = Path(p.chromium.executable_path).exists()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Playwright is unavailable: {exc}")
    if not installed:
        pytest.skip("Playwright Chromium is not installed (playwright install chromium)")


@pytest.fixture(scope="session")
def _site_server(chromium):
    app = TestSite()
    site = FixtureSite({}, apps=[app]).start()
    try:
        yield site, app
    finally:
        site.stop()


@pytest.fixture
def site(_site_server):
    server, _app = _site_server
    server.reset()
    return server


@pytest.fixture
def app(_site_server, site):
    return _site_server[1]

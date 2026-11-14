"""The built-in browser's routes (KAN-138): off by default, then tabs, sites and sign-out."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from operon.browser.builtin import SiteApprovals


@pytest.fixture
def client():
    from operon.api.server import app

    return TestClient(app)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("OPERON_BUILTIN_BROWSER", "true")
    browser = MagicMock()
    browser.started = True
    browser.approvals = SiteApprovals(tmp_path)
    browser.tabs = AsyncMock(return_value=[{"id": 1, "url": "https://a.test", "title": "A", "active": True}])
    for name in ("show", "navigate", "back", "forward", "reload", "open_tab", "select_tab", "close_tab", "sign_out_everywhere"):
        setattr(browser, name, AsyncMock())
    browser.current_page = AsyncMock(return_value=object())
    monkeypatch.setattr("operon.api.routes.builtin_browser.get_builtin_browser", lambda: browser)
    return browser


def test_the_browser_is_off_by_default(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("OPERON_BUILTIN_BROWSER", raising=False)
    assert client.get("/app/api/browser").json() == {"enabled": False}
    resp = client.post("/app/api/browser/navigate", json={"address": "example.com"})
    assert resp.status_code == 404 and "OPERON_BUILTIN_BROWSER" in resp.json()["detail"]


def test_state_lists_tabs_and_sites(client: TestClient, fake) -> None:
    fake.approvals.decide("a.test", "always")
    fake.approvals.ask("b.test", "run-1", "https://b.test/")
    state = client.get("/app/api/browser").json()
    assert state["enabled"] and state["tabs"][0]["title"] == "A"
    assert state["approved_sites"] == ["a.test"]
    assert state["pending_sites"][0]["host"] == "b.test"


def test_open_puts_the_current_tab_in_the_live_view(client: TestClient, fake) -> None:
    assert client.post("/app/api/browser/open").status_code == 200
    fake.show.assert_awaited_once()


def test_navigation_and_tabs_reach_the_browser(client: TestClient, fake) -> None:
    assert client.post("/app/api/browser/navigate", json={"address": "example.com"}).status_code == 200
    fake.navigate.assert_awaited_once_with("example.com")
    client.post("/app/api/browser/back")
    client.post("/app/api/browser/forward")
    client.post("/app/api/browser/reload")
    client.post("/app/api/browser/tabs", json={"address": "b.test"})
    client.post("/app/api/browser/tabs/1/select")
    client.delete("/app/api/browser/tabs/1")
    fake.back.assert_awaited_once()
    fake.open_tab.assert_awaited_once_with("b.test")
    fake.select_tab.assert_awaited_once_with(1)
    fake.close_tab.assert_awaited_once_with(1)


def test_a_bad_address_is_a_400_and_a_missing_tab_a_404(client: TestClient, fake) -> None:
    fake.navigate.side_effect = ValueError("Type an address or a search.")
    fake.select_tab.side_effect = KeyError("no tab 9")
    assert client.post("/app/api/browser/navigate", json={"address": " "}).status_code == 400
    assert client.post("/app/api/browser/tabs/9/select").status_code == 404


def test_answering_a_site_and_forgetting_it(client: TestClient, fake) -> None:
    fake.approvals.ask("b.test", "run-1", "https://b.test/")
    resp = client.post("/app/api/browser/sites", json={"host": "b.test", "answer": "always"})
    assert resp.json()["approved_sites"] == ["b.test"] and resp.json()["pending_sites"] == []
    assert client.post("/app/api/browser/sites", json={"host": "b.test", "answer": "maybe"}).status_code == 422
    assert client.delete("/app/api/browser/sites/b.test").json()["approved_sites"] == []


def test_sign_out_everywhere(client: TestClient, fake) -> None:
    assert client.post("/app/api/browser/sign-out").status_code == 200
    fake.sign_out_everywhere.assert_awaited_once()


def test_input_needs_an_open_browser_then_reaches_the_page(client: TestClient, fake, monkeypatch) -> None:
    monkeypatch.setattr("operon.browser.manager.get_active_manager", lambda: None)
    body = {"input_type": "text", "text": "hi"}
    assert client.post("/app/api/browser/input", json=body).status_code == 409

    manager = MagicMock(is_connected=True)
    manager.inject_input = AsyncMock()
    monkeypatch.setattr("operon.browser.manager.get_active_manager", lambda: manager)
    assert client.post("/app/api/browser/input", json=body).status_code == 200
    args, kwargs = manager.inject_input.call_args
    assert args == (0.0, 0.0, "text") and kwargs["text"] == "hi"
    assert client.post("/app/api/browser/input", json={"input_type": "drag"}).status_code == 422

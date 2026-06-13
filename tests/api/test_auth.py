"""API-key authentication: opt-in via API_KEYS, /health stays open."""

import pytest
from fastapi.testclient import TestClient

from operon.api.server import app

_RUN_BODY = {"intent": "Navigate and click a button.", "headless": True}


def _client() -> TestClient:
    return TestClient(app)


def test_no_keys_configured_allows_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    resp = _client().post("/run-task", json=_RUN_BODY)
    assert resp.status_code == 202  # auth disabled → unchanged behavior


def test_missing_key_is_rejected_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    resp = _client().post("/run-task", json=_RUN_BODY)
    assert resp.status_code == 401


def test_wrong_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    resp = _client().post("/run-task", json=_RUN_BODY, headers={"X-API-Key": "nope"})
    assert resp.status_code == 401


def test_correct_key_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret,other-key")
    resp = _client().post("/run-task", json=_RUN_BODY, headers={"X-API-Key": "other-key"})
    assert resp.status_code == 202


def test_health_stays_open_even_with_keys_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

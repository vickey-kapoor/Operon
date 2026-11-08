"""API-key authentication: on by default with a generated key, /health stays open."""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from operon.api.auth import api_key_file, client_api_key, local_api_key
from operon.api.server import app, create_app

_RUN_BODY = {"intent": "Navigate and click a button.", "headless": True}


def _client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def default_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No API_KEYS, no opt-out, and a fresh runtime dir for the generated key."""
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("OPERON_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("OPERON_RUNTIME_ROOT", str(tmp_path))
    return tmp_path / "api-key"


def test_no_keys_configured_requires_generated_key(default_auth: Path) -> None:
    client = TestClient(create_app())
    assert default_auth.is_file()
    key = default_auth.read_text(encoding="utf-8").strip()
    assert len(key) >= 32
    assert client.post("/run-task", json=_RUN_BODY).status_code == 401
    resp = client.post("/run-task", json=_RUN_BODY, headers={"X-API-Key": key})
    assert resp.status_code == 202


def test_generated_key_survives_restart(default_auth: Path) -> None:
    first = local_api_key()
    assert api_key_file() == default_auth
    # a fresh process reads the same file instead of generating a new key
    from operon.api import auth

    auth._local_keys.clear()
    assert local_api_key() == first


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_generated_key_file_is_private(default_auth: Path) -> None:
    local_api_key()
    assert os.stat(default_auth).st_mode & 0o077 == 0


def test_client_api_key_reads_env_then_file(default_auth: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client_api_key() is None  # never creates the file
    assert not default_auth.exists()
    key = local_api_key()
    assert client_api_key() == key
    monkeypatch.setenv("API_KEYS", "first, second")
    assert client_api_key() == "first"


def test_opt_out_allows_requests(default_auth: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_ALLOW_UNAUTHENTICATED", "true")
    resp = TestClient(create_app()).post("/run-task", json=_RUN_BODY)
    assert resp.status_code == 202
    assert not default_auth.exists()


def test_api_keys_win_over_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_ALLOW_UNAUTHENTICATED", "true")
    monkeypatch.setenv("API_KEYS", "s3cret")
    assert _client().post("/run-task", json=_RUN_BODY).status_code == 401


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


def test_rebound_host_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEYS", raising=False)
    resp = _client().post("/run-task", json=_RUN_BODY, headers={"Host": "attacker.example:8080"})
    assert resp.status_code == 400


def test_loopback_hosts_are_accepted() -> None:
    for host in ("127.0.0.1:8080", "localhost", "[::1]:8080"):
        assert _client().get("/health", headers={"Host": host}).status_code == 200


def test_extra_allowed_host_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_ALLOWED_HOSTS", "operon.internal")
    assert _client().get("/health", headers={"Host": "operon.internal:8080"}).status_code == 200
    assert _client().get("/health", headers={"Host": "evil.operon.internal"}).status_code == 400

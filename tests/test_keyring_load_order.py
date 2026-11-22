"""KAN-186: key order for the CLI and the API: real environment, credential store, .env, operon.env."""

from __future__ import annotations

import os

import keyring
import keyring.backend
import pytest
from fastapi.testclient import TestClient

from operon import cli
from operon.api.server import create_app


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """A working folder with a .env and a per-user operon.env, both holding GEMINI_API_KEY."""
    work = tmp_path / "work"
    work.mkdir()
    (work / ".env").write_text("GEMINI_API_KEY=from-dotenv\n")
    config = tmp_path / "config"
    config.mkdir()
    (config / "operon.env").write_text("GEMINI_API_KEY=from-operon-env\nOPENAI_API_KEY=user-file-openai\n")
    monkeypatch.chdir(work)
    monkeypatch.setenv("OPERON_CONFIG_DIR", str(config))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPERON_KEYRING", "on")
    return work


def test_cli_keyring_beats_dotenv(folder, memory_keyring, monkeypatch) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", "from-keyring")
    cli._load_env()
    assert os.environ["GEMINI_API_KEY"] == "from-keyring"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")


def test_cli_real_env_beats_keyring(folder, memory_keyring, monkeypatch) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", "from-keyring")
    monkeypatch.setenv("GEMINI_API_KEY", "from-real-env")
    cli._load_env()
    assert os.environ["GEMINI_API_KEY"] == "from-real-env"
    monkeypatch.delenv("OPENAI_API_KEY")


def test_cli_dotenv_still_beats_operon_env(folder, monkeypatch) -> None:
    cli._load_env()  # the null backend: no store, so the files decide
    assert os.environ["GEMINI_API_KEY"] == "from-dotenv"
    assert os.environ["OPENAI_API_KEY"] == "user-file-openai"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")


def test_cli_keyring_off_leaves_files_in_charge(folder, memory_keyring, monkeypatch) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", "from-keyring")
    monkeypatch.setenv("OPERON_KEYRING", "off")
    cli._load_env()
    assert os.environ["GEMINI_API_KEY"] == "from-dotenv"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")


def test_create_app_keyring_beats_dotenv_despite_override(folder, memory_keyring, monkeypatch) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", "from-keyring")
    create_app()
    assert os.environ["GEMINI_API_KEY"] == "from-keyring"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_create_app_real_env_beats_dotenv_and_keyring(folder, memory_keyring, monkeypatch) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", "from-keyring")
    monkeypatch.setenv("GEMINI_API_KEY", "from-real-env")
    create_app()
    assert os.environ["GEMINI_API_KEY"] == "from-real-env"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_create_app_survives_a_raising_backend(folder, monkeypatch) -> None:
    class Boom(keyring.backend.KeyringBackend):
        priority = 1  # type: ignore[assignment]

        def get_password(self, service, username):
            raise RuntimeError("store locked")

        def set_password(self, service, username, password):
            raise RuntimeError("store locked")

        def delete_password(self, service, username):
            raise RuntimeError("store locked")

    previous = keyring.get_keyring()
    keyring.set_keyring(Boom())
    try:
        client = TestClient(create_app())
    finally:
        keyring.set_keyring(previous)
    assert client.get("/health").status_code == 200
    assert os.environ["GEMINI_API_KEY"] == "from-dotenv"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_create_app_real_env_beats_dotenv_with_no_store(folder, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "from-real-env")  # null backend: no store at all
    create_app()
    assert os.environ["GEMINI_API_KEY"] == "from-real-env"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_serve_path_reuses_the_first_snapshot(folder, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "from-real-env")
    cli._load_env()  # then create_app sees the already-loaded environment, not a new snapshot
    create_app()
    from operon.core.paths import loaded_key_state

    assert loaded_key_state()[1]["GEMINI_API_KEY"] == "environment"
    assert os.environ["GEMINI_API_KEY"] == "from-real-env"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

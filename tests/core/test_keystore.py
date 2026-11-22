"""KAN-186: model API keys from the OS credential store."""

from __future__ import annotations

import logging
import os
import sys

import keyring
import keyring.backend
import keyring.backends.fail
import keyring.backends.null
import pytest

from operon.core import keystore
from operon.core.paths import load_stored_keys

SECRET = "sk-test-secret-value-123"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (*keystore.KNOWN_KEYS, "OPERON_KEYRING", "MY_CUSTOM_KEY", "OPERON_MODEL_REGISTRY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPERON_KEYRING", "on")  # auto is off on Linux; these tests exercise the store


def test_loader_sets_env_from_keyring(memory_keyring, monkeypatch) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", SECRET)
    sources = load_stored_keys({})
    assert os.environ["GEMINI_API_KEY"] == SECRET
    assert sources == {"GEMINI_API_KEY": keystore.store_label()}
    monkeypatch.delenv("GEMINI_API_KEY")


def test_real_env_beats_keyring(memory_keyring, monkeypatch) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("GEMINI_API_KEY", "from-file")  # as a .env override=True would leave it
    sources = load_stored_keys({"GEMINI_API_KEY": "from-real-env"})
    assert os.environ["GEMINI_API_KEY"] == "from-real-env"
    assert sources["GEMINI_API_KEY"] == "environment"


def test_blank_real_env_counts_as_unset(memory_keyring, monkeypatch) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", SECRET)
    load_stored_keys({"GEMINI_API_KEY": "   "})
    assert os.environ["GEMINI_API_KEY"] == SECRET
    monkeypatch.delenv("GEMINI_API_KEY")


def test_unknown_names_are_ignored(memory_keyring) -> None:
    memory_keyring.set_password("operon", "SOMETHING_ELSE", SECRET)
    assert load_stored_keys({}) == {}
    assert "SOMETHING_ELSE" not in os.environ


def test_raising_backend_is_tolerated_and_never_logged(caplog) -> None:
    class Boom(keyring.backend.KeyringBackend):
        priority = 1  # type: ignore[assignment]

        def get_password(self, service, username):
            raise RuntimeError(f"cannot read {SECRET}")

        def set_password(self, service, username, password):
            raise RuntimeError("no")

        def delete_password(self, service, username):
            raise RuntimeError("no")

    previous = keyring.get_keyring()
    keyring.set_keyring(Boom())
    try:
        with caplog.at_level(logging.DEBUG):
            assert load_stored_keys({}) == {}
    finally:
        keyring.set_keyring(previous)
    assert SECRET not in caplog.text


@pytest.mark.parametrize("backend", [keyring.backends.fail.Keyring, keyring.backends.null.Keyring])
def test_fail_and_null_backends_are_tolerated(backend) -> None:
    previous = keyring.get_keyring()
    keyring.set_keyring(backend())
    try:
        assert keystore.available() is False
        assert load_stored_keys({}) == {}
    finally:
        keyring.set_keyring(previous)


def test_missing_keyring_package_is_tolerated(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    assert keystore.available() is False
    assert keystore.get("GEMINI_API_KEY") is None
    assert load_stored_keys({}) == {}


def test_keyring_off_never_calls_the_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_KEYRING", "off")
    called: list[object] = []
    monkeypatch.setattr(keystore, "available", lambda: called.append("available") or True)
    monkeypatch.setattr(keystore, "get", lambda name: called.append(name))
    assert load_stored_keys({}) == {}
    assert called == []


def test_known_names_are_the_fixed_list_only(monkeypatch) -> None:
    from operon.agent.backends.model_adapter import registry as reg

    class Profile:
        api_key_env = "MY_CUSTOM_KEY"

    monkeypatch.setattr(reg, "load_registry", lambda path=None: {"custom": Profile()})
    assert keystore.known_names() == keystore.KNOWN_KEYS


@pytest.mark.parametrize(
    ("platform", "setting", "expected"),
    [
        ("win32", None, True), ("darwin", None, True), ("linux", None, False),
        ("linux", "auto", False), ("linux", "on", True), ("win32", "off", False), ("darwin", "off", False),
    ],
)
def test_store_use_by_platform(monkeypatch, platform, setting, expected) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    if setting is None:
        monkeypatch.delenv("OPERON_KEYRING")
    else:
        monkeypatch.setenv("OPERON_KEYRING", setting)
    assert keystore.enabled() is expected


def test_linux_auto_never_calls_the_backend_but_restores_real_env(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("OPERON_KEYRING")
    monkeypatch.setattr(keystore, "available", lambda: pytest.fail("backend touched"))
    monkeypatch.setenv("GEMINI_API_KEY", "from-file")
    assert load_stored_keys({"GEMINI_API_KEY": "from-real-env"}) == {"GEMINI_API_KEY": "environment"}
    assert os.environ["GEMINI_API_KEY"] == "from-real-env"


def test_real_env_restored_when_keyring_off_or_no_store(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "from-file")
    monkeypatch.setenv("OPERON_KEYRING", "off")
    load_stored_keys({"GEMINI_API_KEY": "real"})
    assert os.environ["GEMINI_API_KEY"] == "real"
    monkeypatch.setenv("GEMINI_API_KEY", "from-file")
    monkeypatch.setenv("OPERON_KEYRING", "on")  # null backend: no store
    load_stored_keys({"GEMINI_API_KEY": "real"})
    assert os.environ["GEMINI_API_KEY"] == "real"

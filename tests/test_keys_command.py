"""KAN-186: ``operon keys`` set, list, rm, import. No output ever holds a value."""

from __future__ import annotations

import getpass
import os

import keyring
import pytest

from operon import cli
from operon.core import keystore

SECRET = "sk-very-secret-value-9876"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("OPERON_CONFIG_DIR", str(config))
    monkeypatch.setenv("OPERON_KEYRING", "on")
    for name in keystore.KNOWN_KEYS:
        monkeypatch.delenv(name, raising=False)
    from operon.core import paths

    paths._real_env.clear()
    paths._key_sources.clear()
    snapshot = dict(os.environ)  # cli.main loads .env files into the process: put it back afterwards
    yield work
    os.environ.clear()
    os.environ.update(snapshot)


def _answers(monkeypatch, *values: str) -> None:
    queue = list(values)
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": queue.pop(0))


def test_set_stores_and_prints_only_the_name(memory_keyring, monkeypatch, capsys) -> None:
    _answers(monkeypatch, SECRET, SECRET)
    assert cli.main(["keys", "set", "GEMINI_API_KEY"]) == 0
    out = capsys.readouterr()
    assert memory_keyring.get_password("operon", "GEMINI_API_KEY") == SECRET
    assert "GEMINI_API_KEY" in out.out
    assert SECRET not in out.out + out.err


def test_set_mismatch_saves_nothing(memory_keyring, monkeypatch, capsys) -> None:
    _answers(monkeypatch, SECRET, "something-else")
    assert cli.main(["keys", "set", "GEMINI_API_KEY"]) == 1
    assert memory_keyring.get_password("operon", "GEMINI_API_KEY") is None
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err


def test_set_unknown_name_is_refused(memory_keyring, monkeypatch) -> None:
    _answers(monkeypatch)  # no prompt may be reached
    assert cli.main(["keys", "set", "NOT_A_KEY"]) == 2


def test_set_without_a_store_explains(monkeypatch, capsys) -> None:
    _answers(monkeypatch)
    assert cli.main(["keys", "set", "GEMINI_API_KEY"]) == 1
    assert "no credential store available here; keep using .env or environment variables" in capsys.readouterr().err


def test_list_shows_sources_and_no_values(memory_keyring, isolated, tmp_path, monkeypatch, capsys) -> None:
    memory_keyring.set_password("operon", "ANTHROPIC_API_KEY", SECRET)
    (isolated / ".env").write_text(f"OPENAI_API_KEY={SECRET}\n")
    (tmp_path / "config" / "operon.env").write_text(f"GOOGLE_API_KEY={SECRET}\n")
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    assert cli.main(["keys", "list"]) == 0
    out = capsys.readouterr().out
    lines = dict(line.split(": ") for line in out.strip().splitlines())
    assert lines["GEMINI_API_KEY"] == "environment"
    assert lines["ANTHROPIC_API_KEY"] == "Credential Manager"
    assert lines["OPENAI_API_KEY"] == ".env"
    assert lines["GOOGLE_API_KEY"] == "operon.env"
    assert lines["API_KEYS"] == "not set"
    assert SECRET not in out
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_rm_present_and_absent(memory_keyring, capsys) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", SECRET)
    assert cli.main(["keys", "rm", "GEMINI_API_KEY"]) == 0
    assert memory_keyring.get_password("operon", "GEMINI_API_KEY") is None
    assert cli.main(["keys", "rm", "GEMINI_API_KEY"]) == 1
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err


def test_import_copies_known_names_only_and_leaves_env_alone(memory_keyring, isolated, capsys) -> None:
    env_file = isolated / ".env"
    env_file.write_bytes(f"GEMINI_API_KEY={SECRET}\nOPERON_PERMISSION_MODE=ask\nOTHER_TOKEN={SECRET}\n".encode())
    before = env_file.read_bytes()
    assert cli.main(["keys", "import"]) == 0
    out = capsys.readouterr()
    assert memory_keyring.get_password("operon", "GEMINI_API_KEY") == SECRET
    assert memory_keyring.get_password("operon", "OTHER_TOKEN") is None
    assert memory_keyring.get_password("operon", "OPERON_PERMISSION_MODE") is None
    assert env_file.read_bytes() == before
    assert "GEMINI_API_KEY=..." in out.out
    assert str(env_file) in out.out
    assert SECRET not in out.out + out.err
    assert keyring.get_keyring() is memory_keyring


def test_unknown_name_and_extra_args_are_never_echoed(memory_keyring, monkeypatch, capsys) -> None:
    _answers(monkeypatch)
    assert cli.main(["keys", "set", SECRET]) == 2
    assert cli.main(["keys", "set", "GEMINI_API_KEY", SECRET]) == 2
    assert cli.main(["keys", "rm", SECRET]) == 2
    assert cli.main(["keys", "rm", "GEMINI_API_KEY", SECRET]) == 2
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    assert "never pass a key value on the command line" in out.err
    assert memory_keyring.data == {}


@pytest.mark.parametrize("command", [["set", "GEMINI_API_KEY"], ["rm", "GEMINI_API_KEY"], ["import"]])
def test_store_off_on_linux_by_default(memory_keyring, isolated, monkeypatch, capsys, command) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("OPERON_KEYRING")
    (isolated / ".env").write_text(f"GEMINI_API_KEY={SECRET}\n")
    _answers(monkeypatch)
    assert cli.main(["keys", *command]) == 1
    err = capsys.readouterr().err
    assert "OPERON_KEYRING=on" in err
    assert memory_keyring.data == {}


def test_rm_without_a_store_says_so(capsys) -> None:
    assert cli.main(["keys", "rm", "GEMINI_API_KEY"]) == 1
    assert "no credential store available here" in capsys.readouterr().err


def test_list_does_not_read_the_store_when_it_is_off(memory_keyring, monkeypatch, capsys) -> None:
    memory_keyring.set_password("operon", "GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("OPERON_KEYRING", "off")
    monkeypatch.setattr(keystore, "get", lambda name: pytest.fail("store read while off"))
    assert cli.main(["keys", "list"]) == 0
    assert "GEMINI_API_KEY: not set" in capsys.readouterr().out

"""The secrets vault and ``operon secrets`` (KAN-103): storage, names, placeholders, CLI."""

from __future__ import annotations

import stat
import sys

import pytest

from operon.cli import main
from operon.core import vault
from operon.models.policy import ActionType, AgentAction


def test_round_trip_with_this_platforms_storage(vault_dir) -> None:
    vault.set_secret("bank_pw", "hunter2-SECRET")
    assert vault.get_secret("bank_pw") == "hunter2-SECRET"
    assert vault.list_secrets() == ["bank_pw"]
    on_disk = (vault_dir / "bank_pw.secret").read_bytes()
    if sys.platform == "win32":
        assert b"hunter2-SECRET" not in on_disk  # DPAPI blob, not plaintext
    assert vault.delete_secret("bank_pw") is True
    assert vault.delete_secret("bank_pw") is False
    assert vault.get_secret("bank_pw") is None


def test_unicode_and_replace(vault_dir) -> None:
    vault.set_secret("a", "pä$$ wörd 東京")
    assert vault.get_secret("a") == "pä$$ wörd 東京"
    vault.set_secret("a", "second")
    assert vault.get_secret("a") == "second"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_posix_files_are_private(vault_dir) -> None:
    vault.set_secret("pw", "x")
    assert stat.S_IMODE((vault_dir / "pw.secret").stat().st_mode) == 0o600
    assert stat.S_IMODE(vault_dir.stat().st_mode) == 0o700


def test_posix_storage_path_with_patched_platform(vault_dir, monkeypatch) -> None:
    """The non-DPAPI branch runs on any OS: the value is stored as plain bytes."""
    monkeypatch.setattr(vault, "_use_dpapi", lambda: False)
    vault.set_secret("pw", "plain-value")
    assert (vault_dir / "pw.secret").read_bytes() == b"plain-value"
    assert vault.get_secret("pw") == "plain-value"


@pytest.mark.parametrize("name", ["", "a b", "../x", "x" * 65, "pw.secret", "naïve", "a/b", "a{b"])
def test_bad_names_are_refused(vault_dir, name) -> None:
    assert not vault.valid_name(name)
    with pytest.raises(vault.SecretError):
        vault.set_secret(name, "v")
    with pytest.raises(vault.SecretError):
        vault.get_secret(name)


@pytest.mark.parametrize("name", ["pw", "A-b_9", "x" * 64])
def test_good_names(name) -> None:
    assert vault.valid_name(name)


def test_empty_value_refused(vault_dir) -> None:
    with pytest.raises(vault.SecretError):
        vault.set_secret("pw", "")


def test_resolve_replaces_placeholders(vault_dir) -> None:
    vault.set_secret("user", "ann")
    vault.set_secret("pw", "s3cret")
    text, values = vault.resolve_text("{{secret:user}}:{{ secret:pw }}!")
    assert text == "ann:s3cret!" and set(values) == {"ann", "s3cret"}
    assert vault.resolve_text("plain {braces}") == ("plain {braces}", ())


def test_unknown_name_fails_without_leaking(vault_dir) -> None:
    with pytest.raises(vault.SecretError, match="unknown secret nope"):
        vault.resolve_text("{{secret:nope}}")


def test_flag_off_fails_instead_of_typing_the_literal(vault_dir, monkeypatch) -> None:
    monkeypatch.setenv("OPERON_SECRETS", "off")
    vault.set_secret("pw", "v")
    with pytest.raises(vault.SecretError, match="secrets are off"):
        vault.resolve_text("{{secret:pw}}")
    assert vault.available_names() == []


def test_action_uses_secret_looks_inside_batches() -> None:
    typed = AgentAction(action_type=ActionType.TYPE, text="{{secret:pw}}")
    plain = AgentAction(action_type=ActionType.TYPE, text="hello")
    assert vault.action_uses_secret(typed) and not vault.action_uses_secret(plain)
    batch = AgentAction(action_type=ActionType.BATCH, actions=[plain, typed])
    assert vault.action_uses_secret(batch)
    # Only typing counts: a placeholder in another action's text is not typed.
    assert not vault.action_uses_secret(AgentAction(action_type=ActionType.WAIT_FOR_USER, text="{{secret:pw}}"))


def test_redact_walks_nested_values() -> None:
    out = vault.redact({"a": "x hunter2 y", "b": ["hunter2", 3], "c": {"d": "hunter2hunter2"}}, ["hunter2"])
    assert out == {"a": "x [secret] y", "b": ["[secret]", 3], "c": {"d": "[secret][secret]"}}


def test_hint_has_names_and_syntax_only(vault_dir) -> None:
    vault.set_secret("pw", "hunter2-SECRET")
    hint = vault.secrets_hint(vault.available_names())
    assert "pw" in hint and "{{secret:pw}}" in hint and "never ask the user to reveal a secret" in hint.lower()
    assert "hunter2-SECRET" not in hint


# -- CLI ---------------------------------------------------------------------------------


def test_cli_set_list_rm(vault_dir, monkeypatch, capsys) -> None:
    answers = iter(["hunter2-SECRET", "hunter2-SECRET"])
    monkeypatch.setattr("operon.secrets_command.getpass.getpass", lambda prompt="": next(answers))
    assert main(["secrets", "set", "bank_pw"]) == 0
    out = capsys.readouterr()
    assert "hunter2-SECRET" not in out.out + out.err
    assert vault.get_secret("bank_pw") == "hunter2-SECRET"

    assert main(["secrets", "list"]) == 0
    listed = capsys.readouterr().out
    assert listed.split() == ["bank_pw"] and "hunter2" not in listed

    assert main(["secrets", "rm", "bank_pw"]) == 0
    assert main(["secrets", "rm", "bank_pw"]) == 1
    assert vault.list_secrets() == []


def test_cli_set_mismatch_saves_nothing(vault_dir, monkeypatch) -> None:
    answers = iter(["one", "two"])
    monkeypatch.setattr("operon.secrets_command.getpass.getpass", lambda prompt="": next(answers))
    assert main(["secrets", "set", "pw"]) == 1
    assert vault.list_secrets() == []


def test_cli_set_never_takes_the_value_as_an_argument(vault_dir) -> None:
    with pytest.raises(SystemExit):
        main(["secrets", "set", "pw", "hunter2"])
    assert vault.list_secrets() == []


def test_cli_set_rejects_bad_name_before_prompting(vault_dir, monkeypatch) -> None:
    def boom(prompt=""):
        raise AssertionError("must not prompt")

    monkeypatch.setattr("operon.secrets_command.getpass.getpass", boom)
    assert main(["secrets", "set", "bad name"]) == 2

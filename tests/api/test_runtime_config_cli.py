"""``python -m operon.api.runtime_config``: the diagnostic printout and its exit code."""

from __future__ import annotations

from operon.api import runtime_config


def test_valid_configuration_prints_both_modes_and_exits_0(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runtime_config, "validate_modes", lambda settings=None: [])

    assert runtime_config.main() == 0

    out = capsys.readouterr().out
    assert "Desktop:" in out and "Browser:" in out
    assert "Configuration valid." in out


def test_invalid_configuration_lists_errors_and_exits_1(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runtime_config, "validate_modes", lambda settings=None: ["OPERON_BROWSER_BACKEND='x' is not one of [...]"])

    assert runtime_config.main() == 1

    out = capsys.readouterr().out
    assert "VALIDATION ERRORS:" in out
    assert "! OPERON_BROWSER_BACKEND='x'" in out
    assert "Configuration valid." not in out


def test_a_non_gemini_model_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_MODEL", "claude-opus")

    errors = runtime_config.validate_modes()

    assert any("OPERON_BROWSER_MODEL='claude-opus' is not a Gemini model" in e for e in errors)

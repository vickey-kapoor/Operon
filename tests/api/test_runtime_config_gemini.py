"""Operon runs Gemini only: every model setting defaults to Gemini 3.5 Flash, and only
the Gemini backends are accepted."""

import pytest

from operon.api.runtime_config import (
    browser_mode_config,
    desktop_mode_config,
    validate_modes,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("BROWSER", "DESKTOP"):
        for name in ("BACKEND", "MODEL"):
            monkeypatch.delenv(f"OPERON_{mode}_{name}", raising=False)


@pytest.mark.parametrize("config", [browser_mode_config, desktop_mode_config])
def test_every_model_defaults_to_gemini_3_5_flash(config) -> None:
    cfg = config()
    assert cfg.backend == "computer_use"
    assert cfg.primary_model == "gemini-3.5-flash"


@pytest.mark.parametrize("env", ["OPERON_BROWSER_BACKEND", "OPERON_DESKTOP_BACKEND"])
def test_the_json_backend_was_removed_with_the_full_harness(monkeypatch, env) -> None:
    monkeypatch.setenv(env, "json")
    errors = validate_modes()
    assert any(env in e and "json backend was removed" in e and "KAN-183" in e for e in errors)


@pytest.mark.parametrize("backend", ["claude_computer_use", "gpt"])
def test_other_model_backends_are_refused(monkeypatch, backend) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", backend)
    monkeypatch.setenv("OPERON_DESKTOP_BACKEND", backend)
    errors = validate_modes()
    assert any("OPERON_BROWSER_BACKEND" in e for e in errors)
    assert any("OPERON_DESKTOP_BACKEND" in e for e in errors)


def test_a_leftover_non_gemini_model_is_refused(monkeypatch) -> None:
    """An old .env naming a Claude or GPT model would fail on every call."""
    monkeypatch.setenv("OPERON_BROWSER_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("OPERON_DESKTOP_MODEL", "gpt-5")
    errors = validate_modes()
    assert any("OPERON_BROWSER_MODEL='claude-sonnet-4-6'" in e for e in errors)
    assert any("OPERON_DESKTOP_MODEL='gpt-5'" in e for e in errors)


def test_gemini_models_pass(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_MODEL", "gemini-2.5-flash")
    assert validate_modes() == []

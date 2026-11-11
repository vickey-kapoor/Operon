"""The full harness's step critic defaults: Gemini 3.5 Flash in both modes, a Claude model
for the Anthropic provider, and a refusal when the model belongs to the other provider."""

import pytest

from operon.api.runtime_config import (
    browser_mode_config,
    desktop_mode_config,
    validate_modes,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("BROWSER", "DESKTOP"):
        for name in ("BACKEND", "VERIFIER_PROVIDER", "VERIFIER_MODEL", "VERIFIER_THINKING"):
            monkeypatch.delenv(f"OPERON_{mode}_{name}", raising=False)


@pytest.mark.parametrize("config", [browser_mode_config, desktop_mode_config])
def test_the_critic_defaults_to_gemini_3_5_flash_in_both_modes(config) -> None:
    cfg = config()
    assert (cfg.verifier_provider, cfg.verifier_model, cfg.verifier_thinking) == ("gemini", "gemini-3.5-flash", "low")


@pytest.mark.parametrize(("mode", "config"), [("BROWSER", browser_mode_config), ("DESKTOP", desktop_mode_config)])
def test_the_anthropic_critic_gets_a_claude_model(monkeypatch, mode, config) -> None:
    monkeypatch.setenv(f"OPERON_{mode}_VERIFIER_PROVIDER", "anthropic")
    assert config().verifier_model == "claude-sonnet-5-5"


def test_a_critic_model_from_the_other_provider_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_VERIFIER_PROVIDER", "anthropic")
    monkeypatch.setenv("OPERON_BROWSER_VERIFIER_MODEL", "gemini-3.5-flash")
    assert any("OPERON_BROWSER_VERIFIER_MODEL" in e for e in validate_modes())
    monkeypatch.setenv("OPERON_BROWSER_VERIFIER_MODEL", "claude-sonnet-5-5")
    assert not any("VERIFIER_MODEL" in e for e in validate_modes())

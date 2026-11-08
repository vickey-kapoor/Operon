"""Model registry: the shipped evals/models.yaml loads, and bad profiles fail loudly."""

from __future__ import annotations

from pathlib import Path

import pytest

from operon.agent.backends.model_adapter.registry import (
    ModelRegistryError,
    load_registry,
    resolve_profile,
)


def test_shipped_registry_loads_every_profile() -> None:
    profiles = load_registry()
    assert {"claude-opus-5-5", "gemini-computer-use", "qwen3-vl-32b"} <= set(profiles)
    qwen = profiles["qwen3-vl-32b"]
    assert qwen.adapter == "openai_compat" and qwen.coords == "relative_1000"
    assert qwen.image.resize == "smart" and qwen.image.factor == 32
    assert qwen.pricing.gpu_hour_usd == 2.5
    assert profiles["gemini-computer-use"].environments == ("browser",)


def test_resolve_profile_reads_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "qwen3-vl-32b")
    assert resolve_profile().model == "Qwen/Qwen3-VL-32B-Instruct"


def test_unset_or_unknown_profile_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_MODEL_PROFILE", raising=False)
    with pytest.raises(ModelRegistryError, match="OPERON_MODEL_PROFILE"):
        resolve_profile()
    with pytest.raises(ModelRegistryError, match="unknown model profile 'nope'"):
        resolve_profile("nope")


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_typo_in_a_field_fails_at_load(tmp_path: Path) -> None:
    path = _write(tmp_path, "m:\n  adapter: openai_compat\n  model: x\n  cords: pixels\n")
    with pytest.raises(ModelRegistryError, match="cords"):
        load_registry(path)


def test_native_adapter_rejects_a_text_parser(tmp_path: Path) -> None:
    path = _write(tmp_path, "m:\n  adapter: claude_native\n  model: x\n  parser: ui_tars\n")
    with pytest.raises(ModelRegistryError, match="own action format"):
        load_registry(path)


def test_base_url_comes_from_the_named_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, "m:\n  adapter: openai_compat\n  model: x\n  base_url_env: MY_URL\n")
    profile = load_registry(path)["m"]
    monkeypatch.delenv("MY_URL", raising=False)
    with pytest.raises(ModelRegistryError, match="MY_URL"):
        profile.resolved_base_url()
    monkeypatch.setenv("MY_URL", "http://gpu:8000/v1/")
    assert profile.resolved_base_url() == "http://gpu:8000/v1"

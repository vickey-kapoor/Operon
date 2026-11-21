"""OPERON_GEMINI_SAFETY: opt-in relaxation of Gemini's own content filter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from operon.clients.gemini import GeminiHttpClient
from operon.clients.gemini_computer_use import GeminiComputerUseHttpClient
from operon.core.settings import Settings, SettingsError, check_startup_settings

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CATEGORIES = {
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
}


def test_default_and_valid_values() -> None:
    assert Settings.from_env({}).gemini_safety == "default"
    assert Settings.from_env({"OPERON_GEMINI_SAFETY": "High_Only"}).gemini_safety == "high_only"


def test_invalid_value_fails_validation() -> None:
    settings = Settings.from_env({"OPERON_GEMINI_SAFETY": "off"})
    with pytest.raises(SettingsError, match="OPERON_GEMINI_SAFETY='off'"):
        check_startup_settings(settings)


def _cu_payload() -> dict[str, Any]:
    client = GeminiComputerUseHttpClient(api_key="k", model="gemini-3.5-flash")
    return client._build_payload(contents=[])


def test_payload_has_no_safety_settings_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPERON_GEMINI_SAFETY", raising=False)
    assert "safetySettings" not in _cu_payload()


def test_payload_high_only_sets_four_categories(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_GEMINI_SAFETY", "high_only")
    safety = _cu_payload()["safetySettings"]
    assert {s["category"] for s in safety} == CATEGORIES
    assert {s["threshold"] for s in safety} == {"BLOCK_ONLY_HIGH"}


@pytest.mark.parametrize("mode,expected", [("default", False), ("high_only", True)])
def test_plain_gemini_client_matches(monkeypatch, mode: str, expected: bool) -> None:
    monkeypatch.setenv("OPERON_GEMINI_SAFETY", mode)
    sent: dict[str, Any] = {}

    class _Http:
        is_closed = False

        async def post(self, url: str, **kw: Any) -> Any:
            sent.update(json.loads(kw["content"]))
            raise RuntimeError("stop after capturing the request")

    client = GeminiHttpClient(api_key="k", model="gemini-3.5-flash")
    client._client = _Http()  # type: ignore[assignment]
    try:
        asyncio.run(client.generate_policy("hi"))
    except Exception:
        pass
    assert sent, "the request was never built"
    assert ("safetySettings" in sent) is expected


@pytest.mark.parametrize("name", ["live-bench.yml", "live-desktop-bench.yml"])
def test_workflow_input_maps_to_env(name: str) -> None:
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    on = yaml.safe_load(text)[True]
    assert on["workflow_dispatch"]["inputs"]["gemini_safety"]["default"] == "default"
    if "workflow_call" in on:
        assert on["workflow_call"]["inputs"]["gemini_safety"]["default"] == "default"
    assert "OPERON_GEMINI_SAFETY: ${{ inputs.gemini_safety || 'default' }}" in text

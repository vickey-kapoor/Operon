"""Sign-in guidance reaches the model only when the built-in browser is on (KAN-138)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.core.paths import prompts_dir
from operon.models.common import RunStatus
from operon.models.state import AgentState


@pytest.fixture
def backend() -> BrowserComputerUseBackend:
    return BrowserComputerUseBackend(
        client=SimpleNamespace(), prompt_path=prompts_dir() / "browser_computer_use_prompt.txt"
    )


def _state() -> AgentState:
    return AgentState(run_id="run-1", intent="Check my inbox", status=RunStatus.RUNNING)


def test_no_sign_in_guidance_by_default(backend, monkeypatch) -> None:
    monkeypatch.delenv("OPERON_BUILTIN_BROWSER", raising=False)
    assert "Signing in" not in backend._render_prompt(_state())


def test_sign_in_guidance_when_the_built_in_browser_is_on(backend, monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BUILTIN_BROWSER", "true")
    prompt = backend._render_prompt(_state())
    assert "Signing in" in prompt and "safety flow" in prompt and "do not type it yourself" in prompt

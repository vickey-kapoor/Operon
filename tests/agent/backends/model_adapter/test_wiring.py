"""backend=model wiring (KAN-168): a registry profile picks the backend, the harness is
thin and the model's own "done" ends the run; config validation names what is missing."""

from __future__ import annotations

import pytest

from operon.agent.backends.browser_claude_computer_use import (
    BrowserClaudeComputerUseBackend,
)
from operon.agent.backends.desktop_claude_computer_use import (
    DesktopClaudeComputerUseBackend,
)
from operon.agent.backends.model_adapter.backend import (
    GAVE_UP_PREFIX,
    ModelAdapterBackend,
)
from operon.agent.backends.model_adapter.openai_native import OpenAIComputerUseBackend
from operon.agent.backends.model_adapter.registry import ModelRegistryError
from operon.agent.policy.thin import OwnDoneVerifier, ThinPolicy, ThinRecovery
from operon.api.runtime.loops import _loop_layers
from operon.api.runtime.services import build_browser_services, build_desktop_services
from operon.api.runtime_config import validate_modes
from operon.models.common import RunStatus, StopReason
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState
from operon.models.verification import VerificationStatus


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPERON_MODEL_PROFILE", "OPENWEIGHT_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")


@pytest.mark.parametrize("backend", ["model", "computer_use"])
def test_every_backend_runs_the_thin_layers(monkeypatch, backend) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", backend)
    if backend == "model":
        monkeypatch.setenv("OPERON_MODEL_PROFILE", "claude-sonnet-5-5")
    layers = _loop_layers(services=build_browser_services(executor=None))
    assert isinstance(layers["policy_service"], ThinPolicy)
    assert isinstance(layers["verifier_service"], OwnDoneVerifier)
    assert isinstance(layers["recovery_manager"], ThinRecovery)
    assert set(layers) == {"policy_service", "verifier_service", "recovery_manager"}


def test_the_json_backend_was_removed(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "json")
    monkeypatch.setenv("OPERON_DESKTOP_BACKEND", "json")
    errors = validate_modes()
    assert any("OPERON_BROWSER_BACKEND" in e and "KAN-183" in e for e in errors)
    assert any("OPERON_DESKTOP_BACKEND" in e and "KAN-183" in e for e in errors)


def test_claude_profile_builds_the_browser_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "claude-sonnet-5-5")
    services = build_browser_services(executor=None)
    assert isinstance(services.policy_delegate, BrowserClaudeComputerUseBackend)
    assert services.perception_service is services.policy_delegate
    assert services.policy_delegate.client.model == "claude-sonnet-5-5"
    assert validate_modes() == []


def test_claude_profile_builds_the_desktop_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_DESKTOP_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "claude-sonnet-5-5")
    services = build_desktop_services()
    assert isinstance(services.policy_delegate, DesktopClaudeComputerUseBackend)
    assert [tool.get("name") for tool in services.policy_delegate.client.custom_tools] == ["launch_app"]


def test_gpt_profile_builds_the_openai_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "openai-computer-use")
    services = build_browser_services(executor=None)
    assert isinstance(services.policy_delegate, OpenAIComputerUseBackend)
    assert validate_modes() == []


def test_openai_compat_profile_builds_the_adapter(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "qwen3-vl-32b")
    monkeypatch.setenv("OPENWEIGHT_BASE_URL", "http://gpu:8000/v1")
    services = build_browser_services(executor=None)
    assert isinstance(services.policy_delegate, ModelAdapterBackend)
    assert validate_modes() == []


def test_validation_names_a_missing_profile_and_a_missing_key(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    errors = validate_modes()
    assert any("OPERON_MODEL_PROFILE" in e for e in errors)

    monkeypatch.setenv("OPERON_MODEL_PROFILE", "claude-sonnet-5-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert any("ANTHROPIC_API_KEY" in e for e in validate_modes())


def test_an_unknown_profile_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "no-such-model")
    with pytest.raises(ModelRegistryError):
        build_browser_services(executor=None)


def _state() -> AgentState:
    return AgentState(
        run_id="r1", intent="task", status=RunStatus.RUNNING, step_count=1, observation_history=[],
        action_history=[], verification_history=[],
    )


async def test_the_models_own_done_ends_the_run_unchecked() -> None:
    decision = PolicyDecision(action=AgentAction(action_type=ActionType.STOP), rationale="all done", confidence=1.0, active_subgoal="done")
    result = await OwnDoneVerifier(gave_up_prefix=GAVE_UP_PREFIX).verify(
        _state(), decision, ExecutedAction(action=decision.action, success=True, detail="ok")
    )
    assert result.status is VerificationStatus.SUCCESS
    assert result.stop_reason is StopReason.TASK_COMPLETED


async def test_a_give_up_never_counts_as_done() -> None:
    decision = PolicyDecision(action=AgentAction(action_type=ActionType.STOP), rationale=f"{GAVE_UP_PREFIX}stuck", confidence=1.0, active_subgoal="x")
    result = await OwnDoneVerifier(gave_up_prefix=GAVE_UP_PREFIX).verify(
        _state(), decision, ExecutedAction(action=decision.action, success=True, detail="ok")
    )
    assert result.status is VerificationStatus.FAILURE
    assert result.stop_reason is StopReason.COMPLETION_UNVERIFIED

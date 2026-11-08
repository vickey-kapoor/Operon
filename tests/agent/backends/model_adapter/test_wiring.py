"""backend=model wiring: profiles pick the backend, the harness is thin and the model's
own "done" ends the run; config validation names what is missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from operon.agent.backends.browser_claude_computer_use import (
    BrowserClaudeComputerUseBackend,
)
from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.agent.backends.desktop_claude_computer_use import (
    DesktopClaudeComputerUseBackend,
)
from operon.agent.backends.model_adapter.backend import (
    GAVE_UP_PREFIX,
    ModelAdapterBackend,
)
from operon.agent.backends.model_adapter.registry import ModelRegistryError
from operon.agent.policy.thin import (
    FINAL_CHECK,
    OWN_DONE,
    THIN,
    FinalCheckVerifier,
    OwnDoneVerifier,
    completion_check,
    harness_profile,
)
from operon.api.runtime.services import build_browser_services, build_desktop_services
from operon.api.runtime_config import validate_modes
from operon.models.common import RunStatus, StopReason
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState
from operon.models.verification import VerificationStatus


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPERON_HARNESS_PROFILE", "OPERON_COMPLETION_CHECK", "OPERON_MODEL_PROFILE", "OPENWEIGHT_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_backend_model_always_runs_thin_with_its_own_done(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", "full")
    assert harness_profile("model") == THIN
    assert completion_check("model") == OWN_DONE


def test_claude_keeps_the_final_check_unless_benchmark_mode_asks(monkeypatch) -> None:
    assert completion_check("claude_computer_use") == FINAL_CHECK
    monkeypatch.setenv("OPERON_COMPLETION_CHECK", "own")
    assert completion_check("claude_computer_use") == OWN_DONE


def test_openai_compat_profile_builds_the_adapter(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "qwen3-vl-32b")
    monkeypatch.setenv("OPENWEIGHT_BASE_URL", "http://gpu:8000/v1")
    services = build_browser_services(executor=None)
    assert isinstance(services.policy_delegate, ModelAdapterBackend)
    assert services.perception_service is services.policy_delegate
    assert services.policy_delegate.profile.name == "qwen3-vl-32b"
    assert validate_modes() == []


def test_native_profiles_reuse_operons_backends(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    monkeypatch.setenv("OPERON_DESKTOP_BACKEND", "model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "claude-sonnet-5-5")
    browser = build_browser_services(executor=None).policy_delegate
    desktop = build_desktop_services().policy_delegate
    assert isinstance(browser, BrowserClaudeComputerUseBackend) and browser.client.model == "claude-sonnet-5-5"
    assert isinstance(desktop, DesktopClaudeComputerUseBackend)
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "gemini-computer-use")
    assert isinstance(build_browser_services(executor=None).policy_delegate, BrowserComputerUseBackend)


def test_browser_only_profile_is_refused_on_the_desktop(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_DESKTOP_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "gemini-computer-use")
    with pytest.raises(ModelRegistryError, match="desktop"):
        build_desktop_services()
    assert any("does not support desktop" in error for error in validate_modes())


def test_validation_names_the_missing_setting(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    assert any("OPERON_MODEL_PROFILE" in error for error in validate_modes())
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "qwen3-vl-32b")
    assert any("OPENWEIGHT_BASE_URL" in error for error in validate_modes())


def test_loop_layers_pick_the_own_done_verifier(monkeypatch) -> None:
    from operon.api.runtime.loops import _loop_layers
    from operon.api.runtime_config import browser_mode_config

    class _Delegate:
        def add_advisory_hints(self, *args, **kwargs) -> None: ...

    services = type("S", (), {"policy_delegate": _Delegate()})()
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    layers = _loop_layers(config=browser_mode_config(), services=services, memory_store=None)
    assert isinstance(layers["verifier_service"], OwnDoneVerifier) and layers["harness_profile"] == THIN
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "claude_computer_use")
    layers = _loop_layers(config=browser_mode_config(), services=services, memory_store=None)
    assert isinstance(layers["verifier_service"], FinalCheckVerifier)


def _verify_args(rationale: str, action: AgentAction, success: bool = True):
    st = AgentState(run_id="r", intent="Buy the mug", status=RunStatus.RUNNING)
    decision = PolicyDecision(action=action, rationale=rationale, confidence=0.7, active_subgoal="x")
    return st, decision, ExecutedAction(action=action, success=success, detail="ok" if success else "missed")


async def test_own_done_ends_the_run_on_the_models_word() -> None:
    result = await OwnDoneVerifier().verify(*_verify_args("Bought it.", AgentAction(action_type=ActionType.STOP)))
    assert result.status is VerificationStatus.SUCCESS and result.stop_condition_met
    assert result.stop_reason is StopReason.TASK_COMPLETED


async def test_own_done_ends_a_given_up_run_as_not_done() -> None:
    args = _verify_args(f"{GAVE_UP_PREFIX}CAPTCHA", AgentAction(action_type=ActionType.STOP))
    result = await OwnDoneVerifier(gave_up_prefix=GAVE_UP_PREFIX).verify(*args)
    assert result.status is VerificationStatus.FAILURE and result.stop_condition_met
    assert result.stop_reason is StopReason.COMPLETION_UNVERIFIED


async def test_own_done_leaves_ordinary_steps_to_the_model() -> None:
    click = AgentAction(action_type=ActionType.CLICK, x=1, y=1)
    ok = await OwnDoneVerifier().verify(*_verify_args("click", click))
    failed = await OwnDoneVerifier().verify(*_verify_args("click", click, success=False))
    assert ok.status is VerificationStatus.SUCCESS and not ok.stop_condition_met
    assert failed.status is VerificationStatus.UNCERTAIN and not failed.stop_condition_met


def test_registry_path_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    from operon.agent.backends.model_adapter.registry import resolve_profile

    path = tmp_path / "models.yaml"
    path.write_text("tiny:\n  adapter: openai_compat\n  model: tiny-vl\n", encoding="utf-8")
    monkeypatch.setenv("OPERON_MODEL_REGISTRY", str(path))
    assert resolve_profile("tiny").model == "tiny-vl"

"""Tests for the upload_file_native cross-environment action type."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.contracts.critic import FailureType
from core.contracts.perception import Environment, PerceptionOutput
from core.contracts.planner import ActionType as ContractActionType
from core.contracts.planner import PlannerAction
from core.router import (
    BROWSER_ACTIONS,
    DESKTOP_ACTIONS,
    RoutingError,
    is_cross_environment_action,
    validate_plan_route,
)
from executors.browser_executor import BrowserExecutor
from runtime.legacy_adapter import _map_action_type
from runtime.orchestrator import UnifiedOrchestrator
from src.models.common import FailureCategory
from src.models.policy import ActionType, AgentAction

# ---------------------------------------------------------------------------
# 1. Enum presence checks
# ---------------------------------------------------------------------------


def test_upload_file_native_action_type_exists() -> None:
    """Legacy ActionType enum must include upload_file_native."""
    assert ActionType.UPLOAD_FILE_NATIVE == "upload_file_native"


def test_contract_action_type_upload_file_native_exists() -> None:
    """Contract ActionType enum must include upload_file_native."""
    assert ContractActionType.UPLOAD_FILE_NATIVE == "upload_file_native"


# ---------------------------------------------------------------------------
# 2. Router: environment gating
# ---------------------------------------------------------------------------


def test_router_allows_upload_file_native_in_browser() -> None:
    """upload_file_native must be allowed in the BROWSER action set."""
    assert ContractActionType.UPLOAD_FILE_NATIVE in BROWSER_ACTIONS


def test_router_blocks_upload_file_native_in_desktop() -> None:
    """upload_file_native must NOT be in the DESKTOP action set."""
    assert ContractActionType.UPLOAD_FILE_NATIVE not in DESKTOP_ACTIONS


def test_is_cross_environment_action_returns_true_for_upload_file_native() -> None:
    """is_cross_environment_action must return True for upload_file_native."""
    assert is_cross_environment_action(ContractActionType.UPLOAD_FILE_NATIVE) is True


def test_is_cross_environment_action_returns_false_for_click() -> None:
    """is_cross_environment_action must return False for plain click."""
    assert is_cross_environment_action(ContractActionType.CLICK) is False


def test_validate_plan_route_accepts_upload_file_native_in_browser() -> None:
    """validate_plan_route must not raise for upload_file_native in browser."""
    from core.contracts.perception import ContractVersion
    from core.contracts.planner import PlannerOutput

    plan = PlannerOutput(
        contract_version=ContractVersion.PHASE1,
        environment=Environment.BROWSER,
        observation_id="obs_ufn_1",
        plan_id="plan_ufn_1",
        subgoal="Upload a file using the native OS picker.",
        rationale="Testing native upload routing.",
        action=PlannerAction(
            action_type=ContractActionType.UPLOAD_FILE_NATIVE,
            target_id="upload_btn",
            file_path="/tmp/test.pdf",
        ),
        expected_outcome="OS file picker opens.",
    )
    # Should not raise
    validate_plan_route(plan)


def test_validate_plan_route_rejects_upload_file_native_in_desktop() -> None:
    """validate_plan_route must raise RoutingError for upload_file_native in desktop."""
    from core.contracts.perception import ContractVersion
    from core.contracts.planner import PlannerOutput

    plan = PlannerOutput(
        contract_version=ContractVersion.PHASE1,
        environment=Environment.DESKTOP,
        observation_id="obs_ufn_2",
        plan_id="plan_ufn_2",
        subgoal="Upload a file using the native OS picker.",
        rationale="Testing desktop rejection.",
        action=PlannerAction(
            action_type=ContractActionType.UPLOAD_FILE_NATIVE,
            target_id="upload_btn",
            file_path="/tmp/test.pdf",
        ),
        expected_outcome="Should be rejected.",
    )
    with pytest.raises(RoutingError):
        validate_plan_route(plan)


# ---------------------------------------------------------------------------
# 3. Failure type checks
# ---------------------------------------------------------------------------


def test_failure_types_include_picker_not_detected() -> None:
    """FailureType enum must include picker_not_detected."""
    assert FailureType.PICKER_NOT_DETECTED == "picker_not_detected"


def test_failure_types_include_file_not_reflected() -> None:
    """FailureType enum must include file_not_reflected."""
    assert FailureType.FILE_NOT_REFLECTED == "file_not_reflected"


def test_failure_category_includes_picker_not_detected() -> None:
    """Legacy FailureCategory enum must include picker_not_detected."""
    assert FailureCategory.PICKER_NOT_DETECTED == "picker_not_detected"


def test_failure_category_includes_file_not_reflected() -> None:
    """Legacy FailureCategory enum must include file_not_reflected."""
    assert FailureCategory.FILE_NOT_REFLECTED == "file_not_reflected"


# ---------------------------------------------------------------------------
# 4. Orchestrator adaptation strategies
# ---------------------------------------------------------------------------


def test_adaptation_for_picker_not_detected() -> None:
    """PICKER_NOT_DETECTED must map to wait_then_retry."""
    strategy = UnifiedOrchestrator.adaptation_strategy_for(FailureType.PICKER_NOT_DETECTED)
    assert strategy == "wait_then_retry"


def test_adaptation_for_file_not_reflected() -> None:
    """FILE_NOT_REFLECTED must map to reperceive_and_replan."""
    strategy = UnifiedOrchestrator.adaptation_strategy_for(FailureType.FILE_NOT_REFLECTED)
    assert strategy == "reperceive_and_replan"


# ---------------------------------------------------------------------------
# 5. Orchestrator: detect_file_picker
# ---------------------------------------------------------------------------


def test_detect_file_picker_true_on_context_label() -> None:
    """detect_file_picker returns True when context_label contains picker signal."""
    perception = PerceptionOutput(
        environment=Environment.DESKTOP,
        observation_id="obs_picker_1",
        summary="The Open File dialog is active.",
        context_label="Open File Dialog",
        visible_targets=[],
        notes=[],
    )
    assert UnifiedOrchestrator.detect_file_picker(perception) is True


def test_detect_file_picker_true_on_notes() -> None:
    """detect_file_picker returns True when notes mention file picker."""
    perception = PerceptionOutput(
        environment=Environment.DESKTOP,
        observation_id="obs_picker_2",
        summary="A dialog window appeared.",
        context_label="Dialog",
        visible_targets=[],
        notes=["native file chooser is active"],
    )
    assert UnifiedOrchestrator.detect_file_picker(perception) is True


def test_detect_file_picker_false_on_normal_page() -> None:
    """detect_file_picker returns False for a normal browser page."""
    perception = PerceptionOutput(
        environment=Environment.BROWSER,
        observation_id="obs_picker_3",
        summary="The upload form is visible.",
        context_label="Upload Form",
        visible_targets=[],
        notes=[],
    )
    assert UnifiedOrchestrator.detect_file_picker(perception) is False


# ---------------------------------------------------------------------------
# 6. Executor adapters
# ---------------------------------------------------------------------------


def test_browser_executor_translates_upload_file_native() -> None:
    """BrowserExecutor._to_legacy_action must translate UPLOAD_FILE_NATIVE correctly."""
    action = PlannerAction(
        action_type=ContractActionType.UPLOAD_FILE_NATIVE,
        target_id="upload_btn",
        file_path="/tmp/test.pdf",
        picker_title="Choose a file",
    )
    legacy = BrowserExecutor._to_legacy_action(action)
    assert legacy.action_type is ActionType.UPLOAD_FILE_NATIVE
    assert legacy.target_element_id == "upload_btn"
    assert legacy.text == "Choose a file"


def test_browser_executor_translates_upload_file_native_no_picker_title() -> None:
    """BrowserExecutor._to_legacy_action handles UPLOAD_FILE_NATIVE without picker_title."""
    action = PlannerAction(
        action_type=ContractActionType.UPLOAD_FILE_NATIVE,
        target_id="file_input",
        file_path="/home/user/report.pdf",
    )
    legacy = BrowserExecutor._to_legacy_action(action)
    assert legacy.action_type is ActionType.UPLOAD_FILE_NATIVE
    assert legacy.target_element_id == "file_input"
    assert legacy.text is None


# ---------------------------------------------------------------------------
# 7. Legacy adapter: _map_action_type
# ---------------------------------------------------------------------------


def test_legacy_adapter_maps_upload_file_native_action_type() -> None:
    """_map_action_type must map upload_file_native to ContractActionType.UPLOAD_FILE_NATIVE."""
    action = AgentAction(
        action_type=ActionType.UPLOAD_FILE_NATIVE,
        target_element_id="upload_btn",
    )
    result = _map_action_type(action)
    assert result is ContractActionType.UPLOAD_FILE_NATIVE


# ---------------------------------------------------------------------------
# 8. NativeBrowserExecutor: upload_file_native execution path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browser_native_executor_upload_file_native_clicks_then_returns() -> None:
    """upload_file_native should click the upload element and return native_picker_triggered."""
    from src.executor.browser_native import NativeBrowserExecutor

    # Build a minimal mock page
    mock_page = MagicMock()
    mock_page.mouse = MagicMock()
    mock_page.mouse.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=MagicMock(first=MagicMock(click=AsyncMock())))
    mock_page.wait_for_load_state = AsyncMock()

    executor = NativeBrowserExecutor.__new__(NativeBrowserExecutor)
    executor._current_run_id = "run_test"
    executor._post_action_delay = 0.0
    executor._artifact_dir = MagicMock()
    executor._sessions = {}

    async def _fake_current_page(**_kwargs):
        return mock_page

    async def _fake_capture_after():
        return "after.png"

    with (
        patch.object(executor, "_current_page", side_effect=_fake_current_page),
        patch.object(executor, "_capture_after", side_effect=_fake_capture_after),
    ):
        action = AgentAction(
            action_type=ActionType.UPLOAD_FILE_NATIVE,
            x=100,
            y=200,
        )
        result = await executor.execute(action)

    assert result.success is True
    assert result.detail == "native_picker_triggered"
    mock_page.mouse.click.assert_called_once_with(100, 200)


@pytest.mark.asyncio
async def test_browser_native_executor_upload_file_native_uses_selector_fallback() -> None:
    """upload_file_native falls back to selector-based click when no coordinates given."""
    from src.executor.browser_native import NativeBrowserExecutor

    mock_locator_instance = MagicMock()
    mock_locator_instance.first = MagicMock()
    mock_locator_instance.first.click = AsyncMock()

    mock_page = MagicMock()
    mock_page.mouse = MagicMock()
    mock_page.mouse.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator_instance)
    mock_page.wait_for_load_state = AsyncMock()

    executor = NativeBrowserExecutor.__new__(NativeBrowserExecutor)
    executor._current_run_id = "run_test2"
    executor._post_action_delay = 0.0
    executor._artifact_dir = MagicMock()
    executor._sessions = {}

    async def _fake_current_page(**_kwargs):
        return mock_page

    async def _fake_capture_after():
        return "after.png"

    with (
        patch.object(executor, "_current_page", side_effect=_fake_current_page),
        patch.object(executor, "_capture_after", side_effect=_fake_capture_after),
    ):
        action = AgentAction(
            action_type=ActionType.UPLOAD_FILE_NATIVE,
            selector="#upload-btn",
        )
        result = await executor.execute(action)

    assert result.success is True
    assert result.detail == "native_picker_triggered"
    mock_page.locator.assert_called_once_with("#upload-btn")
    mock_locator_instance.first.click.assert_called_once()

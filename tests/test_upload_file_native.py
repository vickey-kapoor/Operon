"""Tests for the upload_file_native cross-environment action type."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.contracts.critic import FailureType
from src.core.contracts.perception import Environment, PerceptionOutput
from src.core.contracts.planner import ActionType as ContractActionType
from src.core.contracts.planner import PlannerAction
from src.core.router import (
    BROWSER_ACTIONS,
    DESKTOP_ACTIONS,
    RoutingError,
    is_cross_environment_action,
    validate_plan_route,
)
from src.executor.browser_adapter import BrowserExecutor
from src.models.common import FailureCategory
from src.models.policy import ActionType, AgentAction
from src.runtime.legacy_adapter import _map_action_type
from src.runtime.orchestrator import UnifiedOrchestrator

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
    from src.core.contracts.perception import ContractVersion
    from src.core.contracts.planner import PlannerOutput

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
    from src.core.contracts.perception import ContractVersion
    from src.core.contracts.planner import PlannerOutput

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


def _make_executor_and_page(
    *,
    run_id: str = "run_test",
    headless: bool = False,
):
    """Return (executor, mock_page) wired up for upload_file_native tests."""
    from src.executor.browser_native import NativeBrowserExecutor

    mock_page = MagicMock()
    mock_page.mouse = MagicMock()
    mock_page.mouse.click = AsyncMock()
    mock_locator = MagicMock()
    mock_locator.first = MagicMock()
    mock_locator.first.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)
    mock_page.wait_for_load_state = AsyncMock()

    executor = NativeBrowserExecutor.__new__(NativeBrowserExecutor)
    executor._current_run_id = run_id
    executor._headless = headless
    executor._run_headless = {}
    executor._post_action_delay = 0.0
    executor._artifact_dir = MagicMock()
    executor._sessions = {}
    return executor, mock_page


@pytest.mark.asyncio
async def test_browser_native_executor_upload_file_native_clicks_and_runs_macro() -> None:
    """upload_file_native should click, then delegate to the OS picker macro."""
    from src.executor.os_picker_macro import PickerMacroResult, PickerOutcome

    executor, mock_page = _make_executor_and_page()

    async def _fake_current_page(**_kw):
        return mock_page

    async def _fake_capture_after():
        return "after.png"

    mock_macro_result = PickerMacroResult(
        outcome=PickerOutcome.SUCCESS,
        detail="os_picker_macro typed file path and confirmed: C:\\tmp\\test.txt",
    )

    with (
        patch.object(executor, "_current_page", side_effect=_fake_current_page),
        patch.object(executor, "_capture_after", side_effect=_fake_capture_after),
        patch(
            "src.executor.browser_native.run_os_picker_macro",
            return_value=mock_macro_result,
        ) as mock_macro,
    ):
        action = AgentAction(
            action_type=ActionType.UPLOAD_FILE_NATIVE,
            x=100,
            y=200,
            text=r"C:\tmp\test.txt",
        )
        result = await executor.execute(action)

    assert result.success is True
    assert "os_picker_macro" in result.detail
    mock_page.mouse.click.assert_called_once_with(100, 200)
    mock_macro.assert_called_once_with(r"C:\tmp\test.txt")


@pytest.mark.asyncio
async def test_browser_native_executor_upload_file_native_uses_selector_fallback() -> None:
    """upload_file_native falls back to selector-based click when no coordinates given."""
    from src.executor.os_picker_macro import PickerMacroResult, PickerOutcome

    executor, mock_page = _make_executor_and_page(run_id="run_test2")

    async def _fake_current_page(**_kw):
        return mock_page

    async def _fake_capture_after():
        return "after.png"

    mock_macro_result = PickerMacroResult(
        outcome=PickerOutcome.SUCCESS,
        detail="os_picker_macro typed file path and confirmed: C:\\tmp\\test.txt",
    )

    with (
        patch.object(executor, "_current_page", side_effect=_fake_current_page),
        patch.object(executor, "_capture_after", side_effect=_fake_capture_after),
        patch(
            "src.executor.browser_native.run_os_picker_macro",
            return_value=mock_macro_result,
        ),
    ):
        action = AgentAction(
            action_type=ActionType.UPLOAD_FILE_NATIVE,
            selector="#upload-btn",
            text=r"C:\tmp\test.txt",
        )
        result = await executor.execute(action)

    assert result.success is True
    mock_page.locator.assert_called_once_with("#upload-btn")
    mock_page.locator.return_value.first.click.assert_called_once()


@pytest.mark.asyncio
async def test_browser_native_executor_upload_file_native_headless_guard() -> None:
    """upload_file_native must fail immediately in headless mode."""
    executor, _ = _make_executor_and_page(headless=True)

    action = AgentAction(
        action_type=ActionType.UPLOAD_FILE_NATIVE,
        x=100,
        y=200,
        text=r"C:\tmp\test.txt",
    )
    result = await executor.execute(action)

    assert result.success is False
    assert "headed" in result.detail
    assert result.failure_category is FailureCategory.EXECUTION_ERROR


@pytest.mark.asyncio
async def test_browser_native_executor_upload_file_native_missing_text() -> None:
    """upload_file_native must fail when text (file path) is missing."""
    executor, _ = _make_executor_and_page()

    action = AgentAction(
        action_type=ActionType.UPLOAD_FILE_NATIVE,
        x=100,
        y=200,
    )
    result = await executor.execute(action)

    assert result.success is False
    assert "requires text" in result.detail
    assert result.failure_category is FailureCategory.EXECUTION_ERROR


@pytest.mark.asyncio
async def test_browser_native_executor_upload_file_native_picker_not_detected() -> None:
    """PICKER_NOT_DETECTED outcome maps to the correct failure category."""
    from src.executor.os_picker_macro import PickerMacroResult, PickerOutcome

    executor, mock_page = _make_executor_and_page()

    async def _fake_current_page(**_kw):
        return mock_page

    async def _fake_capture_after():
        return "after.png"

    mock_macro_result = PickerMacroResult(
        outcome=PickerOutcome.PICKER_NOT_DETECTED,
        detail="No OS file picker window appeared within 3.0s",
    )

    with (
        patch.object(executor, "_current_page", side_effect=_fake_current_page),
        patch.object(executor, "_capture_after", side_effect=_fake_capture_after),
        patch(
            "src.executor.browser_native.run_os_picker_macro",
            return_value=mock_macro_result,
        ),
    ):
        action = AgentAction(
            action_type=ActionType.UPLOAD_FILE_NATIVE,
            x=100,
            y=200,
            text=r"C:\tmp\test.txt",
        )
        result = await executor.execute(action)

    assert result.success is False
    assert result.failure_category is FailureCategory.PICKER_NOT_DETECTED
    assert result.artifact_path == "after.png"


@pytest.mark.asyncio
async def test_browser_native_executor_upload_file_native_file_not_reflected() -> None:
    """FILE_NOT_REFLECTED outcome maps to the correct failure category."""
    from src.executor.os_picker_macro import PickerMacroResult, PickerOutcome

    executor, mock_page = _make_executor_and_page()

    async def _fake_current_page(**_kw):
        return mock_page

    async def _fake_capture_after():
        return "after.png"

    mock_macro_result = PickerMacroResult(
        outcome=PickerOutcome.FILE_NOT_REFLECTED,
        detail="OS file picker did not close within 3.0s",
    )

    with (
        patch.object(executor, "_current_page", side_effect=_fake_current_page),
        patch.object(executor, "_capture_after", side_effect=_fake_capture_after),
        patch(
            "src.executor.browser_native.run_os_picker_macro",
            return_value=mock_macro_result,
        ),
    ):
        action = AgentAction(
            action_type=ActionType.UPLOAD_FILE_NATIVE,
            x=100,
            y=200,
            text=r"C:\tmp\test.txt",
        )
        result = await executor.execute(action)

    assert result.success is False
    assert result.failure_category is FailureCategory.FILE_NOT_REFLECTED
    assert result.artifact_path == "after.png"


@pytest.mark.asyncio
async def test_browser_native_executor_upload_file_native_no_coordinates_or_selector() -> None:
    """upload_file_native may fall back to target_element_id-based lookup."""
    executor, mock_page = _make_executor_and_page()

    async def _fake_current_page(**_kw):
        return mock_page

    async def _fake_capture_after():
        return "after.png"

    from src.executor.os_picker_macro import PickerMacroResult, PickerOutcome

    mock_macro_result = PickerMacroResult(
        outcome=PickerOutcome.SUCCESS,
        detail="os_picker_macro typed file path and confirmed: C:\\tmp\\test.txt",
    )

    with (
        patch.object(executor, "_current_page", side_effect=_fake_current_page),
        patch.object(executor, "_capture_after", side_effect=_fake_capture_after),
        patch(
            "src.executor.browser_native.run_os_picker_macro",
            return_value=mock_macro_result,
        ),
    ):
        action = AgentAction(
            action_type=ActionType.UPLOAD_FILE_NATIVE,
            target_element_id="some_id",
            text=r"C:\tmp\test.txt",
        )
        result = await executor.execute(action)

    assert result.success is True
    mock_page.locator.assert_called_once_with(
        '[id="some_id"], [data-element-id="some_id"], [data-testid="some_id"], [name="some_id"]'
    )
    mock_page.locator.return_value.first.click.assert_called_once()


# ---------------------------------------------------------------------------
# 9. os_picker_macro unit tests
# ---------------------------------------------------------------------------


def test_os_picker_macro_unavailable_when_deps_missing() -> None:
    """Macro returns UNAVAILABLE when pyautogui/pygetwindow can't import."""
    from src.executor.os_picker_macro import PickerOutcome

    with (
        patch("src.executor.os_picker_macro._PYAUTOGUI_IMPORT_ERROR", RuntimeError("no display")),
    ):
        from src.executor.os_picker_macro import run_os_picker_macro
        result = run_os_picker_macro(r"C:\tmp\test.txt")

    assert result.outcome is PickerOutcome.UNAVAILABLE
    assert "unavailable" in result.detail.lower()


def test_os_picker_macro_picker_not_detected() -> None:
    """Macro returns PICKER_NOT_DETECTED if no picker window appears."""
    from src.executor.os_picker_macro import PickerOutcome, run_os_picker_macro

    with (
        patch("src.executor.os_picker_macro._PYAUTOGUI_IMPORT_ERROR", None),
        patch("src.executor.os_picker_macro.pygetwindow") as mock_pgw,
    ):
        mock_pgw.getAllWindows.return_value = []
        result = run_os_picker_macro(r"C:\tmp\test.txt", appear_timeout_s=0.1)

    assert result.outcome is PickerOutcome.PICKER_NOT_DETECTED


def test_os_picker_macro_success() -> None:
    """Macro types path and presses Enter when picker window found and closes."""
    from src.executor.os_picker_macro import PickerOutcome, run_os_picker_macro

    mock_window = MagicMock()
    mock_window.title = "Open File Dialog"
    mock_window.activate = MagicMock()

    call_count = {"n": 0}

    def _mock_get_all_windows():
        call_count["n"] += 1
        # First call: picker visible; subsequent: picker closed
        if call_count["n"] <= 2:
            return [mock_window]
        return []

    with (
        patch("src.executor.os_picker_macro._PYAUTOGUI_IMPORT_ERROR", None),
        patch("src.executor.os_picker_macro.pygetwindow") as mock_pgw,
        patch("src.executor.os_picker_macro.pyautogui") as mock_pag,
    ):
        mock_pgw.getAllWindows = _mock_get_all_windows
        result = run_os_picker_macro(r"C:\tmp\test.txt", appear_timeout_s=1.0, close_timeout_s=1.0)

    assert result.outcome is PickerOutcome.SUCCESS
    mock_pag.write.assert_called_once_with(r"C:\tmp\test.txt", interval=0.02)
    mock_pag.press.assert_called_once_with("enter")


def test_os_picker_macro_file_not_reflected() -> None:
    """Macro returns FILE_NOT_REFLECTED when picker doesn't close after Enter."""
    from src.executor.os_picker_macro import PickerOutcome, run_os_picker_macro

    mock_window = MagicMock()
    mock_window.title = "Open File Dialog"
    mock_window.activate = MagicMock()

    with (
        patch("src.executor.os_picker_macro._PYAUTOGUI_IMPORT_ERROR", None),
        patch("src.executor.os_picker_macro.pygetwindow") as mock_pgw,
        patch("src.executor.os_picker_macro.pyautogui"),
    ):
        # Picker always visible — never closes
        mock_pgw.getAllWindows.return_value = [mock_window]
        result = run_os_picker_macro(r"C:\tmp\test.txt", appear_timeout_s=0.1, close_timeout_s=0.1)

    assert result.outcome is PickerOutcome.FILE_NOT_REFLECTED

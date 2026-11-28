"""Offline coverage for the native browser executor's navigate, hover, drag, read-text
and person-takeover (user_input) paths, using a fake Playwright page."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from pydantic import ValidationError

from operon.executor.browser_native import NativeBrowserExecutor
from operon.models.common import FailureCategory
from operon.models.policy import ActionType, AgentAction


def _page(**overrides) -> SimpleNamespace:
    mouse = SimpleNamespace(
        click=AsyncMock(), dblclick=AsyncMock(), move=AsyncMock(), down=AsyncMock(), up=AsyncMock(), wheel=AsyncMock()
    )
    keyboard = SimpleNamespace(press=AsyncMock(), type=AsyncMock())
    page = SimpleNamespace(
        mouse=mouse,
        keyboard=keyboard,
        goto=AsyncMock(),
        evaluate=AsyncMock(return_value=""),
        wait_for_load_state=AsyncMock(),
    )
    for key, value in overrides.items():
        setattr(page, key, value)
    return page


def _executor(tmp_path: Path, page: SimpleNamespace) -> NativeBrowserExecutor:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    executor._current_page = AsyncMock(return_value=page)
    executor._capture_after = AsyncMock(return_value=str(tmp_path / "after.png"))
    executor.settle_now = AsyncMock()
    executor._foreground_if_fresh_session = AsyncMock()
    return executor


# --- NAVIGATE ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_navigate_goes_to_url_and_waits_for_load(tmp_path: Path) -> None:
    page = _page()
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.NAVIGATE, url="https://example.com/a"))

    assert result.success is True
    assert result.detail == "Executed navigate"
    assert result.artifact_path == str(tmp_path / "after.png")
    page.goto.assert_awaited_once_with("https://example.com/a")
    page.wait_for_load_state.assert_awaited_once_with(timeout=5000)
    executor._current_page.assert_awaited_once_with(foreground=False)
    executor._foreground_if_fresh_session.assert_awaited_once()


def test_navigate_without_url_is_rejected_when_the_action_is_built() -> None:
    # The executor's own "navigate requires url" guard is unreachable: the AgentAction
    # validators reject a url-less navigate at construction, so that is the real gate.
    with pytest.raises(ValidationError, match="navigate requires url"):
        AgentAction(action_type=ActionType.NAVIGATE)


@pytest.mark.asyncio
async def test_navigate_timeout_is_reported_as_a_failure(tmp_path: Path) -> None:
    page = _page(goto=AsyncMock(side_effect=TimeoutError("Timeout 30000ms exceeded")))
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.NAVIGATE, url="https://slow.example"))

    assert result.success is False
    assert result.detail.startswith("navigate failed:")
    assert "Timeout 30000ms exceeded" in result.detail
    assert result.failure_category is FailureCategory.EXECUTION_ERROR
    executor._capture_after.assert_not_awaited()


@pytest.mark.asyncio
async def test_navigate_still_succeeds_when_the_load_state_wait_times_out(tmp_path: Path) -> None:
    page = _page(wait_for_load_state=AsyncMock(side_effect=TimeoutError("load")))
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.NAVIGATE, url="https://example.com"))

    assert result.success is True
    page.goto.assert_awaited_once_with("https://example.com")


# --- HOVER ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hover_moves_the_mouse_to_the_coordinates(tmp_path: Path) -> None:
    page = _page()
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.HOVER, x=310, y=95))

    assert result.success is True
    assert result.detail == "Executed hover"
    page.mouse.move.assert_awaited_once_with(310, 95)
    page.mouse.click.assert_not_awaited()
    page.wait_for_load_state.assert_not_awaited()  # hover cannot navigate


def test_hover_without_coordinates_is_rejected_when_the_action_is_built() -> None:
    with pytest.raises(ValidationError, match="hover requires x, y"):
        AgentAction(action_type=ActionType.HOVER)


@pytest.mark.asyncio
async def test_hover_playwright_error_becomes_a_failure(tmp_path: Path) -> None:
    page = _page()
    page.mouse.move = AsyncMock(side_effect=RuntimeError("Target page, context or browser has been closed"))
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.HOVER, x=1, y=2))

    assert result.success is False
    assert result.detail.startswith("hover failed:")
    assert "has been closed" in result.detail
    assert result.failure_category is FailureCategory.EXECUTION_ERROR


# --- DRAG -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drag_presses_moves_and_releases_in_order(tmp_path: Path) -> None:
    events: list[tuple] = []

    async def move(x, y):
        events.append(("move", x, y))

    async def down():
        events.append(("down",))

    async def up():
        events.append(("up",))

    page = _page()
    page.mouse.move, page.mouse.down, page.mouse.up = move, down, up
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.DRAG, x=5, y=6, x_end=70, y_end=80))

    assert result.success is True
    assert result.detail == "Executed drag"
    assert events == [("move", 5, 6), ("down",), ("move", 70, 80), ("up",)]


@pytest.mark.parametrize("missing", ["x", "y", "x_end", "y_end"])
def test_drag_with_a_missing_coordinate_is_rejected_when_the_action_is_built(missing: str) -> None:
    coords: dict[str, int | None] = {"x": 10, "y": 20, "x_end": 30, "y_end": 40}
    coords[missing] = None

    with pytest.raises(ValidationError, match="drag requires x, y, x_end, y_end"):
        AgentAction(action_type=ActionType.DRAG, **coords)


@pytest.mark.asyncio
async def test_drag_accepts_zero_coordinates(tmp_path: Path) -> None:
    page = _page()
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.DRAG, x=0, y=0, x_end=0, y_end=50))

    assert result.success is True
    page.mouse.move.assert_has_awaits([call(0, 0), call(0, 50)])


@pytest.mark.asyncio
async def test_drag_failure_mid_gesture_is_reported(tmp_path: Path) -> None:
    page = _page()
    page.mouse.up = AsyncMock(side_effect=RuntimeError("mouse up failed"))
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.DRAG, x=1, y=2, x_end=3, y_end=4))

    assert result.success is False
    assert result.detail == "drag failed: mouse up failed"
    assert result.failure_category is FailureCategory.EXECUTION_ERROR


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="a drag that fails after mouse.down leaves the button held; no release is attempted")
async def test_drag_releases_the_mouse_button_when_the_move_fails(tmp_path: Path) -> None:
    page = _page()
    page.mouse.move = AsyncMock(side_effect=[None, RuntimeError("boom")])
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.DRAG, x=1, y=2, x_end=3, y_end=4))

    assert result.success is False
    page.mouse.up.assert_awaited_once_with()


# --- READ_TEXT --------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_text_returns_extracted_text_using_the_default_selector(tmp_path: Path) -> None:
    page = _page(evaluate=AsyncMock(return_value="Paragraph one is long enough."))
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.READ_TEXT))

    assert result.success is True
    assert result.detail == "Paragraph one is long enough."
    assert result.artifact_path == str(tmp_path / "after.png")
    page.evaluate.assert_awaited_once()
    assert page.evaluate.await_args.args[1] == "main, article, #content, body"


@pytest.mark.asyncio
async def test_read_text_uses_the_given_selector_and_truncates_detail(tmp_path: Path) -> None:
    page = _page(evaluate=AsyncMock(return_value="x" * 900))
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.READ_TEXT, selector="#story"))

    assert result.success is True
    assert page.evaluate.await_args.args[1] == "#story"
    assert result.detail == "x" * 500


@pytest.mark.asyncio
async def test_read_text_writes_the_full_text_to_the_output_path(tmp_path: Path) -> None:
    long_text = "y" * 900
    out = tmp_path / "nested" / "dir" / "out.txt"
    page = _page(evaluate=AsyncMock(return_value=long_text))
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.READ_TEXT, text=str(out)))

    assert result.success is True
    assert out.read_text(encoding="utf-8") == long_text
    assert len(result.detail) == 500


@pytest.mark.asyncio
async def test_read_text_with_nothing_found_fails_and_writes_no_file(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    page = _page(evaluate=AsyncMock(return_value=""))
    executor = _executor(tmp_path, page)

    result = await executor.execute(
        AgentAction(action_type=ActionType.READ_TEXT, selector="#missing", text=str(out))
    )

    assert result.success is False
    assert result.detail == "No text found at selector '#missing'"
    assert result.failure_category is FailureCategory.EXECUTION_ERROR
    assert not out.exists()
    executor._capture_after.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_text_evaluate_error_becomes_a_failure(tmp_path: Path) -> None:
    page = _page(evaluate=AsyncMock(side_effect=RuntimeError("Execution context was destroyed")))
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.READ_TEXT))

    assert result.success is False
    assert result.detail.startswith("read_text failed:")
    assert "Execution context was destroyed" in result.detail
    assert result.failure_category is FailureCategory.EXECUTION_ERROR


@pytest.mark.asyncio
async def test_read_text_unwritable_output_path_becomes_a_failure(tmp_path: Path) -> None:
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory", encoding="utf-8")
    page = _page(evaluate=AsyncMock(return_value="some extracted text here"))
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.READ_TEXT, text=str(blocker / "out.txt")))

    assert result.success is False
    assert result.detail.startswith("read_text failed:")


# --- user_input (person takeover) -------------------------------------------


def _takeover_executor(tmp_path: Path, page: SimpleNamespace, run_id: str = "run-1") -> NativeBrowserExecutor:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    executor._sessions[run_id] = SimpleNamespace(page=page)  # type: ignore[assignment]
    return executor


@pytest.mark.asyncio
async def test_user_input_without_a_session_returns_false(tmp_path: Path) -> None:
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)

    assert await executor.user_input("nope", input_type="click", x_ratio=0.5, y_ratio=0.5) is False


@pytest.mark.asyncio
async def test_user_input_click_scales_ratios_to_the_viewport(tmp_path: Path) -> None:
    page = _page()
    executor = _takeover_executor(tmp_path, page)
    w, h = executor._viewport_width, executor._viewport_height

    assert await executor.user_input("run-1", input_type="click", x_ratio=0.5, y_ratio=0.25) is True

    page.mouse.click.assert_awaited_once_with(0.5 * w, 0.25 * h)


@pytest.mark.asyncio
async def test_user_input_clamps_out_of_range_ratios(tmp_path: Path) -> None:
    page = _page()
    executor = _takeover_executor(tmp_path, page)

    await executor.user_input("run-1", input_type="click", x_ratio=3.0, y_ratio=-1.0)

    page.mouse.click.assert_awaited_once_with(float(executor._viewport_width), 0.0)


@pytest.mark.asyncio
async def test_user_input_dblclick(tmp_path: Path) -> None:
    page = _page()
    executor = _takeover_executor(tmp_path, page)

    assert await executor.user_input("run-1", input_type="dblclick", x_ratio=1.0, y_ratio=1.0) is True

    page.mouse.dblclick.assert_awaited_once_with(float(executor._viewport_width), float(executor._viewport_height))
    page.mouse.click.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_input_type_sends_text_verbatim(tmp_path: Path) -> None:
    page = _page()
    executor = _takeover_executor(tmp_path, page)

    assert await executor.user_input("run-1", input_type="type", text="hello world") is True

    page.keyboard.type.assert_awaited_once_with("hello world")
    page.mouse.click.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_input_key_normalizes_the_key_name(tmp_path: Path) -> None:
    page = _page()
    executor = _takeover_executor(tmp_path, page)

    assert await executor.user_input("run-1", input_type="key", key="enter") is True

    page.keyboard.press.assert_awaited_once_with("Enter")


@pytest.mark.asyncio
async def test_user_input_scroll_moves_then_wheels(tmp_path: Path) -> None:
    page = _page()
    executor = _takeover_executor(tmp_path, page)
    w, h = executor._viewport_width, executor._viewport_height

    assert await executor.user_input("run-1", input_type="scroll", x_ratio=0.5, y_ratio=0.5, delta_y=-120) is True

    page.mouse.move.assert_awaited_once_with(0.5 * w, 0.5 * h)
    page.mouse.wheel.assert_awaited_once_with(0, -120)


@pytest.mark.asyncio
async def test_user_input_scroll_defaults_to_300(tmp_path: Path) -> None:
    page = _page()
    executor = _takeover_executor(tmp_path, page)

    await executor.user_input("run-1", input_type="scroll")

    page.mouse.wheel.assert_awaited_once_with(0, 300)


@pytest.mark.asyncio
async def test_user_input_unknown_type_raises_value_error(tmp_path: Path) -> None:
    page = _page()
    executor = _takeover_executor(tmp_path, page)

    with pytest.raises(ValueError, match="unsupported input_type 'swipe'"):
        await executor.user_input("run-1", input_type="swipe")


@pytest.mark.asyncio
async def test_user_input_playwright_error_propagates_to_the_caller(tmp_path: Path) -> None:
    page = _page()
    page.mouse.click = AsyncMock(side_effect=RuntimeError("page closed"))
    executor = _takeover_executor(tmp_path, page)

    with pytest.raises(RuntimeError, match="page closed"):
        await executor.user_input("run-1", input_type="click", x_ratio=0.1, y_ratio=0.1)


# --- KAN-191: only http and https links open --------------------------------


@pytest.mark.asyncio
async def test_navigate_to_a_file_url_is_refused_without_going_there(tmp_path: Path) -> None:
    page = _page()
    executor = _executor(tmp_path, page)

    result = await executor.execute(AgentAction(action_type=ActionType.NAVIGATE, url="file:///home/x/.ssh/id_rsa"))

    assert result.success is False
    assert result.detail == "navigate blocked: only http and https links can be opened (got 'file:')"
    assert result.failure_category is FailureCategory.EXECUTION_ERROR
    page.goto.assert_not_awaited()
    executor._current_page.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_navigate_child_to_a_file_url_is_refused(tmp_path: Path) -> None:
    page = _page()
    executor = _executor(tmp_path, page)
    batch = AgentAction(
        action_type=ActionType.BATCH,
        actions=[
            AgentAction(action_type=ActionType.NAVIGATE, url="https://example.com/"),
            AgentAction(action_type=ActionType.NAVIGATE, url="file:///etc/passwd"),
        ],
    )

    result = await executor.execute(batch)

    assert result.success is False
    assert "only http and https links can be opened" in result.detail
    assert result.batch_failed_index == 1
    page.goto.assert_awaited_once_with("https://example.com/")


@pytest.mark.asyncio
async def test_session_reset_never_opens_a_non_web_start_url(tmp_path: Path) -> None:
    page = _page(url="https://example.com/now", reload=AsyncMock())
    executor = _executor(tmp_path, page)

    await executor.session_reset("file:///etc/passwd")

    page.goto.assert_not_awaited()
    page.reload.assert_awaited_once()
    assert "not an http or https link" in executor.last_reset_actions[0]


async def test_settle_now_forwards_min_s_to_the_change_note_buffer(tmp_path: Path) -> None:
    """KAN-264: the first-step minimum reaches the settle wait."""
    executor = NativeBrowserExecutor(artifact_dir=tmp_path, headless=True)
    executor._change_notes = SimpleNamespace(settle=AsyncMock())
    await executor.settle_now(2.5, min_s=1.0)
    executor._change_notes.settle.assert_awaited_once_with(executor._grab_signature, 2.5, min_s=1.0)

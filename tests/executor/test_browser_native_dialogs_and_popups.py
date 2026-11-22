"""JS dialogs and self-closing popups in NativeBrowserExecutor (KAN-172), on test doubles.

No browser is launched. The real-Chromium versions are in tests/browser_e2e.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from operon.core import vault


@pytest.fixture()
def executor():
    os.environ["OPERON_TEST_SAFE_MODE"] = "true"
    from operon.executor.browser_native import NativeBrowserExecutor

    ex = NativeBrowserExecutor(headless=True)
    ex._sessions = {}
    return ex


def _page(name: str = "page", *, closed: bool = False):
    page = MagicMock(name=name)
    page.handlers = {}
    page.on = lambda event, fn: page.handlers.setdefault(event, fn)
    page.closed = closed
    page.is_closed = lambda: page.closed
    return page


def _session(page, pages=None):
    from operon.executor.browser_native import _BrowserSession

    session = _BrowserSession.__new__(_BrowserSession)
    session.page = page
    session.context = MagicMock(pages=pages if pages is not None else [page])
    return session


def _dialog(kind: str, message: str):
    return SimpleNamespace(type=kind, message=message, accept=AsyncMock(), dismiss=AsyncMock())


async def _raise(executor, run_id: str, kind: str, message: str):
    page = _page()
    executor._attach_page_listeners(page, run_id)
    dialog = _dialog(kind, message)
    await page.handlers["dialog"](dialog)
    return dialog


# -- dialogs ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_alert_is_accepted_and_its_message_kept(executor):
    dialog = await _raise(executor, "r1", "alert", "Saved!")

    dialog.accept.assert_awaited_once()
    dialog.dismiss.assert_not_awaited()
    assert executor.pop_last_dialog_message("r1") == "Saved!"
    assert executor.pop_pending_dialog("r1") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["confirm", "prompt", "beforeunload"])
async def test_a_question_dialog_is_cancelled_and_recorded_by_default(executor, kind):
    dialog = await _raise(executor, "r1", kind, "Delete it?")

    dialog.dismiss.assert_awaited_once()
    dialog.accept.assert_not_awaited()
    assert executor.pop_pending_dialog("r1") == (kind, "Delete it?", None)
    assert executor.pop_pending_dialog("r1") is None  # popped
    assert executor.pop_last_dialog_message("r1") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["confirm", "prompt", "beforeunload"])
async def test_the_auto_policy_accepts_a_question_dialog(executor, kind):
    executor.set_dialog_policy("r1", accept=True)

    dialog = await _raise(executor, "r1", kind, "Delete it?")

    dialog.accept.assert_awaited_once()
    dialog.dismiss.assert_not_awaited()
    assert executor.pop_pending_dialog("r1") is None
    assert executor.pop_last_dialog_message("r1") == "Delete it?"


@pytest.mark.asyncio
async def test_an_armed_accept_covers_exactly_one_dialog(executor):
    executor.arm_accept_next_dialog("r1", "confirm", "Delete it?")

    first = await _raise(executor, "r1", "confirm", "Delete it?")
    second = await _raise(executor, "r1", "confirm", "Delete it?")

    first.accept.assert_awaited_once()
    second.dismiss.assert_awaited_once()
    assert executor.pop_pending_dialog("r1") == ("confirm", "Delete it?", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "message"), [("confirm", "Wipe everything?"), ("prompt", "Delete it?")])
async def test_an_armed_accept_does_not_cover_a_different_dialog(executor, kind, message):
    executor.arm_accept_next_dialog("r1", "confirm", "Delete it?")

    other = await _raise(executor, "r1", kind, message)

    other.dismiss.assert_awaited_once()
    other.accept.assert_not_awaited()
    assert executor.pop_pending_dialog("r1") == (kind, message, None)  # asked about again


@pytest.mark.asyncio
async def test_a_dialog_in_a_batch_records_which_child_raised_it(executor):
    from operon.models.execution import ExecutedAction
    from operon.models.policy import ActionType, AgentAction

    page = _page()
    executor._attach_page_listeners(page, "r1")
    executor.set_current_run_id("r1")
    children = [AgentAction(action_type=ActionType.WAIT, wait_ms=1), AgentAction(action_type=ActionType.WAIT, wait_ms=2)]

    async def run_child(child):
        if child.wait_ms == 2:
            await page.handlers["dialog"](_dialog("confirm", "Sure?"))
        return ExecutedAction(action=child, success=True, detail="ok")

    with patch.object(executor, "execute", new=run_child):
        await executor._execute_batch(AgentAction(action_type=ActionType.BATCH, actions=children))

    assert executor.pop_pending_dialog("r1") == ("confirm", "Sure?", 1)
    assert executor._executing_child is None


@pytest.mark.asyncio
async def test_an_alert_does_not_use_up_the_armed_accept(executor):
    executor.arm_accept_next_dialog("r1", "confirm", "Sure?")

    await _raise(executor, "r1", "alert", "Hello")
    confirm = await _raise(executor, "r1", "confirm", "Sure?")

    confirm.accept.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_dialog_policy_is_per_run(executor):
    executor.set_dialog_policy("auto-run", accept=True)

    dialog = await _raise(executor, "other-run", "confirm", "Sure?")

    dialog.dismiss.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_secret_typed_in_the_run_is_masked_in_the_dialog_message(executor):
    executor._run_secret_values["r1"] = {"hunter2-pass"}

    await _raise(executor, "r1", "confirm", "Use password hunter2-pass?")

    kind, message, _child = executor.pop_pending_dialog("r1")
    assert "hunter2-pass" not in message and vault.REDACTED in message


@pytest.mark.asyncio
async def test_dialog_state_is_cleared_when_the_run_closes(executor):
    executor.set_dialog_policy("r1", accept=True)
    executor.arm_accept_next_dialog("r1", "confirm", "x")
    executor._pending_dialogs["r1"] = ("confirm", "x", None)
    executor._last_dialog_message["r1"] = "y"

    await executor.aclose_run("r1")

    assert "r1" not in executor._dialog_accept_all and "r1" not in executor._dialog_accept_next
    assert executor.pop_pending_dialog("r1") is None and executor.pop_last_dialog_message("r1") is None


# -- popups that close themselves ------------------------------------------------------


def _popup_setup(executor, *, mode="batch"):
    opener, popup = _page("opener"), _page("popup")
    executor._run_mode["r1"] = mode
    session = _session(opener, pages=[opener])
    executor._sessions["r1"] = session
    executor._attach_page_listeners(opener, "r1")
    opener.handlers["popup"](popup)  # the popup becomes the session's page
    session.context.pages = [opener, popup]
    assert session.page is popup
    return opener, popup, session


@pytest.mark.asyncio
async def test_a_popup_that_closes_itself_hands_the_session_back_to_the_opener(executor):
    opener, popup, session = _popup_setup(executor)

    popup.closed = True
    session.context.pages = [opener]
    popup.handlers["close"](popup)

    assert session.page is opener


@pytest.mark.asyncio
async def test_closing_a_page_the_run_left_does_not_move_the_session(executor):
    opener, popup, session = _popup_setup(executor)

    opener.closed = True
    opener.handlers["close"](opener)

    assert session.page is popup


@pytest.mark.asyncio
async def test_the_live_view_follows_back_to_the_opener_in_observable_mode(executor):
    opener, popup, session = _popup_setup(executor, mode="observable")

    with patch.object(executor, "_attach_live_view", new=AsyncMock()) as attach:
        popup.closed = True
        popup.handlers["close"](popup)
        await asyncio.gather(*executor._live_view_tasks)

    assert session.page is opener
    attach.assert_awaited_once_with(opener, 9222)


@pytest.mark.asyncio
async def test_capture_works_after_the_popup_closed_without_a_close_event(executor):
    """Even if the close event is missed, the next page access picks an open page."""
    opener, popup, session = _popup_setup(executor)
    executor.set_current_run_id("r1")
    popup.closed = True
    session.context.pages = [opener]

    assert await executor._current_page() is opener
    assert session.page is opener


@pytest.mark.asyncio
async def test_adopting_a_new_tab_survives_a_closed_current_page(executor):
    opener, popup, session = _popup_setup(executor)
    executor.set_current_run_id("r1")
    popup.closed = True
    session.context.pages = [opener]  # one page left: the old early return skipped the switch

    await executor._adopt_new_tab_if_opened()

    assert session.page is opener


@pytest.mark.asyncio
async def test_with_no_page_left_the_error_is_clear(executor):
    opener, popup, session = _popup_setup(executor)
    executor.set_current_run_id("r1")
    popup.closed = True
    session.context.pages = []

    with pytest.raises(RuntimeError, match="no open page left"):
        await executor._current_page()


# -- a dialog stops the batch / the atomic type ----------------------------------------


def _wait_children():
    from operon.models.policy import ActionType, AgentAction

    return [AgentAction(action_type=ActionType.WAIT, wait_ms=1), AgentAction(action_type=ActionType.WAIT, wait_ms=2)]


@pytest.mark.asyncio
async def test_a_cancelled_dialog_stops_the_batch_and_the_replay_runs_each_child_once(executor):
    from operon.models.execution import ExecutedAction
    from operon.models.policy import ActionType, AgentAction

    page = _page()
    executor._attach_page_listeners(page, "r1")
    executor.set_current_run_id("r1")
    children = _wait_children()
    ran: list[int] = []

    async def run_child(child):
        ran.append(child.wait_ms)
        if child.wait_ms == 1:
            await page.handlers["dialog"](_dialog("confirm", "Sure?"))
        return ExecutedAction(action=child, success=True, detail="ok")

    batch = AgentAction(action_type=ActionType.BATCH, actions=children)
    with patch.object(executor, "execute", new=run_child):
        first = await executor._execute_batch(batch)

        assert ran == [1]  # child 1 never ran
        assert first.success and first.batch_failed_index == 0 and "asked a question" in first.detail
        assert executor.pop_pending_dialog("r1") == ("confirm", "Sure?", 0)

        executor.arm_accept_next_dialog("r1", "confirm", "Sure?")  # the person approved: replay from child 0
        second = await executor._execute_batch(batch)

    assert ran == [1, 1, 2]  # child 0 again (accepted this time), then child 1, once each
    assert second.success and second.batch_failed_index is None
    assert executor.pop_pending_dialog("r1") is None


@pytest.mark.asyncio
async def test_a_second_dialog_in_a_batch_cannot_overwrite_the_first(executor):
    from operon.models.execution import ExecutedAction
    from operon.models.policy import ActionType, AgentAction

    page = _page()
    executor._attach_page_listeners(page, "r1")
    executor.set_current_run_id("r1")

    async def run_child(child):
        await page.handlers["dialog"](_dialog("confirm", f"Question {child.wait_ms}?"))
        return ExecutedAction(action=child, success=True, detail="ok")

    with patch.object(executor, "execute", new=run_child):
        await executor._execute_batch(AgentAction(action_type=ActionType.BATCH, actions=_wait_children()))

    assert executor.pop_pending_dialog("r1") == ("confirm", "Question 1?", 0)


@pytest.mark.asyncio
async def test_a_focus_click_that_raises_a_dialog_types_nothing_until_approved(executor):
    from operon.models.policy import ActionType, AgentAction

    page = _page()
    page.keyboard.type = AsyncMock()
    page.keyboard.press = AsyncMock()
    state = {"first": True}

    async def click(x, y, **kwargs):
        if state["first"]:
            state["first"] = False
        await page.handlers["dialog"](_dialog("confirm", "Leave this field?"))

    page.mouse.click = AsyncMock(side_effect=click)
    executor._attach_page_listeners(page, "r1")
    executor._sessions["r1"] = _session(page)
    executor.set_current_run_id("r1")
    action = AgentAction(action_type=ActionType.TYPE, x=10, y=20, text="hello")

    first = await executor.execute(action)

    page.keyboard.type.assert_not_awaited()
    assert first.success and "asked a question" in first.detail
    assert executor.pop_pending_dialog("r1") == ("confirm", "Leave this field?", None)

    executor.arm_accept_next_dialog("r1", "confirm", "Leave this field?")
    with patch.object(executor, "settle_now", new=AsyncMock()), patch.object(
        executor, "_capture_after", new=AsyncMock(return_value="after.png")
    ):
        await executor.execute(action)  # the approved replay: focus, then type

    page.keyboard.type.assert_awaited_once_with("hello")


# -- takeover, two dialogs in one action, a dialog on the last child -------------------


@pytest.mark.asyncio
async def test_a_dialog_raised_by_the_persons_takeover_input_is_accepted(executor):
    page = _page()
    dialogs = []

    async def click(x, y, **kwargs):
        dialog = _dialog("confirm", "Leave this page?")
        dialogs.append(dialog)
        await page.handlers["dialog"](dialog)

    page.mouse.click = AsyncMock(side_effect=click)
    executor._attach_page_listeners(page, "r1")
    executor._sessions["r1"] = _session(page)

    assert await executor.user_input("r1", input_type="click", x_ratio=0.5, y_ratio=0.5)

    dialogs[0].accept.assert_awaited_once()  # the person answers it, as before
    assert executor.pop_pending_dialog("r1") is None
    # The same dialog raised by an agent action, outside the person's input, is still cancelled.
    agent = await _raise(executor, "r1", "confirm", "Leave this page?")
    agent.dismiss.assert_awaited_once()
    assert executor.pop_pending_dialog("r1") == ("confirm", "Leave this page?", None)


@pytest.mark.asyncio
async def test_while_the_person_drives_their_dialogs_are_accepted_until_handed_back(executor):
    executor.set_person_driving("r1", True)
    driving = await _raise(executor, "r1", "confirm", "Sure?")
    executor.set_person_driving("r1", False)
    handed_back = await _raise(executor, "r1", "confirm", "Sure?")

    driving.accept.assert_awaited_once()
    handed_back.dismiss.assert_awaited_once()


@pytest.mark.asyncio
async def test_two_dialogs_in_one_action_keep_the_first(executor):
    page = _page()
    executor._attach_page_listeners(page, "r1")

    first = _dialog("confirm", "First question?")
    second = _dialog("confirm", "Second question?")
    await page.handlers["dialog"](first)
    await page.handlers["dialog"](second)

    assert executor.pop_pending_dialog("r1") == ("confirm", "First question?", None)
    second.dismiss.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_dialog_on_the_last_child_is_attributed_to_it(executor):
    from operon.models.execution import ExecutedAction
    from operon.models.policy import ActionType, AgentAction

    page = _page()
    executor._attach_page_listeners(page, "r1")
    executor.set_current_run_id("r1")
    ran: list[int] = []

    async def run_child(child):
        ran.append(child.wait_ms)
        if child.wait_ms == 2:
            await page.handlers["dialog"](_dialog("confirm", "Sure?"))
        return ExecutedAction(action=child, success=True, detail="ok")

    with patch.object(executor, "execute", new=run_child):
        result = await executor._execute_batch(AgentAction(action_type=ActionType.BATCH, actions=_wait_children()))

    assert ran == [1, 2] and result.batch_failed_index == 1
    assert executor.pop_pending_dialog("r1") == ("confirm", "Sure?", 1)

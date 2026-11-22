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

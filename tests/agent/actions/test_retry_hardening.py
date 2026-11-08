"""Which failed executions the loop may retry by running the same action again."""

from __future__ import annotations

from operon.agent.actions.retry_hardening import should_retry
from operon.models.common import FailureCategory
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction

CLICK = AgentAction(action_type=ActionType.CLICK, x=1, y=1)


def _failed(action: AgentAction, **extra) -> ExecutedAction:
    return ExecutedAction(action=action, success=False, detail="no", failure_category=FailureCategory.CLICK_NO_EFFECT, **extra)


def test_a_failed_click_is_retried() -> None:
    assert should_retry(_failed(CLICK)) is True


def test_a_batch_that_failed_after_its_first_child_is_not_rerun() -> None:
    """Re-running it would repeat the typing or submit that already happened."""
    batch = AgentAction(action_type=ActionType.BATCH, actions=[AgentAction(action_type=ActionType.TYPE, text="hi"), CLICK])

    assert should_retry(_failed(batch, batch_failed_index=1)) is False
    assert should_retry(_failed(batch, batch_failed_index=0)) is True


def test_an_invisible_right_click_menu_is_not_retried() -> None:
    assert should_retry(_failed(AgentAction(action_type=ActionType.RIGHT_CLICK, x=1, y=1))) is False

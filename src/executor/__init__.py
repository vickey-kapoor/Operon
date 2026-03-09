"""Executor module - Playwright browser automation and action execution."""

from .browser import PlaywrightBrowserExecutor
from .actions import ActionType, Action, ActionResult

__all__ = [
    "PlaywrightBrowserExecutor",
    "ActionType",
    "Action",
    "ActionResult",
]

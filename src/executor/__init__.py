"""Executor module - Playwright browser automation and action execution."""

from .actions import Action, ActionResult, ActionType
from .browser import PlaywrightBrowserExecutor

__all__ = [
    "PlaywrightBrowserExecutor",
    "ActionType",
    "Action",
    "ActionResult",
]

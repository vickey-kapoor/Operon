"""Executor tests do not enumerate the real machine's windows."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_real_window_listing():
    """Launch confirmation reads the window list; None means "cannot check", so the
    older tests that mock Popen keep trusting the call. Tests of the check patch it."""
    with patch("operon.executor.desktop._visible_windows", return_value=None):
        yield

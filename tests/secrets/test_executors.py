"""Every TYPE path fills in a ``{{secret:NAME}}`` at the last moment and never changes the action."""

from __future__ import annotations

import asyncio
import ctypes
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from operon.core import vault
from operon.core.paths import test_artifacts_dir as artifact_root_dir
from operon.environments.guest import create_guest_app
from operon.environments.osworld import OSWorldExecutor
from operon.environments.remote import RemoteDesktopExecutor
from operon.executor import desktop as _desktop
from operon.executor import sendinput
from operon.executor.browser_native import NativeBrowserExecutor
from operon.executor.desktop import DesktopExecutor
from operon.executor.linux_desktop import LinuxDesktopExecutor
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from tests.fixtures.paths import unique_artifact_dir

SECRET = "hunter2-SECRET"
PLACEHOLDER = "{{secret:pw}}"


@pytest.fixture
def stored(vault_dir):
    vault.set_secret("pw", SECRET)


def _type(text: str = PLACEHOLDER, **kw) -> AgentAction:
    return AgentAction(action_type=ActionType.TYPE, text=text, x=100, y=100, **kw)


def _assert_clean(result: ExecutedAction, action: AgentAction) -> None:
    assert result.action.text == action.text
    assert SECRET not in result.model_dump_json()
    assert SECRET not in action.model_dump_json()


# -- browser -------------------------------------------------------------------------------


def _browser():
    executor = NativeBrowserExecutor(artifact_dir=artifact_root_dir() / "secrets-native", headless=True)
    keyboard = SimpleNamespace(press=AsyncMock(), type=AsyncMock())
    mouse = SimpleNamespace(click=AsyncMock())
    page = SimpleNamespace(mouse=mouse, keyboard=keyboard, wait_for_load_state=AsyncMock())
    executor._current_page = AsyncMock(return_value=page)
    executor._capture_after = AsyncMock(return_value="after.png")
    return executor, keyboard, mouse


async def test_browser_types_the_value_and_keeps_the_placeholder(stored) -> None:
    executor, keyboard, _ = _browser()
    action = _type()
    result = await executor.execute(action)
    assert result.success
    keyboard.type.assert_awaited_once_with(SECRET)
    _assert_clean(result, action)


async def test_browser_batch_children_are_substituted_too(stored) -> None:
    executor, keyboard, _ = _browser()
    batch = AgentAction(action_type=ActionType.BATCH, actions=[_type("user"), _type(PLACEHOLDER)])
    result = await executor.execute(batch)
    assert result.success
    assert [c.args[0] for c in keyboard.type.await_args_list] == ["user", SECRET]
    assert SECRET not in result.model_dump_json()


async def test_browser_unknown_secret_types_and_clicks_nothing(vault_dir) -> None:
    executor, keyboard, mouse = _browser()
    result = await executor.execute(_type("{{secret:nope}}"))
    assert not result.success and "unknown secret nope" in result.detail
    keyboard.type.assert_not_awaited()
    mouse.click.assert_not_awaited()


async def test_browser_flag_off_fails_instead_of_typing_the_literal(stored, monkeypatch) -> None:
    monkeypatch.setenv("OPERON_SECRETS", "off")
    executor, keyboard, _ = _browser()
    result = await executor.execute(_type())
    assert not result.success and "secrets are off" in result.detail
    keyboard.type.assert_not_awaited()


async def test_browser_ordinary_text_is_untouched(stored) -> None:
    executor, keyboard, _ = _browser()
    assert (await executor.execute(_type("hello {{not a secret}}"))).success
    keyboard.type.assert_awaited_once_with("hello {{not a secret}}")


# -- desktop (Windows and Linux executors) ---------------------------------------------------

_SCREEN = {"left": 0, "top": 0, "width": 1280, "height": 800}


@pytest.fixture
def gui(monkeypatch):
    fake = MagicMock()
    fake.KEYBOARD_KEYS = ["ctrl", "a", "v", "enter"]
    shot = SimpleNamespace(rgb=b"\x80" * (1280 * 800 * 3), size=(1280, 800), width=1280, height=800)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.monitors = [dict(_SCREEN), dict(_SCREEN)]
    ctx.grab.return_value = shot
    monkeypatch.setattr(_desktop, "pyautogui", fake)
    monkeypatch.setattr(_desktop, "_mss", MagicMock(return_value=ctx))
    monkeypatch.setenv("OPERON_TEST_SAFE_MODE", "true")
    return fake


def _desktop_executor(cls, monkeypatch):
    executor = cls(artifact_dir=unique_artifact_dir("secrets-desktop"))
    executor.set_current_run_id("run-1")
    monkeypatch.setattr(executor, "_capture_after", AsyncMock(return_value="after.png"))
    return executor


async def test_linux_desktop_types_the_value_key_by_key(stored, gui, monkeypatch) -> None:
    executor = _desktop_executor(LinuxDesktopExecutor, monkeypatch)
    action = _type(press_enter=True)
    result = await executor.execute(action)
    assert result.success
    gui.write.assert_called_once_with(SECRET, 0.01)
    assert "[secret]" not in result.detail and PLACEHOLDER in result.detail
    _assert_clean(result, action)


async def test_windows_desktop_sends_key_events_and_never_uses_the_clipboard(stored, gui, monkeypatch) -> None:
    copied: list[str] = []
    monkeypatch.setattr(_desktop.pyperclip, "copy", copied.append)
    sent: list[str] = []
    monkeypatch.setattr(_desktop.sendinput, "type_unicode", sent.append)
    monkeypatch.setattr(DesktopExecutor, "_secret_problem", lambda self, text: None)
    executor = _desktop_executor(DesktopExecutor, monkeypatch)
    action = _type()
    result = await executor.execute(action)
    assert result.success
    assert sent == [SECRET]
    assert copied == []  # pyperclip.copy was never called, with the value or anything else
    _assert_clean(result, action)


async def test_read_clipboard_after_a_secret_type_does_not_return_it(stored, gui, monkeypatch) -> None:
    monkeypatch.setattr(_desktop.sendinput, "type_unicode", lambda text: None)
    monkeypatch.setattr(DesktopExecutor, "_secret_problem", lambda self, text: None)
    monkeypatch.setattr(_desktop.pyperclip, "paste", lambda: f"x {SECRET} y")
    executor = _desktop_executor(DesktopExecutor, monkeypatch)
    assert (await executor.execute(_type())).success
    read = await executor.execute(AgentAction(action_type=ActionType.READ_CLIPBOARD))
    assert read.success and SECRET not in read.detail and "[secret]" in read.detail


async def test_desktop_without_a_trustworthy_keyboard_path_fails_closed(stored, gui, monkeypatch) -> None:
    copied: list[str] = []
    monkeypatch.setattr(_desktop.pyperclip, "copy", copied.append)
    monkeypatch.setattr(DesktopExecutor, "_secret_problem", lambda self, text: "no way to type it")
    executor = _desktop_executor(DesktopExecutor, monkeypatch)
    result = await executor.execute(_type())
    assert not result.success and copied == []
    gui.click.assert_not_called()


@pytest.mark.parametrize("secret", ["pä$$wörd", "emoji \U0001F600 end", "東京", "line\nbreak"])
async def test_linux_secret_with_a_non_typeable_character_fails_closed(vault_dir, gui, monkeypatch, secret) -> None:
    vault.set_secret("pw", secret)
    copied: list[str] = []
    monkeypatch.setattr(_desktop.pyperclip, "copy", copied.append)
    executor = _desktop_executor(LinuxDesktopExecutor, monkeypatch)
    result = await executor.execute(_type())
    assert not result.success and "clipboard" in result.detail
    assert copied == [] and secret not in result.detail
    gui.write.assert_not_called()
    gui.click.assert_not_called()


async def test_linux_long_secret_is_typed_not_pasted(vault_dir, gui, monkeypatch) -> None:
    long_secret = "p" * 800
    vault.set_secret("pw", long_secret)
    copied: list[str] = []
    monkeypatch.setattr(_desktop.pyperclip, "copy", copied.append)
    executor = _desktop_executor(LinuxDesktopExecutor, monkeypatch)
    assert (await executor.execute(_type())).success
    gui.write.assert_called_once_with(long_secret, 0.01)
    assert copied == []


# -- SendInput helper ------------------------------------------------------------------------


def test_sendinput_units_and_surrogates() -> None:
    assert sendinput.utf16_units("aé") == [0x61, 0xE9]
    assert sendinput.utf16_units("\U0001F600") == [0xD83D, 0xDE00]  # a surrogate pair
    assert sendinput.utf16_units("東") == [0x6771]


def test_sendinput_sends_unicode_down_and_up_per_unit(monkeypatch) -> None:
    calls: list[list[tuple[int, int]]] = []

    class FakeUser32:
        def SendInput(self, count, array, size):
            assert size == ctypes.sizeof(sendinput.Input)
            calls.append([(array[i].u.ki.wScan, array[i].u.ki.dwFlags) for i in range(count)])
            return count

    monkeypatch.setattr(sendinput, "_user32", lambda: FakeUser32())
    sendinput.type_unicode("a\U0001F600")
    flat = [event for call in calls for event in call]
    down, up = sendinput.KEYEVENTF_UNICODE, sendinput.KEYEVENTF_UNICODE | sendinput.KEYEVENTF_KEYUP
    assert flat == [(0x61, down), (0x61, up), (0xD83D, down), (0xD83D, up), (0xDE00, down), (0xDE00, up)]


def test_sendinput_batches_long_text_and_reports_a_refusal(monkeypatch) -> None:
    sizes: list[int] = []

    class FakeUser32:
        def SendInput(self, count, array, size):
            sizes.append(count)
            return count

    monkeypatch.setattr(sendinput, "_user32", lambda: FakeUser32())
    sendinput.type_unicode("x" * 300)
    assert sum(sizes) == 600 and max(sizes) <= 200

    class Refusing:
        def SendInput(self, count, array, size):
            return 0

    monkeypatch.setattr(sendinput, "_user32", lambda: Refusing())
    with pytest.raises(OSError):
        sendinput.type_unicode("x")


async def test_desktop_unknown_secret_clicks_and_types_nothing(vault_dir, gui, monkeypatch) -> None:
    copied: list[str] = []
    monkeypatch.setattr(_desktop.pyperclip, "copy", copied.append)
    for cls in (DesktopExecutor, LinuxDesktopExecutor):
        executor = _desktop_executor(cls, monkeypatch)
        result = await executor.execute(_type("{{secret:nope}}"))
        assert not result.success and "unknown secret nope" in result.detail
    gui.click.assert_not_called()
    gui.write.assert_not_called()
    assert copied == []


async def test_desktop_failure_detail_never_carries_the_value(stored, gui, monkeypatch) -> None:
    gui.write.side_effect = RuntimeError(f"boom while typing {SECRET}")
    executor = _desktop_executor(LinuxDesktopExecutor, monkeypatch)
    result = await executor.execute(_type())
    assert not result.success and SECRET not in result.detail and "[secret]" in result.detail


async def test_desktop_batch_children_are_substituted(stored, gui, monkeypatch) -> None:
    executor = _desktop_executor(LinuxDesktopExecutor, monkeypatch)
    result = await executor.execute(AgentAction(action_type=ActionType.BATCH, actions=[_type(PLACEHOLDER)]))
    assert result.success
    gui.write.assert_called_once_with(SECRET, 0.01)
    assert SECRET not in result.model_dump_json()


# -- remote guest (Windows Sandbox / Linux container) ----------------------------------------


class _EchoGuest:
    """A guest that behaves like the real desktop executor: it echoes what it typed in its detail."""

    def __init__(self, tmp_path) -> None:
        self.tmp_path = tmp_path
        self.received: list[AgentAction] = []

    async def capture(self):
        raise NotImplementedError

    async def execute(self, action: AgentAction) -> ExecutedAction:
        self.received.append(action)
        return ExecutedAction(action=action, success=True, detail=f"Typed '{action.text}'")


def _remote(tmp_path, url: str, trusted_host: str | None = None):
    guest = _EchoGuest(tmp_path)
    transport = httpx.ASGITransport(app=create_guest_app(guest, "tok"))
    remote = RemoteDesktopExecutor(
        url, "tok", artifact_dir=tmp_path / "host", transport=transport, trusted_host=trusted_host
    )
    return remote, guest


async def test_remote_substitutes_on_the_host_and_masks_the_reply(stored, tmp_path) -> None:
    remote, guest = _remote(tmp_path, "http://10.0.0.5:8765", trusted_host="10.0.0.5")
    action = _type()
    result = await remote.execute(action)
    assert result.success
    assert guest.received[0].text == SECRET  # the guest typed the real value
    assert "[secret]" in result.detail
    _assert_clean(result, action)


async def test_remote_batch_children_are_substituted(stored, tmp_path) -> None:
    remote, guest = _remote(tmp_path, "http://127.0.0.1:9000")
    result = await remote.execute(AgentAction(action_type=ActionType.BATCH, actions=[_type()]))
    assert result.success and guest.received[0].text == SECRET
    assert SECRET not in result.model_dump_json()


@pytest.mark.parametrize(
    "url", ["http://8.8.8.8:8765", "http://10.0.0.5:8765", "http://192.168.1.9:8765", "http://guest:8765"]
)
async def test_remote_refuses_any_address_the_provider_did_not_report(stored, tmp_path, url) -> None:
    """Only loopback and the provider's own reported address: no general private ranges."""
    remote, guest = _remote(tmp_path, url, trusted_host="10.0.0.99")
    result = await remote.execute(_type())
    assert not result.success and "loopback or provider-reported" in result.detail
    assert guest.received == []


async def test_guest_does_not_re_resolve_or_paste_a_host_filled_value(vault_dir, gui, monkeypatch, tmp_path) -> None:
    """The host fills in a value that itself looks like a placeholder: the guest types it as is."""
    vault.set_secret("pw", "{{secret:other}}")
    vault.set_secret("other", "WRONG")
    copied: list[str] = []
    monkeypatch.setattr(_desktop.pyperclip, "copy", copied.append)
    executor = _desktop_executor(LinuxDesktopExecutor, monkeypatch)
    transport = httpx.ASGITransport(app=create_guest_app(executor, "tok"))
    remote = RemoteDesktopExecutor("http://127.0.0.1:9", "tok", artifact_dir=tmp_path / "host", transport=transport)
    result = await remote.execute(_type())
    assert result.success
    gui.write.assert_called_once_with("{{secret:other}}", 0.01)
    assert copied == []
    assert result.action.text == PLACEHOLDER


async def test_remote_unknown_secret_sends_nothing(vault_dir, tmp_path) -> None:
    remote, guest = _remote(tmp_path, "http://127.0.0.1:9000")
    result = await remote.execute(_type("{{secret:nope}}"))
    assert not result.success and "unknown secret nope" in result.detail
    assert guest.received == []


# -- OSWorld VM ------------------------------------------------------------------------------


def test_osworld_substitutes_only_in_the_snippet(stored, tmp_path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["command"][2])
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    executor = OSWorldExecutor("http://127.0.0.1:5000", artifact_dir=tmp_path, transport=httpx.MockTransport(handler))
    action = AgentAction(action_type=ActionType.TYPE, text=PLACEHOLDER)
    result = asyncio.run(executor.execute(action))
    assert result.success and SECRET in seen[0]
    _assert_clean(result, action)


def test_osworld_unknown_secret_sends_nothing(vault_dir, tmp_path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    executor = OSWorldExecutor("http://127.0.0.1:5000", artifact_dir=tmp_path, transport=httpx.MockTransport(handler))
    result = asyncio.run(executor.execute(AgentAction(action_type=ActionType.TYPE, text="{{secret:nope}}")))
    assert not result.success and calls == []


@pytest.mark.parametrize("url", ["http://vm:5000", "http://10.0.0.7:5000", "http://8.8.8.8:5000"])
def test_osworld_refuses_a_non_loopback_vm_for_secrets(stored, tmp_path, url) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    executor = OSWorldExecutor(url, artifact_dir=tmp_path, transport=httpx.MockTransport(handler))
    result = asyncio.run(executor.execute(AgentAction(action_type=ActionType.TYPE, text=PLACEHOLDER)))
    assert not result.success and "loopback" in result.detail and calls == []
    # Ordinary text still goes to such a VM.
    assert asyncio.run(executor.execute(AgentAction(action_type=ActionType.TYPE, text="hello"))).success


def test_osworld_trusted_host_is_allowed(stored, tmp_path) -> None:
    handler = lambda request: httpx.Response(200, json={"status": "success", "returncode": 0})  # noqa: E731
    executor = OSWorldExecutor(
        "http://10.0.0.7:5000", artifact_dir=tmp_path, transport=httpx.MockTransport(handler), trusted_host="10.0.0.7"
    )
    assert asyncio.run(executor.execute(AgentAction(action_type=ActionType.TYPE, text=PLACEHOLDER))).success


async def test_secrets_used_are_forgotten_when_the_run_ends(stored, gui, monkeypatch) -> None:
    monkeypatch.setattr(_desktop.sendinput, "type_unicode", lambda text: None)
    monkeypatch.setattr(DesktopExecutor, "_secret_problem", lambda self, text: None)
    executor = _desktop_executor(DesktopExecutor, monkeypatch)
    assert (await executor.execute(_type())).success
    assert executor._secrets_used == {"run-1": {SECRET}}
    await executor.aclose_run("run-1")
    assert executor._secrets_used == {}
    assert (await executor.execute(_type())).success
    await executor.aclose()
    assert executor._secrets_used == {}


def test_sendinput_refusal_message_does_not_claim_a_cause(monkeypatch) -> None:
    class Refusing:
        def SendInput(self, count, array, size):
            return 0

    monkeypatch.setattr(sendinput, "_user32", lambda: Refusing())
    with pytest.raises(OSError, match="silently"):
        sendinput.type_unicode("x")

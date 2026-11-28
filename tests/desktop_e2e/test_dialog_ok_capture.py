"""Reproduce the OSWorld dialog-OK capture race on the free Linux desktop container.

What went wrong in a real run (OSWorld run e6e79fd8; the same shape in the Calc run
a169a375, steps 12-14): the agent clicked a dialog's OK button and the screenshot taken
right after still showed the dialog. The click had worked. The app was busy for a moment
with no visual change, then the dialog closed, and the frame was captured before that.
``OSWorldExecutor`` returns as soon as the VM replies, with no wait after the action, and
``DesktopExecutor._exec_click`` waits only 0.15 s before it captures.

These tests rebuild that on $0 infrastructure. A small Tk dialog (solid magenta, an OK
button and a Noop button) is started in the desktop container. Clicking OK writes a marker,
stays busy for ``CLOSE_DELAY_S`` seconds with no repaint, then writes another marker and
closes. Two code paths click it for real:

* ``osworld``: the real ``OSWorldExecutor``, with its HTTP transport swapped for one that
  runs each request with ``docker exec`` (a fresh process per action, like OSWorld's server).
* ``remote``: the real ``RemoteDesktopExecutor`` talking to the container's guest agent.

``test_one_click_closes_dialog`` and ``test_frame_after_wait_shows_result`` (the control)
pass: the click works and the dialog does close. The ``xfail(strict=True)`` tests say the
frame captured right after the click still shows the dialog. Each can only xfail on
``SentinelStillVisible``, raised by the final pixel check alone, so a broken setup fails instead.

The osworld fix is ``OPERON_OSWORLD_POST_ACTION_WAIT_S`` (a fixed wait after each input action,
off by default). ``test_frame_after_click_with_wait_shows_result`` passes with it on. The osworld
xfail runs with the default wait (off): it documents that the default still has the race, and if the
default ever changes it XPASSes (strict makes that loud) and can go green. The remote and
settle-detect paths have no fix yet, so their xfails stay.

Measured on CI (GitHub run 44340050002, job "Desktop (Linux container)", 2026-11-18),
click-to-capture time over 7 samples per path:
    osworld: min 0.79 s, median 0.79 s, max 0.80 s
    remote:  min 0.65 s, median 0.66 s, max 0.67 s
CLOSE_DELAY_S was 3.0 s (3 x that worst sample). On 2026-11-23 (run 45104170922) the osworld
path measured max 1.01 s, so 3 x worst = 3.03 s and the probe failed by 0.03 s on noise alone.
CLOSE_DELAY_S = 5.0 s now: at least 3 x the worst sample seen so far with room for runner jitter
(the probe passes up to a 1.67 s worst sample), and at least 2 s.

Opt in: ``pytest tests/desktop_e2e -m desktop_container -o addopts=""``. Needs Docker.
"""

from __future__ import annotations

import asyncio
import io
import json
import statistics
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from PIL import Image

from operon.environments.linux_container import LinuxContainerProvider
from operon.environments.osworld import OSWorldExecutor
from operon.models.policy import ActionType, AgentAction

pytestmark = pytest.mark.desktop_container

# How long the dialog stays busy after OK, with no visual change. It must be well above the
# click-to-capture time of both paths (the probe test checks that), or the race closes by
# itself and the xfail tests would pass for the wrong reason.
CLOSE_DELAY_S = 5.0

SENTINEL = (255, 0, 255)
SENTINEL_TOLERANCE = 12
READY_FILE = "/tmp/dlg-ready.json"
OK_MARKER = "/tmp/dlg-ok"
CLOSED_MARKER = "/tmp/dlg-closed"
DIALOG_PATTERN = "python3 /tmp/dl[g].py"  # the [g] keeps pgrep/pkill from matching themselves

DIALOG_SOURCE = '''\
import json
import os
import sys
import time
import tkinter as tk

delay = float(sys.argv[1])
root = tk.Tk()
root.title("Operon test dialog")
root.geometry("500x300+300+150")
root.configure(bg="#ff00ff")
root.attributes("-topmost", True)


def on_ok():
    open("/tmp/dlg-ok", "w").write("1")
    time.sleep(delay)  # busy: the event loop is blocked, so nothing repaints
    open("/tmp/dlg-closed", "w").write("1")
    root.destroy()


ok = tk.Button(root, text="OK", command=on_ok, width=10, height=2)
ok.place(x=60, y=200)
noop = tk.Button(root, text="Noop", command=lambda: None, width=10, height=2)
noop.place(x=300, y=200)


def centre(widget):
    return [widget.winfo_rootx() + widget.winfo_width() // 2,
            widget.winfo_rooty() + widget.winfo_height() // 2]


def ready():
    root.update_idletasks()
    data = {
        "ok": centre(ok),
        "noop": centre(noop),
        "sentinel": [root.winfo_rootx() + root.winfo_width() // 2, root.winfo_rooty() + 60],
    }
    with open("/tmp/dlg-ready.json.tmp", "w") as handle:
        json.dump(data, handle)
    os.replace("/tmp/dlg-ready.json.tmp", "/tmp/dlg-ready.json")


root.update()
root.after(1500, ready)  # after the window manager has placed the window
root.mainloop()
'''


class SentinelStillVisible(AssertionError):
    """The frame still shows the dialog. Raised only by the final pixel check."""


# ── container helpers ─────────────────────────────────────────────


def docker_exec(container: str, argv: list[str], *, stdin: str | None = None,
                timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", *(["-i"] if stdin is not None else []), "-e", "DISPLAY=:1", container, *argv],
        input=stdin.encode() if stdin is not None else None,
        capture_output=True, timeout=timeout, check=False,
    )


SCREENSHOT_SCRIPT = (
    "import sys, mss, mss.tools\n"
    "with mss.mss() as s:\n"
    "    img = s.grab(s.monitors[1])\n"
    "    sys.stdout.buffer.write(mss.tools.to_png(img.rgb, img.size))\n"
)


def docker_screenshot(container: str) -> bytes:
    proc = docker_exec(container, ["python3", "-c", SCREENSHOT_SCRIPT])
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return proc.stdout


def sentinel_present(png: str | Path | bytes, point: tuple[int, int]) -> bool:
    """True when the pixel at ``point`` is the dialog's magenta (within a small tolerance)."""
    source = io.BytesIO(png) if isinstance(png, bytes) else png
    with Image.open(source) as image:
        pixel = image.convert("RGB").getpixel(point)
    assert isinstance(pixel, tuple)
    return all(abs(a - b) <= SENTINEL_TOLERANCE for a, b in zip(pixel, SENTINEL, strict=True))


def assert_sentinel_gone(frames: dict[str, str], point: tuple[int, int]) -> None:
    """Check every named frame; raise SentinelStillVisible (and only that) if any shows the dialog."""
    shown = [name for name, path in frames.items() if sentinel_present(path, point)]
    if shown:
        raise SentinelStillVisible(f"dialog still visible in: {', '.join(shown)}")


def file_exists(container: str, path: str) -> bool:
    return docker_exec(container, ["test", "-e", path], timeout=30).returncode == 0


def dialog_running(container: str) -> bool:
    return docker_exec(container, ["pgrep", "-f", DIALOG_PATTERN], timeout=30).returncode == 0


@dataclass
class Dialog:
    container: str
    ok: tuple[int, int]
    noop: tuple[int, int]
    sentinel: tuple[int, int]


def stop_dialog(container: str) -> None:
    docker_exec(container, ["pkill", "-f", DIALOG_PATTERN], timeout=30)
    docker_exec(container, ["rm", "-f", READY_FILE, OK_MARKER, CLOSED_MARKER], timeout=30)


def start_dialog(container: str, close_delay_s: float = CLOSE_DELAY_S) -> Dialog:
    """Start a fresh dialog and prove with a real capture that it is on screen."""
    stop_dialog(container)
    time.sleep(0.5)
    written = docker_exec(container, ["sh", "-c", "cat > /tmp/dlg.py"], stdin=DIALOG_SOURCE, timeout=30)
    assert written.returncode == 0, written.stderr.decode(errors="replace")
    started = subprocess.run(
        ["docker", "exec", "-d", "-e", "DISPLAY=:1", container, "python3", "/tmp/dlg.py", str(close_delay_s)],
        capture_output=True, timeout=30, check=False,
    )
    assert started.returncode == 0, started.stderr.decode(errors="replace")
    deadline = time.monotonic() + 15
    raw = b""
    while time.monotonic() < deadline:
        proc = docker_exec(container, ["cat", READY_FILE], timeout=30)
        if proc.returncode == 0 and proc.stdout.strip():
            raw = proc.stdout
            break
        time.sleep(0.3)
    assert raw, "the dialog never wrote its ready file (is tkinter available in the image?)"
    data = json.loads(raw)
    dialog = Dialog(container, tuple(data["ok"]), tuple(data["noop"]), tuple(data["sentinel"]))
    assert sentinel_present(docker_screenshot(container), dialog.sentinel), \
        "precondition: the dialog's sentinel colour is not on screen"
    return dialog


@pytest.fixture
def dialog(desktop):
    d = start_dialog(desktop.container)
    try:
        yield d
    finally:
        stop_dialog(desktop.container)


# ── the OSWorld path: real OSWorldExecutor, docker exec as the VM's server ──


class DockerExecTransport(httpx.AsyncBaseTransport):
    """Stands in for OSWorld's HTTP server: /execute runs the command as a fresh process in
    the container, /screenshot returns a PNG of its screen."""

    def __init__(self, container: str) -> None:
        self.container = container

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/execute":
            argv = json.loads(request.content)["command"]
            proc = await asyncio.to_thread(docker_exec, self.container, argv)
            return httpx.Response(200, json={
                "status": "success" if proc.returncode == 0 else "error",
                "output": proc.stdout.decode(errors="replace"),
                "error": proc.stderr.decode(errors="replace"),
                "returncode": proc.returncode,
            })
        if request.method == "GET" and path == "/screenshot":
            png = await asyncio.to_thread(docker_screenshot, self.container)
            return httpx.Response(200, content=png, headers={"content-type": "image/png"})
        return httpx.Response(404)


def make_executor(path: str, provider, tmp_path: Path, *, post_action_wait_s: float | None = None):
    if path == "osworld":
        return OSWorldExecutor("http://osworld.test", artifact_dir=tmp_path,
                               transport=DockerExecTransport(provider.container),
                               post_action_wait_s=post_action_wait_s)
    return provider.build_executor()


def click_action(point: tuple[int, int]) -> AgentAction:
    return AgentAction(action_type=ActionType.CLICK, x=point[0], y=point[1])


PATHS = pytest.mark.parametrize("path", ["osworld", "remote"])


# ── latency probe ─────────────────────────────────────────────────


def test_transport_latency_is_well_under_close_delay(desktop, dialog, tmp_path):
    """Click Noop and capture, 7 times per path. The wall time from just before execute() to
    just after the following capture() is an upper bound on the click-to-frame gap. The
    dialog's close delay must leave a wide margin over it."""

    async def sample(path: str) -> list[float]:
        executor = make_executor(path, desktop, tmp_path)
        times: list[float] = []
        try:
            for _ in range(7):
                start = time.monotonic()
                result = await executor.execute(click_action(dialog.noop))
                await executor.capture()
                times.append(time.monotonic() - start)
                assert result.success, result.detail
        finally:
            await executor.aclose()
        return times

    worst = 0.0
    for path in ("osworld", "remote"):
        times = asyncio.run(sample(path))
        warnings.warn(
            f"click-to-capture [{path}]: min {min(times):.2f}s median {statistics.median(times):.2f}s "
            f"max {max(times):.2f}s (CLOSE_DELAY_S={CLOSE_DELAY_S})",
            stacklevel=1,
        )
        worst = max(worst, max(times))
    assert CLOSE_DELAY_S >= max(2.0, 3 * worst), (
        f"CLOSE_DELAY_S={CLOSE_DELAY_S} is too close to the worst click-to-capture time {worst:.2f}s"
    )


# ── the race ──────────────────────────────────────────────────────


@PATHS
def test_one_click_closes_dialog(desktop, dialog, tmp_path, path):
    """The click works: OK is handled and the dialog closes by itself. Must pass."""

    async def scenario():
        executor = make_executor(path, desktop, tmp_path)
        try:
            return await executor.execute(click_action(dialog.ok))
        finally:
            await executor.aclose()

    result = asyncio.run(scenario())
    assert result.success, result.detail
    deadline = time.monotonic() + CLOSE_DELAY_S + 10
    while time.monotonic() < deadline:
        if file_exists(desktop.container, CLOSED_MARKER) and not dialog_running(desktop.container):
            break
        time.sleep(0.3)
    assert file_exists(desktop.container, OK_MARKER), "the click never reached OK"
    assert file_exists(desktop.container, CLOSED_MARKER), "the dialog never finished closing"
    assert not dialog_running(desktop.container), "the dialog process is still running"


async def _click_ok_then_capture(executor, dialog: Dialog):
    """One click on OK, then the capture the loop's next step would take."""
    result = await executor.execute(click_action(dialog.ok))
    frame = await executor.capture()
    return result, frame


@PATHS
@pytest.mark.xfail(strict=True, raises=SentinelStillVisible,
                   reason="default (no post-action wait): the frame is taken before the dialog closes; "
                          "OPERON_OSWORLD_POST_ACTION_WAIT_S fixes the osworld path when set")
def test_frame_after_click_shows_result(desktop, dialog, tmp_path, path):
    """Click OK, capture straight away: the dialog should be gone. With the default (wait off) it is
    not, so this documents the race. If the default ever changes, the osworld case XPASSes (strict
    makes that loud) and the mark can go."""

    async def scenario():
        executor = make_executor(path, desktop, tmp_path)
        try:
            return await _click_ok_then_capture(executor, dialog)
        finally:
            await executor.aclose()

    result, frame = asyncio.run(scenario())
    assert result.success, result.detail
    assert file_exists(desktop.container, OK_MARKER), "the click never reached OK"
    frames = {"capture after the click": frame.artifact_path}
    if path == "remote":
        assert result.artifact_path, "the guest sent no after-frame"
        frames["after-frame of the click"] = result.artifact_path
    assert_sentinel_gone(frames, dialog.sentinel)


def test_frame_after_click_with_wait_shows_result(desktop, dialog, tmp_path):
    """The fix: the osworld executor waits past the close delay after the click, so the capture
    taken straight after shows the dialog gone. Must pass."""

    async def scenario():
        executor = make_executor("osworld", desktop, tmp_path, post_action_wait_s=CLOSE_DELAY_S + 1.0)
        try:
            return await _click_ok_then_capture(executor, dialog)
        finally:
            await executor.aclose()

    result, frame = asyncio.run(scenario())
    assert result.success, result.detail
    assert file_exists(desktop.container, OK_MARKER), "the click never reached OK"
    assert_sentinel_gone({"capture after the click and wait": frame.artifact_path}, dialog.sentinel)


@PATHS
def test_frame_after_wait_shows_result(desktop, dialog, tmp_path, path):
    """The control: the same click, but wait past the close delay before capturing. Must pass,
    so the only difference from the xfail test is the wait."""

    async def scenario():
        executor = make_executor(path, desktop, tmp_path)
        try:
            result = await executor.execute(click_action(dialog.ok))
            await asyncio.sleep(CLOSE_DELAY_S + 2.0)
            return result, await executor.capture()
        finally:
            await executor.aclose()

    result, frame = asyncio.run(scenario())
    assert result.success, result.detail
    assert file_exists(desktop.container, OK_MARKER), "the click never reached OK"
    assert_sentinel_gone({"capture after the wait": frame.artifact_path}, dialog.sentinel)


# ── the guest's own settle detection does not fix it ──────────────


class SettleProvider(LinuxContainerProvider):
    """A second desktop container whose guest runs with OPERON_SETTLE_DETECT=on."""

    def run_args(self) -> list[str]:
        args = super().run_args()
        args[-1:-1] = ["-e", "OPERON_SETTLE_DETECT=on"]  # before the image name
        return args


@pytest.fixture(scope="module")
def settle_desktop(desktop):
    mp = pytest.MonkeyPatch()
    mp.setenv("OPERON_DESKTOP_PLATFORM", "linux")
    provider = SettleProvider()
    check = provider.check()
    if not check.ok:
        mp.undo()
        pytest.fail("linux-container not ready: " + " ".join(check.problems))
    provider.start()
    try:
        yield provider
    finally:
        provider.stop()
        mp.undo()


@pytest.mark.xfail(strict=True, raises=SentinelStillVisible,
                   reason="settle detection sees no change while the app is busy, so it does not wait")
def test_settle_detect_does_not_fix_it(settle_desktop, tmp_path):
    """With OPERON_SETTLE_DETECT=on the guest waits for the screen to stop changing, but a busy
    app changes nothing on screen, so the frame is still taken too early."""
    container = settle_desktop.container
    dlg = start_dialog(container)
    try:
        async def scenario():
            executor = settle_desktop.build_executor()
            try:
                return await _click_ok_then_capture(executor, dlg)
            finally:
                await executor.aclose()

        result, frame = asyncio.run(scenario())
        assert result.success, result.detail
        assert file_exists(container, OK_MARKER), "the click never reached OK"
        assert result.artifact_path, "the guest sent no after-frame"
        assert_sentinel_gone({"capture after the click": frame.artifact_path,
                              "after-frame of the click": result.artifact_path}, dlg.sentinel)
    finally:
        stop_dialog(container)

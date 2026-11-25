"""Free desktop regressions for input bugs, on the Linux desktop container (KAN-212).

The real thin-harness loop drives the real XFCE desktop; only the model is scripted. A small Tk
window in the guest logs the X11 events it receives to a file, and each test grades from that
file before the executor closes, never from what the run claims.

Opt in: ``pytest tests/desktop_e2e -m desktop_container -o addopts=""``. Needs Docker.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time

import pytest

from operon.models.common import RunStatus

from .helpers import sh
from .test_linux_desktop import call, run_script, say

pytestmark = pytest.mark.desktop_container

READY_FILE = "/tmp/inp-ready.json"
EVENT_LOG = "/tmp/inp-events.log"
FIXTURE_PATTERN = "python3 /tmp/in[p].py"  # the [p] keeps pkill from matching itself
SCREEN_W, SCREEN_H = 1280, 800  # OPERON_SCREEN in the guest

# A window that appends one line per X11 wheel notch (Button-4 up, Button-5 down) and, in
# "flat" mode, per left press. Solid grey (equal RGB channels, so the servo sees zero variance) with one big text label.
WINDOW_SOURCE = '''\
import json
import os
import sys
import tkinter as tk

mode = sys.argv[1]  # "wheel" or "flat"
root = tk.Tk()
root.title("Operon input fixture")
root.geometry("500x300+300+150")
root.configure(bg="#c8c8c8")
root.attributes("-topmost", True)


def log(line):
    with open("/tmp/inp-events.log", "a") as handle:
        handle.write(line + "\\n")


root.bind("<Button-4>", lambda e: log("wheel-up"))
root.bind("<Button-5>", lambda e: log("wheel-down"))
label = tk.Label(root, text="MMMMMM", font=("DejaVu Sans Mono", 60, "bold"), fg="#000000", bg="#c8c8c8")
if mode == "flat":
    label.place(x=20, y=20)
    root.bind("<ButtonPress-1>", lambda e: log("press"))


def centre(widget):
    return [widget.winfo_rootx() + widget.winfo_width() // 2,
            widget.winfo_rooty() + widget.winfo_height() // 2]


def ready():
    root.update_idletasks()
    data = {"window": [root.winfo_rootx() + root.winfo_width() // 2,
                       root.winfo_rooty() + root.winfo_height() // 2],
            "flat": [root.winfo_rootx() + 400, root.winfo_rooty() + 220],
            "label": centre(label) if mode == "flat" else None}
    with open("/tmp/inp-ready.json.tmp", "w") as handle:
        json.dump(data, handle)
    os.replace("/tmp/inp-ready.json.tmp", "/tmp/inp-ready.json")


root.update()
root.after(1500, ready)
root.mainloop()
'''


def stop_window(desktop) -> None:
    sh(desktop, f"pkill -f '{FIXTURE_PATTERN}'; rm -f {READY_FILE} {EVENT_LOG}")


def start_window(desktop, mode: str) -> dict:
    """Start the fixture window and wait for its ready file (the points to act on, in pixels)."""
    stop_window(desktop)
    time.sleep(0.5)
    written = subprocess.run(
        ["docker", "exec", "-i", desktop.container, "sh", "-c", "cat > /tmp/inp.py"],
        input=WINDOW_SOURCE.encode(), capture_output=True, timeout=30, check=False,
    )
    assert written.returncode == 0, written.stderr.decode(errors="replace")
    started = subprocess.run(
        ["docker", "exec", "-d", "-e", "DISPLAY=:1", desktop.container, "python3", "/tmp/inp.py", mode],
        capture_output=True, timeout=30, check=False,
    )
    assert started.returncode == 0, started.stderr.decode(errors="replace")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        proc = sh(desktop, f"cat {READY_FILE}")
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout)
        time.sleep(0.3)
    raise AssertionError("the fixture window never wrote its ready file (is tkinter in the image?)")


def grid(point: list[int]) -> dict:
    """Pixels to the model's 0-999 grid."""
    return {"x": round(point[0] / SCREEN_W * 999), "y": round(point[1] / SCREEN_H * 999)}


@pytest.fixture
def window(desktop):
    try:
        yield lambda mode: start_window(desktop, mode)
    finally:
        stop_window(desktop)


def events(desktop) -> list[str]:
    return sh(desktop, f"cat {EVENT_LOG} 2>/dev/null").stdout.split()


def test_scroll_at_sends_a_few_wheel_clicks_not_hundreds(desktop, window, tmp_path):
    """Regression for KAN-171: scroll_at sent the pixel amount as wheel clicks, so a ~300 px scroll
    became 300 notches. 300 px must reach the app as 3 wheel-down events."""
    points = window("wheel")
    magnitude = round(300 / SCREEN_H * 999)  # scaled by the 800 px vertical axis: 300 px
    script = [
        call("scroll_at", **grid(points["window"]), direction="down", magnitude=magnitude),
        say("The task is complete."),
    ]
    status, _steps, _client, log = asyncio.run(
        run_script(desktop, tmp_path, "Scroll down in the window", script, grade=lambda: events(desktop))
    )
    assert status is RunStatus.SUCCEEDED
    assert log == ["wheel-down"] * 3


def _drag_script(start: list[int], end: list[int]) -> list[dict]:
    s, e = grid(start), grid(end)
    return [
        call("drag_and_drop", start_x=s["x"], start_y=s["y"], end_x=e["x"], end_y=e["y"]),
        say("The task is complete."),
    ]


def _drag(desktop, tmp_path, start: list[int], end: list[int]) -> list[str]:
    _status, _steps, _client, log = asyncio.run(
        run_script(desktop, tmp_path, "Drag in the window", _drag_script(start, end),
                   grade=lambda: events(desktop))
    )
    return log


def test_drag_from_a_flat_area_is_refused_by_the_visual_servo(desktop, window, tmp_path):
    """Regression for KAN-174: a drag's start point skipped the visual-servo gate, so it pressed on
    empty space. Dragging from the flat area logs no press; a drag from the label logs one."""
    points = window("flat")
    end = [points["flat"][0] - 150, points["flat"][1]]
    assert _drag(desktop, tmp_path / "flat", points["flat"], end) == []
    sh(desktop, f"rm -f {EVENT_LOG}")
    assert _drag(desktop, tmp_path / "label", points["label"], end) == ["press"]

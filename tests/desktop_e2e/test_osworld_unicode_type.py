"""Typing an en dash, curly quotes, accents and a euro sign on an OSWorld machine (KAN-290).

What went wrong in a real run (Nov 25, task 2c1ebcd7): the agent typed "252–272" and the
page got "252272". ``pyautogui.write`` has no key for the en dash on X11 and drops it without
an error, and the old clipboard paste silently fell back to it when the VM had no clipboard tool.

This test uses the free Linux desktop container. A small Tk window with one Entry is started;
pressing Return writes the Entry's text, as UTF-8, to a file. The real ``OSWorldExecutor`` types
into it, with its HTTP transport swapped for ``docker exec`` (a fresh ``python3 -c`` process per
action, like OSWorld's server), and the file must hold the text exactly.

Opt in: ``pytest tests/desktop_e2e -m desktop_container -o addopts=""``. Needs Docker.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time

import pytest

from operon.environments.osworld import OSWorldExecutor
from operon.models.policy import ActionType, AgentAction

from .helpers import sh
from .test_dialog_ok_capture import DockerExecTransport, docker_exec

pytestmark = pytest.mark.desktop_container

READY_FILE = "/tmp/kan290_ready.json"
TYPED_FILE = "/tmp/kan290_typed.txt"
APP_PATTERN = "python3 /tmp/kan290_ap[p].py"  # the [p] keeps pkill from matching itself
TEXT = "Pages 252–272, café “quoted” 5 € naïve"

APP_SOURCE = '''\
import json
import os
import tkinter as tk

root = tk.Tk()
root.title("Operon KAN-290")
root.geometry("600x80+200+150")
root.attributes("-topmost", True)
entry = tk.Entry(root, font=("DejaVu Sans", 20))
entry.pack(fill="both", expand=True)


def on_return(_event):
    with open("/tmp/kan290_typed.txt", "w", encoding="utf-8") as handle:
        handle.write(entry.get())


entry.bind("<Return>", on_return)


def ready():
    root.update_idletasks()
    entry.focus_force()
    data = {"entry": [entry.winfo_rootx() + entry.winfo_width() // 2,
                      entry.winfo_rooty() + entry.winfo_height() // 2]}
    with open("/tmp/kan290_ready.json.tmp", "w") as handle:
        json.dump(data, handle)
    os.replace("/tmp/kan290_ready.json.tmp", "/tmp/kan290_ready.json")


root.update()
root.after(1500, ready)  # after the window manager has placed the window
root.mainloop()
'''


def _stop(container: str) -> None:
    docker_exec(container, ["pkill", "-f", APP_PATTERN], timeout=30)
    docker_exec(container, ["rm", "-f", READY_FILE, TYPED_FILE], timeout=30)


@pytest.fixture
def entry_point(desktop):
    """Start the Tk window and return the centre of its Entry."""
    container = desktop.container
    _stop(container)
    try:
        time.sleep(0.5)
        written = docker_exec(container, ["sh", "-c", "cat > /tmp/kan290_app.py"], stdin=APP_SOURCE, timeout=30)
        assert written.returncode == 0, written.stderr.decode(errors="replace")
        started = subprocess.run(
            ["docker", "exec", "-d", "-e", "DISPLAY=:1", container, "python3", "/tmp/kan290_app.py"],
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
        assert raw, "the Tk window never wrote its ready file (is tkinter available in the image?)"
        x, y = json.loads(raw)["entry"]
        yield int(x), int(y)
    finally:
        _stop(container)


def test_non_ascii_text_arrives_intact(desktop, entry_point, tmp_path):
    x, y = entry_point

    async def scenario():
        executor = OSWorldExecutor("http://osworld.test", artifact_dir=tmp_path,
                                   transport=DockerExecTransport(desktop.container))
        try:
            return await executor.execute(AgentAction(
                action_type=ActionType.TYPE, text=TEXT, x=x, y=y, press_enter=True))
        finally:
            await executor.aclose()

    result = asyncio.run(scenario())
    assert result.success, result.detail
    deadline = time.monotonic() + 10
    typed = ""
    while time.monotonic() < deadline:
        proc = sh(desktop, f"cat {TYPED_FILE}")
        if proc.returncode == 0:
            typed = proc.stdout
            break
        time.sleep(0.3)
    assert typed == TEXT

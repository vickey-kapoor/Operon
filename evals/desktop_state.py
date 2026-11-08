"""Grade the desktop tasks from the machine's state instead of a judge model.

After a desktop run stops, and before its apps are closed, ``probe()`` reads what
is open on the Windows machine the benchmark drives: each window's title, process
and UI Automation names and text (``evals/desktop_probe.ps1``), the folders File
Explorer shows, whether the task's folder exists, and how much has been drawn on
Paint's canvas. Each task in ``evals/desktop_tasks.py`` has a check here that
reads that snapshot and says whether the task is done, and why.

This is grading only. The agent never sees any of it; its perception and policy
stay vision-only.

Windows from before the task started don't count: a window passes only if its
process started after the task did, or belongs to a long-lived host (File
Explorer, UWP apps' ApplicationFrameHost, Windows Terminal) that the benchmark
closes between tasks anyway.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

PROBE_SCRIPT = Path(__file__).with_name("desktop_probe.ps1")
BENCH_FOLDER = "OperonBench"
# Processes that host windows for many apps and outlive any one task.
LONG_LIVED_HOSTS = frozenset({"applicationframehost", "explorer", "windowsterminal"})
# A process that started this long before the task is still counted (clock skew).
START_SLACK = timedelta(seconds=2)

Snapshot = dict
Check = Callable[[Snapshot], tuple[bool, str]]


class ProbeError(RuntimeError):
    """The machine's state could not be read, so the task can't be graded."""


# --- reading the machine ---------------------------------------------------------

def probe(since: datetime) -> Snapshot:
    """Snapshot the desktop. ``since`` is when the task started (UTC)."""
    if sys.platform != "win32":
        raise ProbeError("desktop state checks only run on Windows")
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PROBE_SCRIPT)],
            capture_output=True, text=True, timeout=120, check=True,
        )
        snapshot = json.loads(done.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ProbeError(f"desktop probe failed: {exc}") from exc
    if isinstance(snapshot.get("windows"), dict):  # ConvertTo-Json unwraps a one-item list
        snapshot["windows"] = [snapshot["windows"]]
    snapshot["since"] = since.isoformat()
    documents = Path(snapshot.get("documents") or Path.home() / "Documents")
    snapshot["bench_folder_exists"] = (documents / BENCH_FOLDER).is_dir()
    snapshot["paint_canvas"] = _paint_canvas([w for w in snapshot["windows"] if _proc(w) == "mspaint"])
    return snapshot


def prepare(task_name: str) -> None:
    """Undo what an earlier run of the task left behind, so the check can't pass on it.

    Only removes the benchmark's own folder, and only while it is empty."""
    if task_name != "explorer_new_folder" or sys.platform != "win32":
        return
    folder = Path(os.path.expanduser("~")) / "Documents" / BENCH_FOLDER
    try:
        folder.rmdir()
    except OSError:
        pass


def _paint_canvas(windows: list[dict]) -> dict | None:
    """How much is drawn on Paint's canvas: dark pixels and their bounding box."""
    if not windows:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        import mss

        user32 = ctypes.windll.user32
        found: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd, _):
            name = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, name, 256)
            if name.value == "MSPaintView":
                child = user32.GetWindow(hwnd, 5)  # GW_CHILD: the canvas itself
                if child:
                    found.append(child)
                return False
            return True

        user32.EnumChildWindows(windows[0]["hwnd"], visit, 0)
        if not found:
            return {"found": False}
        rect = wintypes.RECT()
        user32.GetWindowRect(found[0], ctypes.byref(rect))
        # Leave out the canvas edge and its resize handles.
        box = {"left": rect.left + 4, "top": rect.top + 4,
               "width": max(1, rect.right - rect.left - 8), "height": max(1, rect.bottom - rect.top - 8)}
        with mss.mss() as sct:
            shot = sct.grab(box)
        return {"found": True, **canvas_stats(shot.rgb, shot.width, shot.height)}
    except Exception as exc:  # grading must report, not crash the benchmark
        return {"found": False, "error": str(exc)}


def canvas_stats(rgb: bytes, width: int, height: int, threshold: int = 200) -> dict:
    """Count pixels darker than ``threshold`` in any channel and their bounding box."""
    count, xs, ys = 0, [], []
    for i in range(0, min(len(rgb), width * height * 3), 3):
        if min(rgb[i], rgb[i + 1], rgb[i + 2]) < threshold:
            count += 1
            pixel = i // 3
            xs.append(pixel % width)
            ys.append(pixel // width)
    if not count:
        return {"dark_pixels": 0, "bbox": [0, 0]}
    return {"dark_pixels": count, "bbox": [max(xs) - min(xs) + 1, max(ys) - min(ys) + 1]}


# --- reading the snapshot ---------------------------------------------------------

def _proc(window: dict) -> str:
    return str(window.get("process") or "").lower()


def _fresh(window: dict, since: datetime) -> bool:
    if _proc(window) in LONG_LIVED_HOSTS:
        return True
    started = window.get("started")
    if not started:
        return False
    try:
        when = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
    except ValueError:
        return False
    return when >= since - START_SLACK


def windows(snapshot: Snapshot, *processes: str, title: str = "") -> list[dict]:
    """The task's windows: from ``processes`` (or with ``title`` in the title), opened after the task started."""
    since = datetime.fromisoformat(snapshot["since"])
    wanted = {p.lower() for p in processes}
    found = []
    for w in snapshot.get("windows") or []:
        matches = _proc(w) in wanted or (title and title.lower() in str(w.get("title", "")).lower())
        if matches and _fresh(w, since):
            found.append(w)
    return found


def _texts(found: list[dict]) -> list[str]:
    return [str(t) for w in found for t in (w.get("texts") or [])]


def _names(found: list[dict]) -> list[str]:
    return [str(n) for w in found for n in (w.get("names") or [])]


def _selected(found: list[dict]) -> list[str]:
    return [str(n) for w in found for n in (w.get("selected") or [])]


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]


def _path(value: str) -> str:
    return str(value).replace("/", "\\").rstrip("\\").casefold()


# --- the checks -------------------------------------------------------------------

def _notepad_text(want: Callable[[str], bool], what: str) -> Check:
    def check(snapshot: Snapshot) -> tuple[bool, str]:
        found = windows(snapshot, "notepad")
        if not found:
            return False, "no Notepad window opened during the task"
        if any(want(text) for text in _texts(found)):
            return True, f"a Notepad window contains {what}"
        return False, f"Notepad is open but its text lacks {what}"
    return check


def _two_lines(text: str) -> bool:
    lines = [line.casefold() for line in _lines(text)]
    return any(a == "first line" and b == "second line" for a, b in zip(lines, lines[1:]))


def _explorer_at(snapshot: Snapshot, folder: str) -> bool:
    return any(_path(p) == _path(folder) for p in snapshot.get("explorer") or [])


def explorer_windows(snapshot: Snapshot) -> tuple[bool, str]:
    if _explorer_at(snapshot, r"C:\Windows"):
        return True, r"a File Explorer window shows C:\Windows"
    return False, f"no File Explorer window at C:\\Windows (open: {snapshot.get('explorer') or 'none'})"


def explorer_new_folder(snapshot: Snapshot) -> tuple[bool, str]:
    if not snapshot.get("bench_folder_exists"):
        return False, f"no {BENCH_FOLDER} folder in Documents"
    documents = snapshot.get("documents") or ""
    if _explorer_at(snapshot, documents) or _explorer_at(snapshot, f"{documents}\\{BENCH_FOLDER}"):
        return True, f"{BENCH_FOLDER} exists and File Explorer shows Documents"
    return False, f"{BENCH_FOLDER} exists but no File Explorer window shows Documents"


def _console(snapshot: Snapshot, shell: str, title: str) -> list[str]:
    found = windows(snapshot, shell, "conhost", "openconsole", "windowsterminal", title=title)
    return _texts([w for w in found if title.lower() in str(w.get("title", "")).lower()
                   or _proc(w) == shell])


def cmd_echo(snapshot: Snapshot) -> tuple[bool, str]:
    texts = _console(snapshot, "cmd", "Command Prompt") + _console(snapshot, "cmd", "cmd.exe")
    if not texts:
        return False, "no Command Prompt window opened during the task"
    for text in texts:
        lines = [line.casefold() for line in _lines(text)]
        ran = any(line.endswith("echo operon-bench-ok") for line in lines)
        if ran and "operon-bench-ok" in lines:
            return True, "Command Prompt shows the echo command and its output"
    return False, "Command Prompt is open but doesn't show 'echo operon-bench-ok' and its output"


_DATE = re.compile(r"\b(19|20)\d{2}\b.*\b\d{1,2}:\d{2}(:\d{2})?\b|\b\d{1,2}:\d{2}(:\d{2})?\b.*\b(19|20)\d{2}\b")


def powershell_date(snapshot: Snapshot) -> tuple[bool, str]:
    texts = _console(snapshot, "powershell", "PowerShell") + _console(snapshot, "pwsh", "PowerShell")
    if not texts:
        return False, "no PowerShell window opened during the task"
    for text in texts:
        lines = _lines(text)
        for i, line in enumerate(lines):
            if line.casefold().endswith("get-date") and any(_DATE.search(after) for after in lines[i + 1:]):
                return True, "PowerShell shows Get-Date and a date and time below it"
    return False, "PowerShell is open but doesn't show Get-Date with a date and time below it"


def settings_display(snapshot: Snapshot) -> tuple[bool, str]:
    found = windows(snapshot, "systemsettings", title="Settings")
    if not found:
        return False, "no Settings window opened during the task"
    names = [n.casefold() for n in _names(found)]
    on_display = "display" in [n.casefold() for n in _selected(found)] or "display" in names
    if on_display and any("scale" in n or "resolution" in n for n in names):
        return True, "Settings shows the Display page (scale or resolution options listed)"
    return False, "Settings is open but not on the Display page"


def taskmgr_performance(snapshot: Snapshot) -> tuple[bool, str]:
    found = windows(snapshot, "taskmgr")
    if not found:
        return False, "no Task Manager window opened during the task"
    if "performance" in [n.casefold() for n in _selected(found)]:
        return True, "Task Manager's Performance tab is selected"
    names = [n.casefold() for n in _names(found)]
    if "performance" in names and any("utilization" in n for n in names):
        return True, "Task Manager shows the Performance view (utilization graphs)"
    return False, "Task Manager is open but not on the Performance tab"


def calculator_multiply(snapshot: Snapshot) -> tuple[bool, str]:
    found = windows(snapshot, "calculatorapp", "calc", "win32calc", title="Calculator")
    if not found:
        return False, "no Calculator window opened during the task"
    shown = [re.sub(r"[\s,]", "", n.casefold()) for n in _names(found) + _texts(found)]
    if any(n in {"444", "displayis444"} or n.endswith("displayis444") for n in shown):
        return True, "Calculator shows 444"
    return False, "Calculator is open but doesn't show 444"


def paint_rectangle(snapshot: Snapshot) -> tuple[bool, str]:
    if not windows(snapshot, "mspaint"):
        return False, "no Paint window opened during the task"
    canvas = snapshot.get("paint_canvas") or {}
    if not canvas.get("found"):
        return False, f"Paint's canvas could not be read{': ' + canvas['error'] if canvas.get('error') else ''}"
    width, height = canvas.get("bbox") or [0, 0]
    if canvas.get("dark_pixels", 0) >= 40 and width >= 10 and height >= 10:
        return True, f"Paint's canvas has a drawn shape ({width}x{height} px)"
    return False, "Paint's canvas is blank (no shape at least 10x10 px)"


CHECKS: dict[str, Check] = {
    "notepad_type": _notepad_text(lambda t: "hello from operon" in t.casefold(), "'Hello from Operon'"),
    "notepad_two_lines": _notepad_text(_two_lines, "'first line' then 'second line'"),
    "explorer_windows": explorer_windows,
    "explorer_new_folder": explorer_new_folder,
    "cmd_echo": cmd_echo,
    "powershell_date": powershell_date,
    "settings_display": settings_display,
    "taskmgr_performance": taskmgr_performance,
    "calculator_multiply": calculator_multiply,
    "paint_rectangle": paint_rectangle,
}


def grade(task_name: str, snapshot: Snapshot) -> tuple[bool, str]:
    return CHECKS[task_name](snapshot)

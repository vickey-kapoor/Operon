"""Grade the desktop tasks from the machine's state instead of a judge model.

After a desktop run stops, and before its apps are closed, ``probe()`` reads what
is open on the Windows machine the benchmark drives: each window's title, process
and UI Automation names and text (``evals/desktop_probe.ps1``), the folders File
Explorer shows, whether the task's folder exists, and how much has been drawn on
Paint's canvas. Each task in ``evals/desktop_tasks.py`` has a check here that
reads that snapshot and says whether the task is done, and why.

This is grading only. The agent never sees any of it; its perception and policy
stay vision-only.

Windows from before the task don't count. Before each task ``prepare()`` stops the
task's app if it is already running, closes File Explorer's folder windows for the
Explorer tasks, and records which windows are still open; only windows outside that
baseline are graded, so a console or app left open earlier can't pass a task. (When
no baseline is given, a window counts if its process started after the task did, or
belongs to a long-lived host such as File Explorer or ApplicationFrameHost.)
"""

from __future__ import annotations

import json
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

_CONSOLE_HOSTS = ("conhost", "openconsole", "windowsterminal")
# Per task: the processes whose windows the probe reads in depth, and the app
# processes stopped before the task so an earlier attempt's window can't be reused.
# Consoles are never stopped (the person running the benchmark may be in one);
# the baseline below keeps their windows out of the grade instead.
INSPECT: dict[str, tuple[str, ...]] = {
    "notepad_type": ("notepad",),
    "notepad_two_lines": ("notepad",),
    "explorer_windows": (),
    "explorer_new_folder": (),
    "cmd_echo": ("cmd", *_CONSOLE_HOSTS),
    "powershell_date": ("powershell", "pwsh", *_CONSOLE_HOSTS),
    "settings_display": ("systemsettings", "applicationframehost"),
    "taskmgr_performance": ("taskmgr",),
    "calculator_multiply": ("calculatorapp", "calc", "win32calc", "applicationframehost"),
    "paint_rectangle": ("mspaint",),
}
CLOSE_FIRST: dict[str, tuple[str, ...]] = {
    "notepad_type": ("notepad",),
    "notepad_two_lines": ("notepad",),
    "settings_display": ("systemsettings",),
    "taskmgr_performance": ("taskmgr",),
    "calculator_multiply": ("calculatorapp", "calc", "win32calc"),
    "paint_rectangle": ("mspaint",),
}
EXPLORER_TASKS = frozenset({"explorer_windows", "explorer_new_folder"})


def _run_probe(*args: str, timeout: float = 120) -> Snapshot:
    if sys.platform != "win32":
        raise ProbeError("desktop state checks only run on Windows")
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PROBE_SCRIPT), *args],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        snapshot = json.loads(done.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ProbeError(f"desktop probe failed: {exc}") from exc
    return _normalize(snapshot)


def _as_list(value) -> list:
    """ConvertTo-Json writes a one-item list as the bare item, and an empty one as null."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _normalize(snapshot: Snapshot) -> Snapshot:
    snapshot["windows"] = _as_list(snapshot.get("windows"))
    for w in snapshot["windows"]:
        for key in ("names", "selected", "texts"):
            w[key] = _as_list(w.get(key))
    snapshot["explorer"] = [e if isinstance(e, dict) else {"path": e, "hwnd": None}
                            for e in _as_list(snapshot.get("explorer"))]
    return snapshot


def prepare(task_name: str) -> list[int]:
    """Reset the desktop for a task and return the windows already open (their handles).

    Stops the task's app if it is running (an earlier attempt's window would otherwise
    still count) and, for the Explorer tasks, closes File Explorer's folder windows.
    explorer_new_folder also removes the benchmark's own folder, only while it is empty,
    from the same Documents folder the probe checks (it may be redirected, e.g. to
    OneDrive). Windows in the returned baseline never count towards the grade, so a
    console or app left open by anyone before the task can't pass it."""
    args = ["-Close", ",".join(CLOSE_FIRST.get(task_name, ()))]
    if task_name in EXPLORER_TASKS:
        args.append("-CloseExplorer")
    try:
        snapshot = _run_probe(*args, timeout=60)
    except ProbeError as exc:
        raise ProbeError(f"desktop reset failed: {exc}") from exc
    if task_name == "explorer_new_folder":
        documents = snapshot.get("documents")
        if not documents:
            raise ProbeError("desktop reset failed: Windows gave no Documents folder")
        try:
            (Path(documents) / BENCH_FOLDER).rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:  # not empty: an earlier attempt or a person put files there
            raise ProbeError(f"{BENCH_FOLDER} in Documents is not empty, so the task can't be graded: {exc}") from exc
    return sorted({int(w["hwnd"]) for w in snapshot["windows"] if w.get("hwnd")}
                  | {int(e["hwnd"]) for e in snapshot["explorer"] if e.get("hwnd")})


def probe(since: datetime, task_name: str, baseline: list[int] | None = None) -> Snapshot:
    """Snapshot the desktop after a task. ``since`` is when it started (UTC) and
    ``baseline`` the windows open before it (from ``prepare``)."""
    snapshot = _run_probe("-Inspect", ",".join(INSPECT.get(task_name, ())))
    snapshot["since"] = since.isoformat()
    snapshot["baseline"] = list(baseline) if baseline is not None else None
    if not snapshot.get("documents"):
        raise ProbeError("desktop probe failed: Windows gave no Documents folder")
    snapshot["bench_folder_exists"] = (Path(snapshot["documents"]) / BENCH_FOLDER).is_dir()
    snapshot["paint_canvas"] = (_paint_canvas(windows(snapshot, "mspaint"))
                                if task_name == "paint_rectangle" else None)
    if "notepad" in INSPECT.get(task_name, ()):
        # Notepad's text control doesn't always answer UI Automation; read it directly.
        for w in windows(snapshot, "notepad"):
            text = _edit_text(w.get("hwnd"))
            if text:
                w["texts"].append(text)
    return snapshot


def _edit_text(hwnd) -> str:
    """The text of the first Edit or RichEdit child of ``hwnd`` (WM_GETTEXT)."""
    if not hwnd:
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def visit(child, _):
            name = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(child, name, 256)
            if name.value == "Edit" or name.value.startswith("RichEdit"):
                found.append(child)
                return False
            return True

        user32.EnumChildWindows(int(hwnd), visit, 0)
        if not found:
            return ""
        length = user32.SendMessageW(found[0], 0x000E, 0, 0)  # WM_GETTEXTLENGTH
        buffer = ctypes.create_unicode_buffer(min(int(length), 200_000) + 1)
        user32.SendMessageW(found[0], 0x000D, len(buffer), buffer)  # WM_GETTEXT
        return buffer.value
    except Exception:  # the UI Automation read may still have the text
        return ""


def _paint_canvas(found: list[dict]) -> dict | None:
    """How much is drawn on Paint's canvas: dark pixels and their bounding box."""
    if not found:
        return None
    try:
        import ctypes
        import time
        from ctypes import wintypes

        import mss

        user32 = ctypes.windll.user32
        # Window rectangles must be in physical pixels, the same ones mss grabs.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            user32.SetProcessDPIAware()
        top = found[-1]["hwnd"]
        # Bring Paint to the front so nothing covers the canvas when it is grabbed.
        user32.ShowWindow(top, 9)  # SW_RESTORE
        user32.SetForegroundWindow(top)
        time.sleep(0.5)
        canvas: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd, _):
            name = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, name, 256)
            if name.value == "MSPaintView":
                child = user32.GetWindow(hwnd, 5)  # GW_CHILD: the canvas itself
                if child:
                    canvas.append(child)
                return False
            return True

        user32.EnumChildWindows(top, visit, 0)
        if not canvas:
            return {"found": False, "error": "Paint's canvas (MSPaintView) was not found"}
        rect = wintypes.RECT()
        user32.GetWindowRect(canvas[0], ctypes.byref(rect))
        # Leave out the canvas edge and its resize handles.
        box = {"left": rect.left + 4, "top": rect.top + 4,
               "width": max(1, rect.right - rect.left - 8), "height": max(1, rect.bottom - rect.top - 8)}
        with mss.mss() as sct:
            shot = sct.grab(box)
        return {"found": True, **canvas_stats(shot.rgb, shot.width, shot.height)}
    except Exception as exc:  # grading must report, not crash the benchmark
        return {"found": False, "error": str(exc)}


def canvas_stats(rgb: bytes, width: int, height: int, threshold: int = 200, white: int = 250) -> dict:
    """Measure what is drawn on Paint's canvas.

    The grab covers Paint's whole view, which is the white canvas on a gray backdrop
    with resize handles and a shadow. The canvas is taken to be the box around the
    white pixels; inside it, pixels darker than ``threshold`` in any channel count as
    drawing, and their bounding box is returned."""
    n = min(len(rgb) // 3, width * height)
    lows = [min(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]) for i in range(n)]
    whites = [i for i, low in enumerate(lows) if low >= white]
    if whites:
        wx = [i % width for i in whites]
        left, right = min(wx) + 1, max(wx) - 1
        top, bottom = whites[0] // width + 1, whites[-1] // width - 1
    else:  # the canvas is fully painted over
        left, right, top, bottom = 0, width - 1, 0, height - 1
    count, xs, ys = 0, [], []
    for y in range(max(top, 0), min(bottom, height - 1) + 1):
        row = y * width
        for x in range(max(left, 0), min(right, width - 1) + 1):
            if row + x < n and lows[row + x] < threshold:
                count += 1
                xs.append(x)
                ys.append(y)
    canvas = [max(0, right - left + 1), max(0, bottom - top + 1)]
    if not count:
        return {"dark_pixels": 0, "bbox": [0, 0], "canvas": canvas}
    return {"dark_pixels": count, "bbox": [max(xs) - min(xs) + 1, max(ys) - min(ys) + 1], "canvas": canvas}


# --- reading the snapshot ---------------------------------------------------------

def _proc(window: dict) -> str:
    return str(window.get("process") or "").lower()


def _fresh(window: dict, since: datetime, baseline: list[int] | None) -> bool:
    if baseline is not None:  # a window counts only if it wasn't open before the task
        return window.get("hwnd") is not None and int(window["hwnd"]) not in set(baseline)
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
    baseline = snapshot.get("baseline")
    wanted = {p.lower() for p in processes}
    found = []
    for w in snapshot.get("windows") or []:
        matches = _proc(w) in wanted or (title and title.lower() in str(w.get("title", "")).lower())
        if matches and not w.get("offscreen") and _fresh(w, since, baseline):
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
    baseline = snapshot.get("baseline")
    for entry in snapshot.get("explorer") or []:
        if not isinstance(entry, dict):
            entry = {"path": entry, "hwnd": None}
        if baseline is not None and entry.get("hwnd") is not None and int(entry["hwnd"]) in set(baseline):
            continue  # open before the task
        if _path(entry.get("path") or "") == _path(folder):
            return True
    return False


def _explorer_paths(snapshot: Snapshot) -> list[str]:
    return [e.get("path") if isinstance(e, dict) else e for e in snapshot.get("explorer") or []]


def explorer_windows(snapshot: Snapshot) -> tuple[bool, str]:
    if _explorer_at(snapshot, r"C:\Windows"):
        return True, r"a File Explorer window shows C:\Windows"
    return False, f"no File Explorer window at C:\\Windows (open: {_explorer_paths(snapshot) or 'none'})"


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
    texts = list(dict.fromkeys(_console(snapshot, "cmd", "Command Prompt") + _console(snapshot, "cmd", "cmd.exe")))
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
    # The Display page is open when its nav item is selected (Windows 10 / Server) or
    # the page header reads "System > Display" (Windows 11). A "Display" search hit
    # or nav entry alone is not enough.
    on_display = ("display" in [n.casefold() for n in _selected(found)]
                  or any(re.fullmatch(r"system\s*[>›]\s*display", n) for n in names))
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
        # The grader failed to read the drawing; that says nothing about the model.
        raise ProbeError(f"Paint's canvas could not be read: {canvas.get('error') or 'no canvas in the snapshot'}")
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

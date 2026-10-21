"""Real desktop tasks for the live benchmark (``python -m tests.eval.live_desktop_bench``).

Each task starts from whatever is on screen (the desktop loop minimizes windows
first) and names what the final screen must show (``check``). A judge model
reads the run's final screenshot against it, as in the browser benchmark.

Tasks use apps that ship with Windows, so they also run on a GitHub-hosted
``windows-latest`` runner. Nothing here deletes or sends anything; the one task
that writes to disk creates a folder in the runner's Documents folder.

To add a task: write the intent the way a user would, and make ``check``
something a person could confirm from one screenshot.
"""

from __future__ import annotations

from tests.eval.live_tasks import SMOKE, LiveTask

DESKTOP_TASKS: list[LiveTask] = [
    # --- text ----------------------------------------------------------------
    LiveTask(
        "notepad_type", "text",
        "Open Notepad and type: Hello from Operon",
        None,
        "A Notepad window is open and its text area contains 'Hello from Operon'.",
        tags=SMOKE,
    ),
    LiveTask(
        "notepad_two_lines", "text",
        "Open Notepad and write two lines: 'first line' and then 'second line' below it",
        None,
        "A Notepad window shows 'first line' and, on the next line, 'second line'.",
    ),
    # --- files ---------------------------------------------------------------
    LiveTask(
        "explorer_windows", "files",
        "Open File Explorer and go to C:\\Windows",
        None,
        "A File Explorer window shows the contents of C:\\Windows (folders such as System32 are listed).",
        tags=SMOKE,
    ),
    LiveTask(
        "explorer_new_folder", "files",
        "Open File Explorer, go to the Documents folder and create a new folder named OperonBench",
        None,
        "A File Explorer window shows the Documents folder with a folder named 'OperonBench' in it.",
    ),
    # --- shell ---------------------------------------------------------------
    LiveTask(
        "cmd_echo", "shell",
        "Open Command Prompt and run: echo operon-bench-ok",
        None,
        "A Command Prompt window shows the command and the output line 'operon-bench-ok' below it.",
        tags=SMOKE,
    ),
    LiveTask(
        "powershell_date", "shell",
        "Open PowerShell and run Get-Date",
        None,
        "A PowerShell window shows 'Get-Date' and, below it, a date and time printed as its output.",
    ),
    # --- system --------------------------------------------------------------
    LiveTask(
        "settings_display", "system",
        "Open Settings and go to the Display settings page",
        None,
        "The Settings app is open on the Display page (display, scale or resolution options are showing).",
    ),
    LiveTask(
        "taskmgr_performance", "system",
        "Open Task Manager and show the Performance tab",
        None,
        "Task Manager is open on its Performance view, with CPU or memory usage graphs showing.",
    ),
    # --- apps ----------------------------------------------------------------
    LiveTask(
        "calculator_multiply", "apps",
        "Open Calculator and compute 37 times 12",
        None,
        "A Calculator window shows the result 444.",
    ),
    LiveTask(
        "paint_rectangle", "apps",
        "Open Paint and draw a rectangle on the canvas",
        None,
        "A Paint window is open and a rectangle is drawn on its canvas.",
    ),
]

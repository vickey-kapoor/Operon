"""Real desktop tasks for the live benchmark (``python -m evals.live_desktop_bench``).

Each task starts from whatever is on screen (the desktop loop minimizes windows
first) and names what the final screen must show (``check``). It is graded from
the machine's state against that, not by a judge model: each task has a check in
``evals/desktop_state.py`` that reads the open windows, their text and files.

Tasks use apps that ship with Windows, so they also run on a GitHub-hosted
``windows-latest`` runner. Nothing here deletes or sends anything; the one task
that writes to disk creates a folder in the runner's Documents folder.

To add a task: write the intent the way a user would, make ``check`` something a
person could confirm from one screenshot, and add its state check to
``desktop_state.CHECKS``.
"""

from __future__ import annotations

from evals.live_tasks import SMOKE, LiveTask

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
    # --- saved files (Season 1, KAN-65): several steps or two apps, graded from the file
    LiveTask(
        "notepad_save_file", "files",
        "Open Notepad, type 'Operon season one', and save it as season.txt in a folder named OperonFiles "
        "inside your Documents folder (create the folder if it isn't there)",
        None,
        "Documents\\OperonFiles\\season.txt exists and contains 'Operon season one'.",
        max_steps=40,
    ),
    LiveTask(
        "cmd_list_to_file", "shell",
        "Use Command Prompt to save the list of files in C:\\Windows\\System32\\drivers\\etc into a file "
        "named etc-list.txt in Documents\\OperonFiles (create the folder if it isn't there)",
        None,
        "Documents\\OperonFiles\\etc-list.txt lists the files in drivers\\etc, such as hosts and services.",
        max_steps=40,
    ),
    LiveTask(
        "powershell_computer_name", "shell",
        "Use PowerShell to write this computer's name into a file named computer.txt in Documents\\OperonFiles "
        "(create the folder if it isn't there)",
        None,
        "Documents\\OperonFiles\\computer.txt contains the computer's name.",
        max_steps=40,
    ),
    LiveTask(
        "calculator_to_notepad", "apps",
        "Use Calculator to work out 1234 times 56, then type the result into Notepad and save it as result.txt "
        "in Documents\\OperonFiles (create the folder if it isn't there)",
        None,
        "Documents\\OperonFiles\\result.txt contains 69104, the result from Calculator.",
        max_steps=50,
    ),
]

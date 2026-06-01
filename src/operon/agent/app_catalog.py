"""Canonical desktop-application launch catalog.

Single source of truth mapping a spoken application alias (as a user might
phrase it — "google chrome", "vs code") to the command used to launch it.

This module deliberately imports nothing from the policy or executor layers so
both can depend on it without a circular import:

* the policy rule (:mod:`operon.agent.policy.rules`) uses the *keys* to
  recognise "open/launch <app>" intents, and
* the desktop executor (:mod:`operon.executor.desktop`) uses ``.get()`` to
  resolve the alias to an executable at launch time.

Previously each layer kept its own copy, which drifted: the policy map knew
about Office apps the executor could not launch, and the two disagreed on
``.exe`` suffixes.  Keep this the only place app aliases live.
"""

from __future__ import annotations

# Spoken alias / canonical token -> launch command.
APP_LAUNCH_TARGETS: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "vs code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "code": "code",
    "word": "winword.exe",
    "winword": "winword.exe",
    "excel": "excel.exe",
    "powerpoint": "powerpnt.exe",
    "powerpnt": "powerpnt.exe",
    "chrome": "chrome",
    "google chrome": "chrome",
    "browser": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "msedge": "msedge",
    "terminal": "wt.exe",
    "windows terminal": "wt.exe",
    "wt": "wt.exe",
    "powershell": "powershell.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "task manager": "taskmgr.exe",
    "taskmgr": "taskmgr.exe",
    "settings": "ms-settings:",
}

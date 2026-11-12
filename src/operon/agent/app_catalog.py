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


# Linux desktop (KAN-108: the Azure Linux VM, an XFCE session on Debian). Same
# spoken aliases where the Linux machine has a matching app; Office and the
# Windows-only shells have none, so they are left out (and the model is never
# offered them). Each value is a fixed argv string, split on spaces, never a shell line.
LINUX_APP_LAUNCH_TARGETS: dict[str, str] = {
    "notepad": "mousepad",
    "text editor": "mousepad",
    "mousepad": "mousepad",
    "calculator": "galculator",
    "calc": "galculator",
    "file explorer": "thunar",
    "files": "thunar",
    "explorer": "thunar",
    "file manager": "thunar",
    "terminal": "xfce4-terminal",
    "chrome": "chromium",
    "google chrome": "chromium",
    "chromium": "chromium",
    "browser": "chromium",
    "firefox": "firefox-esr",
    "spreadsheet": "libreoffice --calc",
    "excel": "libreoffice --calc",
    "word": "libreoffice --writer",
    "document": "libreoffice --writer",
    "paint": "ristretto",
    "settings": "xfce4-settings-manager",
}

WINDOWS = "windows"
LINUX = "linux"


def desktop_platform() -> str:
    """``OPERON_DESKTOP_PLATFORM``: ``linux`` drives a Linux (X11) desktop; anything
    else, the default, is the Windows desktop."""
    import os

    return LINUX if (os.getenv("OPERON_DESKTOP_PLATFORM") or "").strip().lower() == LINUX else WINDOWS


def launch_targets() -> dict[str, str]:
    """The launch catalog for the configured desktop platform."""
    return LINUX_APP_LAUNCH_TARGETS if desktop_platform() == LINUX else APP_LAUNCH_TARGETS

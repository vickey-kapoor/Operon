"""Operon's tool action channel (RFC 0002, KAN-79): tools the model may call next to its
clicks and key presses.

On by default since 2026-11-14 (KAN-87: Desktop Season 2 scored 10/10 with ``on``
against 8/10 off). ``OPERON_TOOLS`` sets it:

- ``off``: no tools.
- ``on`` (default): ``read_file``, ``list_dir``, ``write_file`` and ``http_get``.
- ``all``: those and ``run_shell``.
- a comma list of tool names, e.g. ``read_file,list_dir``.

Perception does not change: the model still sees the screen only as pixels. A tool's
output is data the model asked for, sent back as the answer to its call. Each tool has a
short skill note in ``prompts/tools/<tool>.md`` that tells the model when to use it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from operon.core.paths import prompts_dir

logger = logging.getLogger(__name__)

READ_FILE = "read_file"
LIST_DIR = "list_dir"
WRITE_FILE = "write_file"
RUN_SHELL = "run_shell"
HTTP_GET = "http_get"

TOOL_NAMES: tuple[str, ...] = (READ_FILE, LIST_DIR, WRITE_FILE, RUN_SHELL, HTTP_GET)
# ``on`` leaves the shell out: it is the one tool no path or domain check can judge.
DEFAULT_TOOLS: tuple[str, ...] = (READ_FILE, LIST_DIR, WRITE_FILE, HTTP_GET)
# Tools whose ``path`` argument the file guard checks.
FILE_TOOLS: frozenset[str] = frozenset({READ_FILE, LIST_DIR, WRITE_FILE})

_OFF = {"", "off", "false", "0", "no"}
_ON = {"on", "true", "1", "yes"}


def enabled_tools(setting: str | None = None) -> tuple[str, ...]:
    """The tools ``OPERON_TOOLS`` (or ``setting``) turns on, in a fixed order; empty when off."""
    raw = (os.getenv("OPERON_TOOLS", "on") if setting is None else setting).strip().lower()
    if raw in _OFF:
        return ()
    if raw in _ON:
        return DEFAULT_TOOLS
    if raw == "all":
        return TOOL_NAMES
    names = {name.strip() for name in raw.split(",") if name.strip()}
    unknown = sorted(names - set(TOOL_NAMES))
    if unknown:
        logger.warning("OPERON_TOOLS names unknown tools %s; known: %s", unknown, ", ".join(TOOL_NAMES))
    return tuple(name for name in TOOL_NAMES if name in names)


def _string(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


_DECLARATIONS: dict[str, dict[str, Any]] = {
    READ_FILE: {
        "name": READ_FILE,
        "description": "Read a text file and get its contents back (the first 20,000 characters).",
        "parameters": {
            "type": "object",
            "properties": {"path": _string("File path, absolute or relative to the user's home folder.")},
            "required": ["path"],
        },
    },
    LIST_DIR: {
        "name": LIST_DIR,
        "description": "List a folder: one line per entry with its name and size, or 'dir' for a folder.",
        "parameters": {
            "type": "object",
            "properties": {"path": _string("Folder path, absolute or relative to the user's home folder.")},
            "required": ["path"],
        },
    },
    WRITE_FILE: {
        "name": WRITE_FILE,
        "description": "Write text to a file, replacing it if it exists and creating missing folders.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": _string("File path, absolute or relative to the user's home folder."),
                "content": _string("The whole text of the file."),
            },
            "required": ["path", "content"],
        },
    },
    RUN_SHELL: {
        "name": RUN_SHELL,
        "description": (
            "Run one shell command (PowerShell on Windows, bash on Linux) in the user's home folder "
            "and get its exit code and output back."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": _string("The command or short script to run."),
                "timeout_seconds": {"type": "integer", "description": "Stop it after this many seconds (default 30, at most 120)."},
            },
            "required": ["command"],
        },
    },
    HTTP_GET: {
        "name": HTTP_GET,
        "description": "Fetch a URL with an HTTP GET and get the status and body text back (the first 20,000 characters).",
        "parameters": {
            "type": "object",
            "properties": {"url": _string("An http or https URL.")},
            "required": ["url"],
        },
    },
}


def function_declarations(names: tuple[str, ...]) -> list[dict[str, Any]]:
    """The Gemini function declarations for ``names``."""
    return [_DECLARATIONS[name] for name in names]


def skill_text(names: tuple[str, ...]) -> str:
    """The skill notes for ``names``, as a block to add to the prompt ('' for none)."""
    if not names:
        return ""
    notes = [(prompts_dir() / "tools" / f"{name}.md").read_text(encoding="utf-8").strip() for name in names]
    return (
        "\n\nTools:\nBesides acting on the screen you can call these tools. The screen stays the "
        "ground truth: check it after a tool changed something the user would see.\n\n"
        + "\n\n".join(notes)
        + "\n"
    )


def tool_path(raw: str) -> Path:
    """A tool's path argument as a full path: relative paths are under the home folder."""
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = Path.home() / path
    return path.resolve()


def summary(name: str, args: dict[str, Any]) -> str:
    """One line naming the call, which the gates judge and the task page shows. A
    file's content is left out, so text inside it can't match a trust-gate phrase."""
    if name == RUN_SHELL:
        detail = args.get("command")
    elif name == HTTP_GET:
        detail = args.get("url")
    else:
        detail = args.get("path")
    detail = " ".join(str(detail or "").split())
    return f"{name}: {detail}" if detail else name

"""Key names for Playwright, which only accepts its own spelling ("Control", "Enter")."""

from __future__ import annotations

import re

# Common spellings a model or the live view may send, mapped to Playwright's names.
_PLAYWRIGHT_KEY_MAP: dict[str, str] = {
    "escape": "Escape",
    "esc": "Escape",
    "enter": "Enter",
    "return": "Enter",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "tab": "Tab",
    "space": "Space",
    "arrowup": "ArrowUp",
    "up": "ArrowUp",
    "arrowdown": "ArrowDown",
    "down": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "left": "ArrowLeft",
    "arrowright": "ArrowRight",
    "right": "ArrowRight",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
    "ctrl": "Control",
    "control": "Control",
    "alt": "Alt",
    "option": "Alt",
    "shift": "Shift",
    "meta": "Meta",
    "cmd": "Meta",
    "command": "Meta",
    "win": "Meta",
    "windows": "Meta",
    "super": "Meta",
    "capslock": "CapsLock",
}

_FUNCTION_KEY = re.compile(r"f([1-9]|1[0-9]|2[0-4])")


def normalize_key(key: str) -> str:
    """One key or a ``+``-joined combination, in Playwright's spelling.

    Unknown names pass through unchanged, so a single character ("a", "+") or a name
    already in Playwright's form keeps working."""
    if "+" in key.strip("+") and len(key) > 1:
        return "+".join(_normalize_one(part.strip()) for part in key.split("+"))
    return _normalize_one(key.strip() or key)


def _normalize_one(key: str) -> str:
    lowered = key.lower()
    if lowered in _PLAYWRIGHT_KEY_MAP:
        return _PLAYWRIGHT_KEY_MAP[lowered]
    if _FUNCTION_KEY.fullmatch(lowered):
        return lowered.upper()
    return key

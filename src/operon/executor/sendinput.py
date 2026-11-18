"""Type text on Windows with ``SendInput`` and ``KEYEVENTF_UNICODE`` (KAN-103).

Used for secrets: the value goes straight to the focused window as key events and never
touches the clipboard (no Win+V history, no cloud clipboard, nothing for READ_CLIPBOARD to
read back). Each UTF-16 code unit is sent as its own key down and key up, so characters
outside the BMP go as their surrogate pair, which is how Windows expects them.
"""

from __future__ import annotations

import ctypes
import struct

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
_INPUT_KEYBOARD = 1
_BATCH = 200  # events per SendInput call


class _KeybdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputUnion(ctypes.Union):
    # MouseInput is the largest member; it sets the union's (and so INPUT's) size.
    _fields_ = [("ki", _KeybdInput), ("mi", _MouseInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("u", _InputUnion)]


def utf16_units(text: str) -> list[int]:
    """The UTF-16 code units of ``text``; a character above U+FFFF is two (a surrogate pair)."""
    raw = text.encode("utf-16-le", errors="surrogatepass")
    return list(struct.unpack(f"<{len(raw) // 2}H", raw))


def _events(units: list[int]) -> list[Input]:
    events: list[Input] = []
    for unit in units:
        for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            item = Input(type=_INPUT_KEYBOARD)
            item.u.ki = _KeybdInput(wVk=0, wScan=unit, dwFlags=flags, time=0, dwExtraInfo=0)
            events.append(item)
    return events


def _user32():
    return ctypes.windll.user32  # type: ignore[attr-defined]


def _send(events: list[Input]) -> None:
    user32 = _user32()
    for start in range(0, len(events), _BATCH):
        chunk = events[start : start + _BATCH]
        array = (Input * len(chunk))(*chunk)
        sent = user32.SendInput(len(chunk), array, ctypes.sizeof(Input))
        if sent != len(chunk):
            raise OSError("Windows refused part of the keyboard input (it can also drop input silently for an elevated window, so check the field)")


def type_unicode(text: str) -> None:
    """Type ``text`` into the focused window. Blocking: call it from a worker thread."""
    _send(_events(utf16_units(text)))

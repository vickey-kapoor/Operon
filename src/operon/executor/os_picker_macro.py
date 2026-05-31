"""Deterministic OS file-picker macro.

This is the executor-layer primitive invoked by ``NativeBrowserExecutor`` after
it clicks an upload control that opens a real OS file dialog (headed mode
only). It is intentionally thin:

1. Poll for a native picker window using ``pygetwindow`` keyword matching.
2. Type the provided absolute file path with ``pyautogui.write``.
3. Press Enter to confirm.
4. Poll for the picker window to close.

The macro does *no* LLM calls, no retries, and no higher-level reasoning. It
returns a :class:`PickerMacroResult` so the caller can translate failures into
the standard executor failure categories (``PICKER_NOT_DETECTED`` /
``FILE_NOT_REFLECTED`` / ``EXECUTION_ERROR``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

try:
    import pyautogui  # type: ignore[import-not-found]
    _PYAUTOGUI_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - headless CI path
    _PYAUTOGUI_IMPORT_ERROR = exc

    def _unavailable(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError("pyautogui is unavailable in this environment") from _PYAUTOGUI_IMPORT_ERROR

    pyautogui = SimpleNamespace(  # type: ignore[assignment]
        write=_unavailable,
        press=_unavailable,
    )

try:
    import pygetwindow  # type: ignore[import-not-found]
    _PYGETWINDOW_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - headless CI path
    _PYGETWINDOW_IMPORT_ERROR = exc
    pygetwindow = None  # type: ignore[assignment]


class PickerOutcome(str, Enum):
    """Outcome of a single OS picker macro invocation."""

    SUCCESS = "success"
    PICKER_NOT_DETECTED = "picker_not_detected"
    FILE_NOT_REFLECTED = "file_not_reflected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class PickerMacroResult:
    """Result returned by :func:`run_os_picker_macro`."""

    outcome: PickerOutcome
    detail: str


# Window-title keywords that identify a native file picker across browsers and
# platforms. Matching is case-insensitive and substring-based.
_PICKER_TITLE_KEYWORDS: tuple[str, ...] = (
    "open file",
    "open files",
    "save as",
    "select file",
    "choose file",
    "file upload",
    "upload file",
    "browse for file",
)


def _find_picker_window() -> object | None:
    """Return the first visible window whose title looks like a file picker."""
    if pygetwindow is None:
        return None
    try:
        windows = pygetwindow.getAllWindows()
    except Exception:  # pragma: no cover - defensive
        return None
    for window in windows:
        title = getattr(window, "title", "") or ""
        lowered = title.lower()
        if not lowered:
            continue
        if any(keyword in lowered for keyword in _PICKER_TITLE_KEYWORDS):
            return window
    return None


def _poll_for_picker(timeout_s: float, interval_s: float = 0.1) -> object | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        window = _find_picker_window()
        if window is not None:
            return window
        time.sleep(interval_s)
    return None


def _poll_for_picker_closed(timeout_s: float, interval_s: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _find_picker_window() is None:
            return True
        time.sleep(interval_s)
    return False


def run_os_picker_macro(
    file_path: str,
    *,
    appear_timeout_s: float = 3.0,
    close_timeout_s: float = 3.0,
) -> PickerMacroResult:
    """Type ``file_path`` into an open OS file picker and confirm.

    Parameters
    ----------
    file_path:
        Absolute path to type into the picker's filename field.
    appear_timeout_s:
        Maximum time to wait for a picker window to appear after the caller
        has already clicked the trigger button.
    close_timeout_s:
        Maximum time to wait for the picker window to close after pressing
        Enter. If it does not close, we assume the filename was not accepted
        and return :attr:`PickerOutcome.FILE_NOT_REFLECTED`.
    """
    if _PYAUTOGUI_IMPORT_ERROR is not None or pygetwindow is None:
        return PickerMacroResult(
            outcome=PickerOutcome.UNAVAILABLE,
            detail=(
                "os_picker_macro unavailable: pyautogui/pygetwindow not importable "
                "(requires headed desktop session)"
            ),
        )

    picker_window = _poll_for_picker(appear_timeout_s)
    if picker_window is None:
        return PickerMacroResult(
            outcome=PickerOutcome.PICKER_NOT_DETECTED,
            detail=(
                f"No OS file picker window appeared within {appear_timeout_s:.1f}s "
                f"(matched titles: {list(_PICKER_TITLE_KEYWORDS)})"
            ),
        )

    # Best-effort focus; ignore failures because many picker windows auto-focus
    # their filename field and ``activate()`` can raise on Windows when the
    # foreground already belongs to the same process.
    try:
        activate = getattr(picker_window, "activate", None)
        if callable(activate):
            activate()
    except Exception:
        pass

    try:
        pyautogui.write(file_path, interval=0.02)
    except Exception as exc:
        return PickerMacroResult(
            outcome=PickerOutcome.UNAVAILABLE,
            detail=f"pyautogui.write failed: {exc}",
        )

    time.sleep(0.2)

    try:
        pyautogui.press("enter")
    except Exception as exc:
        return PickerMacroResult(
            outcome=PickerOutcome.UNAVAILABLE,
            detail=f"pyautogui.press('enter') failed: {exc}",
        )

    if not _poll_for_picker_closed(close_timeout_s):
        return PickerMacroResult(
            outcome=PickerOutcome.FILE_NOT_REFLECTED,
            detail=(
                f"OS file picker window did not close within {close_timeout_s:.1f}s "
                "after pressing Enter; filename may have been rejected"
            ),
        )

    return PickerMacroResult(
        outcome=PickerOutcome.SUCCESS,
        detail=f"os_picker_macro typed file path and confirmed: {file_path}",
    )

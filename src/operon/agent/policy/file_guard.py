"""File-path guard for actions that read or write local files.

``read_text`` writes page text to ``action.text`` and the ``upload_file*``
actions hand ``action.text`` to the page's file chooser. That path comes from
model output, which a page can steer, so without a check a run could overwrite
or upload any file the operator can reach.

Always on (a security boundary, not an experiment). A path is allowed when it:

- appears verbatim in the run's intent — the user asked for that file;
- resolves inside the run's own artifact directory; or
- resolves under a root listed in ``OPERON_ALLOWED_FILE_ROOTS``.

Anything else pauses the run for a human instead of executing.

Config:
    OPERON_ALLOWED_FILE_ROOTS   comma-separated directories file actions may
                                use without the path being named in the intent
"""

from __future__ import annotations

import os
from pathlib import Path

from operon.models.policy import ActionType

FILE_PATH_ACTIONS: frozenset[ActionType] = frozenset({
    ActionType.READ_TEXT,
    ActionType.UPLOAD_FILE,
    ActionType.UPLOAD_FILE_NATIVE,
})

# Characters that continue a path; a match in the intent must not be followed
# or preceded by one, so "C:\out" does not authorize "C:\out\secret".
_PATH_CHARS = frozenset("_-/\\")


def _canon(value: str) -> str:
    value = value.replace("\\", "/")
    return value.casefold() if os.name == "nt" else value


def _named_in_intent(path: str, intent: str) -> bool:
    needle, haystack = _canon(path.strip()), _canon(intent)
    if not needle:
        return False
    start = haystack.find(needle)
    while start != -1:
        before = haystack[start - 1] if start > 0 else " "
        end = start + len(needle)
        after = haystack[end] if end < len(haystack) else " "
        if not (before.isalnum() or before in _PATH_CHARS) and not (after.isalnum() or after in _PATH_CHARS):
            return True
        start = haystack.find(needle, start + 1)
    return False


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def allowed_file_roots() -> list[Path]:
    raw = os.getenv("OPERON_ALLOWED_FILE_ROOTS", "")
    return [Path(p.strip()).expanduser().resolve() for p in raw.split(",") if p.strip()]


def blocked_file_path(action: object, *, intent: str, run_dir: Path) -> str | None:
    """Return the offending path if ``action`` uses a file path it may not, else None."""
    if getattr(action, "action_type", None) not in FILE_PATH_ACTIONS:
        return None
    path = getattr(action, "text", None)
    if not path:
        return None  # the executor rejects a missing path itself
    if _named_in_intent(path, intent):
        return None
    resolved = Path(path).expanduser().resolve()
    if any(_under(resolved, root) for root in [run_dir.resolve(), *allowed_file_roots()]):
        return None
    return path

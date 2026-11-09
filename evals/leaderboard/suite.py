"""Hash a frozen task set and check that a run used the same one.

A season's tasks are frozen; a run records the hashes of the tasks it used, and any
later scoring or comparison refuses tasks whose hashes differ. Changing a task on
purpose (for example retiring a broken one) needs ``allow_change=True`` and logs a
warning, so a changed task can never be mixed into old results silently.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)


class SuiteMismatchError(RuntimeError):
    """The tasks being used are not the ones a run was made with."""


def hash_bytes(parts: Iterable[tuple[str, bytes]]) -> str:
    """Hash (name, content) pairs in name order, ignoring CRLF/LF differences.

    Each field is length-prefixed, so no file's bytes can imitate a field boundary
    (a file containing NUL bytes can't hash like a different set of files).
    """
    digest = hashlib.sha256()
    for name, content in sorted(parts, key=lambda part: part[0]):
        for field in (name.encode("utf-8"), content.replace(b"\r\n", b"\n")):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def hash_directory(path: str | Path) -> str:
    """Hash every file's relative path and contents under ``path`` (empty if missing)."""
    root = Path(path)
    if not root.exists():
        return hash_bytes([])
    return hash_bytes(
        (file.relative_to(root).as_posix(), file.read_bytes())
        for file in root.rglob("*")
        if file.is_file()
    )


def hash_text(text: str) -> str:
    """Hash one task definition given as text (for example its canonical JSON)."""
    return hash_bytes([("", text.encode("utf-8"))])


def check_hashes_match(
    recorded: Mapping[str, str],
    current: Mapping[str, str],
    *,
    allow_change: bool = False,
    what: str = "task set",
) -> list[str]:
    """Compare named hashes (e.g. {"working": ..., "held_out": ...}) with a run's record.

    Returns the names that differ. Raises SuiteMismatchError on any difference,
    including a name present on one side only, unless ``allow_change`` is set, in
    which case it logs a warning and returns the differing names.
    """
    changed = sorted(name for name in set(recorded) | set(current) if recorded.get(name) != current.get(name))
    if not changed:
        return []
    message = (
        f"the {what}'s {' and '.join(changed)} hash doesn't match the run's record, "
        "so it isn't the one the run was made with"
    )
    if not allow_change:
        raise SuiteMismatchError(
            message + ". Use the original tasks, or allow the change on purpose "
            "(for example after retiring a broken task)."
        )
    logger.warning("%s. Continuing because the change was allowed.", message)
    return changed

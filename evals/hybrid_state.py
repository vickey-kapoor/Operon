"""The Windows side of the Hybrid tasks: set up their files before a run, read them after.

``prepare(task)`` stops the apps the task may have left open (the browser, Notepad),
removes the files it saves from an earlier attempt and writes the files it is given,
all under ``Documents\\OperonFiles`` (the Documents folder Windows reports, which may
be redirected). ``probe(task)`` reads those files back, with the computer's name,
for the grader in ``evals/hybrid_tasks.py``. Only the task's own files are touched.

This is grading only: the agent never sees any of it, and its perception and policy
stay vision-only. It reuses the desktop probe (``evals/desktop_probe.ps1``) and the
file helpers in ``evals/desktop_state.py`` without changing them, so the Desktop tasks' grading (and
their season fingerprints) stay as they are.
"""

from __future__ import annotations

import os
from pathlib import Path

# Imported by name, not through the module, so a season fingerprint follows them
# (evals/leaderboard/season.py hashes the helpers a grader names).
from evals.desktop_state import (
    OUTPUT_FOLDER,
    ProbeError,
    _outputs,
    _remove_outputs,
    _run_probe,
)
from evals.hybrid_tasks import HybridTask


def _documents(snapshot: dict) -> str:
    documents = snapshot.get("documents")
    if not documents:
        raise ProbeError("Windows gave no Documents folder")
    return documents


def reset_files(documents: str, task: HybridTask) -> None:
    """Remove the task's files from an earlier attempt, then write the ones it is given.
    The folder itself is created, so a task never fails on a missing folder."""
    _remove_outputs(documents, (*task.outputs, *task.inputs))
    root = Path(documents) / OUTPUT_FOLDER
    for relative, text in task.inputs.items():
        path = root / relative
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # CRLF and a BOM-less UTF-8 file: what Notepad itself writes.
            path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
        except OSError as exc:
            raise ProbeError(f"could not write the task's file {path}: {exc}") from exc
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProbeError(f"could not create {root}: {exc}") from exc


def read_files(documents: str, task: HybridTask) -> dict[str, str | None]:
    """The text of every file the task saves or was given; None for one that isn't there."""
    return _outputs(documents, (*task.outputs, *task.inputs))


def prepare(task: HybridTask) -> None:
    """Reset the desktop and the task's files before a run."""
    try:
        snapshot = _run_probe("-Close", ",".join(task.close_first), timeout=60)
        reset_files(_documents(snapshot), task)
    except ProbeError as exc:
        raise ProbeError(f"hybrid reset failed: {exc}") from exc


def probe(task: HybridTask) -> dict:
    """The task's files and the computer's name after a run (the site's state is added by the caller)."""
    snapshot = _run_probe("-Inspect", "")
    documents = _documents(snapshot)
    return {
        "documents": documents,
        "files": read_files(documents, task),
        "computer_name": os.environ.get("COMPUTERNAME", ""),
        "windows": [{"title": w.get("title"), "process": w.get("process")} for w in snapshot.get("windows") or []],
    }

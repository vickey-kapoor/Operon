"""Which process owns a run, so a scan never calls a live run interrupted (KAN-80).

The process that creates or resumes a run writes ``<run>/owner.json`` with its pid, its
process creation time and its host, and removes it when the run ends. A pid alone is not
proof (pids are reused), so an owner is alive only if the pid exists AND its creation time
still matches. The probes are small helpers (``pid_alive``, ``process_create_time``) so tests
can patch them. No psutil: ``os.kill(pid, 0)`` / ``/proc`` on POSIX, ctypes on Windows.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

OWNER_FILE = "owner.json"
_STILL_ACTIVE = 259
_ERROR_INVALID_PARAMETER = 87
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _open_process(pid: int):
    if sys.platform != "win32":
        raise OSError("Windows only")
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    return kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)


def _win_last_error() -> int:
    if sys.platform != "win32":
        raise OSError("Windows only")
    import ctypes

    return ctypes.get_last_error()


def _windows_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    handle = _open_process(pid)
    if not handle:
        # Only "no such pid" means dead. Access denied (5: another account's or a protected
        # process) and every other failure mean it exists or we cannot tell: alive.
        return _win_last_error() != _ERROR_INVALID_PARAMETER
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True  # cannot tell: treat as alive
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def process_create_time(pid: int) -> int | None:
    """An opaque per-process start stamp (equal for the same process), or None when unknown."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            try:
                created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
                ok = kernel32.GetProcessTimes(
                    handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)
                )
                return (created.dwHighDateTime << 32) | created.dwLowDateTime if ok else None
            finally:
                kernel32.CloseHandle(handle)
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return int(stat.rsplit(")", 1)[1].split()[19])  # field 22: start time in ticks
    except (OSError, ValueError, IndexError):
        return None


def _path(run_dir: Path) -> Path:
    return run_dir / OWNER_FILE


def claim(run_dir: Path) -> None:
    """Record this process as the run's owner. Never raises: a failed write only means the
    run falls back to the age rule."""
    try:
        pid = os.getpid()
        data = {"pid": pid, "created": process_create_time(pid), "host": socket.gethostname()}
        run_dir.mkdir(parents=True, exist_ok=True)
        _path(run_dir).write_text(json.dumps(data), encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.warning("could not write %s: %s", _path(run_dir), exc)


class RunOwnedError(ValueError):
    """Another live process already owns (is resuming) the run."""


_GUARD_FILE = "owner.takeover"
_STALE_PARTIAL_S = 5.0


def _owner_data() -> str:
    pid = os.getpid()
    return json.dumps({"pid": pid, "created": process_create_time(pid), "host": socket.gethostname()})


def _create_exclusive(path: Path, text: str) -> bool:
    """Create ``path`` with ``text`` only if it does not exist (atomic across processes)."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return True


def _held_by_other(run_dir: Path) -> int | None:
    """The pid of a live owner other than this process, or None when free to take over.
    An unreadable file counts as held until it is a few seconds old (its writer may be mid-write)."""
    path = _path(run_dir)
    owner = read(run_dir)
    if owner is None:
        try:
            return -1 if time.time() - path.stat().st_mtime < _STALE_PARTIAL_S else None
        except OSError:
            return None  # gone meanwhile
    if owner["pid"] == os.getpid() and owner.get("host") in (None, socket.gethostname()):
        return None
    return owner["pid"] if owner_alive(owner) else None


def acquire(run_dir: Path) -> None:
    """Exclusively claim the run for this process, or raise ``RunOwnedError`` when another
    live process owns it. The claim is ``owner.json`` created with O_EXCL, so of two
    processes only one wins. A dead owner's file is taken over under a short-lived guard
    file, so two processes that both see it stale cannot both remove and recreate it.
    Re-acquiring a run this process already owns succeeds."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = _path(run_dir)
    text = _owner_data()
    for _ in range(3):
        if _create_exclusive(path, text):
            return
        holder = _held_by_other(run_dir)
        if holder is not None:
            raise RunOwnedError(f"run {run_dir.name} is already being resumed by pid {holder}")
        guard = run_dir / _GUARD_FILE
        if not _create_exclusive(guard, text):
            try:
                if time.time() - guard.stat().st_mtime > 10 * _STALE_PARTIAL_S:
                    guard.unlink(missing_ok=True)  # a crashed taker-over
            except OSError:
                pass
            time.sleep(0.05)
            continue  # someone is taking over right now: look again
        try:
            if _held_by_other(run_dir) is not None:
                continue  # it was claimed while we took the guard: loop re-reads and rejects
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue  # Windows: another process has it open (WinError 32): treat as held, retry
            if _create_exclusive(path, text):
                return
        finally:
            try:
                guard.unlink(missing_ok=True)
            except OSError:
                pass
    raise RunOwnedError(f"run {run_dir.name} is being claimed by another process")


def release(run_dir: Path) -> None:
    try:
        _path(run_dir).unlink(missing_ok=True)
    except OSError:
        pass


def read(run_dir: Path) -> dict | None:
    try:
        data = json.loads(_path(run_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("pid"), int) else None


def owner_alive(owner: dict) -> bool:
    """Whether the process that wrote ``owner`` is still running. Another host's owner
    cannot be probed from here, so it counts as alive (never mark what we cannot judge)."""
    if owner.get("host") not in (None, socket.gethostname()):
        return True
    pid = owner["pid"]
    if not pid_alive(pid):
        return False
    recorded = owner.get("created")
    current = process_create_time(pid)
    return recorded is None or current is None or recorded == current

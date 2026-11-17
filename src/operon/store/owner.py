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
from pathlib import Path

logger = logging.getLogger(__name__)

OWNER_FILE = "owner.json"
_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True  # cannot tell: treat as alive
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
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
    pid = os.getpid()
    data = {"pid": pid, "created": process_create_time(pid), "host": socket.gethostname()}
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        _path(run_dir).write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:
        logger.warning("could not write %s: %s", _path(run_dir), exc)


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

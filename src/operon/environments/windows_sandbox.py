"""Windows Sandbox provider (KAN-80, stage 2): a clean, throwaway Windows on this PC.

``start()`` writes a ``.wsb`` configuration and opens Windows Sandbox with it. The
sandbox maps this machine's Python, Operon's virtualenv and Operon's source folder
read-only at the same paths, so the same interpreter and packages run inside with
nothing to install. Its logon command starts ``operon guest`` (``guest.py``), which
writes its address to a small writable "exchange" folder. The host then drives the
sandbox's desktop through ``RemoteDesktopExecutor``.

What crosses into the sandbox: the read-only folders above and a one-time guest
token. Not the repo root, so not ``.env``, and not ``~/.operon``: the Gemini key
stays on the host, which makes every model call. Clipboard, printer, microphone and
camera redirection are off. Everything inside is wiped when the sandbox closes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

import operon
from operon.core.paths import user_config_dir
from operon.environments.base import EnvironmentProvider, ProviderCheck
from operon.environments.guest import DEFAULT_GUEST_PORT, new_token

logger = logging.getLogger(__name__)

ENABLE_COMMAND = "Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All"
# Process names Windows Sandbox has used across Windows 10 and 11 releases.
_SANDBOX_PROCESSES = (
    "WindowsSandboxRemoteSession.exe",
    "WindowsSandboxClient.exe",
    "WindowsSandbox.exe",
)
# Where the guest keeps its own files inside the sandbox (wiped on close).
_GUEST_RUNTIME = r"C:\Users\WDAGUtilityAccount\operon-guest"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def sandbox_exe() -> Path:
    return Path(os.getenv("SystemRoot", r"C:\Windows")) / "System32" / "WindowsSandbox.exe"


def source_dir() -> Path:
    """The folder that holds the ``operon`` package (``src`` in a checkout)."""
    return Path(operon.__file__).resolve().parent.parent


def mapped_folders() -> list[Path]:
    """Read-only folders the guest needs, at the same paths inside: base Python,
    the virtualenv (if any) and Operon's source folder (if not inside the venv)."""
    folders: list[Path] = []
    for raw in (sys.base_prefix, sys.prefix):
        path = Path(raw).resolve()
        if path not in folders:
            folders.append(path)
    src = source_dir()
    if not any(src == f or f in src.parents for f in folders):
        folders.append(src)
    return folders


def guest_python() -> Path:
    """The interpreter the guest runs: this one (the venv's, when there is one)."""
    exe = Path(sys.executable)
    windowless = exe.with_name("pythonw.exe")
    return windowless if windowless.exists() else exe


class WindowsSandboxProvider(EnvironmentProvider):
    name = "windows-sandbox"
    summary = "A clean, throwaway Windows on this PC (Windows Sandbox), wiped when it closes."

    def __init__(self, *, work_dir: Path | None = None) -> None:
        self.work_dir = work_dir or (user_config_dir() / "windows-sandbox")
        self.exchange = self.work_dir / "exchange"
        self.port = DEFAULT_GUEST_PORT
        self.memory_mb = _int_env("OPERON_SANDBOX_MEMORY_MB", 4096)
        self.start_timeout_s = _int_env("OPERON_SANDBOX_START_TIMEOUT", 240)
        self._token: str | None = None
        self._endpoint: str | None = None
        self._started = False

    def platform(self) -> str:
        return "windows"

    # ── check ────────────────────────────────────────────────────

    def check(self) -> ProviderCheck:
        problems: list[str] = []
        if sys.platform != "win32":
            problems.append("Windows Sandbox runs only on Windows 10/11 Pro, Enterprise or Education.")
        elif not sandbox_exe().exists():
            problems.append(
                "Windows Sandbox is not turned on. In an administrator PowerShell run "
                f"`{ENABLE_COMMAND}`, then restart."
            )
        if "WindowsApps" in Path(sys.base_prefix).parts:
            problems.append(
                "Operon's Python is the Microsoft Store one, which cannot be shared with the sandbox; "
                "use Python from python.org."
            )
        return ProviderCheck(ok=not problems, problems=tuple(problems))

    # ── configuration files ──────────────────────────────────────

    def guest_command(self) -> str:
        """The batch file the sandbox runs at logon."""
        ex = self.exchange
        python = guest_python()
        return "\r\n".join(
            [
                "@echo off",
                f'netsh advfirewall firewall add rule name="Operon guest" dir=in action=allow protocol=TCP localport={self.port} >nul',
                "set PYTHONDONTWRITEBYTECODE=1",
                f"set OPERON_RUNTIME_ROOT={_GUEST_RUNTIME}",
                f"set OPERON_CONFIG_DIR={_GUEST_RUNTIME}",
                f'set PYTHONPATH={source_dir()}',
                f'"{python}" -m operon.cli guest --port {self.port} --token-file "{ex / "token.txt"}" '
                f'--ready-file "{ex / "ready.json"}" --log-file "{ex / "guest.log"}"',
                "",
            ]
        )

    def wsb_config(self) -> str:
        """The Windows Sandbox configuration (``.wsb``) XML."""
        rows = [
            "<MappedFolder>"
            f"<HostFolder>{escape(str(folder))}</HostFolder>"
            f"<SandboxFolder>{escape(str(folder))}</SandboxFolder>"
            "<ReadOnly>true</ReadOnly>"
            "</MappedFolder>"
            for folder in mapped_folders()
        ]
        rows.append(
            "<MappedFolder>"
            f"<HostFolder>{escape(str(self.exchange))}</HostFolder>"
            f"<SandboxFolder>{escape(str(self.exchange))}</SandboxFolder>"
            "<ReadOnly>false</ReadOnly>"
            "</MappedFolder>"
        )
        start = escape(f'cmd.exe /c "{self.exchange / "start-guest.cmd"}"')
        return "\n".join(
            [
                "<Configuration>",
                "  <VGpu>Default</VGpu>",
                "  <Networking>Enable</Networking>",
                f"  <MemoryInMB>{self.memory_mb}</MemoryInMB>",
                "  <ClipboardRedirection>Disable</ClipboardRedirection>",
                "  <PrinterRedirection>Disable</PrinterRedirection>",
                "  <AudioInput>Disable</AudioInput>",
                "  <VideoInput>Disable</VideoInput>",
                "  <MappedFolders>",
                *(f"    {row}" for row in rows),
                "  </MappedFolders>",
                f"  <LogonCommand><Command>{start}</Command></LogonCommand>",
                "</Configuration>",
                "",
            ]
        )

    def write_files(self) -> Path:
        """Fresh token, logon script and ``.wsb``; clears the last start's ready file."""
        self.exchange.mkdir(parents=True, exist_ok=True)
        for stale in ("ready.json", "ready.tmp", "guest.log"):
            (self.exchange / stale).unlink(missing_ok=True)
        self._token = new_token()
        (self.exchange / "token.txt").write_text(self._token, encoding="utf-8")
        (self.exchange / "start-guest.cmd").write_text(self.guest_command(), encoding="utf-8")
        wsb = self.work_dir / "operon.wsb"
        wsb.write_text(self.wsb_config(), encoding="utf-8")
        return wsb

    # ── lifecycle ────────────────────────────────────────────────

    def _open(self, wsb: Path) -> None:  # pragma: no cover - needs Windows
        os.startfile(str(wsb))  # type: ignore[attr-defined]

    def wait_until_ready(self, *, interval_s: float = 1.0) -> dict:
        """Wait for the guest's ready file, then for its ``/health``."""
        from operon.environments.remote import wait_for_guest

        ready = self.exchange / "ready.json"
        deadline = time.monotonic() + self.start_timeout_s
        while not ready.exists():
            if time.monotonic() >= deadline:
                log = self.exchange / "guest.log"
                tail = log.read_text(encoding="utf-8", errors="replace")[-800:] if log.exists() else "(no guest log)"
                raise RuntimeError(
                    f"Windows Sandbox did not start Operon's guest within {self.start_timeout_s}s. Guest log:\n{tail}"
                )
            time.sleep(interval_s)
        info = json.loads(ready.read_text(encoding="utf-8"))
        self._endpoint = f"http://{info['host']}:{info['port']}"
        remaining = max(5.0, deadline - time.monotonic())
        return wait_for_guest(self._endpoint, self._token or "", timeout_s=remaining, interval_s=interval_s)

    def start(self) -> None:
        if self._started:
            return
        wsb = self.write_files()
        logger.info("windows-sandbox: opening %s", wsb)
        self._open(wsb)
        self._started = True
        try:
            health = self.wait_until_ready()
        except Exception:
            self.stop()
            raise
        logger.info("windows-sandbox: guest ready at %s (%s)", self._endpoint, health)

    def build_executor(self):
        if self._endpoint is None or self._token is None:
            raise RuntimeError("windows-sandbox: start() must finish before build_executor()")
        from operon.environments.remote import RemoteDesktopExecutor

        return RemoteDesktopExecutor(self._endpoint, self._token)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._endpoint = None
        if sys.platform == "win32" and shutil.which("taskkill"):
            for name in _SANDBOX_PROCESSES:
                subprocess.run(
                    ["taskkill", "/IM", name, "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        (self.exchange / "token.txt").unlink(missing_ok=True)

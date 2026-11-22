"""Runs the tools of the tool action channel (RFC 0002)."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import signal
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

from operon.agent.tools import (
    HTTP_GET,
    LIST_DIR,
    READ_FILE,
    RUN_SHELL,
    TOOL_NAMES,
    WRITE_FILE,
    tool_path,
)
from operon.core.launch_env import launch_env
from operon.models.execution import ToolOutput

TEXT_LIMIT = 20_000
STREAM_LIMIT = 8_000
SHELL_TIMEOUT = 30
SHELL_TIMEOUT_MAX = 120
HTTP_TIMEOUT = 20.0


class ToolError(Exception):
    """The tool could not run; the message goes back to the model."""


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[cut: {len(text) - limit} more characters]"


def _decode(data: bytes) -> str:
    """A file's text: UTF-16 when it starts with that byte-order mark (PowerShell's
    default on Windows), else UTF-8, with any bad bytes replaced."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def _string_arg(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"needs a non-empty '{key}'")
    return value


async def _kill_tree(process: asyncio.subprocess.Process) -> None:
    """Kill the process and everything it started, then reap it."""
    if process.returncode is None:
        try:
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill", "/F", "/T", "/PID", str(process.pid),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await killer.wait()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        except OSError:
            pass
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        # An IPv4 address hidden in an IPv6 one: mapped (::ffff:a.b.c.d), compatible
        # (::a.b.c.d) or NAT64 (64:ff9b::a.b.c.d): judge the IPv4 address.
        if ip.ipv4_mapped is not None:
            return _is_public(ip.ipv4_mapped)
        if int(ip) >> 32 == 0 and int(ip) > 1:
            return _is_public(ipaddress.IPv4Address(int(ip)))
        if ip in _NAT64:
            return _is_public(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if ip.is_site_local:
            return False
    return ip.is_global and not ip.is_multicast


async def _resolve(host: str, port: int) -> list[str]:
    """The addresses a host name resolves to (patched in tests)."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ToolError(f"failed: could not resolve {host}: {exc}") from None
    return [str(info[4][0]).split("%")[0] for info in infos]


async def _public_addresses(host: str, port: int) -> list[str]:
    """The addresses to try for a host; refuse if any resolved address is local,
    private or a cloud-metadata address (SSRF)."""
    addresses = await _resolve(host, port)
    if not addresses:
        raise ToolError(f"failed: could not resolve {host}")
    for address in addresses:
        if not _is_public(ipaddress.ip_address(address)):
            raise ToolError(f"refused: {host} resolves to a local, private or cloud-metadata address ({address})")
    return addresses


class ToolRunner:
    """Runs one tool call and returns its output. Tools the run did not turn on are refused."""

    def __init__(self, tools: tuple[str, ...]) -> None:
        self.tools = tools
        self._shells: dict[asyncio.subprocess.Process, str | None] = {}
        self._stopped: set[asyncio.subprocess.Process] = set()

    async def kill_running(self, run_id: str | None = None) -> int:
        """Kill the shell commands still running (of one run, or all when run_id is None)
        because the run was stopped; returns how many."""
        live = [p for p, owner in list(self._shells.items()) if p.returncode is None and run_id in (None, owner)]
        self._stopped.update(live)
        for process in live:
            await _kill_tree(process)
        return len(live)

    async def run(self, name: str, args: dict[str, Any] | None, run_id: str | None = None) -> ToolOutput:
        args = args or {}
        if name not in TOOL_NAMES:
            return ToolOutput(tool=name, ok=False, output=f"There is no tool named {name}.")
        if name not in self.tools:
            return ToolOutput(tool=name, ok=False, output=f"The {name} tool is not turned on for this run.")
        handler = {
            READ_FILE: self._read_file,
            LIST_DIR: self._list_dir,
            WRITE_FILE: self._write_file,
            RUN_SHELL: lambda a: self._run_shell(a, run_id),
            HTTP_GET: self._http_get,
        }[name]
        try:
            return ToolOutput(tool=name, ok=True, output=await handler(args))
        except ToolError as exc:
            return ToolOutput(tool=name, ok=False, output=f"{name} {exc}")
        except OSError as exc:
            return ToolOutput(tool=name, ok=False, output=f"{name} failed: {exc}")

    async def _read_file(self, args: dict[str, Any]) -> str:
        path = tool_path(_string_arg(args, "path"))
        if not path.is_file():
            raise ToolError(f"failed: {path} is not a file")
        return _cut(_decode(path.read_bytes()), TEXT_LIMIT)

    async def _list_dir(self, args: dict[str, Any]) -> str:
        path = tool_path(_string_arg(args, "path"))
        if not path.is_dir():
            raise ToolError(f"failed: {path} is not a folder")
        lines = []
        for entry in sorted(path.iterdir(), key=lambda p: p.name.casefold()):
            lines.append(f"{entry.name}\t{'dir' if entry.is_dir() else entry.stat().st_size}")
        return _cut("\n".join(lines) or "(empty folder)", TEXT_LIMIT)

    async def _write_file(self, args: dict[str, Any]) -> str:
        path = tool_path(_string_arg(args, "path"))
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolError("needs 'content' as text")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        path.write_bytes(data)
        return f"Wrote {len(data)} bytes to {path}"

    async def _run_shell(self, args: dict[str, Any], run_id: str | None = None) -> str:
        command = _string_arg(args, "command")
        try:
            timeout = int(args.get("timeout_seconds") or SHELL_TIMEOUT)
        except (TypeError, ValueError):
            timeout = SHELL_TIMEOUT
        timeout = max(1, min(timeout, SHELL_TIMEOUT_MAX))
        if os.name == "nt":
            argv = [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "[Console]::OutputEncoding = [Text.Encoding]::UTF8; " + command,
            ]
        else:
            argv = ["bash", "-c", command]
        # Output goes to temporary files, not pipes: a program the command leaves running
        # (``start file.svg`` opens a viewer) inherits the shell's output handles, and
        # reading a pipe would then wait for that program to close, not for the shell.
        with tempfile.TemporaryFile() as out_file, tempfile.TemporaryFile() as err_file:
            # Own process group / session, so a stop or timeout can kill the whole tree.
            group: dict[str, Any] = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
                if os.name == "nt"
                else {"start_new_session": True}
            )
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(Path.home()),
                env=launch_env(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=out_file,
                stderr=err_file,
                **group,
            )
            self._shells[process] = run_id
            try:
                await asyncio.wait_for(process.wait(), timeout)
            except TimeoutError:
                await _kill_tree(process)
                raise ToolError(f"timed out after {timeout} seconds and was stopped") from None
            except BaseException:
                await asyncio.shield(_kill_tree(process))
                raise
            finally:
                self._shells.pop(process, None)
                stopped = process in self._stopped
                self._stopped.discard(process)
            if stopped:
                raise ToolError("was stopped (the run was stopped)")
            out_file.seek(0)
            err_file.seek(0)
            out = _cut(out_file.read().decode("utf-8", errors="replace"), STREAM_LIMIT)
            err = _cut(err_file.read().decode("utf-8", errors="replace"), STREAM_LIMIT)
        return f"exit code: {process.returncode}\nstdout:\n{out}\nstderr:\n{err}"

    async def _http_get(self, args: dict[str, Any]) -> str:
        url = _string_arg(args, "url").strip()
        if not url.lower().startswith(("http://", "https://")):
            raise ToolError("only fetches http and https URLs")
        try:
            # Redirects are not followed: a redirect could leave the trust gate's domain
            # allowlist, so the model sees where it points and may fetch that URL next.
            try:
                parsed = httpx.URL(url)
                host = parsed.raw_host.decode("ascii")
            except (ValueError, UnicodeError, httpx.InvalidURL):
                raise ToolError("needs a valid URL") from None
            if not host:
                raise ToolError("needs a URL with a host")
            https = parsed.scheme == "https"
            port = parsed.port or (443 if https else 80)
            addresses = await _public_addresses(host, port)
            # Connect to an address that was checked, not to a second DNS answer
            # (rebinding); Host and the TLS name stay the real hostname.
            shown = f"[{host}]" if ":" in host else host
            host_header = shown if parsed.port is None else f"{shown}:{parsed.port}"
            extensions = {"sni_hostname": host} if https else {}
            # trust_env=False: an HTTPS_PROXY would tunnel to the pinned IP (httpcore then
            # ignores sni_hostname and fails certificate checks), and no environment proxy
            # may be a way around the address check, so the client always connects directly.
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=False, trust_env=False) as client:
                for index, ip in enumerate(addresses):
                    try:
                        response = await client.get(
                            parsed.copy_with(host=ip), headers={"Host": host_header}, extensions=extensions
                        )
                        break
                    except httpx.ConnectError:
                        if index == len(addresses) - 1:
                            raise
        except httpx.HTTPError as exc:
            raise ToolError(f"failed: {exc.__class__.__name__}: {exc}") from None
        head = f"status: {response.status_code}\ncontent-type: {response.headers.get('content-type', '')}"
        location = response.headers.get("location")
        if location:
            head += f"\nlocation: {location}"
        return f"{head}\n\n{_cut(response.text, TEXT_LIMIT)}"

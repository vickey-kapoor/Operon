"""Runs the tools of the tool action channel (RFC 0002)."""

from __future__ import annotations

import asyncio
import os
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


class ToolRunner:
    """Runs one tool call and returns its output. Tools the run did not turn on are refused."""

    def __init__(self, tools: tuple[str, ...]) -> None:
        self.tools = tools

    async def run(self, name: str, args: dict[str, Any] | None) -> ToolOutput:
        args = args or {}
        if name not in TOOL_NAMES:
            return ToolOutput(tool=name, ok=False, output=f"There is no tool named {name}.")
        if name not in self.tools:
            return ToolOutput(tool=name, ok=False, output=f"The {name} tool is not turned on for this run.")
        handler = {
            READ_FILE: self._read_file,
            LIST_DIR: self._list_dir,
            WRITE_FILE: self._write_file,
            RUN_SHELL: self._run_shell,
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

    async def _run_shell(self, args: dict[str, Any]) -> str:
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
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(Path.home()),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ToolError(f"timed out after {timeout} seconds and was stopped") from None
        out = _cut(stdout.decode("utf-8", errors="replace"), STREAM_LIMIT)
        err = _cut(stderr.decode("utf-8", errors="replace"), STREAM_LIMIT)
        return f"exit code: {process.returncode}\nstdout:\n{out}\nstderr:\n{err}"

    async def _http_get(self, args: dict[str, Any]) -> str:
        url = _string_arg(args, "url").strip()
        if not url.lower().startswith(("http://", "https://")):
            raise ToolError("only fetches http and https URLs")
        try:
            # Redirects are not followed: a redirect could leave the trust gate's domain
            # allowlist, so the model sees where it points and may fetch that URL next.
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=False) as client:
                response = await client.get(url)
        except httpx.HTTPError as exc:
            raise ToolError(f"failed: {exc.__class__.__name__}: {exc}") from None
        head = f"status: {response.status_code}\ncontent-type: {response.headers.get('content-type', '')}"
        location = response.headers.get("location")
        if location:
            head += f"\nlocation: {location}"
        return f"{head}\n\n{_cut(response.text, TEXT_LIMIT)}"

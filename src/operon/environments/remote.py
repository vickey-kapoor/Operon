"""``RemoteDesktopExecutor`` (KAN-80): the host's side of the guest agent.

The loop drives it like any desktop executor. Each call goes to ``operon guest``
on the run environment's machine (see ``guest.py``); screenshots and traces come
back as bytes and are written under this host's artifact folder, so the loop, the
verifier and the run store only ever see local files.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from operon.core.paths import desktop_artifacts_dir
from operon.core.vault import SecretError, redact, resolve_text, secret_endpoint_ok
from operon.environments.guest import FILE_FIELDS
from operon.models.capture import CaptureFrame
from operon.models.common import FailureCategory, LoopStage
from operon.models.execution import ExecutedAction, ToolOutput
from operon.models.policy import ActionType, AgentAction

logger = logging.getLogger(__name__)

# An action can include waits and app launches; screenshots are a few MB.
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class GuestUnavailableError(RuntimeError):
    """The guest agent did not answer."""


class RemoteDesktopExecutor:
    """Drives a desktop through its guest agent at ``base_url``."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        artifact_dir: str | Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        trusted_host: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # The one non-loopback address a secret may be sent to: the guest address the
        # provider itself reported (a Windows Sandbox's host-only IP).
        self._trusted_host = trusted_host
        self._headers = {"Authorization": f"Bearer {token}"}
        self._transport = transport
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else desktop_artifacts_dir()
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._client: httpx.AsyncClient | None = None
        self._current_run_id: str | None = None
        self._close_when_done: set[str] = set()
        # run_id -> each secret value this host filled in (KAN-103), one by one, never the
        # composite text. Every later guest reply of the run is masked with them; forgotten
        # when the run ends.
        self._secrets_used: dict[str, set[str]] = {}
        # Set by the loop around a batch: checked before each child after the first
        # (a stop, a trust gate). A callable cannot cross to the guest, so batches are
        # split here and each child goes to the guest on its own.
        self.batch_child_check = None

    # ── transport ────────────────────────────────────────────────

    def _http(self) -> httpx.AsyncClient:
        # Built on first use, inside the event loop that will use it.
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, headers=self._headers, timeout=_TIMEOUT, transport=self._transport,
                # The guest is a machine on this host's own network: never via a proxy.
                trust_env=False,
            )
        return self._client

    async def _post(self, path: str, payload: dict[str, Any] | None = None) -> httpx.Response:
        try:
            response = await self._http().post(path, json=payload or {})
        except httpx.HTTPError as exc:
            raise GuestUnavailableError(f"guest agent at {self.base_url} did not answer {path}: {exc}") from exc
        response.raise_for_status()
        return response

    async def _call(self, method: str, **kwargs: Any) -> Any:
        return (await self._post("/call", {"method": method, "kwargs": kwargs})).json().get("result")

    def _save(self, b64: str | None, suffix: str) -> str | None:
        if not b64:
            return None
        path = self._artifact_dir / f"remote_{uuid4().hex[:8]}{suffix}"
        raw = base64.b64decode(b64)
        known = self._secrets_used.get(self._current_run_id or "")
        if known and suffix == ".json":
            raw = redact(raw.decode("utf-8", errors="replace"), known).encode("utf-8")
        path.write_bytes(raw)
        return str(path)

    # ── the executor surface the loop uses ───────────────────────

    async def capture(self) -> CaptureFrame:
        data = (await self._post("/capture")).json()
        frame = CaptureFrame.model_validate(data["frame"])
        local = self._save(data.get("png"), ".png")
        if local is None:
            raise GuestUnavailableError("guest agent sent a frame without its image")
        return frame.model_copy(update={"artifact_path": local})

    async def peek_frame(self) -> CaptureFrame | None:
        """The fresh look's extra screenshot from the guest (KAN-267). None when the guest has
        no ``peek_frame`` (204) or is an older guest without the route (404)."""
        try:
            response = await self._http().post("/peek", json={})
        except httpx.HTTPError as exc:
            raise GuestUnavailableError(f"guest agent at {self.base_url} did not answer /peek: {exc}") from exc
        if response.status_code in (204, 404):
            return None
        response.raise_for_status()
        data = response.json()
        frame = CaptureFrame.model_validate(data["frame"])
        local = self._save(data.get("png"), ".png")
        if local is None:
            raise GuestUnavailableError("guest agent sent a frame without its image")
        return frame.model_copy(update={"artifact_path": local})

    async def execute(self, action: AgentAction) -> ExecutedAction:
        if action.action_type is ActionType.BATCH:
            return await self._execute_batch(action)
        return await self._execute_one(action)

    async def _execute_batch(self, action: AgentAction) -> ExecutedAction:
        """A turn's calls in order, stopping at the first failure or a blocked child,
        as the local desktop and tool executors do."""
        outputs: list[ToolOutput] = []
        done: list[str] = []
        artifact_path: str | None = None
        for index, child in enumerate(action.actions or []):
            check = self.batch_child_check
            if index > 0 and check is not None:
                blocked = await check(child)
                if blocked is not None:
                    return _batch_failed(
                        action, index, f"Batch stopped before {child.action_type.value}: {blocked}", outputs, artifact_path
                    )
            result = await self._execute_one(child)
            outputs.extend(result.tool_outputs or [])
            artifact_path = result.artifact_path or artifact_path
            done.append(f"{child.action_type.value}:{'ok' if result.success else 'failed'}")
            if not result.success:
                return _batch_failed(
                    action, index, f"Batch failed on {child.action_type.value}: {result.detail}", outputs,
                    artifact_path, result,
                )
        return ExecutedAction(
            action=action,
            success=True,
            detail=f"Executed batch: {', '.join(done)}",
            artifact_path=artifact_path,
            tool_outputs=outputs or None,
        )

    def _local_endpoint(self) -> bool:
        """True when the guest is on loopback, or at exactly the address the provider reported.
        A secret is never sent anywhere else (no general private ranges)."""
        return secret_endpoint_ok(self.base_url, self._trusted_host)

    async def _execute_one(self, action: AgentAction) -> ExecutedAction:
        # KAN-103: the guest has no vault, so a {{secret:NAME}} placeholder is filled in here,
        # on the host, in the copy sent over the token-protected local connection. `action`
        # itself keeps the placeholder, and the guest's reply is masked before it is returned.
        send = action
        secret_values: tuple[str, ...] = ()
        run_key = self._current_run_id or ""
        if action.action_type is ActionType.TYPE and action.text:
            try:
                resolved, secret_values = resolve_text(action.text)
                if secret_values and not self._local_endpoint():
                    raise SecretError("the run environment is not at a loopback or provider-reported address")
            except SecretError as exc:
                return ExecutedAction(
                    action=action, success=False, detail=f"type failed: {exc}",
                    failure_category=FailureCategory.EXECUTION_ERROR,
                )
            send = action.model_copy(update={"text": resolved})
            if secret_values:
                self._secrets_used.setdefault(run_key, set()).update(secret_values)
        known = self._secrets_used.get(run_key, ())
        try:
            body: dict[str, Any] = {"action": send.model_dump(mode="json")}
            if secret_values:
                body["secret_filled"] = True
            data = (await self._post("/execute", body)).json()
        except (GuestUnavailableError, httpx.HTTPStatusError) as exc:
            return ExecutedAction(
                action=action,
                success=False,
                detail=redact(f"Run environment unreachable: {exc}", known),
                failure_category=FailureCategory.EXECUTION_ERROR,
            )
        result = dict(data["result"])
        if known:
            # Any reply of the run (a clipboard read, a tool output) may carry a value typed
            # earlier, so all replies are masked, longest value first.
            result = redact(result, known)
            result["action"] = action.model_dump(mode="json")
        files = data.get("files") or {}
        for name in FILE_FIELDS:
            suffix = ".json" if name == "execution_trace_artifact_path" else ".png"
            result[name] = self._save(files.get(name), suffix)
        # A recording lives on the guest; the host cannot open its path.
        result["recording_path"] = None
        return ExecutedAction.model_validate(result)

    def set_current_run_id(self, run_id: str) -> None:
        # Sync in the executor interface: remember it here, tell the guest in the background.
        self._current_run_id = run_id
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._call("set_current_run_id", run_id=run_id))
        task.add_done_callback(_log_task_error)

    async def reset_desktop(self) -> None:
        await self._call("reset_desktop")

    async def context_reset(self) -> None:
        await self._call("context_reset")

    async def session_reset(self, start_url: str | None = None) -> None:
        await self._call("session_reset", start_url=start_url)

    def close_apps_when_done(self, run_id: str) -> None:
        # Remembered here and passed on when the run ends (aclose_run), in order.
        self._close_when_done.add(run_id)

    async def aclose_run(self, run_id: str) -> int:
        self._secrets_used.pop(run_id, None)
        if run_id in self._close_when_done:
            self._close_when_done.discard(run_id)
            await self._call("close_apps_when_done", run_id=run_id)
        return int(await self._call("aclose_run", run_id=run_id) or 0)

    async def live_frame_png(self, run_id: str) -> bytes | None:
        if run_id is None or run_id != self._current_run_id:
            return None
        try:
            response = await self._post("/live_frame", {"run_id": run_id})
        except (GuestUnavailableError, httpx.HTTPStatusError):
            return None
        return response.content if response.status_code == 200 else None

    async def user_input(self, run_id: str, *, input_type: str, **kwargs: Any) -> bool:
        try:
            response = await self._post("/user_input", {"run_id": run_id, "input_type": input_type, **kwargs})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                raise ValueError(exc.response.json().get("detail", "bad input")) from exc
            raise
        return bool(response.json().get("ok"))

    async def aclose(self) -> None:
        self._secrets_used.clear()
        try:
            await self._call("aclose")
        except Exception:  # noqa: BLE001 - the machine may already be gone at shutdown
            logger.debug("remote aclose failed", exc_info=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _batch_failed(
    action: AgentAction,
    index: int,
    detail: str,
    outputs: list[ToolOutput],
    artifact_path: str | None,
    result: ExecutedAction | None = None,
) -> ExecutedAction:
    return ExecutedAction(
        action=action,
        success=False,
        detail=detail,
        artifact_path=artifact_path,
        execution_trace=result.execution_trace if result else None,
        failure_category=(result.failure_category if result else None) or FailureCategory.EXECUTION_ERROR,
        failure_stage=(result.failure_stage if result else None) or LoopStage.EXECUTE,
        batch_failed_index=index,
        tool_outputs=outputs or None,
    )


def _log_task_error(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.warning("guest set_current_run_id failed: %s", task.exception())


def wait_for_guest(base_url: str, token: str, *, timeout_s: float = 120.0, interval_s: float = 1.0,
                   transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """Block until the guest's ``/health`` answers; returns its health JSON."""
    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    with httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {token}"}, timeout=5.0,
                      transport=transport, trust_env=False) as client:
        while True:
            try:
                response = client.get("/health")
                if response.status_code == 200:
                    return response.json()
                last = RuntimeError(f"HTTP {response.status_code}")
            except httpx.HTTPError as exc:
                last = exc
            if time.monotonic() >= deadline:
                raise GuestUnavailableError(f"guest agent at {base_url} did not start within {timeout_s:.0f}s: {last}")
            time.sleep(interval_s)

"""Public Python SDK (KAN-35): run one Operon task from code.

    from operon import Agent
    result = Agent().run("find the weather in Paris", url="https://example.com")

It drives the same in-process path as ``operon run`` (``cli.LocalClient``, i.e. the
front door), so a run made here behaves like one made from the terminal. Safe defaults:
the trust gate is on, mode is ``ask`` and the browser is an isolated batch Chromium.
A step that needs approval is denied unless ``on_approval`` is given and says yes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

logger = logging.getLogger(__name__)

__all__ = ["Agent", "ApprovalRequest", "RunResult"]

STATUS_DENIED = "denied"  # an approval was refused (by the callback, or by default with no callback)


@dataclass(frozen=True)
class ApprovalRequest:
    """What a paused run asks the caller to approve."""

    run_id: str
    reason: str
    kind: str  # "approval" (a trust/confidence gate) or "needs_you" (the model asks to confirm)
    gate: str | None = None


@dataclass(frozen=True)
class RunResult:
    """The outcome of one run. ``status`` is the run status (``succeeded``, ``failed``,
    ``cancelled``, ...) or ``denied`` when an approval was refused."""

    status: str
    final_answer: str | None  # the model's answer; None unless status == "succeeded"
    steps: int
    cost_usd: float | None
    run_id: str
    artifacts_dir: str | None = None
    summary: str = ""
    message: str | None = None  # the model's last word whatever the status (a failure note, say)
    denied_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


ApprovalCallback = Callable[[ApprovalRequest], "bool | Awaitable[bool]"]


class Agent:
    """Runs tasks in this process. One ``run()`` is one task."""

    def __init__(
        self,
        *,
        desktop: bool = False,
        env: str | None = None,
        mode: Literal["ask", "risky", "auto"] = "ask",
        max_steps: int | None = None,
        on_approval: ApprovalCallback | None = None,
        poll_s: float = 0.5,
    ) -> None:
        if mode not in ("ask", "risky", "auto"):
            raise ValueError(f"mode must be 'ask', 'risky' or 'auto', not {mode!r}")
        if env is not None and not desktop:
            raise ValueError("env picks the desktop environment: pass desktop=True with it")
        if env is not None:
            from operon.environments import provider_names

            env = env.strip().lower()  # as environment_name() reads OPERON_ENVIRONMENT
            if env not in provider_names():
                raise ValueError(f"unknown env {env!r}; choose one of {', '.join(provider_names())}")
        self.desktop = desktop
        self.env = env
        self.mode = mode
        self.max_steps = max_steps
        self.on_approval = on_approval
        self._poll_s = poll_s

    def run(self, task: str, *, url: str | None = None) -> RunResult:
        """Run ``task`` and return its result. Use ``arun`` inside a running event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(task, url=url))
        raise RuntimeError("Agent.run() cannot be called from a running event loop: use `await agent.arun(...)`.")

    async def arun(self, task: str, *, url: str | None = None) -> RunResult:
        """Async ``run``. One SDK run at a time per process."""
        from operon.cli import LocalClient
        from operon.core.paths import load_user_env
        from operon.core.runtime import current_runtime
        from operon.core.settings import startup_problems

        runtime = current_runtime()
        if runtime.sdk_run_active:
            raise RuntimeError("Another Operon SDK run is in progress in this process: one run at a time.")
        if runtime.browser_loop is not None or runtime.desktop_loop is not None or runtime.desktop_provider is not None:
            raise RuntimeError(
                "This process already holds Operon agent loops (for example `operon serve` runs here): "
                "the SDK will not tear them down. Run it in its own process."
            )
        runtime.sdk_run_active = True
        env_key = "OPERON_ENVIRONMENT"
        had_env = env_key in os.environ
        saved_env = os.environ.get(env_key)
        started: dict[str, Any] = {}
        client = None
        try:
            load_user_env()  # the key `operon setup` saved; never overrides a value already set
            if self.env:
                os.environ[env_key] = self.env
            errors, _warnings = startup_problems()
            if errors:
                raise RuntimeError("Invalid Operon configuration: " + "; ".join(errors))
            client = LocalClient()
            return await self._drive(client, self._payload(task, url), started)
        finally:
            try:
                if client is not None:
                    if started.get("run_id") and not started.get("done"):
                        # Cancelled or failed partway: do not leave the run going.
                        try:
                            await client.stop(started["run_id"])
                        except Exception:  # noqa: BLE001 - cleanup must not mask the real error
                            pass
                    try:
                        await client.close()
                    except BaseException:
                        logger.warning("SDK cleanup was interrupted: an executor or sandbox may still be running")
                        raise
            finally:
                if client is not None:
                    self._forget_built_loops()  # even if a second cancel interrupted stop/close
                if self.env:
                    if had_env and saved_env is not None:
                        os.environ[env_key] = saved_env
                    else:
                        os.environ.pop(env_key, None)
                runtime.sdk_run_active = False

    def _payload(self, task: str, url: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "intent": task,
            "environment": "desktop" if self.desktop else "browser",
            "permission_mode": self.mode,
        }
        if self.max_steps is not None:
            payload["max_steps"] = self.max_steps
        if url:
            payload["start_url"] = url
        return payload

    @staticmethod
    def _forget_built_loops() -> None:
        # The executors are closed; the next run (maybe on a new event loop) builds fresh ones.
        from operon.core.runtime import current_runtime

        runtime = current_runtime()
        runtime.browser_loop = None
        runtime.desktop_loop = None
        runtime.desktop_provider = None

    async def _ask(self, request: ApprovalRequest) -> bool:
        if self.on_approval is None:
            return False  # never auto-approve by default
        answer = self.on_approval(request)
        if inspect.isawaitable(answer):
            answer = await answer
        return bool(answer)

    async def _drive(self, client, payload: dict[str, Any], started: dict[str, Any]) -> RunResult:
        status = await client.start(payload)
        run_id = status["run_id"]
        started["run_id"] = run_id
        denied: str | None = None
        while not status["done"]:
            waiting = status.get("waiting")
            if waiting is None:
                await asyncio.sleep(self._poll_s)
                status = await client.status(run_id)
                continue
            if waiting["kind"] == "takeover":
                denied = "The run was handed to a person; the SDK cannot take over."
                status = await client.stop(run_id)
                continue
            request = ApprovalRequest(
                run_id=run_id, reason=waiting["reason"], kind=waiting["kind"], gate=waiting.get("gate")
            )
            if await self._ask(request):
                status = await client.approve(run_id)
            else:
                denied = request.reason
                status = await client.stop(run_id)
        started["done"] = True
        report = await client.report(run_id)
        succeeded = report["status"] == "succeeded" and denied is None
        return RunResult(
            status=STATUS_DENIED if denied is not None else report["status"],
            final_answer=report.get("final_message") if succeeded else None,
            message=report.get("final_message"),
            steps=report["steps"],
            cost_usd=report.get("cost_usd"),
            run_id=run_id,
            artifacts_dir=report.get("run_dir"),
            summary=report.get("summary", ""),
            denied_reason=denied,
        )

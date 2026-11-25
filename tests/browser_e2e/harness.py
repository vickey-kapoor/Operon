"""Run a scripted model through Operon's real browser pipeline (headless Chromium, $0).

The real ``AgentLoop`` (capture, perceive, decide, gate, execute, verify, recover), the real
production policy, verifier and recovery layers (the thin harness), the real ``BrowserComputerUseBackend``
and the real ``NativeBrowserExecutor`` on a real headless Chromium in batch mode. Only the
model is a script: ``ScriptedClient`` stands in for the Gemini computer-use HTTP client, as
in ``tests/desktop_e2e``. The thin harness has no second model: the run ends on the script's own "done".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.agent.loop import AgentLoop
from operon.agent.perception.capture import ScreenCaptureService
from operon.agent.tools import enabled_tools, function_declarations
from operon.api.runtime.loops import _loop_layers
from operon.api.runtime.services import RuntimeServices
from operon.clients.gemini_computer_use import GeminiComputerUseHttpClient
from operon.core.contracts.perception import Environment
from operon.core.paths import prompts_dir
from operon.executor.browser_native import NativeBrowserExecutor
from operon.executor.tools import ToolExecutor
from operon.models.common import RunStatus, RunTaskRequest, StepRequest
from operon.store.background_writer import bg_writer
from operon.store.memory import FileBackedMemoryStore
from operon.store.run_store import FileBackedRunStore

from .site import VIEWPORT

TERMINAL = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


def call(name: str, **args: Any) -> dict:
    return {"function_call": {"name": name, "args": args}}


def calls(*items: dict) -> dict:
    """One model turn with several function calls (a multi-call turn)."""
    return {"function_calls": list(items)}


def say(text: str) -> dict:
    return {"text": text}


class ScriptedClient:
    """Stands in for the Gemini computer-use HTTP client: the next scripted turn, a function
    call or text (how the model says it is done). When the script runs out the last turn
    repeats: a model that is stuck keeps trying, it does not claim success."""

    def __init__(self, script: list[dict], *, tools: tuple[str, ...] = ()) -> None:
        self.function_declarations = function_declarations(tools)
        self._script = list(script)
        self._last = say("I cannot make progress.")
        self.turns = 0
        self.served: list[dict] = []

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns += 1
        reply = self._script.pop(0) if self._script else self._last
        self._last = reply
        self.served.append(reply)
        return self._payload(reply)

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[{"role": "user", "parts": [{"text": prompt}]}])

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None, **_) -> dict:
        response = {"error": error} if error else {}
        return {
            "role": "user",
            "parts": [
                {"function_response": {"name": function_name, "response": response}},
                {"inline_data": {"mime_type": "image/png", "data": ""}},
            ],
        }

    @staticmethod
    def _payload(reply: dict) -> dict:
        perception = {"summary": "scripted", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5}
        if "text" in reply:
            text = reply["text"]
            return {
                "perception": perception,
                "action": {"action_type": "wait", "wait_ms": 1000},
                "rationale": text,
                "confidence": 0.4,
                "active_subgoal": "finish",
                "model_content": {"parts": [{"text": text}]},
            }
        if "function_calls" in reply:
            fcs = [item["function_call"] for item in reply["function_calls"]]
            return {
                "perception": perception,
                "function_call": fcs[0],
                "function_calls": fcs,
                "rationale": "several calls",
                "confidence": 0.7,
                "active_subgoal": "several calls",
                "model_content": {"parts": [{"function_call": fc} for fc in fcs]},
            }
        fc = reply["function_call"]
        return {
            "perception": perception,
            "function_call": fc,
            "function_calls": [fc],
            "rationale": fc["name"],
            "confidence": 0.7,
            "active_subgoal": fc["name"],
            "model_content": {"parts": [{"function_call": fc}]},
        }


# -- the model API boundary: the REAL Gemini HTTP client on an httpx.MockTransport -------------


def _usage() -> dict:
    return {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110}


def api_call(name: str, **args: Any) -> httpx.Response:
    """A 200 generateContent reply with one function call."""
    part = {"functionCall": {"name": name, "args": args}}
    body = {"candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP"}],
            "usageMetadata": _usage()}
    return httpx.Response(200, json=body)


def api_text(text: str, finish: str = "STOP") -> httpx.Response:
    """A 200 reply with only text and the given finishReason (how the model says it is done)."""
    body = {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}],
            "usageMetadata": _usage()}
    return httpx.Response(200, json=body)


def api_status(code: int, retry_after: str | None = None) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(code, headers=headers, text=f"scripted error {code}")


def api_blocked(reason: str = "SAFETY") -> httpx.Response:
    """A 200 reply with no candidates and a promptFeedback blockReason."""
    return httpx.Response(200, json={"promptFeedback": {"blockReason": reason}, "usageMetadata": _usage()})


def count_inline_images(request_body: dict) -> int:
    """How many inline screenshot parts a recorded generateContent request carried."""
    return sum(
        1
        for content in request_body.get("contents", [])
        for part in content.get("parts", [])
        if "inline_data" in part or "inlineData" in part
    )


@dataclass
class ApiLog:
    """What the scripted API saw: each request's JSON body, in order."""

    requests: list[dict] = field(default_factory=list)


def _api_client(script: list[httpx.Response], log: ApiLog, tools: tuple[str, ...]) -> GeminiComputerUseHttpClient:
    queue = list(script)
    last = api_text("I cannot make progress.")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal last
        log.requests.append(json.loads(request.content))
        if queue:
            last = queue.pop(0)
        return last  # an exhausted script repeats its last reply, like ScriptedClient

    return GeminiComputerUseHttpClient(
        model="gemini-e2e",
        api_key="fake-test-key",
        api_base_url="http://gemini.invalid",
        retry_backoff_seconds=0.0,
        function_declarations=function_declarations(tools),
        transport=httpx.MockTransport(handler),
    )


@dataclass
class Result:
    status: RunStatus
    steps: int
    client: Any  # ScriptedClient, or the real GeminiComputerUseHttpClient on the api path
    run_dir: Path
    state: Any  # the saved AgentState
    pauses: list[str] | None = None  # the text of each pause the run answered (on_pause)
    api: ApiLog | None = None  # the requests the scripted API saw (api path only)

    @property
    def stop_reason(self) -> str | None:
        reason = getattr(self.state, "stop_reason", None)
        return reason.value if reason is not None else None

    def trace(self) -> list[dict]:
        path = self.run_dir / "run.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_script(
    tmp_path: Path,
    intent: str,
    script: list[dict],
    *,
    start_url: str,
    max_steps: int = 12,
    tools: bool | str = False,
    permission_mode: str = "auto",
    on_pause: str | None = None,
    api: list[httpx.Response] | None = None,
) -> Result:
    """Run ``intent`` through the real loop on headless Chromium.

    ``on_pause``: "approve" or "decline" answers each pause the run reaches (resume_run), as the
    person would; None leaves a pause unanswered (the run stays waiting).

    ``tools``: True enables the "on" tool set, "all" adds run_shell. ``api``: instead of
    ``script``, serve these raw generateContent replies to the real Gemini HTTP client
    (``Result.api`` records each request)."""
    executor = NativeBrowserExecutor(
        artifact_dir=tmp_path / "artifacts",
        viewport_width=VIEWPORT[0],
        viewport_height=VIEWPORT[1],
        headless=True,
        record_video=False,
    )
    tool_names = enabled_tools("all" if tools == "all" else "on") if tools else ()
    log = ApiLog() if api is not None else None
    client: Any = _api_client(api, log, tool_names) if api is not None and log is not None else ScriptedClient(
        script, tools=tool_names)
    backend = BrowserComputerUseBackend(
        client=client,
        prompt_path=prompts_dir() / "browser_computer_use_prompt.txt",
        browser_runtime=executor,
        tools=tool_names,
    )
    run_store = FileBackedRunStore(tmp_path / "runs")
    memory_store = FileBackedMemoryStore(tmp_path / "memory")
    layers = _loop_layers(services=RuntimeServices(perception_service=backend, policy_delegate=backend))
    loop = AgentLoop(
        capture_service=ScreenCaptureService(executor=executor, root_dir=run_store.root_dir),
        perception_service=backend,
        run_store=run_store,
        executor=ToolExecutor(executor, tool_names) if tool_names else executor,
        memory_store=memory_store,
        environment=Environment.BROWSER,
        # Production wiring (the thin policy, own-done verifier and recovery), so a change there is followed here.
        **layers,
    )
    try:
        run = await loop.start_run(
            RunTaskRequest(intent=intent, start_url=start_url, headless=True, max_steps=max_steps,
                           permission_mode=permission_mode)
        )
        status, steps = run.status, 0
        pauses: list[str] = []
        # One call past the budget: the loop ends a run that has spent its steps on the next call.
        for _ in range(max_steps + 1):
            if status in TERMINAL:
                break
            response = await loop.step_run(StepRequest(run_id=run.run_id))
            status, steps = response.status, response.step_count
            if status is RunStatus.WAITING_FOR_USER and on_pause is not None:
                paused = await run_store.get_run(run.run_id)
                pauses.append(getattr(paused, "pause_reason", None) or "")
                response = await loop.resume_run(run.run_id, approve_gated_action=on_pause == "approve")
                status = response.status
        await bg_writer.flush()
        state = await run_store.get_run(run.run_id)
        return Result(status, steps, client, run_store.root_dir / run.run_id, state, pauses, log)
    finally:
        await executor.aclose()

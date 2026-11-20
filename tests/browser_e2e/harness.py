"""Run a scripted model through Operon's real browser pipeline (headless Chromium, $0).

The real ``AgentLoop`` (capture, perceive, decide, gate, execute, verify, recover), the real
production policy, verifier and recovery layers (the thin harness), the real ``BrowserComputerUseBackend``
and the real ``NativeBrowserExecutor`` on a real headless Chromium in batch mode. Only the
model is a script: ``ScriptedClient`` stands in for the Gemini computer-use HTTP client, as
in ``tests/desktop_e2e``. The thin harness has no second model: the run ends on the script's own "done".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.agent.loop import AgentLoop
from operon.agent.perception.capture import ScreenCaptureService
from operon.agent.tools import enabled_tools, function_declarations
from operon.api.runtime.loops import _loop_layers
from operon.api.runtime.services import RuntimeServices
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


@dataclass
class Result:
    status: RunStatus
    steps: int
    client: ScriptedClient
    run_dir: Path
    state: Any  # the saved AgentState

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
    tools: bool = False,
) -> Result:
    """Run ``intent`` through the real loop on headless Chromium."""
    executor = NativeBrowserExecutor(
        artifact_dir=tmp_path / "artifacts",
        viewport_width=VIEWPORT[0],
        viewport_height=VIEWPORT[1],
        headless=True,
        record_video=False,
    )
    tool_names = enabled_tools("on") if tools else ()
    client = ScriptedClient(script, tools=tool_names)
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
                           permission_mode="auto")
        )
        status, steps = run.status, 0
        # One call past the budget: the loop ends a run that has spent its steps on the next call.
        for _ in range(max_steps + 1):
            if status in TERMINAL:
                break
            response = await loop.step_run(StepRequest(run_id=run.run_id))
            status, steps = response.status, response.step_count
        await bg_writer.flush()
        state = await run_store.get_run(run.run_id)
        return Result(status, steps, client, run_store.root_dir / run.run_id, state)
    finally:
        await executor.aclose()

"""KAN-93: free end-to-end desktop tests on a real Windows desktop in Windows Sandbox.

The real agent loop (thin harness), the real guest agent and a real Windows desktop in
Windows Sandbox (KAN-80): the mouse clicks and the keyboard types for real. Only the
model is a script, so a run costs nothing. Each task is graded from machine state: the
sandbox's writable ``exchange`` folder is mapped to the same path on the host, so a file
the guest writes there is read straight from the host.

The sandbox maps the source folder the tests import (``source_dir()``) read-only, so the
guest runs the same code as the test run.

Opt in: ``pytest tests/desktop_e2e_windows -m windows_sandbox -o addopts=""``. Needs
Windows Sandbox turned on; the sandbox takes about two minutes to boot. One sandbox
serves every test in the module (Windows allows only one at a time).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from operon.agent.backends.desktop_computer_use import (
    DESKTOP_ENVIRONMENT,
    DesktopComputerUseBackend,
    launch_app_function,
)
from operon.agent.perception.capture import ScreenCaptureService
from operon.agent.policy.thin import OwnDoneVerifier, ThinPolicy, ThinRecovery
from operon.core.contracts.perception import Environment
from operon.core.paths import prompts_dir
from operon.environments.windows_sandbox import WindowsSandboxProvider
from operon.models.common import RunStatus, RunTaskRequest, StepRequest
from operon.store.background_writer import bg_writer
from operon.store.memory import FileBackedMemoryStore
from operon.store.run_store import FileBackedRunStore

pytestmark = pytest.mark.windows_sandbox

# Gemini coordinates are a 0-999 grid. Notepad opens in the middle of the screen, so this
# point (the middle) is inside its text area.
CENTER = {"x": 500, "y": 500}
MAX_STEPS = 14


def call(name: str, **args: Any) -> dict:
    return {"function_call": {"name": name, "args": args}}


def say(text: str) -> dict:
    return {"text": text}


class ScriptedClient:
    """Stands in for the Gemini computer-use HTTP client: replies with the next scripted
    turn (a function call, or text, which is how the model says it is done)."""

    environment = DESKTOP_ENVIRONMENT

    def __init__(self, script: list[dict]) -> None:
        self.function_declarations = [launch_app_function()]
        self._script = list(script)
        self.turns = 0

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns += 1
        reply = self._script.pop(0) if self._script else say("The task is complete.")
        return self._payload(reply)

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[{"role": "user", "parts": [{"text": prompt}]}])

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
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


@pytest.fixture(scope="module")
def sandbox():
    """One Windows Sandbox for the whole module."""
    if sys.platform != "win32":
        pytest.skip("Windows Sandbox runs only on Windows")
    # tests/conftest.py points OPERON_CONFIG_DIR at a relative ".." path inside the repo, which
    # the sandbox cannot map; use the provider's normal folder in the user's home instead.
    provider = WindowsSandboxProvider(work_dir=Path.home() / ".operon" / "windows-sandbox")
    check = provider.check()
    if not check.ok:
        pytest.skip("windows-sandbox not ready: " + " ".join(check.problems))
    provider.start()
    try:
        yield provider
    finally:
        provider.stop()


async def run_script(provider, tmp_path: Path, intent: str, script: list[dict]):
    """Run ``intent`` through the real thin-harness loop on the sandbox's desktop."""
    executor = provider.build_executor()
    client = ScriptedClient(script)
    backend = DesktopComputerUseBackend(
        client=client, prompt_path=prompts_dir() / "desktop_computer_use_prompt.txt",
    )
    run_store = FileBackedRunStore(tmp_path / "runs")
    memory_store = FileBackedMemoryStore(tmp_path / "memory")
    from operon.agent.loop import AgentLoop

    loop = AgentLoop(
        capture_service=ScreenCaptureService(executor=executor, root_dir=run_store.root_dir),
        perception_service=backend,
        run_store=run_store,
        policy_service=ThinPolicy(backend),
        executor=executor,
        verifier_service=OwnDoneVerifier(),
        recovery_manager=ThinRecovery(),
        memory_store=memory_store,
        environment=Environment.DESKTOP,
    )
    try:
        run = await loop.start_run(RunTaskRequest(intent=intent, max_steps=MAX_STEPS))
        status, steps = run.status, 0
        while steps < MAX_STEPS and status not in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
            response = await loop.step_run(StepRequest(run_id=run.run_id))
            status, steps = response.status, response.step_count
        await bg_writer.flush()
        return status, steps, client
    finally:
        await executor.aclose()


def test_the_host_sees_the_real_sandbox_desktop(sandbox):
    """The host captures the sandbox's actual screen, not a blank frame."""

    async def grab():
        executor = sandbox.build_executor()
        try:
            return await executor.capture()
        finally:
            await executor.aclose()

    frame = asyncio.run(grab())
    assert frame.width > 800 and frame.height > 500
    assert Path(frame.artifact_path).stat().st_size > 5_000


def test_a_typed_command_writes_the_file(sandbox, tmp_path):
    """launch_app and typing on a real desktop: the console the agent opens must write the file.

    The sandbox image ships no Notepad (it is a Store app: ``where notepad`` finds nothing in the
    guest, and the launch opens no window), so a real text editor cannot be launched there. ``cmd``
    is on every Windows image and takes keyboard input the same way, so it proves launch, focus and
    typing end to end.
    """
    target = sandbox.exchange / "operon-e2e-note.txt"
    target.unlink(missing_ok=True)
    script = [
        call("launch_app", app="cmd"),
        call("wait_5_seconds"),
        # The new console takes the keyboard focus: type without clicking.
        call("type", text=f'echo hello from the scripted model> "{target}"', press_enter=True),
        call("wait_5_seconds"),
        say("The task is complete."),
    ]
    status, _steps, client = asyncio.run(
        run_script(sandbox, tmp_path, "Write a note with a command and save it", script)
    )
    assert status is RunStatus.SUCCEEDED
    assert client.turns == len(script)
    assert target.exists(), "the typed command did not run in the guest's console"
    assert target.read_text(encoding="utf-8", errors="replace").strip() == "hello from the scripted model"


def test_an_app_that_is_not_in_the_catalog_never_starts(sandbox, tmp_path):
    """The executor refuses anything outside the catalog, so the run cannot claim a launch."""
    evil = sandbox.exchange / "operon-e2e-evil.txt"
    evil.unlink(missing_ok=True)
    script = [call("launch_app", app=f'cmd /c echo x > "{evil}"'), call("wait_5_seconds")]
    asyncio.run(run_script(sandbox, tmp_path, "Open an app", script))
    assert not evil.exists()

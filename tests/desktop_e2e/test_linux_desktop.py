"""KAN-166: free end-to-end desktop tests on the Linux desktop container.

The real agent loop (thin harness), the real guest agent and the real XFCE desktop in
Docker (KAN-80, ``docker/linux-desktop``): the mouse clicks and the keyboard types for
real. Only the model is a script, so a run costs nothing. Each task is graded from
the machine's own state (a file the typed command wrote, a process that is running),
never from what the run claims.

Opt in: ``pytest tests/desktop_e2e -m desktop_container -o addopts=""``. Needs Docker;
the first run builds the image (a few minutes). CI runs it as the "Desktop (Linux
container)" job.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
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
from operon.environments import get_provider
from operon.models.common import RunStatus, RunTaskRequest, StepRequest
from operon.store.background_writer import bg_writer
from operon.store.memory import FileBackedMemoryStore
from operon.store.run_store import FileBackedRunStore

pytestmark = pytest.mark.desktop_container

# The desktop is 1280x800 (OPERON_SCREEN); Gemini coordinates are a 0-999 grid. New windows
# open at the top left, so this point (about 384,240 px) is inside them.
CENTER = {"x": 300, "y": 300}
MAX_STEPS = 12


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


@pytest.fixture(scope="session")
def desktop():
    """One Linux desktop container for the whole module."""
    # In CI a missing desktop is a failure, never a quiet skip.
    unavailable = pytest.fail if os.environ.get("OPERON_E2E_REQUIRE_DESKTOP") else pytest.skip
    if shutil.which("docker") is None:
        unavailable("Docker is not installed")
    mp = pytest.MonkeyPatch()
    mp.setenv("OPERON_DESKTOP_PLATFORM", "linux")
    provider = get_provider("linux-container")
    check = provider.check()
    if not check.ok:
        mp.undo()
        unavailable("linux-container not ready: " + " ".join(check.problems))
    provider.start()
    try:
        yield provider
    finally:
        provider.stop()
        mp.undo()


def sh(provider, command: str) -> subprocess.CompletedProcess:
    """A shell command inside the desktop container (to grade from its own state)."""
    return subprocess.run(
        ["docker", "exec", provider.container, "sh", "-c", command],
        capture_output=True, text=True, timeout=30, check=False,
    )


async def run_script(provider, tmp_path: Path, intent: str, script: list[dict], grade=None):
    """Run ``intent`` through the real thin-harness loop on the container's desktop.
    ``grade`` runs against the machine as the run leaves it, before the executor closes
    (which closes the apps the run opened); its result is returned last."""
    executor = provider.build_executor()
    client = ScriptedClient(script)
    backend = DesktopComputerUseBackend(
        client=client, prompt_path=prompts_dir() / "desktop_computer_use_prompt_linux.txt",
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
        return status, steps, client, (grade() if grade is not None else None)
    finally:
        await executor.aclose()


def test_type_a_command_in_the_terminal(desktop, tmp_path):
    """launch_app, click and type on the real desktop: the typed command must run."""
    marker = "/tmp/operon-e2e-terminal.txt"
    sh(desktop, f"rm -f {marker}")
    script = [
        call("launch_app", app="terminal"),
        call("wait_5_seconds"),
        call("type_text_at", **CENTER, text=f"echo operon-e2e-ok > {marker}", press_enter=True),
        call("wait_5_seconds"),
        say("The task is complete."),
    ]
    status, _steps, client, written = asyncio.run(
        run_script(desktop, tmp_path, "Run a command in the terminal", script,
                   grade=lambda: sh(desktop, f"cat {marker}").stdout.strip())
    )
    assert status is RunStatus.SUCCEEDED
    assert client.turns == 5
    assert written == "operon-e2e-ok"


def test_launch_a_second_app(desktop, tmp_path):
    """The text editor opens as its own window and keeps running."""
    script = [
        call("launch_app", app="text editor"),
        call("wait_5_seconds"),
        call("type_text_at", **CENTER, text="hello from the scripted model"),
        say("The task is complete."),
    ]
    status, _steps, _client, running = asyncio.run(
        run_script(desktop, tmp_path, "Open the text editor", script,
                   grade=lambda: sh(desktop, "pgrep -x mousepad").returncode == 0)
    )
    assert status is RunStatus.SUCCEEDED
    assert running


def test_an_app_that_is_not_in_the_catalog_never_starts(desktop, tmp_path):
    """The executor refuses anything outside the catalog, so the run cannot claim a launch."""
    sh(desktop, "rm -f /tmp/operon-e2e-evil")
    script = [call("launch_app", app="touch /tmp/operon-e2e-evil")]
    asyncio.run(run_script(desktop, tmp_path, "Open an app", script))
    assert sh(desktop, "test -e /tmp/operon-e2e-evil").returncode != 0

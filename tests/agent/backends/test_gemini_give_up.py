"""KAN-202: Gemini computer use can give up, on the browser and the desktop.

Before, a text reply after Gemini had acted always counted as "done", so "I can't do
this" was recorded as a success. The backends now declare a give_up function; calling
it ends the run as failed (model gave up) with the model's reason saved."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from operon.agent.backends.browser_computer_use import (
    GIVE_UP,
    GIVE_UP_FUNCTION,
    BrowserComputerUseBackend,
)
from operon.agent.backends.desktop_computer_use import (
    GEMINI_LAUNCH_APP_FUNCTION,
    DesktopComputerUseBackend,
)
from operon.agent.backends.model_adapter.backend import GAVE_UP_PREFIX
from operon.agent.loop import AgentLoop
from operon.agent.policy.thin import OwnDoneVerifier, ThinPolicy, ThinRecovery
from operon.api.runtime.services import desktop_prompt_name
from operon.clients.gemini_computer_use import GeminiComputerUseHttpClient
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import (
    FailureCategory,
    RunFailureCause,
    RunStatus,
    RunTaskRequest,
    StepRequest,
)
from operon.models.execution import ExecutedAction
from operon.models.policy import AgentAction
from operon.store.run_store import FileBackedRunStore


def _reply(*calls: dict, text: str = "") -> dict:
    """A normalized Gemini turn, as the HTTP client returns it."""
    parts = ([{"text": text}] if text else []) + [{"function_call": c} for c in calls]
    return {
        "perception": {"summary": text or "Computer Use step evaluated.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "function_call": calls[0],
        "function_calls": list(calls),
        "rationale": text or "Computer Use step evaluated.",
        "confidence": 0.7,
        "active_subgoal": calls[0]["name"],
        "model_content": {"role": "model", "parts": parts},
        "raw_candidate": {"finish_reason": "STOP", "had_function_call": True},
    }


def _click() -> dict:
    return {"name": "click_at", "args": {"x": 500, "y": 500}}


def _give_up(reason: str | None = "the site needs a login I do not have") -> dict:
    return {"name": GIVE_UP, "args": {} if reason is None else {"reason": reason}}


class ScriptedClient:
    def __init__(self, replies: list[dict], *, environment: str = "ENVIRONMENT_BROWSER", declarations=None) -> None:
        self._replies = list(replies)
        self.environment = environment
        self.function_declarations = list(declarations or [])
        self.turns = 0

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns += 1
        return self._replies.pop(0)

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[])

    @staticmethod
    def build_function_response_content(*, function_name, screenshot_path, current_url, error=None) -> dict:
        return {
            "role": "user",
            "parts": [{"function_response": {"name": function_name, "response": {}}}, {"text": "screen"}],
        }


def _loop(tmp_path: Path, backend) -> tuple[AgentLoop, FileBackedRunStore, Mock]:
    run_store = FileBackedRunStore(root_dir=tmp_path / "runs")

    async def capture(record) -> CaptureFrame:
        path = tmp_path / "frames" / f"{record.run_id}-{record.step_count + 1}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1000, 1000), "white").save(path)
        return CaptureFrame(artifact_path=str(path), width=1000, height=1000)

    async def execute(action: AgentAction) -> ExecutedAction:
        return ExecutedAction(action=action, success=True, detail="ok")

    executor = Mock()
    executor.execute = AsyncMock(side_effect=execute)
    loop = AgentLoop(
        capture_service=SimpleNamespace(capture=capture),
        perception_service=backend,
        run_store=run_store,
        policy_service=ThinPolicy(backend),
        executor=executor,
        verifier_service=OwnDoneVerifier(gave_up_prefix=GAVE_UP_PREFIX),
        recovery_manager=ThinRecovery(),
    )
    return loop, run_store, executor


def _browser(replies: list[dict]) -> tuple[BrowserComputerUseBackend, ScriptedClient]:
    client = ScriptedClient(replies)
    backend = BrowserComputerUseBackend(client=client, prompt_path=prompts_dir() / "browser_computer_use_prompt.txt")
    return backend, client


def _desktop(replies: list[dict]) -> tuple[DesktopComputerUseBackend, ScriptedClient]:
    client = ScriptedClient(replies, environment="ENVIRONMENT_DESKTOP", declarations=[GEMINI_LAUNCH_APP_FUNCTION])
    backend = DesktopComputerUseBackend(client=client, prompt_path=prompts_dir() / desktop_prompt_name())
    return backend, client


async def _run(loop: AgentLoop, steps: int) -> tuple[str, RunStatus]:
    run = await loop.start_run(RunTaskRequest(intent="Buy the blue shirt", max_steps=10))
    status = RunStatus.RUNNING
    for _ in range(steps):
        status = (await loop.step_run(StepRequest(run_id=run.run_id))).status
    return run.run_id, status


def _rows(run_store: FileBackedRunStore, run_id: str) -> list[dict]:
    text = (run_store.root_dir / run_id / "run.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.mark.parametrize("make", [_browser, _desktop], ids=["browser", "desktop"])
async def test_give_up_ends_the_run_as_failed_with_the_reason(tmp_path, make) -> None:
    backend, client = make([_reply(_click()), _reply(_give_up())])
    loop, run_store, executor = _loop(tmp_path, backend)

    run_id, status = await _run(loop, 2)

    assert status is RunStatus.FAILED
    assert client.turns == 2
    ran = [call.args[0].action_type.value for call in executor.execute.await_args_list]
    assert ran[0] == "click" and all(kind == "stop" for kind in ran[1:])
    state = await run_store.get_run(run_id)
    assert state is not None
    assert state.failure_cause is RunFailureCause.MODEL_GAVE_UP
    assert "login I do not have" in (state.failure_detail or "")
    step = _rows(run_store, run_id)[-1]
    assert step["failure"]["category"] == FailureCategory.MODEL_GAVE_UP.value


@pytest.mark.parametrize("make", [_browser, _desktop], ids=["browser", "desktop"])
async def test_a_text_reply_after_acting_still_counts_as_done(tmp_path, make) -> None:
    done = {
        "perception": {"summary": "Bought it.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "action": {"action_type": "wait", "wait_ms": 1000},
        "rationale": "Bought it.",
        "confidence": 0.4,
        "model_content": {"role": "model", "parts": [{"text": "Bought it."}]},
        "raw_candidate": {"finish_reason": "STOP", "had_function_call": False},
    }
    backend, _ = make([_reply(_click()), done])
    loop, run_store, _ = _loop(tmp_path, backend)

    run_id, status = await _run(loop, 2)

    assert status is RunStatus.SUCCEEDED
    state = await run_store.get_run(run_id)
    assert state is not None and state.failure_cause is None


async def test_give_up_with_other_calls_runs_none_of_them(tmp_path) -> None:
    backend, _ = _browser([_reply(_click(), _give_up("the shirt is sold out"))])
    loop, run_store, executor = _loop(tmp_path, backend)

    run_id, status = await _run(loop, 1)

    assert status is RunStatus.FAILED
    assert all(call.args[0].action_type.value == "stop" for call in executor.execute.await_args_list)
    state = await run_store.get_run(run_id)
    assert state is not None and "sold out" in (state.failure_detail or "")


async def test_give_up_without_a_reason_uses_the_models_text(tmp_path) -> None:
    backend, _ = _browser([_reply(_give_up(None), text="The store is closed today.")])
    loop, run_store, _ = _loop(tmp_path, backend)

    run_id, status = await _run(loop, 1)

    assert status is RunStatus.FAILED
    state = await run_store.get_run(run_id)
    assert state is not None and "The store is closed today." in (state.failure_detail or "")


async def test_give_up_with_no_reason_and_no_text_says_so(tmp_path) -> None:
    backend, _ = _browser([_reply(_give_up(None))])
    loop, run_store, _ = _loop(tmp_path, backend)

    run_id, _ = await _run(loop, 1)

    state = await run_store.get_run(run_id)
    assert state is not None
    assert "no reason given" in (state.failure_detail or "")
    assert "Computer Use step evaluated" not in (state.failure_detail or "")


async def test_give_up_marked_for_confirmation_is_not_held(tmp_path) -> None:
    call = {"name": GIVE_UP, "args": {"reason": "blocked", "safety_decision": {"decision": "require_confirmation"}}}
    backend, _ = _browser([_reply(call)])
    loop, run_store, _ = _loop(tmp_path, backend)

    run_id, status = await _run(loop, 1)

    assert status is RunStatus.FAILED
    assert not backend.has_held_actions(run_id)


def test_the_real_client_sends_give_up_on_browser_and_desktop() -> None:
    browser = GeminiComputerUseHttpClient(model="m", api_key="k")
    BrowserComputerUseBackend(client=browser, prompt_path=prompts_dir() / "browser_computer_use_prompt.txt")
    desktop = GeminiComputerUseHttpClient(
        model="m", api_key="k", environment="ENVIRONMENT_DESKTOP", function_declarations=[GEMINI_LAUNCH_APP_FUNCTION]
    )
    DesktopComputerUseBackend(client=desktop, prompt_path=prompts_dir() / desktop_prompt_name())

    for client, names in ((browser, [GIVE_UP]), (desktop, ["launch_app", GIVE_UP])):
        tools = client._build_payload(contents=[])["tools"]
        declared = [d["name"] for tool in tools for d in tool.get("functionDeclarations", [])]
        assert declared == names


def test_give_up_is_declared_once_even_when_the_client_is_shared() -> None:
    client = GeminiComputerUseHttpClient(model="m", api_key="k")
    for _ in range(2):
        BrowserComputerUseBackend(client=client, prompt_path=prompts_dir() / "browser_computer_use_prompt.txt")
    assert [d["name"] for d in client.function_declarations] == [GIVE_UP]
    assert GIVE_UP_FUNCTION["parameters"]["required"] == ["reason"]


@pytest.mark.parametrize(
    "name",
    ["browser_computer_use_prompt.txt", "desktop_computer_use_prompt.txt", "desktop_computer_use_prompt_linux.txt"],
)
def test_every_gemini_prompt_tells_the_model_about_give_up(name) -> None:
    assert "give_up" in (prompts_dir() / name).read_text(encoding="utf-8")

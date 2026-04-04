"""Smoke test for the browser Computer Use runtime path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.agent.browser_computer_use import BrowserComputerUseBackend
from src.agent.capture import ScreenCaptureService
from src.agent.loop import AgentLoop
from src.agent.policy_coordinator import PolicyCoordinator
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.clients.gemini import PlaceholderGeminiClient
from src.models.capture import CaptureFrame
from src.models.common import RunStatus, RunTaskRequest, StepRequest
from src.models.state import AgentState
from src.store.memory import FileBackedMemoryStore
from src.store.run_store import FileBackedRunStore


def _local_test_dir(name: str) -> Path:
    path = Path(".test-artifacts") / f"{name}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class StubComputerUseClient:
    def __init__(self) -> None:
        self.calls = 0

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        self.calls += 1
        return self._tool_response()

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.calls += 1
        return self._tool_response()

    @staticmethod
    def build_function_response_content(
        *,
        function_name: str,
        screenshot_path: str,
        current_url: str | None,
        error: str | None = None,
    ) -> dict:
        return {"role": "user", "parts": [{"function_response": {"name": function_name, "response": {"url": current_url or ""}}}]}

    @staticmethod
    def _tool_response() -> dict:
        return {
            "perception": {
                "summary": "Example page visible",
                "page_hint": "unknown",
                "visible_elements": [],
                "confidence": 0.9,
            },
            "function_call": {
                "name": "click_at",
                "args": {"x": 500, "y": 250},
            },
            "model_content": {"parts": [{"function_call": {"name": "click_at", "args": {"x": 500, "y": 250}}}]},
            "rationale": "Click the main content area.",
            "confidence": 0.8,
            "active_subgoal": "inspect page",
        }


class StubBrowserExecutor:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.current_run_id: str | None = None
        self.executed_actions = []

    def set_current_run_id(self, run_id: str) -> None:
        self.current_run_id = run_id

    async def reset_desktop(self) -> None:
        return None

    async def capture(self) -> CaptureFrame:
        path = self.artifact_dir / f"capture-{uuid4().hex[:8]}.png"
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xa5\x1d\xb8"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return CaptureFrame(
            artifact_path=str(path),
            width=1200,
            height=800,
            mime_type="image/png",
        )

    async def execute(self, action):
        self.executed_actions.append(action)
        after_path = self.artifact_dir / f"after-{uuid4().hex[:8]}.png"
        after_path.write_bytes(b"after")
        from src.models.execution import ExecutedAction

        return ExecutedAction(
            action=action,
            success=True,
            detail=f"Executed {action.action_type.value}",
            artifact_path=str(after_path),
        )


@pytest.mark.asyncio
async def test_browser_computer_use_single_step_smoke() -> None:
    root = _local_test_dir("browser-cu-smoke")
    prompts_dir = root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompts_dir / "browser_computer_use_prompt.txt"
    prompt_path.write_text(
        "intent={intent}\nsubgoal={current_subgoal}\nstep={step_count}\nprev={previous_summary}\nretry={retry_counts}\nhints={advisory_hints}",
        encoding="utf-8",
    )

    executor = StubBrowserExecutor(root / "captures")
    executor.artifact_dir.mkdir(parents=True, exist_ok=True)
    run_store = FileBackedRunStore(root_dir=root / "runs")
    memory_store = FileBackedMemoryStore(root_dir=root / "runs")
    client = StubComputerUseClient()
    backend = BrowserComputerUseBackend(
        client=client,
        prompt_path=prompt_path,
    )
    loop = AgentLoop(
        capture_service=ScreenCaptureService(
            executor=executor,
            root_dir=root / "runs",
        ),
        perception_service=backend,
        run_store=run_store,
        policy_service=PolicyCoordinator(
            delegate=backend,
            memory_store=memory_store,
        ),
        executor=executor,
        verifier_service=DeterministicVerifierService(gemini_client=PlaceholderGeminiClient()),
        recovery_manager=RuleBasedRecoveryManager(),
        memory_store=memory_store,
        gemini_client=PlaceholderGeminiClient(),
    )

    with patch.object(loop, "_maybe_video_verify", new_callable=AsyncMock, return_value=None):
        start = await loop.start_run(RunTaskRequest(intent="Inspect the current browser page"))
        step = await loop.step_run(StepRequest(run_id=start.run_id))

    assert step.status in {RunStatus.RUNNING, RunStatus.SUCCEEDED}
    assert executor.current_run_id == start.run_id
    assert executor.executed_actions
    assert executor.executed_actions[0].action_type.value == "click"
    assert client.calls == 1

    run_dir = root / "runs" / start.run_id / "step_1"
    assert (run_dir / "before.png").exists()
    assert (run_dir / "computer_use_prompt.txt").exists()
    assert (run_dir / "computer_use_raw.json").exists()
    assert (run_dir / "computer_use_parsed.json").exists()
    assert (root / "runs" / start.run_id / "run.jsonl").exists()


@pytest.mark.asyncio
async def test_browser_computer_use_retries_weak_wait_response_once() -> None:
    root = _local_test_dir("browser-cu-retry")
    prompt_path = root / "browser_computer_use_prompt.txt"
    prompt_path.write_text(
        "intent={intent}\nsubgoal={current_subgoal}\nstep={step_count}\nprev={previous_summary}\nretry={retry_counts}\nhints={advisory_hints}",
        encoding="utf-8",
    )

    class WeakThenToolClient:
        def __init__(self) -> None:
            self.calls = 0

        async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
            self.calls += 1
            return self._response()

        async def generate_turn(self, *, contents: list[dict]) -> dict:
            self.calls += 1
            return self._response()

        @staticmethod
        def build_function_response_content(
            *,
            function_name: str,
            screenshot_path: str,
            current_url: str | None,
            error: str | None = None,
        ) -> dict:
            return {"role": "user", "parts": [{"function_response": {"name": function_name, "response": {"url": current_url or ""}}}]}

        def _response(self) -> dict:
            if self.calls == 1:
                return {
                    "perception": {
                        "summary": "Computer Use step evaluated.",
                        "page_hint": "unknown",
                        "visible_elements": [],
                        "confidence": 0.5,
                    },
                    "action": {"action_type": "wait", "wait_ms": 1000},
                    "rationale": "Need more context",
                    "confidence": 0.4,
                    "active_subgoal": "gather more browser context",
                    "model_content": {"parts": [{"text": "Need more context"}]},
                }
            return {
                "perception": {
                    "summary": "Prominent page content visible",
                    "page_hint": "unknown",
                    "visible_elements": [],
                    "confidence": 0.8,
                },
                "function_call": {"name": "scroll_at", "args": {"x": 500, "y": 500, "direction": "down", "magnitude": 600}},
                "model_content": {"parts": [{"function_call": {"name": "scroll_at", "args": {"x": 500, "y": 500, "direction": "down", "magnitude": 600}}}]},
                "rationale": "Scroll to reveal more content.",
                "confidence": 0.8,
                "active_subgoal": "inspect page",
            }

    backend = BrowserComputerUseBackend(
        client=WeakThenToolClient(),
        prompt_path=prompt_path,
    )
    screenshot = CaptureFrame(
        artifact_path=str(root / "before.png"),
        width=1200,
        height=800,
        mime_type="image/png",
    )
    Path(screenshot.artifact_path).write_bytes(b"png")
    state = AgentState(run_id="run-1", intent="Inspect the current browser page", status=RunStatus.RUNNING)

    perception = await backend.perceive(screenshot, state)
    decision = await backend.choose_action(state, perception)

    assert decision.action.action_type.value == "scroll"

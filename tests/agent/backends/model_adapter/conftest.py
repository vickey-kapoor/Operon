"""Shared helpers for model-adapter tests: sample replies, frames and states."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction
from operon.models.state import AgentState

REPLIES = Path(__file__).parent / "replies"


def sample(name: str) -> dict:
    """A sample chat-completions reply (vLLM response shape) from ``replies/<name>.json``."""
    return json.loads((REPLIES / f"{name}.json").read_text(encoding="utf-8"))


def chat_reply(content: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240},
    }


def tool_call(name: str, call_id: str = "call_1", **arguments) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class ScriptedChat:
    """Stands in for ``OpenAICompatClient``: returns the scripted replies in order."""

    def __init__(self, replies: list[dict | Exception]) -> None:
        self._replies = replies
        self.requests: list[dict] = []

    async def chat(self, messages: list[dict], *, tools: list[dict] | None) -> dict:
        self.requests.append({"messages": messages, "tools": tools})
        reply = self._replies[min(len(self.requests), len(self._replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def latest_usage(self):
        return None


@pytest.fixture
def frame(tmp_path: Path):
    def make(name: str = "frame.png", size: tuple[int, int] = (1280, 800)) -> CaptureFrame:
        step = tmp_path / name.removesuffix(".png")
        step.mkdir(parents=True, exist_ok=True)
        path = step / "frame.png"
        Image.new("RGB", size, "white").save(path)
        return CaptureFrame(artifact_path=str(path), width=size[0], height=size[1])

    return make


def state(*executed: tuple[ActionType, bool], intent: str = "Open the Products page") -> AgentState:
    return AgentState(
        run_id="run-1",
        intent=intent,
        status=RunStatus.RUNNING,
        action_history=[
            ExecutedAction(
                action=AgentAction(action_type=a, x=1, y=1) if a is ActionType.CLICK else AgentAction(action_type=a, wait_ms=500),
                success=ok,
                detail="ok" if ok else "element not found",
            )
            for a, ok in executed
        ],
    )

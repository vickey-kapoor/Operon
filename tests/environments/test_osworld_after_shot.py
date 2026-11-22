from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from operon.environments.osworld import OSWorldExecutor
from operon.models.policy import ActionType, AgentAction

PNG = b"\x89PNG fake after shot"


def _executor(handler, tmp_path):
    return OSWorldExecutor("http://vm:5000", artifact_dir=tmp_path, transport=httpx.MockTransport(handler),
                           post_action_wait_s=0)


def _ok(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        return httpx.Response(200, content=PNG)
    return httpx.Response(200, json={"status": "success", "returncode": 0})


def test_input_action_saves_an_after_screenshot(tmp_path):
    action = AgentAction(action_type=ActionType.PRESS_KEY, key="enter")
    result = asyncio.run(_executor(_ok, tmp_path).execute(action))
    assert result.success and result.artifact_path
    assert Path(result.artifact_path).read_bytes() == PNG


def test_no_screenshot_when_no_input_was_sent(tmp_path):
    result = asyncio.run(_executor(_ok, tmp_path).execute(AgentAction(action_type=ActionType.STOP)))
    assert result.success and result.artifact_path is None


def test_failed_capture_does_not_fail_the_action(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(500)
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    result = asyncio.run(_executor(handler, tmp_path).execute(
        AgentAction(action_type=ActionType.PRESS_KEY, key="enter")))
    assert result.success and result.artifact_path is None


def test_batch_captures_once(tmp_path):
    gets: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            gets.append("g")
        return _ok(request)

    batch = AgentAction(action_type=ActionType.BATCH, actions=[
        AgentAction(action_type=ActionType.PRESS_KEY, key="a"),
        AgentAction(action_type=ActionType.PRESS_KEY, key="b")])
    result = asyncio.run(_executor(handler, tmp_path).execute(batch))
    assert result.success and result.artifact_path and len(gets) == 1

"""`operon mcp` (KAN-96): the tool bodies over a real loop with scripted model calls, offline."""

from __future__ import annotations

import builtins
import io

import pytest

pytest.importorskip("mcp")

from PIL import Image  # noqa: E402

from operon import cli, mcp_server  # noqa: E402
from operon.api import ws_stream  # noqa: E402
from operon.api.runtime import loops, tasks  # noqa: E402
from tests.fixtures.frontdoor import CLICK_SAVE, PURCHASE, build_loop  # noqa: E402


@pytest.fixture
def scheduled(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Runs are created but not stepped by a background task; tests step them by hand."""
    calls: list[str] = []
    monkeypatch.setattr(tasks, "schedule_auto_run", lambda loop, run_id, max_steps: calls.append(run_id))
    return calls


def _tools(monkeypatch, tmp_path, actions):
    loop, executor = build_loop(monkeypatch, tmp_path, actions)
    monkeypatch.setattr(loops, "get_agent_loop", lambda: loop)
    monkeypatch.setattr(loops, "desktop_loop_holds_run", lambda run_id: False)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)
    return mcp_server.OperonMCPTools(cli.LocalClient(), poll_s=0.01), loop, executor


async def _step(loop, run_id: str) -> None:
    from operon.models.common import StepRequest

    await loop.step_run(StepRequest(run_id=run_id))


async def test_start_wait_approve_flow(monkeypatch, tmp_path, scheduled) -> None:
    tools, loop, executor = _tools(monkeypatch, tmp_path, [PURCHASE])
    started = await tools.start_task("Buy it", url="https://shop.test")
    run_id = started["run_id"]
    assert started["status"] == "pending" and scheduled == [run_id]
    assert (await loop.run_store.get_run(run_id)).permission_mode == "ask"

    await _step(loop, run_id)
    waiting = await tools.wait_for_task(run_id, timeout_s=5)
    assert waiting["status"] == "waiting_for_user" and not waiting["done"]
    assert waiting["waiting"]["kind"] == "approval" and waiting["waiting"]["reason"]
    assert waiting["waiting"]["category"]
    assert not [a for a in executor.executed if a.text == "purchase now"]  # nothing ran without the client's approval

    await tools.approve(run_id)
    assert scheduled == [run_id, run_id]  # approving resumed the run


async def test_deny_tells_the_model_not_to_do_it(monkeypatch, tmp_path, scheduled) -> None:
    from operon.agent import control

    tools, loop, executor = _tools(monkeypatch, tmp_path, [PURCHASE])
    run_id = (await tools.start_task("Buy it"))["run_id"]
    await _step(loop, run_id)
    seen: list[str] = []
    real = control.set_override_hint
    monkeypatch.setattr(control, "set_override_hint", lambda rid, text: (seen.append(text), real(rid, text)))

    await tools.deny(run_id)

    assert seen and "declined" in seen[0]
    assert executor.executed == []


async def test_stop_cancels_and_frees_the_slot(monkeypatch, tmp_path, scheduled) -> None:
    tools, loop, _ = _tools(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = (await tools.start_task("Save"))["run_id"]
    stopped = await tools.stop_task(run_id)
    assert stopped["status"] == "cancelled" and stopped["done"]
    assert stopped["summary"].startswith("Stopped")
    assert (await tools.start_task("Again"))["run_id"] != run_id


async def test_one_run_at_a_time(monkeypatch, tmp_path, scheduled) -> None:
    tools, _, _ = _tools(monkeypatch, tmp_path, [CLICK_SAVE])
    first = (await tools.start_task("One"))["run_id"]
    with pytest.raises(ValueError, match="one task at a time"):
        await tools.start_task("Two")
    assert tools.active_run_id == first


async def test_wait_times_out_with_the_current_state(monkeypatch, tmp_path, scheduled) -> None:
    tools, _, _ = _tools(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = (await tools.start_task("One"))["run_id"]
    state = await tools.wait_for_task(run_id, timeout_s=0)
    assert state["status"] == "pending" and not state["done"]


async def test_env_needs_desktop_and_unknown_runs_are_clear_errors(monkeypatch, tmp_path, scheduled) -> None:
    tools, _, _ = _tools(monkeypatch, tmp_path, [CLICK_SAVE])
    with pytest.raises(ValueError, match="desktop=true"):
        await tools.start_task("x", env="linux-container")
    with pytest.raises(ValueError, match="not found"):
        await tools.get_task("nope")


class _WithShot:
    def __init__(self, inner, path: str) -> None:
        self.inner, self.path = inner, path

    async def status(self, run_id: str) -> dict:
        return {**(await self.inner.status(run_id)), "latest_screenshot": self.path}


async def test_screenshot_is_downscaled_to_1280(monkeypatch, tmp_path, scheduled) -> None:
    tools, _, _ = _tools(monkeypatch, tmp_path, [CLICK_SAVE])
    run_id = (await tools.start_task("Save"))["run_id"]
    with pytest.raises(ValueError, match="no screenshot yet"):
        await tools.screenshot_png(run_id)

    big = tmp_path / "big.png"
    Image.new("RGB", (2560, 1440), "white").save(big)
    tools._client = _WithShot(tools.client, str(big))
    png = await tools.screenshot_png(run_id)
    assert Image.open(io.BytesIO(png)).size == (1280, 720)

    small = tmp_path / "small.png"
    Image.new("RGB", (400, 300), "white").save(small)
    assert Image.open(io.BytesIO(mcp_server._downscaled_png(str(small)))).size == (400, 300)


async def test_list_environments_reports_readiness() -> None:
    rows = await mcp_server.OperonMCPTools(object()).list_environments()
    assert {"live", "linux-container"} <= {row["name"] for row in rows}
    assert all({"ready", "problems", "summary", "selected"} <= row.keys() for row in rows)


async def test_server_lists_tools_and_starts_a_task_end_to_end(monkeypatch, tmp_path, scheduled) -> None:
    from mcp.client import Client

    tools, _, _ = _tools(monkeypatch, tmp_path, [CLICK_SAVE])
    server = mcp_server.build_server(tools)
    async with Client(server) as client:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(listed) == {
            "start_task", "get_task", "wait_for_task", "approve", "deny", "stop_task", "get_screenshot",
            "list_environments",
        }
        assert "real mouse and keyboard" in " ".join(listed["start_task"].description.split())
        assert "whole screen" in " ".join(listed["get_screenshot"].description.split())
        result = await client.call_tool("start_task", {"task": "Save the form"})
        assert not result.is_error
        assert result.structured_content["status"] == "pending"
        second = await client.call_tool("start_task", {"task": "Another"})
        assert second.is_error


def test_missing_extra_prints_a_hint_and_exits_2(monkeypatch, capsys) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("no mcp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert mcp_server.main() == 2
    assert "operon-agent[mcp]" in capsys.readouterr().err


def test_the_command_is_registered() -> None:
    assert cli.build_parser().parse_args(["mcp"]).command == "mcp"

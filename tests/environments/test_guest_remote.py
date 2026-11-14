"""KAN-80: the guest agent and RemoteDesktopExecutor, end to end in one process."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from operon.environments.guest import create_guest_app, local_ip, write_ready_file
from operon.environments.remote import (
    GuestUnavailableError,
    RemoteDesktopExecutor,
    wait_for_guest,
)
from operon.models.capture import CaptureFrame
from operon.models.common import FailureCategory
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction

TOKEN = "t0ken"
PNG = b"\x89PNG\r\n\x1a\nfake"


class FakeGuestExecutor:
    """Stands in for the guest machine's DesktopExecutor."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.calls: list[tuple] = []
        self._current_run_id: str | None = None

    async def capture(self) -> CaptureFrame:
        path = self.tmp / "guest_frame.png"
        path.write_bytes(PNG)
        return CaptureFrame(artifact_path=str(path), width=1280, height=800, visual_velocity=0.01)

    async def execute(self, action: AgentAction) -> ExecutedAction:
        self.calls.append(("execute", action.action_type.value))
        after = self.tmp / "guest_after.png"
        after.write_bytes(PNG + b"after")
        return ExecutedAction(
            action=action, success=True, detail="clicked", artifact_path=str(after), recording_path=str(self.tmp / "rec.mp4")
        )

    def set_current_run_id(self, run_id: str) -> None:
        self.calls.append(("set_current_run_id", run_id))
        self._current_run_id = run_id

    async def reset_desktop(self) -> None:
        self.calls.append(("reset_desktop",))

    async def aclose_run(self, run_id: str) -> int:
        self.calls.append(("aclose_run", run_id))
        return 2

    async def live_frame_png(self, run_id: str) -> bytes | None:
        return b"jpeg" if run_id == self._current_run_id else None

    async def user_input(self, run_id: str, **kwargs) -> bool:
        if kwargs["input_type"] == "bogus":
            raise ValueError("unsupported input_type 'bogus'")
        self.calls.append(("user_input", run_id, kwargs["input_type"], kwargs["x_ratio"]))
        return True


@pytest.fixture
def guest(tmp_path):
    return FakeGuestExecutor(tmp_path)


@pytest.fixture
def remote(guest, tmp_path):
    app = create_guest_app(guest, TOKEN)
    transport = httpx.ASGITransport(app=app)
    return RemoteDesktopExecutor("http://guest", TOKEN, artifact_dir=tmp_path / "host", transport=transport)


# ── guest auth ──────────────────────────────────────────────────


def test_guest_refuses_requests_without_the_token(guest):
    client = TestClient(create_guest_app(guest, TOKEN))
    assert client.get("/health").status_code == 401
    assert client.post("/capture", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = client.get("/health", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200 and ok.json()["ok"] is True


def test_guest_needs_a_token(guest):
    with pytest.raises(ValueError):
        create_guest_app(guest, "")


def test_guest_only_calls_listed_methods(guest):
    client = TestClient(create_guest_app(guest, TOKEN))
    response = client.post(
        "/call", json={"method": "__init__", "kwargs": {}}, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 400


def test_guest_reports_a_missing_optional_method(guest):
    client = TestClient(create_guest_app(guest, TOKEN))
    response = client.post(
        "/call", json={"method": "context_reset"}, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.json() == {"present": False, "result": None}


# ── remote executor through the guest ───────────────────────────


def test_capture_brings_the_screenshot_to_a_host_file(remote, tmp_path):
    frame = asyncio.run(remote.capture())
    local = Path(frame.artifact_path)
    assert local.parent == tmp_path / "host"
    assert local.read_bytes() == PNG
    assert (frame.width, frame.height, frame.visual_velocity) == (1280, 800, 0.01)
    # The guest's own copy is removed once sent.
    assert not (tmp_path / "guest_frame.png").exists()


def test_execute_runs_on_the_guest_and_localizes_files(remote, guest, tmp_path):
    action = AgentAction(action_type=ActionType.CLICK, x=10, y=20)
    result = asyncio.run(remote.execute(action))
    assert result.success and result.detail == "clicked"
    assert guest.calls == [("execute", "click")]
    assert Path(result.artifact_path).parent == tmp_path / "host"
    assert Path(result.artifact_path).read_bytes() == PNG + b"after"
    assert result.recording_path is None  # guest paths never reach the host


def test_execute_reports_an_unreachable_guest_as_a_failed_action(tmp_path):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    remote = RemoteDesktopExecutor(
        "http://guest", TOKEN, artifact_dir=tmp_path, transport=httpx.MockTransport(refuse)
    )
    result = asyncio.run(remote.execute(AgentAction(action_type=ActionType.WAIT, wait_ms=100)))
    assert not result.success
    assert result.failure_category is FailureCategory.EXECUTION_ERROR
    assert "unreachable" in result.detail


def test_capture_from_an_unreachable_guest_raises(tmp_path):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    remote = RemoteDesktopExecutor(
        "http://guest", TOKEN, artifact_dir=tmp_path, transport=httpx.MockTransport(refuse)
    )
    with pytest.raises(GuestUnavailableError):
        asyncio.run(remote.capture())


def test_run_hooks_reach_the_guest(remote, guest):
    async def scenario():
        remote.set_current_run_id("run-1")
        await asyncio.sleep(0.05)  # the guest is told in the background
        await remote.reset_desktop()
        await remote.context_reset()  # absent on the guest: a no-op, not an error
        closed = await remote.aclose_run("run-1")
        frame = await remote.live_frame_png("run-1")
        other = await remote.live_frame_png("run-2")
        ok = await remote.user_input("run-1", input_type="click", x_ratio=0.5)
        return closed, frame, other, ok

    closed, frame, other, ok = asyncio.run(scenario())
    assert ("set_current_run_id", "run-1") in guest.calls
    assert ("reset_desktop",) in guest.calls
    assert closed == 2
    assert frame == b"jpeg" and other is None
    assert ok and ("user_input", "run-1", "click", 0.5) in guest.calls


def test_bad_takeover_input_is_a_value_error(remote):
    async def scenario():
        await remote.user_input("run-1", input_type="bogus")

    with pytest.raises(ValueError, match="bogus"):
        asyncio.run(scenario())


def test_set_current_run_id_outside_a_loop_only_remembers(tmp_path):
    remote = RemoteDesktopExecutor("http://guest", TOKEN, artifact_dir=tmp_path)
    remote.set_current_run_id("run-9")
    assert remote._current_run_id == "run-9"


def test_aclose_tolerates_a_vanished_guest(tmp_path):
    def refuse(request):
        raise httpx.ConnectError("gone", request=request)

    remote = RemoteDesktopExecutor(
        "http://guest", TOKEN, artifact_dir=tmp_path, transport=httpx.MockTransport(refuse)
    )
    asyncio.run(remote.aclose())


# ── readiness ───────────────────────────────────────────────────


def test_wait_for_guest_retries_until_healthy():
    answers = iter([httpx.ConnectError("not yet"), 503, 200])

    def handler(request):
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return httpx.Response(answer, json={"ok": answer == 200, "platform": "windows"})

    health = wait_for_guest("http://guest", TOKEN, timeout_s=5, interval_s=0, transport=httpx.MockTransport(handler))
    assert health["platform"] == "windows"


def test_wait_for_guest_gives_up_after_the_timeout():
    def handler(request):
        raise httpx.ConnectError("never", request=request)

    with pytest.raises(GuestUnavailableError, match="did not start"):
        wait_for_guest("http://guest", TOKEN, timeout_s=0, interval_s=0, transport=httpx.MockTransport(handler))


def test_ready_file_is_written_whole(tmp_path):
    path = tmp_path / "ready.json"
    write_ready_file(path, host="172.20.1.5", port=8765)
    assert path.read_text() == '{"host": "172.20.1.5", "port": 8765}'
    assert not (tmp_path / "ready.tmp").exists()


def test_local_ip_is_an_address():
    assert local_ip().count(".") == 3


def test_png_round_trip_is_byte_exact(remote):
    frame = asyncio.run(remote.capture())
    assert base64.b64encode(Path(frame.artifact_path).read_bytes()) == base64.b64encode(PNG)


# ── batches are split on the host (Codex P1 on #272) ────────────


def _batch(*children):
    return AgentAction(action_type=ActionType.BATCH, actions=list(children))


def _click(x):
    return AgentAction(action_type=ActionType.CLICK, x=x, y=1)


def test_a_blocked_batch_child_never_reaches_the_guest(remote, guest):
    seen = []

    async def check(child):
        seen.append(child.x)
        return "run stopped" if child.x == 2 else None

    remote.batch_child_check = check
    result = asyncio.run(remote.execute(_batch(_click(1), _click(2), _click(3))))
    assert not result.success
    assert result.batch_failed_index == 1
    assert "Batch stopped before click: run stopped" in result.detail
    assert guest.calls == [("execute", "click")]  # only the first child ran
    assert seen == [2]  # never checked before the first child


def test_a_whole_batch_runs_child_by_child(remote, guest):
    result = asyncio.run(remote.execute(_batch(_click(1), _click(2))))
    assert result.success and result.detail == "Executed batch: click:ok, click:ok"
    assert guest.calls == [("execute", "click"), ("execute", "click")]
    assert result.artifact_path and Path(result.artifact_path).exists()


def test_a_failed_child_stops_the_batch(remote, guest):
    async def fail_second(action):
        guest.calls.append(("execute", action.action_type.value))
        ok = len([c for c in guest.calls if c[0] == "execute"]) == 1
        return ExecutedAction(
            action=action, success=ok, detail="ok" if ok else "missed",
            failure_category=None if ok else FailureCategory.EXECUTION_ERROR,
        )

    guest.execute = fail_second
    result = asyncio.run(remote.execute(_batch(_click(1), _click(2), _click(3))))
    assert not result.success and result.batch_failed_index == 1
    assert "Batch failed on click: missed" in result.detail
    assert len(guest.calls) == 2


def test_batch_tool_outputs_come_back_together(remote, guest):
    from operon.models.execution import ToolOutput

    async def tool_result(action):
        guest.calls.append(("execute", action.action_type.value))
        return ExecutedAction(
            action=action, success=True, detail="ran", tool_outputs=[ToolOutput(tool="read_file", ok=True, output="x")]
        )

    guest.execute = tool_result
    result = asyncio.run(remote.execute(_batch(_click(1), _click(2))))
    assert [o.output for o in result.tool_outputs] == ["x", "x"]


# ── closing a run's apps reaches the guest (Codex P2 on #272) ───


def test_close_apps_when_done_is_passed_on_when_the_run_ends(remote, guest):
    asked = []
    guest.close_apps_when_done = lambda run_id: asked.append(run_id)
    remote.close_apps_when_done("run-1")
    assert asked == []  # nothing until the run ends
    closed = asyncio.run(remote.aclose_run("run-1"))
    assert asked == ["run-1"] and closed == 2
    asyncio.run(remote.aclose_run("run-1"))
    assert asked == ["run-1"]  # once per request


# ── the guest runs the tools itself ─────────────────────────────


def test_guest_executor_carries_the_tools_the_host_asked_for(monkeypatch):
    from operon.environments import guest as guest_mod
    from operon.executor.tools import ToolExecutor

    monkeypatch.setenv("OPERON_DESKTOP_PLATFORM", "linux")
    monkeypatch.setattr("operon.executor.linux_desktop.LinuxDesktopExecutor", lambda: "x11")
    monkeypatch.setenv("OPERON_TOOLS", "on")
    wrapped = guest_mod.build_guest_executor()
    assert isinstance(wrapped, ToolExecutor) and wrapped.inner == "x11"
    monkeypatch.setenv("OPERON_TOOLS", "off")
    assert guest_mod.build_guest_executor() == "x11"

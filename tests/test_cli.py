"""`operon run` (in process and --request) and `operon serve` (KAN-89), offline."""

from __future__ import annotations

import httpx
import pytest

from operon import cli
from operon.api import ws_stream
from operon.api.runtime import loops
from operon.api.server import app
from tests.fixtures.frontdoor import CLICK_SAVE, build_loop


def _use(monkeypatch: pytest.MonkeyPatch, loop) -> None:
    monkeypatch.setattr(loops, "get_agent_loop", lambda: loop)
    monkeypatch.setattr(loops, "desktop_loop_holds_run", lambda run_id: False)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)


def _answers(*replies: str):
    queue = list(replies)
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return queue.pop(0)

    return ask, asked


def test_parser_builds_the_task_payload() -> None:
    args = cli.build_parser().parse_args(
        ["run", "Order lunch", "--desktop", "--mode", "ask", "--max-steps", "9", "--url", "https://x.test"]
    )
    assert cli._payload(args) == {
        "intent": "Order lunch",
        "environment": "desktop",
        "max_steps": 9,
        "start_url": "https://x.test",
        "permission_mode": "ask",
    }
    assert args.request is False


def test_parser_rejects_an_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "x", "--mode", "yolo"])


async def test_local_run_asks_approves_and_reports(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    ask, asked = _answers("y", "s")
    lines: list[str] = []

    code, report = await cli.drive(
        cli.LocalClient(),
        {"intent": "Save the form", "permission_mode": "ask", "max_steps": 5},
        ask=ask,
        out=lines.append,
        poll_s=0.01,
    )

    assert code == 1  # stopped at the second approval
    assert len(asked) == 2
    assert report["status"] == "cancelled" and report["approvals"] == 1
    assert len(executor.executed) == 1
    assert any(line.startswith("Approval needed (ask mode)") for line in lines)
    assert "\nArtifacts: " in lines[-1]
    assert (await loop.run_store.get_run(report["run_id"])).mode == "batch"  # no page watches it


async def test_a_correction_typed_at_the_prompt_declines(monkeypatch, tmp_path) -> None:
    from operon.agent import control

    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    seen_hints: list[str | None] = []

    def ask(prompt: str) -> str:
        if seen_hints:
            return "s"
        seen_hints.append("asked")
        return "click Cancel instead"

    real_resume = loop.resume_run

    async def resume(run_id: str, *, approve_gated_action: bool = True):
        seen_hints.append(control.consume_override_hint(run_id))
        control.set_override_hint(run_id, seen_hints[-1])
        return await real_resume(run_id, approve_gated_action=approve_gated_action)

    loop.resume_run = resume
    code, _ = await cli.drive(
        cli.LocalClient(), {"intent": "x", "permission_mode": "ask"}, ask=ask, out=lambda _: None, poll_s=0.01
    )

    assert code == 1
    assert "click Cancel instead" in seen_hints[1]
    assert executor.executed == []


async def test_ctrl_c_takes_over_and_enter_hands_back(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    interrupts = cli._Interrupts()
    interrupts(2, None)  # one Ctrl+C before the first poll
    ask, asked = _answers("I logged in", "")

    async def finish_after_hand_back(run_id, note=None):
        response = await real_hand_back(run_id, note)
        from operon.models.common import RunStatus

        await loop.run_store.set_status(run_id, RunStatus.SUCCEEDED)
        return response

    real_hand_back = loop.hand_back
    loop.hand_back = finish_after_hand_back
    code, report = await cli.drive(
        cli.LocalClient(), {"intent": "x", "permission_mode": "auto"}, ask=ask, out=lambda _: None,
        poll_s=0.01, interrupts=interrupts,
    )

    assert code == 0
    assert asked[0].startswith("Press Enter to hand back")
    assert report["takeovers"] == 1 and report["outcome"] == "Done"


def test_a_second_ctrl_c_quits() -> None:
    interrupts = cli._Interrupts()
    interrupts(2, None)
    with pytest.raises(KeyboardInterrupt):
        interrupts(2, None)


async def test_without_a_terminal_a_local_run_stops_instead_of_hanging(monkeypatch, tmp_path) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    code, report = await cli.drive(
        cli.LocalClient(), {"intent": "x", "permission_mode": "ask"}, ask=lambda _: pytest.fail("asked"),
        out=lambda _: None, interactive=False, poll_s=0.01,
    )
    assert code == 1 and report["status"] == "cancelled"
    assert executor.executed == []


async def test_request_mode_drives_a_running_api(monkeypatch, tmp_path) -> None:
    from operon.api.routes import frontdoor as routes

    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    monkeypatch.setattr(routes, "test_safe_mode_enabled", lambda: False)

    async def no_cdp(mode: str) -> None:
        return None

    monkeypatch.setattr("operon.api.dependencies.ensure_cdp_ready", no_cdp)
    client = cli.RemoteClient("http://127.0.0.1:8080", None, transport=httpx.ASGITransport(app=app))
    try:
        code, report = await cli.drive(
            client, {"intent": "Save the form", "permission_mode": "auto", "max_steps": 2},
            ask=lambda _: pytest.fail("asked"), out=lambda _: None, poll_s=0.01,
        )
    finally:
        await client.close()

    assert code == 1  # the scripted model never finishes, so the run hits max_steps
    assert report["steps"] == 2 and len(executor.executed) == 2


async def test_request_mode_leaves_a_paused_run_for_the_page(monkeypatch, tmp_path) -> None:
    from operon.api.routes import frontdoor as routes

    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    monkeypatch.setattr(routes, "test_safe_mode_enabled", lambda: False)

    async def no_cdp(mode: str) -> None:
        return None

    monkeypatch.setattr("operon.api.dependencies.ensure_cdp_ready", no_cdp)
    client = cli.RemoteClient("http://127.0.0.1:8080", None, transport=httpx.ASGITransport(app=app))
    lines: list[str] = []
    try:
        code, report = await cli.drive(
            client, {"intent": "x", "permission_mode": "ask"}, ask=lambda _: pytest.fail("asked"),
            out=lines.append, interactive=False, poll_s=0.01, leave_paused=True,
        )
    finally:
        await client.close()

    assert code == 2 and report is None
    assert any(line.startswith("Left paused") for line in lines)
    link = [line for line in lines if line.startswith("Follow it on ")]
    assert link and "/app#run=" in link[0] and len(link[0].split("#run=")[1]) > 8


async def test_remote_errors_carry_the_api_detail() -> None:
    client = cli.RemoteClient("http://127.0.0.1:8080", None, transport=httpx.ASGITransport(app=app))
    try:
        with pytest.raises(RuntimeError, match="404"):
            await client.status("missing-run")
    finally:
        await client.close()


def test_serve_prints_the_page_link_with_the_key_in_the_fragment(monkeypatch, capsys) -> None:
    import uvicorn

    monkeypatch.setenv("API_KEYS", "k1,k2")
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    assert cli.main(["serve", "--port", "9000"]) == 0
    assert capsys.readouterr().out.strip() == "Task page: http://127.0.0.1:9000/app#key=k1"


async def test_settle_subprocess_pipes_waits_only_on_windows(monkeypatch) -> None:
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(cli.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    await cli.settle_subprocess_pipes()
    assert slept == []
    monkeypatch.setattr(cli.sys, "platform", "win32")
    await cli.settle_subprocess_pipes()
    assert slept == [0.25]


def test_closed_pipe_noise_is_dropped_but_other_errors_still_print(monkeypatch) -> None:
    seen: list[BaseException] = []
    monkeypatch.setattr(cli.sys, "unraisablehook", lambda u: seen.append(u.exc_value))
    cli.quiet_closed_pipe_noise()

    class Unraisable:
        def __init__(self, exc: BaseException) -> None:
            self.exc_value = exc

    cli.sys.unraisablehook(Unraisable(ValueError("I/O operation on closed pipe")))
    other = ValueError("something else")
    cli.sys.unraisablehook(Unraisable(other))
    cli.sys.unraisablehook(Unraisable(OSError("closed pipe")))
    assert seen[0] is other and isinstance(seen[1], OSError) and len(seen) == 2


def test_run_output_keeps_library_warnings_out(monkeypatch) -> None:
    import logging

    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.WARNING)
    cli._console_logging("run")
    try:
        assert root.level == logging.ERROR and root.handlers
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    cli._console_logging("serve")
    assert root.handlers == []

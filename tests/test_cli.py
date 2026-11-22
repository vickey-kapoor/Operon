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


async def test_request_mode_leaves_a_paused_run(monkeypatch, tmp_path) -> None:
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
    link = [line for line in lines if line.startswith("Run ") and " is on " in line]
    assert link and "is on http://" in link[0] and "/app" not in link[0]


async def test_remote_errors_carry_the_api_detail() -> None:
    client = cli.RemoteClient("http://127.0.0.1:8080", None, transport=httpx.ASGITransport(app=app))
    try:
        with pytest.raises(RuntimeError, match="404"):
            await client.status("missing-run")
    finally:
        await client.close()


def _serve_calls(monkeypatch) -> list:
    import uvicorn

    calls: list = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    return calls


def test_serve_prints_the_api_address_and_where_the_key_comes_from(monkeypatch, capsys) -> None:
    monkeypatch.setenv("API_KEYS", "k1,k2")
    calls = _serve_calls(monkeypatch)

    assert cli.main(["serve", "--port", "9000"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == ["API: http://127.0.0.1:9000", "API key: from API_KEYS"]
    assert "k1" not in out and len(calls) == 1


def test_serve_never_prints_a_generated_key(monkeypatch, capsys, tmp_path) -> None:
    from operon.api import auth

    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("OPERON_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("OPERON_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(auth, "auth_disabled", lambda: False)
    _serve_calls(monkeypatch)

    assert cli.main(["serve"]) == 0
    out = capsys.readouterr().out
    key_file = auth.api_key_file()
    key = key_file.read_text(encoding="utf-8").strip()
    assert key and key not in out
    assert f"API key file: {key_file}" in out


def test_serve_says_when_authentication_is_off(monkeypatch, capsys) -> None:
    from operon.api import auth

    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setattr(auth, "auth_disabled", lambda: True)
    _serve_calls(monkeypatch)

    assert cli.main(["serve"]) == 0
    assert "Authentication is off" in capsys.readouterr().out


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.com"])
def test_serve_refuses_a_non_loopback_host(monkeypatch, capsys, host) -> None:
    calls = _serve_calls(monkeypatch)

    assert cli.main(["serve", "--host", host]) == 2
    captured = capsys.readouterr()
    assert f"refusing to listen on {host}" in captured.err and "--allow-remote" in captured.err
    assert captured.out == "" and calls == []


def test_serve_allow_remote_warns_and_starts(monkeypatch, capsys) -> None:
    monkeypatch.setenv("API_KEYS", "k1")
    calls = _serve_calls(monkeypatch)

    assert cli.main(["serve", "--host", "0.0.0.0", "--allow-remote", "--port", "9001"]) == 0
    captured = capsys.readouterr()
    assert "warning" in captured.err and "0.0.0.0" in captured.err
    assert "API: http://127.0.0.1:9001" in captured.out
    assert calls and calls[0][1]["host"] == "0.0.0.0"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "LocalHost"])
def test_serve_allows_loopback_hosts(monkeypatch, capsys, host) -> None:
    monkeypatch.setenv("API_KEYS", "k1")
    calls = _serve_calls(monkeypatch)

    assert cli.main(["serve", "--host", host]) == 0
    assert capsys.readouterr().err == "" and len(calls) == 1


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com"])
def test_serve_refuses_remote_when_authentication_is_off(monkeypatch, capsys, host) -> None:
    from operon.api import auth

    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setattr(auth, "auth_disabled", lambda: True)
    calls = _serve_calls(monkeypatch)

    assert cli.main(["serve", "--host", host, "--allow-remote"]) == 2
    captured = capsys.readouterr()
    assert "authentication off" in captured.err and "OPERON_ALLOW_UNAUTHENTICATED" in captured.err
    assert captured.out == "" and calls == []


@pytest.mark.parametrize(
    ("host", "shown"),
    [
        ("0.0.0.0", "127.0.0.1"),
        ("::", "127.0.0.1"),
        ("", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "[::1]"),
        ("fe80::1", "[fe80::1]"),
        ("example.com", "example.com"),
    ],
)
def test_display_host(host, shown) -> None:
    assert cli._display_host(host) == shown


def _run_args(*extra: str):
    return cli.build_parser().parse_args(["run", "--request", *extra, "task"])


def test_api_key_file_is_read_and_stripped(tmp_path) -> None:
    key_file = tmp_path / "key"
    key_file.write_text("  filekey\n", encoding="utf-8")

    assert cli._resolve_api_key(_run_args("--api-key-file", str(key_file))) == "filekey"


def test_api_key_wins_over_api_key_file(tmp_path) -> None:
    key_file = tmp_path / "key"
    key_file.write_text("filekey", encoding="utf-8")

    assert cli._resolve_api_key(_run_args("--api-key", "flag", "--api-key-file", str(key_file))) == "flag"


def test_no_key_flags_resolve_to_none() -> None:
    assert cli._resolve_api_key(_run_args()) is None


@pytest.mark.parametrize("content", ["", "  \n"])
def test_run_with_an_empty_api_key_file_exits_2(tmp_path, capsys, content) -> None:
    key_file = tmp_path / "key"
    key_file.write_text(content, encoding="utf-8")

    assert cli.main(["run", "--request", "--api-key-file", str(key_file), "task"]) == 2
    assert "empty" in capsys.readouterr().err


def test_run_with_an_unreadable_api_key_file_exits_2(tmp_path, capsys) -> None:
    assert cli.main(["run", "--request", "--api-key-file", str(tmp_path / "missing"), "task"]) == 2
    assert "cannot read --api-key-file" in capsys.readouterr().err


def test_resume_with_an_unreadable_api_key_file_exits_2(tmp_path, capsys) -> None:
    assert cli.main(["resume", "r1", "--request", "--api-key-file", str(tmp_path / "missing")]) == 2
    assert "cannot read --api-key-file" in capsys.readouterr().err


def test_resume_and_runs_accept_api_key_file() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["runs", "--interrupted", "--api-key-file", "p"]).api_key_file == "p"
    assert parser.parse_args(["resume", "r1", "--api-key-file", "p"]).api_key_file == "p"


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


class _FakeClient:
    """A run that pauses on the model's own confirmation (needs_you), then finishes."""

    def __init__(self, kind: str = "needs_you") -> None:
        self.kind, self.approved = kind, 0

    async def start(self, payload):
        return {"run_id": "r1", "environment": "desktop", "permission_mode": "auto", "recent_steps": [],
                "done": False, "in_takeover": False,
                "waiting": {"kind": self.kind, "gate": None, "reason": "Confirm the change"}}

    async def approve(self, run_id):
        self.approved += 1
        return {"run_id": run_id, "recent_steps": [], "done": True, "in_takeover": False, "waiting": None}

    async def stop(self, run_id):
        return {"run_id": run_id, "recent_steps": [], "done": True, "in_takeover": False, "waiting": None}

    async def report(self, run_id):
        return {"status": "succeeded"}

    def format_report(self, report):
        return "report"


async def test_auto_confirm_answers_the_models_pause_when_on(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_AUTO_CONFIRM", "on")
    client, lines = _FakeClient(), []
    code, _ = await cli.drive(client, {"intent": "x"}, ask=lambda _: "n", out=lines.append,
                              interactive=False, poll_s=0.01)
    assert code == 0 and client.approved == 1
    assert any("Auto-confirming" in line for line in lines)


async def test_auto_confirm_is_off_by_default_and_never_answers_gates(monkeypatch) -> None:
    monkeypatch.delenv("OPERON_AUTO_CONFIRM", raising=False)
    off = _FakeClient()
    await cli.drive(off, {"intent": "x"}, ask=lambda _: "n", out=lambda _: None, interactive=False, poll_s=0.01)
    assert off.approved == 0  # no terminal: stopped, not approved
    monkeypatch.setenv("OPERON_AUTO_CONFIRM", "on")
    gate = _FakeClient(kind="approval")
    await cli.drive(gate, {"intent": "x"}, ask=lambda _: "n", out=lambda _: None, interactive=False, poll_s=0.01)
    assert gate.approved == 0


def test_cli_payload_defaults_max_steps_to_25() -> None:
    args = cli.build_parser().parse_args(["run", "do it"])
    assert args.max_steps is None and cli._payload(args)["max_steps"] == 25
    args = cli.build_parser().parse_args(["run", "do it", "--max-steps", "7"])
    assert cli._payload(args)["max_steps"] == 7


def test_run_without_a_task_is_a_usage_error(capsys) -> None:
    assert cli.main(["run"]) == 2
    assert "give a task" in capsys.readouterr().err


def test_version_flag(capsys) -> None:
    from operon import __version__

    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"operon {__version__}"

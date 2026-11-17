"""KAN-80 stage 2: the Windows Sandbox provider (offline: no sandbox is opened)."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from operon.environments import get_provider
from operon.environments import windows_sandbox as ws
from operon.environments.remote import RemoteDesktopExecutor
from operon.environments.windows_sandbox import WindowsSandboxProvider


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.delenv("OPERON_SANDBOX_MEMORY_MB", raising=False)
    monkeypatch.delenv("OPERON_SANDBOX_START_TIMEOUT", raising=False)
    monkeypatch.setattr(ws, "sandbox_running", lambda: False)
    return WindowsSandboxProvider(work_dir=tmp_path / "sb")


def test_registered_under_its_name():
    assert isinstance(get_provider("windows-sandbox"), WindowsSandboxProvider)
    assert get_provider("windows-sandbox").platform() == "windows"


def test_settings_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPERON_SANDBOX_MEMORY_MB", "8192")
    monkeypatch.setenv("OPERON_SANDBOX_START_TIMEOUT", "oops")
    p = WindowsSandboxProvider(work_dir=tmp_path)
    assert p.memory_mb == 8192
    assert p.start_timeout_s == 240  # a bad value keeps the default


# ── check ───────────────────────────────────────────────────────


def test_check_on_another_os_says_so(provider, monkeypatch):
    monkeypatch.setattr(ws.sys, "platform", "linux")
    check = provider.check()
    assert not check.ok and "only on Windows" in check.problems[0]


def test_check_without_the_feature_gives_the_enable_command(provider, monkeypatch, tmp_path):
    monkeypatch.setattr(ws.sys, "platform", "win32")
    monkeypatch.setattr(ws, "sandbox_exe", lambda: tmp_path / "missing.exe")
    check = provider.check()
    assert not check.ok and ws.ENABLE_COMMAND in check.problems[0]


def test_check_ready_with_the_feature_on(provider, monkeypatch, tmp_path):
    exe = tmp_path / "WindowsSandbox.exe"
    exe.write_text("")
    monkeypatch.setattr(ws.sys, "platform", "win32")
    monkeypatch.setattr(ws, "sandbox_exe", lambda: exe)
    monkeypatch.setattr(ws.sys, "base_prefix", str(tmp_path / "Python314"))
    assert provider.check().ok


def test_check_refuses_store_python(provider, monkeypatch, tmp_path):
    exe = tmp_path / "WindowsSandbox.exe"
    exe.write_text("")
    monkeypatch.setattr(ws.sys, "platform", "win32")
    monkeypatch.setattr(ws, "sandbox_exe", lambda: exe)
    monkeypatch.setattr(ws.sys, "base_prefix", str(tmp_path / "WindowsApps" / "Python"))
    check = provider.check()
    assert not check.ok and "Microsoft Store" in check.problems[0]


# ── files ───────────────────────────────────────────────────────


def _folders(xml: str) -> list[tuple[str, str, str]]:
    root = ET.fromstring(xml)
    return [
        (m.findtext("HostFolder"), m.findtext("SandboxFolder"), m.findtext("ReadOnly"))
        for m in root.iter("MappedFolder")
    ]


def test_wsb_maps_python_and_source_read_only_and_exchange_writable(provider):
    folders = _folders(provider.wsb_config())
    *code, exchange = folders
    assert all(ro == "true" and host == inside for host, inside, ro in code)
    hosts = {Path(h) for h, _, _ in code}
    assert Path(sys.base_prefix).resolve() in hosts
    assert any(f == ws.source_dir() or f in ws.source_dir().parents for f in hosts)
    assert exchange == (str(provider.exchange), str(provider.exchange), "false")


def test_wsb_never_maps_the_repo_root_or_user_config(provider):
    hosts = {Path(h) for h, _, _ in _folders(provider.wsb_config())}
    repo_root = ws.source_dir().parent
    if (repo_root / "pyproject.toml").exists():  # a source checkout: .env lives here
        assert repo_root not in hosts
    assert Path.home() / ".operon" not in hosts


def test_wsb_turns_off_host_redirection_and_sets_memory(provider):
    root = ET.fromstring(provider.wsb_config())
    for tag in ("ClipboardRedirection", "PrinterRedirection", "AudioInput", "VideoInput"):
        assert root.findtext(tag) == "Disable"
    assert root.findtext("MemoryInMB") == "4096"
    assert root.findtext("Networking") == "Enable"
    assert "start-guest.cmd" in root.find("LogonCommand").findtext("Command")


def test_guest_command_starts_the_guest_with_the_exchange_files(provider):
    cmd = provider.guest_command()
    assert "-m operon.cli guest" in cmd
    assert f"--port {provider.port}" in cmd
    for name in ("token.txt", "ready.json", "guest.log"):
        assert str(provider.exchange / name) in cmd
    assert "localport=8765" in cmd
    assert "OPERON_RUNTIME_ROOT=" in cmd
    assert "GEMINI" not in cmd


def test_write_files_makes_a_fresh_token_and_clears_old_readiness(provider):
    provider.exchange.mkdir(parents=True)
    (provider.exchange / "ready.json").write_text("{}")
    wsb = provider.write_files()
    first = (provider.exchange / "token.txt").read_text()
    assert wsb.exists() and (provider.exchange / "start-guest.cmd").exists()
    assert not (provider.exchange / "ready.json").exists()
    provider.write_files()
    assert (provider.exchange / "token.txt").read_text() != first
    assert len(first) >= 32


# ── lifecycle ───────────────────────────────────────────────────


def _guest_comes_up(provider, monkeypatch, *, health=None):
    def fake_open(wsb: Path):
        (provider.exchange / "ready.json").write_text(json.dumps({"host": "172.30.0.9", "port": 8765}))

    seen = {}

    def fake_wait(endpoint, token, *, timeout_s, interval_s):
        seen.update(endpoint=endpoint, token=token)
        if isinstance(health, Exception):
            raise health
        return {"ok": True}

    monkeypatch.setattr(provider, "_open", fake_open)
    monkeypatch.setattr("operon.environments.remote.wait_for_guest", fake_wait)
    return seen


def test_start_opens_the_sandbox_and_builds_a_remote_executor(provider, monkeypatch):
    seen = _guest_comes_up(provider, monkeypatch)
    provider.start()
    provider.start()  # idempotent
    assert seen["endpoint"] == "http://172.30.0.9:8765"
    assert seen["token"] == (provider.exchange / "token.txt").read_text()
    executor = provider.build_executor()
    assert isinstance(executor, RemoteDesktopExecutor)
    assert executor.base_url == "http://172.30.0.9:8765"


def test_build_executor_before_start_is_an_error(provider):
    with pytest.raises(RuntimeError, match="start"):
        provider.build_executor()


def test_a_guest_that_never_answers_stops_the_sandbox(provider, monkeypatch):
    _guest_comes_up(provider, monkeypatch, health=RuntimeError("no health"))
    stopped = []
    monkeypatch.setattr(ws.subprocess, "run", lambda *a, **k: stopped.append(a[0]))
    with pytest.raises(RuntimeError, match="no health"):
        provider.start()
    assert provider._started is False
    assert not (provider.exchange / "token.txt").exists()


def test_timeout_without_ready_file_shows_the_guest_log(provider, monkeypatch):
    provider.start_timeout_s = 0
    monkeypatch.setattr(provider, "_open", lambda wsb: (provider.exchange / "guest.log").write_text("ImportError: mss"))
    with pytest.raises(RuntimeError, match="ImportError: mss"):
        provider.start()


def test_stop_closes_the_sandbox_once(provider, monkeypatch):
    _guest_comes_up(provider, monkeypatch)
    provider.start()
    killed: list[list[str]] = []
    monkeypatch.setattr(ws.sys, "platform", "win32")
    monkeypatch.setattr(ws.shutil, "which", lambda name: "taskkill")
    monkeypatch.setattr(ws.subprocess, "run", lambda argv, **k: killed.append(argv))
    provider.stop()
    provider.stop()
    assert [argv[2] for argv in killed] == list(ws._SANDBOX_PROCESSES)
    assert not (provider.exchange / "token.txt").exists()


def test_logon_script_writes_the_marker_first(provider):
    lines = provider.guest_command().split("\r\n")
    assert lines[0] == "@echo off"
    assert "logon.txt" in lines[1]


def test_timeout_says_whether_the_logon_command_ran(provider, monkeypatch):
    provider.start_timeout_s = 0
    monkeypatch.setattr(provider, "_open", lambda wsb: None)
    with pytest.raises(RuntimeError, match="never ran its logon command"):
        provider.start()
    monkeypatch.setattr(provider, "_open", lambda wsb: (provider.exchange / ws.LOGON_MARKER).write_text("now"))
    with pytest.raises(RuntimeError, match="ran its logon command"):
        provider.start()


def test_start_refuses_while_another_sandbox_runs(provider, monkeypatch):
    monkeypatch.setattr(ws, "sandbox_running", lambda: True)
    monkeypatch.setattr(ws.time, "sleep", lambda s: None)
    ticks = iter(range(0, 10_000, 50))
    monkeypatch.setattr(ws.time, "monotonic", lambda: next(ticks))
    opened = []
    monkeypatch.setattr(provider, "_open", lambda wsb: opened.append(wsb))
    with pytest.raises(RuntimeError, match="still running"):
        provider.start()
    assert opened == []


def test_start_retries_once_when_the_sandbox_closes_before_logon(provider, monkeypatch):
    opens: list[Path] = []
    waits = iter([ws.SandboxClosedEarly("closed"), {"status": "ok"}])

    def fake_wait():
        outcome = next(waits)
        if isinstance(outcome, Exception):
            raise outcome
        provider._endpoint = "http://x:1"
        return outcome

    monkeypatch.setattr(provider, "_open", lambda wsb: opens.append(wsb))
    monkeypatch.setattr(provider, "wait_until_ready", fake_wait)
    provider.start()
    assert len(opens) == 2


def test_wait_notices_a_sandbox_that_vanished_before_logon(provider, monkeypatch):
    provider.start_timeout_s = 1000
    states = iter([True, False])
    monkeypatch.setattr(ws, "sandbox_running", lambda: next(states))
    clock = iter(range(0, 100_000, 11))
    monkeypatch.setattr(ws.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(ws.time, "sleep", lambda s: None)
    provider.exchange.mkdir(parents=True)
    with pytest.raises(ws.SandboxClosedEarly):
        provider.wait_until_ready()


def test_a_slow_boot_still_gets_time_after_logon(provider, monkeypatch):
    provider.start_timeout_s = 100
    provider.exchange.mkdir(parents=True)
    (provider.exchange / ws.LOGON_MARKER).write_text("now")
    clock = {"t": 0.0}

    def tick(_s):
        clock["t"] += 5
        if clock["t"] == 140:  # past the 100 s budget, inside the post-logon grace
            (provider.exchange / "ready.json").write_text(json.dumps({"host": "h", "port": 1}))

    monkeypatch.setattr(ws.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(ws.time, "sleep", tick)
    monkeypatch.setattr(ws, "sandbox_running", lambda: True)
    monkeypatch.setattr("operon.environments.remote.wait_for_guest", lambda *a, **k: {"status": "ok"})
    assert provider.wait_until_ready() == {"status": "ok"}


def test_start_waits_for_a_closing_sandbox_to_finish(provider, monkeypatch):
    states = iter([True, True, False])
    monkeypatch.setattr(ws, "sandbox_running", lambda: next(states))
    monkeypatch.setattr(ws.time, "sleep", lambda s: None)
    assert ws.wait_for_sandbox_gone(60) is True


def test_sandbox_running_reads_tasklist(monkeypatch):
    monkeypatch.setattr(ws.sys, "platform", "win32")
    monkeypatch.setattr(ws.shutil, "which", lambda name: "tasklist")

    class Done:
        stdout = "WindowsSandboxServer.exe   1234 Services  0  9,000 K"

    monkeypatch.setattr(ws.subprocess, "run", lambda *a, **k: Done())
    assert ws.sandbox_running() is True
    Done.stdout = "INFO: No tasks are running which match the specified criteria."
    assert ws.sandbox_running() is False


def test_server_and_vm_processes_are_never_killed_by_name():
    assert not {"WindowsSandboxServer.exe", "vmwp.exe"} & set(ws._SANDBOX_PROCESSES)


def test_stop_before_start_does_nothing(provider):
    provider.stop()


def test_guest_command_passes_the_hosts_tool_setting(provider, monkeypatch):
    monkeypatch.setattr(ws, "guest_tools_setting", lambda: "all")
    assert "set OPERON_TOOLS=all" in provider.guest_command()
    monkeypatch.setattr(ws, "guest_tools_setting", lambda: "")
    assert "OPERON_TOOLS" not in provider.guest_command()


def test_guest_command_passes_the_dpi_awareness_flag(provider, monkeypatch):
    monkeypatch.setenv("OPERON_DPI_AWARE", "on")
    assert "set OPERON_DPI_AWARE=on" in provider.guest_command()
    monkeypatch.setenv("OPERON_DPI_AWARE", "off")
    assert "OPERON_DPI_AWARE" not in provider.guest_command()


def test_sandbox_tools_never_run_on_the_host():
    assert WindowsSandboxProvider.runs_on_host is False


def test_a_reported_private_ip_becomes_the_trusted_secret_host(provider, monkeypatch):
    _guest_comes_up(provider, monkeypatch)
    provider.start()
    assert provider.build_executor()._trusted_host == "172.30.0.9"


@pytest.mark.parametrize("host", ["evil.example", "8.8.8.8", "172.30.0.9/x", "not an ip"])
def test_a_guest_reported_host_that_is_not_a_private_ip_is_not_trusted_for_secrets(provider, monkeypatch, host):
    def fake_open(wsb: Path):
        (provider.exchange / "ready.json").write_text(json.dumps({"host": host, "port": 8765}))

    monkeypatch.setattr(provider, "_open", fake_open)
    monkeypatch.setattr("operon.environments.remote.wait_for_guest", lambda *a, **k: {"ok": True})
    provider.start()
    assert provider.build_executor()._trusted_host is None

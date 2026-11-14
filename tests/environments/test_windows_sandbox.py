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


def test_stop_before_start_does_nothing(provider):
    provider.stop()

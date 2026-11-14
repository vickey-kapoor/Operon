"""KAN-80: environment providers (interface, registry, live desktop, loop wiring)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from operon.api.runtime import loops
from operon.environments import (
    DEFAULT_ENVIRONMENT,
    EnvironmentProvider,
    ProviderCheck,
    environment_name,
    get_provider,
    provider_names,
)
from operon.environments.cli import list_environments
from operon.environments.live import LiveDesktopProvider


class _FakeProvider(EnvironmentProvider):
    name = "fake"
    summary = "A test machine."

    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[str] = []
        self.executor = SimpleNamespace(kind="fake-executor")

    def platform(self) -> str:
        return "linux"

    def check(self) -> ProviderCheck:
        self.calls.append("check")
        return ProviderCheck(ok=self.ok, problems=() if self.ok else ("Docker is not running.",))

    def start(self) -> None:
        self.calls.append("start")

    def build_executor(self):
        self.calls.append("build_executor")
        return self.executor

    def stop(self) -> None:
        self.calls.append("stop")


@pytest.fixture(autouse=True)
def _no_built_runtime(monkeypatch):
    monkeypatch.setattr(loops, "_desktop_provider", None)
    monkeypatch.setattr(loops, "_desktop_agent_loop", None)
    monkeypatch.setattr(loops, "_agent_loop", None)


# ── registry ────────────────────────────────────────────────────


def test_default_environment_is_live(monkeypatch):
    monkeypatch.delenv("OPERON_ENVIRONMENT", raising=False)
    assert environment_name() == DEFAULT_ENVIRONMENT == "live"
    assert isinstance(get_provider(), LiveDesktopProvider)


def test_blank_environment_means_live(monkeypatch):
    monkeypatch.setenv("OPERON_ENVIRONMENT", "  ")
    assert environment_name() == "live"


def test_environment_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("OPERON_ENVIRONMENT", " LIVE ")
    assert isinstance(get_provider(), LiveDesktopProvider)


def test_unknown_environment_is_refused_with_the_known_names(monkeypatch):
    monkeypatch.setenv("OPERON_ENVIRONMENT", "moon")
    with pytest.raises(ValueError, match=r"'moon'.*'live'"):
        get_provider()


def test_provider_names_include_live():
    assert "live" in provider_names()


def test_startup_validation_reports_an_unknown_environment(monkeypatch):
    from operon.api.runtime_config import validate_modes

    monkeypatch.setenv("OPERON_ENVIRONMENT", "moon")
    assert any("OPERON_ENVIRONMENT='moon'" in e for e in validate_modes())
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    assert not any("OPERON_ENVIRONMENT" in e for e in validate_modes())


# ── live desktop ────────────────────────────────────────────────


def test_live_follows_desktop_platform(monkeypatch):
    provider = LiveDesktopProvider()
    monkeypatch.delenv("OPERON_DESKTOP_PLATFORM", raising=False)
    assert provider.platform() == "windows"
    monkeypatch.setenv("OPERON_DESKTOP_PLATFORM", "linux")
    assert provider.platform() == "linux"


def test_live_is_always_ready_and_start_stop_are_noops():
    provider = LiveDesktopProvider()
    assert provider.check().ok
    provider.start()
    provider.start()
    provider.stop()
    provider.stop()


def test_live_builds_the_default_desktop_executor(monkeypatch):
    monkeypatch.delenv("OPERON_DESKTOP_PLATFORM", raising=False)
    with patch("operon.executor.desktop.DesktopExecutor", return_value="win") as win:
        assert LiveDesktopProvider().build_executor() == "win"
    win.assert_called_once_with()


def test_live_builds_the_linux_executor_on_linux(monkeypatch):
    monkeypatch.setenv("OPERON_DESKTOP_PLATFORM", "linux")
    with patch("operon.executor.linux_desktop.LinuxDesktopExecutor", return_value="x11") as linux:
        assert LiveDesktopProvider().build_executor() == "x11"
    linux.assert_called_once_with()


# ── loop wiring ─────────────────────────────────────────────────


def test_desktop_provider_checks_then_starts_once(monkeypatch):
    fake = _FakeProvider()
    monkeypatch.setattr(loops, "get_provider", lambda: fake)
    assert loops.desktop_provider() is fake
    assert loops.desktop_provider() is fake
    assert fake.calls == ["check", "start"]


def test_desktop_provider_that_cannot_run_is_refused_and_not_started(monkeypatch):
    fake = _FakeProvider(ok=False)
    monkeypatch.setattr(loops, "get_provider", lambda: fake)
    with pytest.raises(RuntimeError, match="'fake' cannot run here: Docker is not running."):
        loops.desktop_provider()
    assert fake.calls == ["check"]
    assert loops._desktop_provider is None


def test_desktop_loop_runs_on_the_providers_executor(monkeypatch):
    fake = _FakeProvider()
    monkeypatch.setattr(loops, "get_provider", lambda: fake)
    built = {}

    class _Loop:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.executor = kwargs["executor"]

    with (
        patch.object(loops, "build_desktop_services", return_value=SimpleNamespace(perception_service="p", policy_delegate="d")),
        patch.object(loops, "_loop_layers", return_value={}),
        patch.object(loops, "_with_tools", side_effect=lambda executor, config: executor),
        patch.object(loops, "FileBackedRunStore", return_value=SimpleNamespace(root_dir="/tmp/x")),
        patch.object(loops, "FileBackedMemoryStore"),
        patch.object(loops, "ScreenCaptureService") as capture,
        patch.object(loops, "AgentLoop", _Loop),
        patch.object(loops._ws_stream, "set_desktop_executor") as stream,
    ):
        loop = loops.get_desktop_agent_loop()
    assert loop.executor is fake.executor
    assert capture.call_args.kwargs["executor"] is fake.executor
    stream.assert_called_once_with(fake.executor)
    assert fake.calls == ["check", "start", "build_executor"]


def test_close_built_runtime_closes_executors_then_stops_the_provider(monkeypatch):
    order: list[str] = []

    class _Exec:
        async def aclose(self):
            order.append("aclose")

    fake = _FakeProvider()
    fake.stop = lambda: order.append("stop")  # type: ignore[method-assign]
    monkeypatch.setattr(loops, "_desktop_agent_loop", SimpleNamespace(executor=_Exec()))
    monkeypatch.setattr(loops, "_desktop_provider", fake)
    asyncio.run(loops.close_built_runtime())
    assert order == ["aclose", "stop"]


def test_close_built_runtime_survives_a_failing_stop(monkeypatch, caplog):
    fake = _FakeProvider()

    def boom():
        raise OSError("sandbox already gone")

    fake.stop = boom  # type: ignore[method-assign]
    monkeypatch.setattr(loops, "_desktop_provider", fake)
    asyncio.run(loops.close_built_runtime())
    assert "sandbox already gone" in caplog.text


def test_close_built_runtime_without_anything_built_is_a_noop():
    asyncio.run(loops.close_built_runtime())


# ── CLI ─────────────────────────────────────────────────────────


def test_operon_envs_lists_live_as_selected(monkeypatch, capsys):
    monkeypatch.delenv("OPERON_ENVIRONMENT", raising=False)
    assert list_environments() == 0
    out = capsys.readouterr().out
    assert "* live" in out and "ready" in out


def test_operon_envs_flags_an_unknown_selection(monkeypatch, capsys):
    monkeypatch.setenv("OPERON_ENVIRONMENT", "moon")
    assert list_environments() == 1
    assert "'moon' is not a known environment" in capsys.readouterr().out


def test_operon_envs_is_a_cli_command(monkeypatch, capsys):
    from operon import cli

    monkeypatch.delenv("OPERON_ENVIRONMENT", raising=False)
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    assert cli.main(["envs"]) == 0
    assert "live" in capsys.readouterr().out


def test_run_env_flag_sets_the_environment_for_a_local_run(monkeypatch):
    from operon import cli

    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    args = cli.build_parser().parse_args(["run", "--desktop", "--env", "moon", "task"])
    # An unknown name stops at config validation, before any machine is touched.
    assert asyncio.run(cli._run(args)) == 1
    import os

    assert os.environ["OPERON_ENVIRONMENT"] == "moon"


def test_run_env_flag_is_refused_with_request():
    from operon import cli

    args = cli.build_parser().parse_args(["run", "--request", "--env", "live", "task"])
    assert asyncio.run(cli._run(args)) == 2


# ── operon envs --try ───────────────────────────────────────────


def test_try_environment_starts_screenshots_and_stops(monkeypatch, tmp_path, capsys):
    from operon.environments import cli as env_cli
    from operon.models.capture import CaptureFrame

    shot = tmp_path / "frame.png"
    shot.write_bytes(b"png")
    closed = []

    class _Exec:
        async def capture(self):
            return CaptureFrame(artifact_path=str(shot), width=1024, height=768)

        async def aclose(self):
            closed.append(True)

    fake = _FakeProvider()
    fake.executor = _Exec()
    monkeypatch.setattr(env_cli, "get_provider", lambda name: fake)
    assert env_cli.try_environment("fake", out_dir=tmp_path / "out") == 0
    out = capsys.readouterr().out
    assert "1024x768" in out
    saved = list((tmp_path / "out").glob("fake-*.png"))
    assert len(saved) == 1 and saved[0].read_bytes() == b"png"
    assert fake.calls == ["check", "start", "build_executor", "stop"]
    assert closed == [True]


def test_try_environment_reports_a_failed_start_and_still_stops(monkeypatch, capsys):
    from operon.environments import cli as env_cli

    fake = _FakeProvider()

    def boom():
        fake.calls.append("start")
        raise RuntimeError("guest did not start; Guest log: ImportError")

    fake.start = boom  # type: ignore[method-assign]
    monkeypatch.setattr(env_cli, "get_provider", lambda name: fake)
    assert env_cli.try_environment("fake") == 1
    assert "ImportError" in capsys.readouterr().out
    assert fake.calls[-1] == "stop"


def test_try_environment_that_is_not_ready_never_starts(monkeypatch, capsys):
    from operon.environments import cli as env_cli

    fake = _FakeProvider(ok=False)
    monkeypatch.setattr(env_cli, "get_provider", lambda name: fake)
    assert env_cli.try_environment("fake") == 1
    assert fake.calls == ["check"]
    assert "Docker is not running" in capsys.readouterr().out


def test_try_unknown_environment(capsys):
    from operon.environments import cli as env_cli

    assert env_cli.try_environment("moon") == 2


def test_envs_try_flag_routes_to_the_smoke_test(monkeypatch):
    from operon import cli
    from operon.environments import cli as env_cli

    seen = []
    monkeypatch.setattr(env_cli, "try_environment", lambda name: seen.append(name) or 0)
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    assert cli.main(["envs", "--try", "live"]) == 0
    assert seen == ["live"]


# ── tools stay inside a guest machine (Codex P1 on #272) ────────


def _build_desktop_loop_with(monkeypatch, provider):
    monkeypatch.setattr(loops, "get_provider", lambda: provider)
    built = {}

    class _Loop:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.executor = kwargs["executor"]

    with (
        patch.object(loops, "build_desktop_services", return_value=SimpleNamespace(perception_service="p", policy_delegate="d")),
        patch.object(loops, "_loop_layers", return_value={}),
        patch.object(loops, "_with_tools", side_effect=lambda executor, config: ("tools", executor)),
        patch.object(loops, "FileBackedRunStore", return_value=SimpleNamespace(root_dir="/tmp/x")),
        patch.object(loops, "FileBackedMemoryStore"),
        patch.object(loops, "ScreenCaptureService"),
        patch.object(loops, "AgentLoop", _Loop),
        patch.object(loops._ws_stream, "set_desktop_executor"),
    ):
        loops.get_desktop_agent_loop()
    return built["executor"]


def test_host_tools_wrap_a_machine_on_the_host(monkeypatch):
    fake = _FakeProvider()
    assert _build_desktop_loop_with(monkeypatch, fake) == ("tools", fake.executor)


def test_host_tools_never_wrap_a_guest_machine(monkeypatch):
    fake = _FakeProvider()
    fake.runs_on_host = False
    assert _build_desktop_loop_with(monkeypatch, fake) is fake.executor


def test_guest_tools_setting_follows_the_host(monkeypatch):
    from operon.api.runtime_config import RuntimeModeConfig
    from operon.environments import base

    cu = RuntimeModeConfig(backend="computer_use", primary_model="gemini-3.5-flash")
    js = RuntimeModeConfig(backend="json", primary_model="gemini-3.5-flash")
    monkeypatch.setattr("operon.api.runtime_config.desktop_mode_config", lambda: cu)
    monkeypatch.setenv("OPERON_TOOLS", " all ")
    assert base.guest_tools_setting() == "all"
    monkeypatch.delenv("OPERON_TOOLS")
    assert base.guest_tools_setting() == ""  # the guest's default is the host's default
    monkeypatch.setattr("operon.api.runtime_config.desktop_mode_config", lambda: js)
    monkeypatch.setenv("OPERON_TOOLS", "on")
    assert base.guest_tools_setting() == "off"  # only computer use declares tools

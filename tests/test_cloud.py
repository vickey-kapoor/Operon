"""`operon cloud` (KAN-108): the az and ssh command lines, and that every path deallocates."""

from __future__ import annotations

import argparse

import pytest

from operon import cloud
from operon.cli import build_parser

CFG = cloud.CloudConfig()


class FakeRunner:
    def __init__(self, fail_ssh: int = 0) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.fail_ssh = fail_ssh

    def __call__(self, argv, input_text=None) -> str:
        argv = list(argv)
        self.calls.append((argv, input_text))
        if argv[:3] == ["az", "vm", "show"]:
            return "20.1.2.3\n"
        if argv[0] == "ssh" and "test -f /var/lib/operon/ready" in argv and self.fail_ssh:
            self.fail_ssh -= 1
            raise RuntimeError("not yet")
        if argv[0] == "ssh" and "cat /etc/operon/api-key" in argv:
            return "k3y\n"
        if argv[:3] == ["az", "vm", "create"]:
            return '{"publicIpAddress": "20.1.2.3"}'
        return ""

    def verbs(self) -> list[str]:
        return [" ".join(argv[:3]) for argv, _ in self.calls]


def test_settings_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_CLOUD_VM", "other")
    monkeypatch.setenv("OPERON_CLOUD_LOCAL_PORT", "9000")
    cfg = cloud.CloudConfig.from_env()
    assert cfg.vm == "other" and cfg.local_port == 9000 and cfg.resource_group == "operon-bench"


def test_create_opens_no_public_port_but_ssh_from_one_ip(tmp_path) -> None:
    create = cloud.az_create(CFG, tmp_path / "ci.yaml")
    assert create[create.index("--nsg-rule") + 1] == "NONE"
    assert create[create.index("--image") + 1] == cloud.IMAGE
    rule = cloud.az_allow_ssh(CFG, "1.2.3.4")
    assert rule[rule.index("--source-address-prefixes") + 1] == "1.2.3.4/32"
    assert rule[rule.index("--destination-port-ranges") + 1] == "22"


def test_the_tunnel_binds_only_localhost() -> None:
    tunnel = cloud.ssh_tunnel(CFG, "20.1.2.3")
    assert tunnel[tunnel.index("-L") + 1] == "127.0.0.1:8090:127.0.0.1:8080"
    assert tunnel[-1] == "azureuser@20.1.2.3"


def test_cloud_init_installs_the_requested_branch(tmp_path) -> None:
    text = cloud.render_cloud_init("my-branch")
    assert "OPERON_REF=my-branch" in text and "__OPERON_REF__" not in text
    assert "OPERON_DESKTOP_PLATFORM=linux" in text
    assert "--host 127.0.0.1 --port 8080" in text


def test_create_locks_ssh_and_sets_the_auto_shutdown(monkeypatch) -> None:
    run = FakeRunner()
    args = argparse.Namespace(ref="main", allow_ip="5.6.7.8")
    assert cloud.cmd_create(args, CFG, run, out=lambda *_: None) == 0
    assert run.verbs() == ["az vm create", "az network nsg", "az vm auto-shutdown"]
    custom = run.calls[0][0][run.calls[0][0].index("--custom-data") + 1]
    assert custom.endswith("cloud-init.yaml")


def test_set_key_sends_the_key_on_stdin_and_deallocates(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret")
    monkeypatch.setattr(cloud.time, "sleep", lambda s: None)
    run = FakeRunner(fail_ssh=1)
    assert cloud.cmd_set_key(argparse.Namespace(keep=False), CFG, run, out=lambda *_: None) == 0
    sent = [(argv, data) for argv, data in run.calls if data]
    assert len(sent) == 1 and sent[0][1] == "GEMINI_API_KEY=gem-secret\n"
    assert all("gem-secret" not in " ".join(argv) for argv, _ in run.calls)
    assert run.verbs()[0] == "az vm start" and run.verbs()[-1] == "az vm deallocate"


def test_a_failed_start_still_deallocates(monkeypatch) -> None:
    monkeypatch.setattr(cloud, "_port_free", lambda port: True)

    def wait_never(*args, **kwargs):
        raise RuntimeError("setup never finished")

    monkeypatch.setattr(cloud, "wait_for_ssh", wait_never)
    run = FakeRunner()
    with pytest.raises(RuntimeError, match="never finished"):
        with cloud._Session(CFG, run, out=lambda *_: None):
            pass
    assert run.verbs()[-1] == "az vm deallocate"


def test_keep_leaves_the_vm_running(monkeypatch) -> None:
    run = FakeRunner()
    session = cloud._Session(CFG, run, keep=True, out=lambda *_: None)
    session.__exit__(None, None, None)
    assert "az vm deallocate" not in run.verbs()


def test_stop_deallocates() -> None:
    run = FakeRunner()
    cloud.cmd_stop(argparse.Namespace(), CFG, run, out=lambda *_: None)
    assert run.verbs() == ["az vm deallocate"]


def test_the_cli_has_the_cloud_commands() -> None:
    args = build_parser().parse_args(["cloud", "run", "open mousepad", "--keep"])
    assert args.cloud_command == "run" and args.task == "open mousepad" and args.keep
    for name in ("create", "set-key", "open", "stop"):
        assert build_parser().parse_args(["cloud", name]).cloud_command == name


def test_set_key_deallocates_when_the_start_fails(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret")

    class NoIp(FakeRunner):
        def __call__(self, argv, input_text=None) -> str:
            out = super().__call__(argv, input_text)
            return "" if list(argv)[:3] == ["az", "vm", "show"] else out

    run = NoIp()
    with pytest.raises(RuntimeError, match="no public IP"):
        cloud.cmd_set_key(argparse.Namespace(keep=False), CFG, run, out=lambda *_: None)
    assert run.verbs()[-1] == "az vm deallocate"


def test_run_command_returns_stdout_and_raises_with_stderr(monkeypatch) -> None:
    import subprocess

    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        code = 0 if argv[1] == "ok" else 3
        return subprocess.CompletedProcess(argv, code, stdout="out\n", stderr="boom")

    monkeypatch.setattr(cloud.shutil, "which", lambda name: f"/bin/{name}.cmd")
    monkeypatch.setattr(cloud.subprocess, "run", fake_run)

    assert cloud.run_command(["az", "ok"]) == "out\n"
    with pytest.raises(RuntimeError, match="az bad failed: boom"):
        cloud.run_command(["az", "bad"])
    assert seen[0][0] == "/bin/az.cmd"  # az resolves to az.cmd on Windows


def test_read_api_key_refuses_an_empty_key() -> None:
    class Empty(FakeRunner):
        def __call__(self, argv, input_text=None) -> str:
            return ""

    with pytest.raises(RuntimeError, match="no API key"):
        cloud.read_api_key(CFG, "20.1.2.3", Empty())


def test_wait_for_ssh_gives_up_after_its_timeout(monkeypatch) -> None:
    clock = iter([0.0, 100.0, 400.0])
    monkeypatch.setattr(cloud.time, "monotonic", lambda: next(clock))
    run = FakeRunner(fail_ssh=99)

    with pytest.raises(RuntimeError, match="within 300 s"):
        cloud.wait_for_ssh(CFG, "20.1.2.3", run, sleep=lambda s: None)
    assert len(run.calls) == 2


def test_wait_for_api_returns_once_health_answers(monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    attempts: list[str] = []

    def urlopen(url, timeout):
        attempts.append(url)
        if len(attempts) < 3:
            raise OSError("connection refused")
        return Response()

    monkeypatch.setattr(cloud.urllib.request, "urlopen", urlopen)
    cloud.wait_for_api(8090, sleep=lambda s: None)
    assert attempts == ["http://127.0.0.1:8090/health"] * 3


def test_wait_for_api_gives_up_after_its_timeout(monkeypatch) -> None:
    clock = iter([0.0, 200.0])
    monkeypatch.setattr(cloud.time, "monotonic", lambda: next(clock))

    def refuse(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(cloud.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="did not answer"):
        cloud.wait_for_api(8090, sleep=lambda s: None)


def test_a_busy_local_port_fails_before_the_vm_starts(monkeypatch) -> None:
    monkeypatch.setattr(cloud, "_port_free", lambda port: False)
    run = FakeRunner()

    with pytest.raises(RuntimeError, match="port 8090 is in use"):
        with cloud._Session(CFG, run, out=lambda *_: None):
            pass
    assert run.calls == []  # nothing started, so nothing billed


def test_exit_kills_a_tunnel_that_will_not_stop() -> None:
    import subprocess

    class StuckTunnel:
        killed = False

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("ssh", timeout)

        def kill(self) -> None:
            self.killed = True

    run = FakeRunner()
    session = cloud._Session(CFG, run, out=lambda *_: None)
    tunnel = session.tunnel = StuckTunnel()
    session.__exit__(None, None, None)

    assert tunnel.killed and session.tunnel is None
    assert run.verbs() == ["az vm deallocate"]


def test_set_key_with_no_key_starts_nothing(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cloud.getpass, "getpass", lambda prompt: "  ")
    run = FakeRunner()

    assert cloud.cmd_set_key(argparse.Namespace(keep=False), CFG, run, out=lambda *_: None) == 1
    assert run.calls == []


def test_set_key_with_keep_leaves_the_vm_running(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret")
    run = FakeRunner()

    assert cloud.cmd_set_key(argparse.Namespace(keep=True), CFG, run, out=lambda *_: None) == 0
    assert "az vm deallocate" not in run.verbs()


@pytest.mark.parametrize(
    ("browser", "mode", "expected"),
    [
        (False, None, {"intent": "t", "environment": "desktop", "max_steps": 7}),
        (True, "ask", {"intent": "t", "environment": "browser", "max_steps": 7, "permission_mode": "ask"}),
    ],
)
def test_run_sends_the_task_through_the_session(monkeypatch, browser, mode, expected) -> None:
    import contextlib

    from operon import cli

    sessions: list[bool] = []

    @contextlib.contextmanager
    def fake_session(cfg, run, *, keep=False, out=print):
        sessions.append(keep)
        yield argparse.Namespace(base_url="http://127.0.0.1:8090", api_key="k3y")

    sent: list[dict] = []

    async def fake_drive(client, payload, **kwargs):
        sent.append(payload)
        return 0, None

    monkeypatch.setattr(cloud, "_Session", fake_session)
    monkeypatch.setattr(cli, "drive", fake_drive)
    args = argparse.Namespace(task="t", browser=browser, mode=mode, max_steps=7, keep=False)

    assert cloud.cmd_run(args, CFG, FakeRunner(), out=lambda *_: None) == 0
    assert sent == [expected] and sessions == [False]


async def test_run_prints_a_link_to_the_started_run() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={"run_id": "r-42"})

    lines: list[str] = []
    client = cloud._LinkingClient("http://127.0.0.1:8090", "k3y", out=lines.append)
    client._http = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(handler))
    await client.start({"intent": "x"})
    await client.close()
    assert lines == ["Watch it live: http://127.0.0.1:8090/app#key=k3y&run=r-42"]

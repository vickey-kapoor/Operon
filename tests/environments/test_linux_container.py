"""KAN-80 stage 3: the Linux desktop container provider (offline: docker is faked)."""

from __future__ import annotations

import subprocess

import pytest

from operon.environments import get_provider
from operon.environments import linux_container as lc
from operon.environments.linux_container import LinuxContainerProvider, image_tag
from operon.environments.remote import RemoteDesktopExecutor


class FakeDocker:
    """Answers `docker ...` argv lists; records each call."""

    def __init__(self, *, running=True, image=True, port="127.0.0.1:49153\n", run_rc=0):
        self.calls: list[list[str]] = []
        self.envs: list[dict | None] = []
        self.running, self.image, self.port, self.run_rc = running, image, port, run_rc
        self.built = False

    def __call__(self, argv, **kwargs):
        assert argv[0] == "docker"
        args = argv[1:]
        self.calls.append(args)
        self.envs.append(kwargs.get("env"))
        verb = args[0]
        if verb == "info":
            return _done(0 if self.running else 1, "29.0.0")
        if verb == "image":
            return _done(0 if (self.image or self.built) else 1)
        if verb == "build":
            self.built = True
            return _done(0)
        if verb == "run":
            return _done(self.run_rc, "abc123", "" if self.run_rc == 0 else "port is already allocated")
        if verb == "port":
            return _done(0, self.port)
        if verb == "logs":
            return _done(0, "ImportError: no display")
        return _done(0)

    def verbs(self):
        return [c[0] for c in self.calls]


def _done(rc, out="", err=""):
    return subprocess.CompletedProcess([], rc, out, err)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "docker" / "linux-desktop").mkdir(parents=True)
    (tmp_path / "docker" / "linux-desktop" / "Dockerfile").write_text("FROM python\n")
    (tmp_path / "docker" / "linux-desktop" / "entrypoint.sh").write_text("#!/bin/sh\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='operon'\n")
    (tmp_path / "src" / "operon").mkdir(parents=True)
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("OPERON_CONTAINER_IMAGE", "OPERON_CONTAINER_START_TIMEOUT", "OPERON_CONTAINER_CA_BUNDLE",
                 "OPERON_TOOLS"):
        monkeypatch.delenv(name, raising=False)


def _healthy(monkeypatch, seen=None):
    def fake_wait(endpoint, token, *, timeout_s):
        if seen is not None:
            seen.update(endpoint=endpoint, token=token)
        return {"ok": True}

    monkeypatch.setattr("operon.environments.remote.wait_for_guest", fake_wait)


def test_registered_and_linux():
    provider = get_provider("linux-container")
    assert isinstance(provider, LinuxContainerProvider)
    assert provider.platform() == "linux"
    assert provider.runs_on_host is False


def test_image_tag_follows_the_files_that_shape_it(repo):
    first = image_tag(repo)
    assert first.startswith("operon-linux-desktop:") and len(first.split(":")[1]) == 12
    assert image_tag(repo) == first
    (repo / "pyproject.toml").write_text("[project]\nname='operon'\ndependencies=['x']\n")
    assert image_tag(repo) != first


def test_image_override(monkeypatch, repo):
    monkeypatch.setenv("OPERON_CONTAINER_IMAGE", "me/desktop:1")
    assert LinuxContainerProvider(runner=FakeDocker(), root=repo).image == "me/desktop:1"


# ── check ───────────────────────────────────────────────────────


def test_check_ready_when_docker_runs(repo):
    assert LinuxContainerProvider(runner=FakeDocker(), root=repo).check().ok


def test_check_says_start_docker_when_it_is_not_running(repo):
    check = LinuxContainerProvider(runner=FakeDocker(running=False), root=repo).check()
    assert not check.ok and "Start Docker Desktop" in check.problems[0]


def test_check_without_docker_installed(monkeypatch, repo):
    monkeypatch.setattr(lc.shutil, "which", lambda name: None)
    p = LinuxContainerProvider(root=repo)  # real runner, but no docker on PATH
    check = p.check()
    assert not check.ok and "Install Docker Desktop" in check.problems[0]


def test_check_without_image_or_checkout(monkeypatch):
    monkeypatch.setattr(lc, "source_checkout", lambda: None)
    p = LinuxContainerProvider(runner=FakeDocker(image=False), root=None)
    check = p.check()
    assert not check.ok and "OPERON_CONTAINER_IMAGE" in check.problems[0]


# ── start / stop ────────────────────────────────────────────────


def test_start_runs_a_container_and_builds_a_remote_executor(monkeypatch, repo):
    docker = FakeDocker()
    seen = {}
    _healthy(monkeypatch, seen)
    p = LinuxContainerProvider(runner=docker, root=repo)
    p.start()
    p.start()  # idempotent
    assert docker.verbs() == ["image", "run", "port"]
    run = docker.calls[1]
    assert "--rm" in run and "-d" in run
    assert run[run.index("-p") + 1] == "127.0.0.1::8765"  # loopback only, port picked by docker
    assert run[run.index("-e") + 1] == "OPERON_GUEST_TOKEN"  # value from the environment, not argv
    assert not any(p._token in arg for arg in run)
    assert docker.envs[1]["OPERON_GUEST_TOKEN"] == p._token
    assert f"{repo / 'src'}:/opt/operon-src:ro" in run
    assert run[-1] == p.image
    assert seen == {"endpoint": "http://127.0.0.1:49153", "token": p._token}
    executor = p.build_executor()
    assert isinstance(executor, RemoteDesktopExecutor) and executor.base_url == "http://127.0.0.1:49153"


def test_start_builds_the_image_the_first_time(monkeypatch, repo):
    docker = FakeDocker(image=False)
    _healthy(monkeypatch)
    monkeypatch.setenv("OPERON_CONTAINER_CA_BUNDLE", "/certs/ca.pem")
    LinuxContainerProvider(runner=docker, root=repo).start()
    build = docker.calls[1]
    assert build[0] == "build"
    assert build[build.index("-f") + 1] == str(repo / "docker" / "linux-desktop" / "Dockerfile")
    assert "id=ca_bundle,src=/certs/ca.pem" in build
    assert build[-1] == str(repo)


def test_start_passes_the_hosts_tool_setting(monkeypatch, repo):
    docker = FakeDocker()
    _healthy(monkeypatch)
    monkeypatch.setattr(lc, "guest_tools_setting", lambda: "on")
    LinuxContainerProvider(runner=docker, root=repo).start()
    assert "OPERON_TOOLS=on" in docker.calls[1]


def test_a_failed_docker_run_is_reported(monkeypatch, repo):
    p = LinuxContainerProvider(runner=FakeDocker(run_rc=125), root=repo)
    with pytest.raises(RuntimeError, match="port is already allocated"):
        p.start()
    assert p.container is None


def test_a_guest_that_never_answers_removes_the_container_and_shows_its_log(monkeypatch, repo):
    docker = FakeDocker()

    def never(endpoint, token, *, timeout_s):
        raise RuntimeError("did not start within 120s")

    monkeypatch.setattr("operon.environments.remote.wait_for_guest", never)
    p = LinuxContainerProvider(runner=docker, root=repo)
    with pytest.raises(RuntimeError, match="ImportError: no display"):
        p.start()
    assert docker.verbs()[-1] == "rm" and p.container is None


def test_no_published_port_is_an_error(monkeypatch, repo):
    _healthy(monkeypatch)
    p = LinuxContainerProvider(runner=FakeDocker(port=""), root=repo)
    with pytest.raises(RuntimeError, match="no published guest port"):
        p.start()


def test_stop_removes_the_container_once(monkeypatch, repo):
    docker = FakeDocker()
    _healthy(monkeypatch)
    p = LinuxContainerProvider(runner=docker, root=repo)
    p.start()
    name = p.container
    p.stop()
    p.stop()
    assert docker.calls[-1] == ["rm", "-f", name]
    assert docker.verbs().count("rm") == 1


def test_build_executor_before_start_is_an_error(repo):
    with pytest.raises(RuntimeError, match="start"):
        LinuxContainerProvider(runner=FakeDocker(), root=repo).build_executor()

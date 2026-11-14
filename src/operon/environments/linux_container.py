"""Linux desktop container provider (KAN-80, stage 3): an XFCE desktop in Docker.

``start()`` makes sure the image exists (``docker/linux-desktop/Dockerfile``, built
from a source checkout the first time, a few minutes), then runs one container:
Xvfb + XFCE + ``operon guest``. Its guest port is published on 127.0.0.1 only, at a
port Docker picks; the token goes in through the environment of the ``docker``
client, never on its command line. The host's current ``src`` folder is mounted
read-only over the image's copy, so code changes need no rebuild.

The image tag carries a hash of the files that shape it (Dockerfile, entrypoint,
``pyproject.toml``), so a change to the dependencies builds a new image by itself.
The Gemini key, ``.env`` and ``~/.operon`` never go in. ``stop()`` removes the container.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from operon.core.paths import source_checkout
from operon.environments.base import (
    EnvironmentProvider,
    ProviderCheck,
    guest_tools_setting,
)
from operon.environments.guest import DEFAULT_GUEST_PORT, new_token

logger = logging.getLogger(__name__)

IMAGE_REPO = "operon-linux-desktop"
DOCKERFILE = Path("docker") / "linux-desktop" / "Dockerfile"
_HASHED_FILES = (DOCKERFILE, Path("docker") / "linux-desktop" / "entrypoint.sh", Path("pyproject.toml"))

Runner = Callable[..., subprocess.CompletedProcess]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def image_tag(root: Path) -> str:
    """``operon-linux-desktop:<hash of the files that shape the image>``."""
    digest = hashlib.sha256()
    for rel in _HASHED_FILES:
        path = root / rel
        digest.update(rel.as_posix().encode())
        digest.update(path.read_bytes() if path.exists() else b"")
    return f"{IMAGE_REPO}:{digest.hexdigest()[:12]}"


class LinuxContainerProvider(EnvironmentProvider):
    name = "linux-container"
    summary = "A Linux (XFCE) desktop in a Docker container on this machine; needs Docker Desktop."
    runs_on_host = False

    def __init__(self, *, runner: Runner | None = None, root: Path | None = None) -> None:
        self._run = runner or subprocess.run
        self.root = root if root is not None else source_checkout()
        self.image = os.getenv("OPERON_CONTAINER_IMAGE", "").strip() or (
            image_tag(self.root) if self.root is not None else ""
        )
        self.start_timeout_s = _int_env("OPERON_CONTAINER_START_TIMEOUT", 120)
        self.container: str | None = None
        self._token: str | None = None
        self._endpoint: str | None = None

    def platform(self) -> str:
        return "linux"

    # ── docker ───────────────────────────────────────────────────

    def _docker(self, args: Sequence[str], *, timeout: float = 60, env: dict | None = None,
                capture: bool = True) -> subprocess.CompletedProcess:
        return self._run(
            ["docker", *args],
            capture_output=capture,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )

    def _image_present(self) -> bool:
        return bool(self.image) and self._docker(["image", "inspect", self.image]).returncode == 0

    # ── check ────────────────────────────────────────────────────

    def check(self) -> ProviderCheck:
        if shutil.which("docker") is None and self._run is subprocess.run:
            return ProviderCheck(False, ("Docker is not installed. Install Docker Desktop, then start it.",))
        try:
            info = self._docker(["info", "--format", "{{.ServerVersion}}"], timeout=20)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ProviderCheck(False, (f"Docker did not answer ({exc}). Start Docker Desktop.",))
        if info.returncode != 0:
            return ProviderCheck(False, ("Docker is not running. Start Docker Desktop and wait for it to say it is running.",))
        if not self._image_present() and self.root is None:
            return ProviderCheck(
                False,
                ("The desktop image is not built and this is not a source checkout to build it from; "
                 "set OPERON_CONTAINER_IMAGE to a built image.",),
            )
        return ProviderCheck(ok=True)

    # ── lifecycle ────────────────────────────────────────────────

    def build_image(self) -> None:
        """Build the image from the source checkout (first start, or after a dependency change)."""
        if self.root is None:
            raise RuntimeError("linux-container: no source checkout to build the image from")
        logger.info("linux-container: building %s (first time takes a few minutes)", self.image)
        print(f"linux-container: building the desktop image {self.image} (first time only, a few minutes)...",
              flush=True)
        args = ["build", "-f", str(self.root / DOCKERFILE), "-t", self.image]
        ca_bundle = os.getenv("OPERON_CONTAINER_CA_BUNDLE", "").strip()
        if ca_bundle:  # behind a TLS-inspecting proxy: pip in the build trusts this CA
            args += ["--secret", f"id=ca_bundle,src={ca_bundle}"]
        result = self._docker([*args, str(self.root)], timeout=1800, capture=False)
        if result.returncode != 0:
            raise RuntimeError(f"linux-container: docker build failed (exit {result.returncode})")

    def run_args(self) -> list[str]:
        """``docker run`` arguments (the token comes from the client's environment)."""
        assert self.container is not None
        args = [
            "run", "-d", "--rm",
            "--name", self.container,
            "-p", f"127.0.0.1::{DEFAULT_GUEST_PORT}",
            "--shm-size", "1g",
            "-e", "OPERON_GUEST_TOKEN",
        ]
        tools = guest_tools_setting()
        if tools:
            args += ["-e", f"OPERON_TOOLS={tools}"]
        if self.root is not None and (self.root / "src" / "operon").is_dir():
            args += ["-v", f"{self.root / 'src'}:/opt/operon-src:ro"]
        args.append(self.image)
        return args

    def published_endpoint(self) -> str:
        out = self._docker(["port", self.container or "", f"{DEFAULT_GUEST_PORT}/tcp"]).stdout or ""
        for line in out.splitlines():
            host, _, port = line.strip().rpartition(":")
            if port.isdigit() and host in {"127.0.0.1", "0.0.0.0"}:
                return f"http://127.0.0.1:{port}"
        raise RuntimeError(f"linux-container: no published guest port ({out.strip() or 'docker port said nothing'})")

    def start(self) -> None:
        if self.container is not None:
            return
        if not self._image_present():
            self.build_image()
        self._token = new_token()
        self.container = f"operon-desktop-{uuid.uuid4().hex[:8]}"
        env = {**os.environ, "OPERON_GUEST_TOKEN": self._token}
        result = self._docker(self.run_args(), env=env, timeout=120)
        if result.returncode != 0:
            self.container = None
            raise RuntimeError(f"linux-container: docker run failed: {(result.stderr or result.stdout).strip()}")
        try:
            self._endpoint = self.published_endpoint()
            from operon.environments.remote import wait_for_guest

            wait_for_guest(self._endpoint, self._token, timeout_s=self.start_timeout_s)
        except Exception as exc:
            logs = self._docker(["logs", "--tail", "30", self.container]).stdout or ""
            self.stop()
            raise RuntimeError(f"{exc}\nContainer log:\n{logs.strip()}") from exc
        logger.info("linux-container: %s ready at %s", self.container, self._endpoint)

    def build_executor(self):
        if self._endpoint is None or self._token is None:
            raise RuntimeError("linux-container: start() must finish before build_executor()")
        from operon.environments.remote import RemoteDesktopExecutor

        return RemoteDesktopExecutor(self._endpoint, self._token)

    def stop(self) -> None:
        if self.container is None:
            return
        name, self.container, self._endpoint = self.container, None, None
        try:
            self._docker(["rm", "-f", name], timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("linux-container: could not remove %s", name, exc_info=True)

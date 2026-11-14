"""``operon cloud``: Operon's own computer on an Azure Linux VM (KAN-108).

Runs on your PC and needs only the Azure CLI (``az``, signed in) and ``ssh``. The VM
runs an XFCE desktop and the Operon API on its own 127.0.0.1:8080; nothing on it
listens on a public port except SSH. This command starts the VM, opens an SSH
tunnel to the API, prints the task page's link, and deallocates the VM when you
are done, so it is billed only while a task runs (its disk and IP are kept).

    operon cloud create                 create the VM (once; asks Azure, costs money)
    operon cloud set-key                put your GEMINI_API_KEY on the VM
    operon cloud open                   start it, open the task page, deallocate on Ctrl+C
    operon cloud run "task"             start it, run one desktop task, deallocate
    operon cloud stop                   deallocate it now

Settings (environment or .env): OPERON_CLOUD_RESOURCE_GROUP (operon-bench),
OPERON_CLOUD_VM (operon-linux), OPERON_CLOUD_LOCATION (westus2),
OPERON_CLOUD_SIZE (Standard_D2s_v4), OPERON_CLOUD_ADMIN (azureuser),
OPERON_CLOUD_LOCAL_PORT (8090).
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from operon.cli import RemoteClient

IMAGE = "Debian:debian-12:12-gen2:latest"
REMOTE_API_PORT = 8080
# A daily safety net: Azure deallocates the VM at this UTC time even if a session was
# left open (08:00 UTC is 2 AM in Chicago).
AUTO_SHUTDOWN_UTC = "0800"
CLOUD_INIT = Path(__file__).resolve().parents[2] / "scripts" / "azure-linux-computer" / "cloud-init.yaml"


@dataclass(frozen=True)
class CloudConfig:
    resource_group: str = "operon-bench"
    vm: str = "operon-linux"
    location: str = "westus2"
    size: str = "Standard_D2s_v4"
    admin: str = "azureuser"
    local_port: int = 8090

    @classmethod
    def from_env(cls) -> CloudConfig:
        env = os.getenv
        return cls(
            resource_group=env("OPERON_CLOUD_RESOURCE_GROUP") or cls.resource_group,
            vm=env("OPERON_CLOUD_VM") or cls.vm,
            location=env("OPERON_CLOUD_LOCATION") or cls.location,
            size=env("OPERON_CLOUD_SIZE") or cls.size,
            admin=env("OPERON_CLOUD_ADMIN") or cls.admin,
            local_port=int(env("OPERON_CLOUD_LOCAL_PORT") or cls.local_port),
        )


# run(argv) or run(argv, input_text=...) -> stdout
Runner = Callable[..., str]


def run_command(argv: Sequence[str], *, input_text: str | None = None) -> str:
    """Run a command and return its stdout; raise with its stderr when it fails."""
    exe = shutil.which(argv[0]) or argv[0]  # az is az.cmd on Windows
    result = subprocess.run([exe, *argv[1:]], capture_output=True, text=True, input=input_text)
    if result.returncode != 0:
        raise RuntimeError(f"{argv[0]} {argv[1] if len(argv) > 1 else ''} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


# ── az command lines (pure, so tests can check them) ─────────────────────────


def az_start(cfg: CloudConfig) -> list[str]:
    return ["az", "vm", "start", "-g", cfg.resource_group, "-n", cfg.vm, "-o", "none"]


def az_deallocate(cfg: CloudConfig) -> list[str]:
    return ["az", "vm", "deallocate", "-g", cfg.resource_group, "-n", cfg.vm, "-o", "none"]


def az_public_ip(cfg: CloudConfig) -> list[str]:
    return ["az", "vm", "show", "-d", "-g", cfg.resource_group, "-n", cfg.vm, "--query", "publicIps", "-o", "tsv"]


def az_create(cfg: CloudConfig, custom_data: Path) -> list[str]:
    return [
        "az", "vm", "create",
        "-g", cfg.resource_group, "-n", cfg.vm, "-l", cfg.location,
        "--image", IMAGE, "--size", cfg.size,
        "--admin-username", cfg.admin, "--generate-ssh-keys",
        "--authentication-type", "ssh",
        "--os-disk-size-gb", "30", "--storage-sku", "StandardSSD_LRS",
        "--public-ip-sku", "Standard",
        "--nsg-rule", "NONE",
        "--custom-data", str(custom_data),
        "-o", "json",
    ]


def az_allow_ssh(cfg: CloudConfig, source_ip: str) -> list[str]:
    return [
        "az", "network", "nsg", "rule", "create",
        "-g", cfg.resource_group, "--nsg-name", f"{cfg.vm}NSG",
        "-n", "ssh-from-owner", "--priority", "1000",
        "--access", "Allow", "--protocol", "Tcp", "--direction", "Inbound",
        "--source-address-prefixes", f"{source_ip}/32", "--destination-port-ranges", "22",
        "-o", "none",
    ]


def az_auto_shutdown(cfg: CloudConfig) -> list[str]:
    return ["az", "vm", "auto-shutdown", "-g", cfg.resource_group, "-n", cfg.vm, "--time", AUTO_SHUTDOWN_UTC, "-o", "none"]


def ssh_base(cfg: CloudConfig, ip: str) -> list[str]:
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=30",
        "-o", "ConnectTimeout=10",
        f"{cfg.admin}@{ip}",
    ]


def ssh_tunnel(cfg: CloudConfig, ip: str) -> list[str]:
    base = ssh_base(cfg, ip)
    return [base[0], "-N", "-o", "ExitOnForwardFailure=yes",
            "-L", f"127.0.0.1:{cfg.local_port}:127.0.0.1:{REMOTE_API_PORT}", *base[1:]]


def render_cloud_init(ref: str, source: Path = CLOUD_INIT) -> str:
    return source.read_text(encoding="utf-8").replace("__OPERON_REF__", ref)


# ── steps ─────────────────────────────────────────────────────────────────────


def start_vm(cfg: CloudConfig, run: Runner, out=print) -> str:
    out(f"Starting {cfg.vm} (billing starts now)...")
    run(az_start(cfg))
    ip = run(az_public_ip(cfg)).strip()
    if not ip:
        raise RuntimeError(f"{cfg.vm} has no public IP")
    return ip


def deallocate_vm(cfg: CloudConfig, run: Runner, out=print) -> None:
    out(f"Deallocating {cfg.vm}...")
    run(az_deallocate(cfg))
    out("Deallocated: no compute billing until the next start.")


def wait_for_ssh(cfg: CloudConfig, ip: str, run: Runner, *, timeout_s: float = 300.0, sleep=None) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            run([*ssh_base(cfg, ip), "test -f /var/lib/operon/ready"])
            return
        except RuntimeError:
            if time.monotonic() > deadline:
                raise RuntimeError(f"{cfg.vm} did not finish setting up within {int(timeout_s)} s") from None
            (sleep or time.sleep)(5)  # looked up per call so tests can patch time.sleep


def read_api_key(cfg: CloudConfig, ip: str, run: Runner) -> str:
    key = run([*ssh_base(cfg, ip), "cat /etc/operon/api-key"]).strip()
    if not key:
        raise RuntimeError("the VM has no API key yet (setup still running?)")
    return key


def wait_for_api(port: int, *, timeout_s: float = 120.0, sleep=None) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - fixed localhost URL
                if response.status < 500:
                    return
        except Exception:  # noqa: BLE001 - not up yet
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("the Operon API on the VM did not answer through the tunnel")
        (sleep or time.sleep)(2)


def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


class _Session:
    """Start the VM and open the tunnel; always deallocate on exit unless ``keep``."""

    def __init__(self, cfg: CloudConfig, run: Runner, *, keep: bool = False, out=print) -> None:
        self.cfg, self.run, self.keep, self.out = cfg, run, keep, out
        self.tunnel: subprocess.Popen | None = None
        self.api_key = ""

    def __enter__(self) -> _Session:
        if not _port_free(self.cfg.local_port):
            raise RuntimeError(f"port {self.cfg.local_port} is in use here; set OPERON_CLOUD_LOCAL_PORT")
        try:
            ip = start_vm(self.cfg, self.run, self.out)
            self.out(f"Waiting for {ip} to be ready...")
            wait_for_ssh(self.cfg, ip, self.run)
            self.api_key = read_api_key(self.cfg, ip, self.run)
            argv = ssh_tunnel(self.cfg, ip)
            self.tunnel = subprocess.Popen([shutil.which(argv[0]) or argv[0], *argv[1:]])
            wait_for_api(self.cfg.local_port)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.cfg.local_port}"

    def __exit__(self, *_exc) -> None:
        if self.tunnel is not None:
            self.tunnel.terminate()
            try:
                self.tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tunnel.kill()
            self.tunnel = None
        if self.keep:
            self.out(f"Left {self.cfg.vm} running (--keep). Stop it with: operon cloud stop")
        else:
            deallocate_vm(self.cfg, self.run, self.out)


# ── commands ──────────────────────────────────────────────────────────────────


def _my_ip() -> str:
    with urllib.request.urlopen("https://api.ipify.org", timeout=10) as response:  # noqa: S310 - fixed URL
        return response.read().decode().strip()


def cmd_create(args: argparse.Namespace, cfg: CloudConfig, run: Runner = run_command, out=print) -> int:
    source_ip = args.allow_ip or _my_ip()
    with tempfile.TemporaryDirectory() as tmp:
        custom = Path(tmp) / "cloud-init.yaml"
        custom.write_text(render_cloud_init(args.ref), encoding="utf-8")
        out(f"Creating {cfg.vm} ({cfg.size}, {cfg.location}); setup takes about 10 minutes...")
        created = json.loads(run(az_create(cfg, custom)) or "{}")
    run(az_allow_ssh(cfg, source_ip))
    out(f"SSH allowed only from {source_ip}.")
    run(az_auto_shutdown(cfg))
    out(f"Daily auto-deallocate at {AUTO_SHUTDOWN_UTC} UTC.")
    out(f"Created, public IP {created.get('publicIpAddress', '?')}. Next: operon cloud set-key")
    return 0


def cmd_set_key(args: argparse.Namespace, cfg: CloudConfig, run: Runner = run_command, out=print) -> int:
    key = os.getenv("GEMINI_API_KEY") or getpass.getpass("GEMINI_API_KEY (not shown): ")
    if not key.strip():
        out("No key given.")
        return 1
    try:
        ip = start_vm(cfg, run, out)
        wait_for_ssh(cfg, ip, run)
        # The key goes over SSH's stdin, so it never appears in a command line or in Azure's logs.
        script = "sudo tee /etc/operon/gemini.env >/dev/null && sudo chmod 600 /etc/operon/gemini.env && sudo systemctl restart operon-api"
        run([*ssh_base(cfg, ip), script], input_text=f"GEMINI_API_KEY={key.strip()}\n")
        out("Gemini key saved on the VM.")
    finally:
        if not args.keep:
            deallocate_vm(cfg, run, out)
    return 0


def cmd_open(args: argparse.Namespace, cfg: CloudConfig, run: Runner = run_command, out=print) -> int:
    with _Session(cfg, run, keep=args.keep, out=out) as session:
        out(f"Task page: {session.base_url}/app#key={session.api_key}")
        out("Pick Desktop to use the VM's own desktop. Press Ctrl+C here when you are done.")
        try:
            while session.tunnel is not None and session.tunnel.poll() is None:
                time.sleep(1)
            out("The tunnel closed.")
        except KeyboardInterrupt:
            out("")
    return 0


def cmd_run(args: argparse.Namespace, cfg: CloudConfig, run: Runner = run_command, out=print) -> int:
    from operon.cli import drive

    payload = {"intent": args.task, "environment": "browser" if args.browser else "desktop", "max_steps": args.max_steps}
    if args.mode:
        payload["permission_mode"] = args.mode
    with _Session(cfg, run, keep=args.keep, out=out) as session:
        async def go() -> int:
            client = _LinkingClient(session.base_url, session.api_key, out=out)
            try:
                code, _ = await drive(client, payload, ask=input, interactive=sys.stdin.isatty())
                return code
            finally:
                await client.close()

        return asyncio.run(go())


def live_link(base_url: str, api_key: str, run_id: str) -> str:
    """The task page for one run: the key and the run both ride in the fragment."""
    return f"{base_url}/app#key={api_key}&run={run_id}"


class _LinkingClient(RemoteClient):
    """Prints the started run's task-page link, so a fresh tab opens that run."""

    def __init__(self, base_url: str, api_key: str, *, out=print) -> None:
        super().__init__(base_url, api_key)
        self._key, self._out = api_key, out

    async def start(self, payload):
        status = await super().start(payload)
        self._out(f"Watch it live: {live_link(self.base_url, self._key, status['run_id'])}")
        return status


def cmd_stop(args: argparse.Namespace, cfg: CloudConfig, run: Runner = run_command, out=print) -> int:  # noqa: ARG001
    deallocate_vm(cfg, run, out)
    return 0


def add_parser(sub) -> None:
    cloud = sub.add_parser("cloud", help="Operon's own computer on an Azure Linux VM")
    cs = cloud.add_subparsers(dest="cloud_command", required=True)

    create = cs.add_parser("create", help="create the VM (once)")
    create.add_argument("--ref", default="main", help="Operon branch the VM installs (default main)")
    create.add_argument("--allow-ip", default=None, help="the only IP allowed to SSH in (default: this PC's public IP)")
    create.set_defaults(func=lambda a: cmd_create(a, CloudConfig.from_env()))

    set_key = cs.add_parser("set-key", help="save GEMINI_API_KEY on the VM")
    set_key.add_argument("--keep", action="store_true", help="leave the VM running afterwards")
    set_key.set_defaults(func=lambda a: cmd_set_key(a, CloudConfig.from_env()))

    open_ = cs.add_parser("open", help="start the VM and open the task page; deallocate on Ctrl+C")
    open_.add_argument("--keep", action="store_true", help="leave the VM running afterwards")
    open_.set_defaults(func=lambda a: cmd_open(a, CloudConfig.from_env()))

    run = cs.add_parser("run", help="start the VM, run one task, deallocate")
    run.add_argument("task")
    run.add_argument("--browser", action="store_true", help="run in the VM's browser instead of on its desktop")
    run.add_argument("--mode", choices=["ask", "risky", "auto"], default=None)
    run.add_argument("--max-steps", type=int, default=25)
    run.add_argument("--keep", action="store_true", help="leave the VM running afterwards")
    run.set_defaults(func=lambda a: cmd_run(a, CloudConfig.from_env()))

    stop = cs.add_parser("stop", help="deallocate the VM now")
    stop.set_defaults(func=lambda a: cmd_stop(a, CloudConfig.from_env()))

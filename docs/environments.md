# Run environments (KAN-80)

A desktop run happens on one machine. An **environment provider** owns that machine:
it checks that the machine can be used, gets it ready, builds the desktop executor
that drives it, and cleans it up at shutdown. The control loop only sees the executor,
so adding a new place to run is a new provider, not a change to the loop.

```
OPERON_ENVIRONMENT=<name>  ->  get_provider()  ->  check() -> start() -> build_executor()
                                                        ... runs ...
                         API / `operon run` shutdown  ->  executor.aclose() -> stop()
```

Code: `src/operon/environments/` (`base.py` is the interface, `registry.py` the names).

## Choosing one

| How | Effect |
|---|---|
| `OPERON_ENVIRONMENT=<name>` in `.env` | Every desktop run on this API uses it |
| `operon run --desktop --env <name> "task"` | One local run uses it (not with `--request`: the API you send to uses its own setting) |
| `operon envs` | Lists every environment, whether it is ready here, and which one is selected |
| `operon envs --try <name>` | Free smoke test: starts it, takes one screenshot through it (saved under `.var/env-try/`), closes it. No model calls |

An unknown name is refused at startup, and a provider whose `check()` fails stops the
desktop loop from being built, with the missing pieces in the error.

## Environments

| Name | Status | What it is |
|---|---|---|
| `live` | built | This machine's own screen, mouse and keyboard (today's behavior and the default). `OPERON_DESKTOP_PLATFORM=linux` still switches it to an X11 Linux desktop such as the Azure Linux computer. |
| `windows-sandbox` | built, being tried on a real PC | A clean, throwaway Windows on a Windows 10/11 Pro PC, wiped on close (below) |
| `linux-container` | built, being tried on a real PC | A Linux (XFCE) desktop in a Docker container on this machine (below) |
| Azure VM | planned | The Azure VMs Operon already uses (`operon cloud`, the Desktop bench runner) |
| dockur/windows | planned | Windows inside Docker on a Linux host with KVM |
| Windows 365 for Agents | planned | Microsoft's Cloud PCs for agents |

They are built one at a time, each tested before the next starts.

## Machines Operon does not run on: the guest agent

The live desktop is driven in-process. Every other environment runs a small **guest
agent** on its machine (`operon guest`, `src/operon/environments/guest.py`) and the host
drives it with `RemoteDesktopExecutor` (`remote.py`):

```
host: API, loop, Gemini calls, run files          machine: operon guest (token-guarded HTTP)
  RemoteDesktopExecutor  -- capture / execute -->   the machine's DesktopExecutor
                         <-- PNG bytes, results --  (visual servo, app catalog as usual)
```

The Gemini key, the model calls and the run artifacts stay on the host; screenshots and
traces travel back as bytes and are written under the host's artifact folder. The guest
holds only a one-time token. The task page's live view and takeover go through the guest too.

## Windows Sandbox (`windows-sandbox`)

One-time setup on the PC (admin PowerShell, then restart):

```powershell
Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All
```

Then `operon envs` shows `windows-sandbox  ready`. Run with
`operon run --desktop --env windows-sandbox "task"`, or set `OPERON_ENVIRONMENT=windows-sandbox`
for the API (it opens the sandbox at startup and closes it at shutdown).

How it starts: the provider writes `~/.operon/windows-sandbox/operon.wsb` and opens it. The
sandbox maps, read-only and at the same paths, this PC's Python, Operon's virtualenv and
Operon's source folder, so nothing is installed inside. Its logon command (`exchange/start-guest.cmd`)
opens the guest's port in the sandbox firewall and starts `operon guest` with a windowless
Python; the guest writes its address to `exchange/ready.json` and logs to `exchange/guest.log`.

What is shared: those read-only folders and the writable `exchange` folder (token, logon
script, ready file, log). Not the repo root (so not `.env`) and not `~/.operon`. Clipboard,
printer, microphone and camera redirection are off. Everything inside is wiped on close.

Limits: one sandbox at a time (a Windows rule); the Microsoft Store Python cannot be shared,
so use Python from python.org; screen recordings stay inside the sandbox. Settings:
`OPERON_SANDBOX_MEMORY_MB` (4096), `OPERON_SANDBOX_START_TIMEOUT` (240 seconds).

## Linux desktop container (`linux-container`)

Needs Docker Desktop (or any Docker engine) running. `operon envs` shows `linux-container  ready`
once Docker answers.

The first start builds the image from `docker/linux-desktop/Dockerfile` (Debian trixie with
Python 3.14, Xvfb, XFCE, Mousepad, Galculator, Thunar, Ristretto, Chromium, Firefox ESR, and
Operon's dependencies), which takes a few minutes. Later starts take seconds. The image tag
is a hash of the Dockerfile, the entrypoint and `pyproject.toml`, so a dependency change
builds a new image by itself. Each start runs one fresh container (`docker run --rm`):

- the guest port is published on `127.0.0.1` only, at a port Docker picks;
- the guest token goes in through the `docker` client's environment, not its command line;
- the host's `src` folder is mounted read-only over the image's copy, so code changes need no rebuild;
- not `.env`, not `~/.operon`: the Gemini key stays on the host.

`stop()` removes the container, so every run starts from a clean desktop. The screen is
1280x800. LibreOffice is not in the image (it would add about 600 MB), so the `spreadsheet`
and `document` apps cannot be launched there.

Settings: `OPERON_CONTAINER_IMAGE` (use a prebuilt image), `OPERON_CONTAINER_START_TIMEOUT`
(120 seconds), `OPERON_CONTAINER_CA_BUNDLE` (a CA bundle for pip during the build, behind a
TLS-inspecting proxy).

## Tools inside a guest machine

The model's tools (`OPERON_TOOLS`: files, web) run where the desktop is. For `windows-sandbox`
and `linux-container`, the host does not wrap the executor with its own tools. It passes its
tool setting to the guest instead, and the guest runs the tools inside the machine, so a
file a task writes lands in the sandbox or container, never on the host. A batch of actions
is split on the host, so a stop or a trust-gate block between two actions still holds.

## Writing a provider

Subclass `EnvironmentProvider`, set `name` and `summary` (and `runs_on_host = False` for a machine reached through the guest agent), implement `platform()`
(`windows` or `linux`) and `build_executor()`, and override `check()`, `start()` and
`stop()` when the machine needs them (`start` and `stop` must be safe to call twice).
Register it in `registry._PROVIDERS` with a factory that imports lazily, so an unused
provider's dependencies are never loaded. Add offline tests under `tests/environments/`.

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

An unknown name is refused at startup, and a provider whose `check()` fails stops the
desktop loop from being built, with the missing pieces in the error.

## Environments

| Name | Status | What it is |
|---|---|---|
| `live` | built | This machine's own screen, mouse and keyboard (today's behavior and the default). `OPERON_DESKTOP_PLATFORM=linux` still switches it to an X11 Linux desktop such as the Azure Linux computer. |
| `windows-sandbox` | built, being tried on a real PC | A clean, throwaway Windows on a Windows 10/11 Pro PC, wiped on close (below) |
| Linux desktop container | planned | An Xvfb desktop in Docker Desktop on the PC |
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

## Writing a provider

Subclass `EnvironmentProvider`, set `name` and `summary`, implement `platform()`
(`windows` or `linux`) and `build_executor()`, and override `check()`, `start()` and
`stop()` when the machine needs them (`start` and `stop` must be safe to call twice).
Register it in `registry._PROVIDERS` with a factory that imports lazily, so an unused
provider's dependencies are never loaded. Add offline tests under `tests/environments/`.

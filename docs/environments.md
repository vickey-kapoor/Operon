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
| Windows Sandbox | next | A clean, throwaway Windows on a Windows 11 Pro PC, wiped on close |
| Linux desktop container | planned | An Xvfb desktop in Docker Desktop on the PC |
| Azure VM | planned | The Azure VMs Operon already uses (`operon cloud`, the Desktop bench runner) |
| dockur/windows | planned | Windows inside Docker on a Linux host with KVM |
| Windows 365 for Agents | planned | Microsoft's Cloud PCs for agents |

They are built one at a time, each tested before the next starts.

## Writing a provider

Subclass `EnvironmentProvider`, set `name` and `summary`, implement `platform()`
(`windows` or `linux`) and `build_executor()`, and override `check()`, `start()` and
`stop()` when the machine needs them (`start` and `stop` must be safe to call twice).
Register it in `registry._PROVIDERS` with a factory that imports lazily, so an unused
provider's dependencies are never loaded. Add offline tests under `tests/environments/`.

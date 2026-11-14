"""``operon envs``: list the run environments and whether each can run here (KAN-80).

``operon envs --try NAME`` is the free smoke test: start that environment, take one
screenshot through its executor, close it. No model is called.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from operon.environments.registry import environment_name, get_provider, provider_names


def try_environment(name: str, *, out_dir: Path | None = None) -> int:
    """Start ``name``, take one screenshot through it, stop it. Returns an exit code."""
    from operon.core.paths import runtime_dir

    try:
        provider = get_provider(name)
    except ValueError as exc:
        print(exc)
        return 2
    check = provider.check()
    if not check.ok:
        print(f"{name}: not ready: " + " ".join(check.problems))
        return 1
    print(f"{name}: starting (a sandbox or VM can take a minute or two)...", flush=True)
    began = time.monotonic()
    executor = None
    try:
        provider.start()
        print(f"{name}: started in {time.monotonic() - began:.0f}s; taking a screenshot...", flush=True)
        executor = provider.build_executor()
        frame = asyncio.run(executor.capture())
        target_dir = out_dir or (runtime_dir() / "env-try")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.png"
        shutil.copyfile(frame.artifact_path, target)
        print(f"{name}: screenshot {frame.width}x{frame.height} saved to {target}")
        return 0
    except Exception as exc:  # noqa: BLE001 - report it, then still close the environment
        print(f"{name}: failed: {exc}")
        return 1
    finally:
        if executor is not None and hasattr(executor, "aclose"):
            try:
                asyncio.run(executor.aclose())
            except Exception:  # noqa: BLE001 - closing anyway
                pass
        print(f"{name}: closing.", flush=True)
        provider.stop()


def list_environments(args=None) -> int:
    if getattr(args, "try_name", None):
        return try_environment(args.try_name)
    current = environment_name()
    for name in provider_names():
        provider = get_provider(name)
        check = provider.check()
        mark = "*" if name == current else " "
        status = "ready" if check.ok else "not ready: " + " ".join(check.problems)
        print(f"{mark} {name:<10} {status}")
        print(f"  {'':<10} {provider.summary}")
    if current not in provider_names():
        print(f"OPERON_ENVIRONMENT={current!r} is not a known environment.")
        return 1
    print("\n* = selected (OPERON_ENVIRONMENT, or `operon run --env NAME`)")
    return 0


def add_parser(sub) -> None:
    parser = sub.add_parser("envs", help="list the places a desktop run can happen")
    parser.add_argument(
        "--try", dest="try_name", metavar="NAME", default=None,
        help="free smoke test: start NAME, take one screenshot through it, close it (no model calls)",
    )
    parser.set_defaults(func=list_environments)

"""``operon envs``: list the run environments and whether each can run here (KAN-80)."""

from __future__ import annotations

from operon.environments.registry import environment_name, get_provider, provider_names


def list_environments(_args=None) -> int:
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
    parser.set_defaults(func=list_environments)

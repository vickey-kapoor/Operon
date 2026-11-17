"""Operon - vision-driven computer-use engine for UI-native automation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("operon-agent")
except PackageNotFoundError:  # pragma: no cover - running from a tree that was never installed
    __version__ = "0+unknown"
__author__ = "Operon"

_SDK_NAMES = ("Agent", "RunResult", "ApprovalRequest")


def __getattr__(name: str):
    # Lazy, so `import operon` stays cheap: the SDK pulls in the agent loop only when used.
    if name in _SDK_NAMES:
        from operon import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Agent", "ApprovalRequest", "RunResult", "__author__", "__version__"]

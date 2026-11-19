"""Operon - vision-driven computer-use engine for UI-native automation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("operon-agent")
except PackageNotFoundError:  # pragma: no cover - running from a tree that was never installed
    __version__ = "0+unknown"
__author__ = "Operon"

"""Environment enum shared by the browser and desktop execution paths."""

from __future__ import annotations

from enum import StrEnum


class Environment(StrEnum):
    BROWSER = "browser"
    DESKTOP = "desktop"

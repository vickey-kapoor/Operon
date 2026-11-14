"""Shared filesystem path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    """Return ``$name`` as a path, or ``default`` when it is unset or blank.

    A blank value (``OPERON_RUNS_ROOT=``) would otherwise become ``Path("")``,
    which silently means the current working directory.
    """
    value = os.getenv(name, "")
    return Path(value) if value.strip() else default


def project_root() -> Path:
    """Return the repository root containing ``pyproject.toml``."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[3]


def user_config_dir() -> Path:
    """Return the per-user settings folder (``~/.operon``) that ``operon setup`` writes.

    It holds ``operon.env``, so a pip-installed Operon finds its key from any folder.
    """
    return _env_path("OPERON_CONFIG_DIR", Path.home() / ".operon")


def user_env_file() -> Path:
    """Return the per-user settings file that ``operon setup`` writes the model key to."""
    return user_config_dir() / "operon.env"


def load_user_env() -> bool:
    """Load ``user_env_file()`` into the environment without overriding anything already set.

    Call it after the working folder's ``.env``, so a value there (or in the real
    environment) wins over the per-user file. Returns whether the file existed.
    """
    path = user_env_file()
    if not path.is_file():
        return False
    from dotenv import load_dotenv

    load_dotenv(path, override=False)
    return True


def prompts_dir() -> Path:
    """Return the prompts directory shipped inside the ``operon`` package."""
    return Path(__file__).resolve().parents[1] / "prompts"


def runtime_dir() -> Path:
    """Return the root for generated runtime artifacts."""
    return _env_path("OPERON_RUNTIME_ROOT", project_root() / ".var")


def runs_dir() -> Path:
    """Return the local run-artifact directory."""
    return _env_path("OPERON_RUNS_ROOT", runtime_dir() / "runs")


def browser_artifacts_dir() -> Path:
    """Return the browser session artifact directory."""
    return _env_path("OPERON_BROWSER_ARTIFACTS_ROOT", runtime_dir() / "browser-artifacts")


def desktop_artifacts_dir() -> Path:
    """Return the desktop session artifact directory."""
    return _env_path("OPERON_DESKTOP_ARTIFACTS_ROOT", runtime_dir() / "desktop-artifacts")


def test_artifacts_dir() -> Path:
    """Return the test artifact directory."""
    return _env_path("OPERON_TEST_ARTIFACTS_ROOT", runtime_dir() / "test-artifacts")

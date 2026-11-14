"""Shared filesystem path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    """Return ``$name`` as a path, or ``default`` when it is unset or blank.

    A blank value (``OPERON_RUNS_ROOT=``) would otherwise become ``Path("")``,
    which silently means the current working directory.
    """
    from operon.core.settings import get_settings

    value = getattr(get_settings(), _PATH_FIELDS[name])
    return value if value is not None else default


# The folder settings, by variable (KAN-157: parsed in operon.core.settings).
_PATH_FIELDS = {
    "OPERON_CONFIG_DIR": "config_dir",
    "OPERON_RUNTIME_ROOT": "runtime_root",
    "OPERON_RUNS_ROOT": "runs_root",
    "OPERON_BROWSER_ARTIFACTS_ROOT": "browser_artifacts_root",
    "OPERON_DESKTOP_ARTIFACTS_ROOT": "desktop_artifacts_root",
    "OPERON_TEST_ARTIFACTS_ROOT": "test_artifacts_root",
}


def project_root() -> Path:
    """Return the repository root containing ``pyproject.toml``."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[3]


def source_checkout() -> Path | None:
    """Return the repository root when Operon runs from a source checkout, else ``None``.

    A checkout (or an editable install of one) has this file at
    ``<root>/src/operon/core/paths.py`` next to ``<root>/pyproject.toml``. A pip install
    keeps it under ``site-packages``, where any ``pyproject.toml`` further up belongs to
    someone else's project, so it does not count.
    """
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    if package.parent.name == "src" and (root / "pyproject.toml").is_file():
        return root
    return None


def user_config_dir() -> Path:
    """Return the per-user settings folder (``~/.operon``) that ``operon setup`` writes.

    It holds ``operon.env``, so a pip-installed Operon finds its key from any folder.
    """
    return _env_path("OPERON_CONFIG_DIR", Path.home() / ".operon")


def user_env_file() -> Path:
    """Return the per-user settings file that ``operon setup`` writes the model key to."""
    return user_config_dir() / "operon.env"


_shell_env: dict[str, str] | None = None


def remember_shell_env() -> None:
    """Note the environment as the shell gave it, before any ``.env`` file is loaded.

    ``operon setup`` uses it to tell a key set in the shell from one read out of a file.
    """
    global _shell_env
    if _shell_env is None:
        _shell_env = dict(os.environ)


def shell_env() -> dict[str, str]:
    """The environment from before ``.env`` files were loaded (now, if never noted)."""
    return dict(_shell_env) if _shell_env is not None else dict(os.environ)


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
    """Return the root for generated runtime artifacts (runs, screenshots, the local API key).

    ``<checkout>/.var`` in a source checkout; otherwise the per-user folder
    (``~/.operon``), so a pip install never writes inside Python's own folders.
    """
    checkout = source_checkout()
    return _env_path("OPERON_RUNTIME_ROOT", checkout / ".var" if checkout else user_config_dir())


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

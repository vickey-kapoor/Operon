"""Shared filesystem path helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
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
    """Return the per-user settings folder (``~/.operon``).

    It optionally holds ``operon.env``, so Operon finds its key from any folder.
    """
    return _env_path("OPERON_CONFIG_DIR", Path.home() / ".operon")


def user_env_file() -> Path:
    """Return the per-user settings file (``~/.operon/operon.env``), which is optional: it can hold the model key."""
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


_real_env: dict[str, str] = {}
_key_sources: dict[str, str] = {}
_snapshot_taken = False


def real_environment() -> dict[str, str]:
    """The environment as the process started: the first snapshot if the loader ran, else the live one."""
    return dict(_real_env) if _snapshot_taken else dict(os.environ)


def reset_key_state() -> None:
    global _snapshot_taken
    _real_env.clear()
    _key_sources.clear()
    _snapshot_taken = False


def load_stored_keys(real_env: Mapping[str, str]) -> dict[str, str]:
    """Settle where the known key variables come from (KAN-186); return name -> source.

    ``real_env`` is the environment as the process started, before any .env file was read. Order:
    real environment, credential store, working-folder .env, ``operon.env``. A non-blank real value
    is always put back (undoing ``create_app``'s ``override=True``), even with no store or with
    ``OPERON_KEYRING=off``. The store itself is read only when ``keystore.enabled()`` and a real
    backend exists. Never raises and never logs a value.
    """
    global _snapshot_taken
    import logging

    _real_env.clear()
    _real_env.update(real_env)
    _snapshot_taken = True
    _key_sources.clear()
    log = logging.getLogger(__name__)
    try:
        from operon.core import keystore

        names = keystore.known_names()
        for name in names:
            if real_env.get(name, "").strip():
                os.environ[name] = real_env[name]
                _key_sources[name] = "environment"
        if not keystore.enabled():
            log.debug("credential store not used (%s)", keystore.mode())
            return dict(_key_sources)
        if not keystore.available():
            log.debug("no credential store available; keys come from the environment and .env files")
            return dict(_key_sources)
        for name in names:
            if name in _key_sources:
                continue
            value = keystore.get(name)
            if value:
                os.environ[name] = value
                _key_sources[name] = "Credential Manager"
    except Exception as exc:
        log.debug("credential store skipped (%s)", type(exc).__name__)
    return dict(_key_sources)


def loaded_key_state() -> tuple[dict[str, str], dict[str, str]]:
    """The real-environment snapshot and the sources from the last ``load_stored_keys`` call."""
    return dict(_real_env), dict(_key_sources)


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

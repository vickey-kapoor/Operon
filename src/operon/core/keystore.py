"""Model API keys in the operating system's credential store (KAN-186).

On Windows that is Credential Manager; macOS and desktop Linux use their own keystore through the
Python ``keyring`` package. Nothing here prints or logs a value: failures log the exception type only.
``keyring`` is imported inside the functions, so a missing or broken install never stops startup.
"""

from __future__ import annotations

import logging

SERVICE = "operon"

# Variables the loader may fill from the store. Profiles in the model registry add their own key variable.
KNOWN_KEYS: tuple[str, ...] = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENWEIGHT_API_KEY",
    "API_KEYS",
    "HITL_WEBHOOK_URL",
)

_log = logging.getLogger(__name__)


class KeystoreError(RuntimeError):
    """The credential store is missing or refused the request (the message holds no value)."""


def known_names() -> tuple[str, ...]:
    """The names that may live in the store: the fixed list plus each registry profile's key variable."""
    names = list(KNOWN_KEYS)
    try:
        from operon.agent.backends.model_adapter.registry import load_registry

        for profile in load_registry().values():
            if profile.api_key_env not in names:
                names.append(profile.api_key_env)
    except Exception as exc:  # a bad registry must not hide the fixed names
        _log.debug("keystore: registry names skipped (%s)", type(exc).__name__)
    return tuple(names)


def _backend():
    import keyring

    return keyring.get_keyring()


def available() -> bool:
    """Whether a real credential store is here (not missing, not the fail or null backend)."""
    try:
        module = type(_backend()).__module__
    except Exception:  # ImportError, or a backend that breaks on load
        return False
    return not module.startswith(("keyring.backends.fail", "keyring.backends.null"))


def get(name: str) -> str | None:
    """The stored value, or None when absent or the store fails."""
    try:
        import keyring

        return keyring.get_password(SERVICE, name) or None
    except Exception as exc:
        _log.debug("keystore: read of %s failed (%s)", name, type(exc).__name__)
        return None


def set(name: str, value: str) -> None:  # noqa: A001 - mirrors keyring's own verbs
    try:
        import keyring

        keyring.set_password(SERVICE, name, value)
    except Exception as exc:
        raise KeystoreError(f"could not save {name} ({type(exc).__name__})") from None


def delete(name: str) -> bool:
    """Remove a stored value; False when there was none."""
    try:
        import keyring
        from keyring.errors import PasswordDeleteError

        try:
            keyring.delete_password(SERVICE, name)
        except PasswordDeleteError:
            return False
        return True
    except Exception as exc:
        raise KeystoreError(f"could not remove {name} ({type(exc).__name__})") from None

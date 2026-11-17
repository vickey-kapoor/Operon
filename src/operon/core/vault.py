"""The secrets vault (KAN-103, ``OPERON_SECRETS``, default off).

The person stores a secret once with ``operon secrets set NAME``. The model never sees
the value: it writes ``{{secret:NAME}}`` as the text of a TYPE action, and the executor
swaps in the value at the last moment, in a local variable inside its type call. The
:class:`~operon.models.policy.AgentAction` is never changed, so every record of the action
(run files, logs, the model's conversation, the live view, reports) holds the placeholder.

Storage is one file per secret under ``<config dir>/secrets/``. On Windows each value is
encrypted with DPAPI (current-user scope, no new dependency). Elsewhere the file has mode
0600 in a 0700 folder and is NOT encrypted at rest: see docs/secrets.md.
"""

from __future__ import annotations

import contextvars
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from operon.core.paths import user_config_dir
from operon.core.settings import get_settings

NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
_PLACEHOLDER = re.compile(r"\{\{\s*secret:([^{}]*?)\s*\}\}")
_SUFFIX = ".secret"
REDACTED = "[secret]"


# Set by the guest agent around a request the host already filled in (KAN-103): the text is
# the secret itself, so the executor must not look for placeholders in it, must not use the
# clipboard for it, and must mask it in everything it returns.
HOST_FILLED: contextvars.ContextVar[bool] = contextvars.ContextVar("operon_secret_host_filled", default=False)


class SecretError(Exception):
    """A secret could not be stored or used. The message never contains a value."""


def valid_name(name: str) -> bool:
    return NAME_PATTERN.fullmatch(name) is not None


def vault_dir() -> Path:
    return user_config_dir() / "secrets"


def secrets_enabled() -> bool:
    return get_settings().secrets


def _path(name: str) -> Path:
    if not valid_name(name):
        raise SecretError(f"invalid secret name {name!r}: use 1-64 letters, digits, '_' or '-'")
    return vault_dir() / f"{name}{_SUFFIX}"


# -- storage ------------------------------------------------------------------------------


def _use_dpapi() -> bool:
    return sys.platform == "win32"


def _blob_call(function_name: str, data: bytes) -> bytes:
    """Run ``CryptProtectData`` / ``CryptUnprotectData`` over ``data`` (current user, no UI)."""
    import ctypes
    from ctypes import wintypes

    class DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32: Any = ctypes.WinDLL("crypt32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    function = getattr(crypt32, function_name)
    function.restype = wintypes.BOOL
    # (data in, description, entropy, reserved, prompt, flags, data out): the description
    # slot is a pointer out for Unprotect and unused for Protect, so a plain pointer fits both.
    function.argtypes = [
        ctypes.POINTER(DataBlob), ctypes.c_void_p, ctypes.POINTER(DataBlob),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DataBlob),
    ]
    ui_forbidden = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
    source = ctypes.create_string_buffer(data, len(data))
    blob_in = DataBlob(len(data), ctypes.cast(source, ctypes.POINTER(ctypes.c_char)))
    blob_out = DataBlob()
    ok = function(ctypes.byref(blob_in), None, None, None, None, ui_forbidden, ctypes.byref(blob_out))
    if not ok:
        raise SecretError(f"Windows could not {'encrypt' if function_name == 'CryptProtectData' else 'decrypt'} the secret")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(blob_out.pbData, ctypes.c_void_p))


def _encode(value: str) -> bytes:
    raw = value.encode("utf-8")
    return _blob_call("CryptProtectData", raw) if _use_dpapi() else raw


def _decode(blob: bytes) -> str:
    raw = _blob_call("CryptUnprotectData", blob) if _use_dpapi() else blob
    return raw.decode("utf-8")


def set_secret(name: str, value: str) -> None:
    """Store ``value`` under ``name`` (replacing any earlier one)."""
    path = _path(name)
    if not value:
        raise SecretError("a secret cannot be empty")
    folder = path.parent
    folder.mkdir(parents=True, exist_ok=True)
    if not _use_dpapi():
        os.chmod(folder, 0o700)
    data = _encode(value)
    tmp = path.with_suffix(".tmp")
    if _use_dpapi():
        # Whole or not at all: write beside it, then swap in.
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return
    # POSIX: create the file private from the first byte.
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def get_secret(name: str) -> str | None:
    """The stored value, or ``None`` when ``name`` is not in the vault."""
    path = _path(name)
    try:
        blob = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SecretError(f"could not read secret {name}: {exc.strerror or 'read error'}") from None
    return _decode(blob)


def list_secrets() -> list[str]:
    """The stored names, sorted. Never values."""
    folder = vault_dir()
    if not folder.is_dir():
        return []
    return sorted(p.name[: -len(_SUFFIX)] for p in folder.glob(f"*{_SUFFIX}") if valid_name(p.name[: -len(_SUFFIX)]))


def delete_secret(name: str) -> bool:
    path = _path(name)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


# -- placeholders -------------------------------------------------------------------------


def has_placeholder(text: str | None) -> bool:
    return bool(text) and _PLACEHOLDER.search(text or "") is not None


def action_uses_secret(action: object) -> bool:
    """True when ``action``, or any child of a batch, types a ``{{secret:NAME}}`` placeholder."""
    if getattr(getattr(action, "action_type", None), "value", None) == "type" and has_placeholder(getattr(action, "text", None)):
        return True
    return any(action_uses_secret(child) for child in (getattr(action, "actions", None) or ()))


def resolve_text(text: str) -> tuple[str, tuple[str, ...]]:
    """``(text with every placeholder replaced, the values used)``.

    Text without a placeholder comes back unchanged. Raises :class:`SecretError` when
    ``OPERON_SECRETS`` is off or a name is not in the vault, so the caller types nothing.
    """
    if not has_placeholder(text):
        return text, ()
    if not secrets_enabled():
        raise SecretError("secrets are off: set OPERON_SECRETS=on to type {{secret:NAME}} placeholders")
    if not get_settings().trust_gate:
        raise SecretError("secrets need the trust gate on (OPERON_TRUST_GATE)")
    values: list[str] = []

    def swap(match: re.Match[str]) -> str:
        name = match.group(1)
        if not valid_name(name):
            raise SecretError(f"invalid secret name {name!r}")
        value = get_secret(name)
        if value is None:
            raise SecretError(f"unknown secret {name}")
        values.append(value)
        return value

    return _PLACEHOLDER.sub(swap, text), tuple(values)


def resolve_for_type(text: str) -> tuple[str, tuple[str, ...]]:
    """``resolve_text``, except on a guest for a request the host already filled in: then the
    whole text is the secret and is not searched for placeholders again."""
    if HOST_FILLED.get():
        return text, ((text,) if text else ())
    return resolve_text(text)


def redact(value: Any, secret_values: Iterable[str]) -> Any:
    """``value`` (a string, or a nested dict or list of them) with each secret masked."""
    needles = sorted({v for v in secret_values if v}, key=len, reverse=True)
    if not needles:
        return value
    if isinstance(value, str):
        for needle in needles:
            value = value.replace(needle, REDACTED)
        return value
    if isinstance(value, dict):
        return {k: redact(v, needles) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, needles) for v in value]
    return value


def available_names() -> list[str]:
    """The names the model may use: empty when the flag is off."""
    return list_secrets() if secrets_enabled() else []


def secrets_hint(names: list[str]) -> str:
    """The advisory hint that tells the model what it may type. Names only, never values."""
    return (
        "Secrets are available by name: "
        + ", ".join(names)
        + ". To enter one, use it inside the text of a type action as {{secret:NAME}} (for example "
        "{{secret:" + names[0] + "}}); it is filled in for you and you will never see the value. "
        "Never ask the user to reveal a secret."
    )

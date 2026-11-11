"""Short-lived tokens for ``/ws/stream`` (KAN-89).

The stream shows the live screen and accepts control messages (resume, override,
input), so whoever holds a stream URL holds the run. Rather than putting the
long-lived API key on every stream connection, a client that already has the key
asks ``POST /app/api/stream-token`` for a token that:

- expires after ``OPERON_STREAM_TOKEN_TTL_S`` seconds (default 600) — it is checked
  only at the WebSocket handshake, so an open stream is not cut off;
- has a scope: ``control`` may send control messages, ``view`` may only watch,
  so a watch link can be handed out without handing out control.

Tokens live in this process's memory only; a server restart invalidates them all.
The client presents one as the subprotocol pair ``["x-stream-token", "<token>"]``.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass

SCOPES = ("view", "control")
_DEFAULT_TTL_S = 600
_MAX_TOKENS = 256


@dataclass(frozen=True)
class StreamToken:
    token: str
    scope: str
    expires_at: float


_tokens: dict[str, StreamToken] = {}


def token_ttl_s() -> int:
    """``OPERON_STREAM_TOKEN_TTL_S``, clamped to 10 s .. 1 day; default 600."""
    try:
        ttl = int(os.getenv("OPERON_STREAM_TOKEN_TTL_S", str(_DEFAULT_TTL_S)))
    except ValueError:
        ttl = _DEFAULT_TTL_S
    return max(10, min(ttl, 86400))


def _prune(now: float) -> None:
    for key in [k for k, t in _tokens.items() if t.expires_at <= now]:
        del _tokens[key]
    if len(_tokens) > _MAX_TOKENS:
        # Oldest first: a client that keeps minting cannot grow the table without bound.
        for key, _ in sorted(_tokens.items(), key=lambda kv: kv[1].expires_at)[: len(_tokens) - _MAX_TOKENS]:
            del _tokens[key]


def mint(scope: str = "control", *, now: float | None = None) -> StreamToken:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    now = time.time() if now is None else now
    _prune(now)
    token = StreamToken(secrets.token_urlsafe(24), scope, now + token_ttl_s())
    _tokens[token.token] = token
    return token


def check(presented: str | None, *, now: float | None = None) -> str | None:
    """The token's scope when ``presented`` is a live token, else None."""
    if not presented:
        return None
    now = time.time() if now is None else now
    for key, token in list(_tokens.items()):
        # Constant-time compare so a guess learns nothing from timing.
        if secrets.compare_digest(key, presented):
            if token.expires_at <= now:
                _tokens.pop(key, None)
                return None
            return token.scope
    return None


def revoke_all() -> None:
    _tokens.clear()

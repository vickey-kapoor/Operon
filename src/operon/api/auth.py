"""Optional API-key authentication for the Operon HTTP surface.

Opt-in by design: when ``API_KEYS`` is unset or empty, authentication is
**disabled** and every request passes — this preserves the existing local/dev
experience. When ``API_KEYS`` is set (a comma-separated list), every request
except a small allowlist (health probes) must present a matching ``X-API-Key``
header, or it is rejected with 401.

Why this matters: the Operon API can drive a real browser/desktop. An
unauthenticated server is reachable by any local process — and by a web page via
DNS-rebinding / CSRF to ``127.0.0.1`` — so setting ``API_KEYS`` is strongly
recommended for any non-throwaway deployment.

Keys are compared in constant time to avoid leaking length/prefix via timing.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Header, HTTPException, Request, WebSocket, status

logger = logging.getLogger(__name__)

# Paths that must stay reachable without a key (liveness/readiness probes).
_OPEN_PATHS: frozenset[str] = frozenset({"/health"})


def configured_api_keys() -> set[str]:
    """The set of accepted API keys from ``API_KEYS`` (comma-separated). Empty ⇒ auth off."""
    raw = os.getenv("API_KEYS", "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def _key_matches(presented: str, accepted: set[str]) -> bool:
    # constant-time compare against each configured key
    return any(secrets.compare_digest(presented, key) for key in accepted)


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency enforcing ``X-API-Key`` when ``API_KEYS`` is configured."""
    accepted = configured_api_keys()
    if not accepted:
        return  # auth disabled — backward compatible
    if request.url.path in _OPEN_PATHS:
        return  # health probes stay open
    if x_api_key is None or not _key_matches(x_api_key, accepted):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


def websocket_api_key_ok(websocket: WebSocket) -> bool:
    """Whether a WebSocket handshake is authorized.

    No-op (returns True) when API_KEYS is unset. Browsers cannot set custom
    headers on a WebSocket, so the key is read from the ``api_key`` query
    parameter (primary) or, failing that, an ``x-api-key`` subprotocol value
    (client connects with subprotocols ``["x-api-key", "<key>"]``).
    """
    accepted = configured_api_keys()
    if not accepted:
        return True
    presented = websocket.query_params.get("api_key")
    if presented is None:
        protocols = [
            p.strip()
            for p in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if p.strip()
        ]
        if len(protocols) >= 2 and protocols[0] == "x-api-key":
            presented = protocols[1]
    return presented is not None and _key_matches(presented, accepted)

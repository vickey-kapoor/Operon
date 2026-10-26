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
from urllib.parse import urlsplit

from fastapi import Header, HTTPException, Request, WebSocket, status

logger = logging.getLogger(__name__)

# Paths that must stay reachable without a key (liveness/readiness probes).
_OPEN_PATHS: frozenset[str] = frozenset({"/health"})


def configured_api_keys() -> set[str]:
    """The set of accepted API keys from ``API_KEYS`` (comma-separated). Empty ⇒ auth off."""
    raw = os.getenv("API_KEYS", "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def configured_cors_origins() -> set[str]:
    """The browser origins allowed by ``CORS_ORIGINS`` (comma-separated). Empty ⇒ none."""
    raw = os.getenv("CORS_ORIGINS", "")
    return {origin.strip() for origin in raw.split(",") if origin.strip()}


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


def _offered_subprotocols(websocket: WebSocket) -> list[str]:
    return [
        p.strip()
        for p in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if p.strip()
    ]


def websocket_subprotocol(websocket: WebSocket) -> str | None:
    """The subprotocol to echo on accept: ``x-api-key`` when the client offered it.

    A browser that offers subprotocols drops the connection unless the server
    selects one, so the key-carrying ``x-api-key`` offer must be echoed back
    (whether or not auth is enabled) — never the key itself.
    """
    protocols = _offered_subprotocols(websocket)
    return "x-api-key" if protocols[:1] == ["x-api-key"] else None


def websocket_api_key_ok(websocket: WebSocket) -> bool:
    """Whether a WebSocket handshake is authorized.

    No-op (returns True) when API_KEYS is unset. Browsers cannot set custom
    headers on a WebSocket, so the key comes as an ``x-api-key`` subprotocol value
    (client connects with subprotocols ``["x-api-key", "<key>"]``). A query-string
    key is not accepted: it would be written to access logs.
    """
    accepted = configured_api_keys()
    if not accepted:
        return True
    protocols = _offered_subprotocols(websocket)
    presented = protocols[1] if len(protocols) >= 2 and protocols[0] == "x-api-key" else None
    return presented is not None and _key_matches(presented, accepted)


_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


def configured_allowed_hosts() -> set[str]:
    """Host names accepted in the ``Host`` header: loopback plus ``OPERON_ALLOWED_HOSTS``."""
    raw = os.getenv("OPERON_ALLOWED_HOSTS", "")
    return set(_LOOPBACK_HOSTS) | {host.strip().lower() for host in raw.split(",") if host.strip()}


def _host_name(host_header: str) -> str:
    """The name part of a ``Host`` header value, without port or IPv6 brackets."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class TrustedHostMiddleware:
    """Refuse HTTP and WebSocket requests whose ``Host`` is not an allowed name.

    Always on. A web page that re-points its own domain at 127.0.0.1 (DNS
    rebinding) makes the browser treat this API as same-origin, which bypasses
    CORS; its requests still carry the page's domain in ``Host``, so they stop
    here. Extra names (a LAN host, a reverse proxy) go in ``OPERON_ALLOWED_HOSTS``.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        host = ""
        for name, value in scope.get("headers", []):
            if name == b"host":
                host = value.decode("latin-1")
                break
        if _host_name(host) in configured_allowed_hosts():
            await self.app(scope, receive, send)
            return
        logger.warning("refused request with untrusted Host header %r", host)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await send({
            "type": "http.response.start",
            "status": 400,
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        })
        await send({"type": "http.response.body", "body": b"Invalid host header"})


def websocket_origin_ok(websocket: WebSocket) -> bool:
    """Whether a WebSocket handshake comes from an allowed browser origin.

    CORS does not apply to WebSockets, so without this check any page open in
    any browser on the host could connect to the stream and inject input. Unlike
    the API-key gate this is always on:

    - no ``Origin`` header ⇒ allowed (a non-browser client; browsers always send it)
    - an origin listed in ``CORS_ORIGINS`` ⇒ allowed
    - a same-origin page served by this server on a loopback host ⇒ allowed;
      loopback-only so a DNS-rebound name (whose Host matches its Origin)
      does not pass
    - anything else, including ``null`` ⇒ rejected
    """
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    if origin in configured_cors_origins():
        return True
    parts = urlsplit(origin)
    host_header = websocket.headers.get("host", "").lower()
    if parts.hostname not in _LOOPBACK_HOSTS:
        return False
    return parts.netloc.lower() == host_header

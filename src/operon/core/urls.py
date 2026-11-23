"""Which URLs the agent may open (KAN-191).

A page the agent opens renders in the model's own screenshots, so a ``file:`` URL would let a
malicious page steer the model into reading local files. Only plain web links are allowed.
This is a hard refusal, separate from the trust gate (whose DENY means "ask a person").
"""

from __future__ import annotations

from urllib.parse import urlparse

WEB_SCHEMES = frozenset({"http", "https"})


def _scheme(url: str) -> str:
    try:
        return urlparse(url.strip()).scheme.lower()
    except ValueError:
        return ""


def is_web_url(url: str) -> bool:
    """True only for an ``http`` or ``https`` URL that names a host."""
    try:
        parsed = urlparse(url.strip())
        host = parsed.hostname
    except ValueError:
        return False
    return parsed.scheme.lower() in WEB_SCHEMES and bool(host)


def web_url_refusal(url: str, what: str = "navigate") -> str:
    """The message for a refused URL. Names the scheme only, never the full URL."""
    scheme = _scheme(url)
    shown = f"{scheme}:" if scheme else "no scheme"
    return f"{what} blocked: only http and https links can be opened (got '{shown}')"

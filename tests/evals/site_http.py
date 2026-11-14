"""HTTP clients for the local fixture site (plain HTTP on 127.0.0.1).

A default httpx client loads certifi's CA bundle when it is created, which takes ~0.2 s with
some OpenSSL builds (3.0.x on Windows) and buys nothing over plain HTTP. Tests create hundreds
of these, so they skip it with verify=False.
"""

from __future__ import annotations

import httpx


def site_client(base_url: str, **kwargs) -> httpx.Client:
    return httpx.Client(base_url=base_url, verify=False, **kwargs)


def site_get(url: str, **kwargs) -> httpx.Response:
    return httpx.get(url, verify=False, **kwargs)

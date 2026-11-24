"""Helpers for keeping sensitive URL components out of diagnostics."""

from urllib.parse import urlsplit


def url_origin_for_log(url: object) -> str | None:
    """Keep only an HTTP(S) origin, including the port when supplied."""
    if url is None:
        return None
    if not isinstance(url, str):
        return "<unavailable URL origin>"
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
        if parts.scheme in {"http", "https"} and host:
            authority = f"[{host}]" if ":" in host else host
            return f"{parts.scheme}://{authority}" + (f":{port}" if port is not None else "")
    except ValueError:
        pass
    return "<unavailable URL origin>"

"""Pluggable, site-specific search-URL adapters.

The policy rule engine is deliberately site-agnostic: it decides *when* to fall
back to a direct search URL, never *how* a particular site builds one. That
per-site knowledge lives here, in opt-in adapters, so no site-specific logic is
hardcoded into the rules.

No adapters are registered by default. ``resolve_search_url`` then falls back to
a generic ``<base>/search?q=<query>`` form. Register built-in or custom adapters
explicitly via :func:`register_site_adapter` / :func:`register_builtin_site_adapters`,
or set the ``OPERON_SITE_ADAPTERS`` environment variable (comma-separated adapter
names, or ``all``) to load the built-ins at import time.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Protocol
from urllib.parse import quote_plus, urljoin, urlparse


class SiteAdapter(Protocol):
    """Builds a direct search URL for a family of sites it recognises."""

    name: str

    def matches(self, base_url: str) -> bool:
        """Return True if this adapter knows how to search ``base_url``'s site."""
        ...

    def search_url(self, query: str, base_url: str) -> str:
        """Return a direct search URL for ``query`` on ``base_url``'s site."""
        ...


def generic_search_url(query: str, base_url: str) -> str:
    """Site-agnostic fallback: append a conventional ``search?q=`` query string."""
    return urljoin(base_url or "", f"search?q={quote_plus(query)}")


class WikipediaSiteAdapter:
    """MediaWiki-style ``/w/index.php?search=`` lookups."""

    name = "wikipedia"

    def matches(self, base_url: str) -> bool:
        return "wikipedia.org" in (base_url or "")

    def search_url(self, query: str, base_url: str) -> str:
        parts = urlparse(base_url)
        origin = f"{parts.scheme}://{parts.netloc}"
        return f"{origin}/w/index.php?search={quote_plus(query)}"


class GitHubSiteAdapter:
    """GitHub repository search, translating common natural-language patterns."""

    name = "github"

    _LANG_RE = re.compile(
        r"\b(python|javascript|typescript|rust|go|java|c\+\+|ruby|swift|kotlin)\b",
        re.IGNORECASE,
    )
    _STARS_RE = re.compile(r"(\d[\d,]*)\s*(?:k|,?000)?\s*\+?\s*stars?", re.IGNORECASE)
    _STARS_THRESHOLD_RE = re.compile(r"more than\s+(\d[\d,]*)", re.IGNORECASE)

    def matches(self, base_url: str) -> bool:
        return "github.com" in (base_url or "")

    def search_url(self, query: str, base_url: str) -> str:
        lang_m = self._LANG_RE.search(query)
        stars_m = self._STARS_RE.search(query)
        threshold_m = self._STARS_THRESHOLD_RE.search(query)
        if lang_m and (stars_m or threshold_m):
            lang = lang_m.group(1).lower().replace("+", "%2B")
            stars = int((threshold_m or stars_m).group(1).replace(",", ""))
            return (
                f"https://github.com/search?q=language%3A{lang}+stars%3A%3E{stars}"
                "&type=repositories&s=stars&o=desc"
            )
        if lang_m:
            return f"https://github.com/trending/{lang_m.group(1).lower()}"
        return f"https://github.com/search?q={quote_plus(query)}&type=repositories"


_BUILTIN_ADAPTERS: dict[str, SiteAdapter] = {
    adapter.name: adapter for adapter in (WikipediaSiteAdapter(), GitHubSiteAdapter())
}

_REGISTRY: list[SiteAdapter] = []


def register_site_adapter(adapter: SiteAdapter) -> None:
    """Register a site adapter. Earlier registrations win on ``matches``."""
    _REGISTRY.append(adapter)


def register_builtin_site_adapters(names: Iterable[str] | None = None) -> None:
    """Register the built-in adapters (all of them, or the named subset)."""
    chosen = (
        _BUILTIN_ADAPTERS.values()
        if names is None
        else [_BUILTIN_ADAPTERS[n] for n in names if n in _BUILTIN_ADAPTERS]
    )
    for adapter in chosen:
        register_site_adapter(adapter)


def clear_site_adapters() -> None:
    """Remove all registered adapters (primarily for tests)."""
    _REGISTRY.clear()


def registered_site_adapters() -> tuple[SiteAdapter, ...]:
    """Return the currently registered adapters, in registration order."""
    return tuple(_REGISTRY)


def resolve_search_url(query: str, base_url: str) -> str:
    """Build a search URL for ``query``.

    Uses the first registered adapter that matches ``base_url``; otherwise falls
    back to the site-agnostic :func:`generic_search_url`.
    """
    for adapter in _REGISTRY:
        if adapter.matches(base_url):
            return adapter.search_url(query, base_url)
    return generic_search_url(query, base_url)


def _load_from_env() -> None:
    raw = os.getenv("OPERON_SITE_ADAPTERS", "").strip()
    if not raw:
        return
    if raw.lower() == "all":
        register_builtin_site_adapters()
    else:
        register_builtin_site_adapters([name.strip() for name in raw.split(",") if name.strip()])


_load_from_env()

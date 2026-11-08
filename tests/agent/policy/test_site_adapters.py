"""Tests for the pluggable site-specific search-URL adapters."""

from __future__ import annotations

import pytest

from operon.agent.policy import site_adapters
from operon.agent.policy.site_adapters import (
    GitHubSiteAdapter,
    WikipediaSiteAdapter,
    clear_site_adapters,
    generic_search_url,
    register_builtin_site_adapters,
    register_site_adapter,
    registered_site_adapters,
    resolve_search_url,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test starts and ends with an empty registry."""
    clear_site_adapters()
    yield
    clear_site_adapters()


# --- default behaviour: no adapters registered -> generic only ----------------


def test_no_adapters_registered_by_default_in_a_clean_registry() -> None:
    assert registered_site_adapters() == ()


def test_resolve_falls_back_to_generic_when_no_adapter_matches() -> None:
    # Even for github/wikipedia URLs, with nothing registered we stay generic.
    # The generic fallback uses urljoin semantics (relative to the base path).
    assert (
        resolve_search_url("Operon", "https://github.com")
        == "https://github.com/search?q=Operon"
    )
    assert (
        resolve_search_url("Markov chain", "https://en.wikipedia.org/wiki/Main_Page")
        == "https://en.wikipedia.org/wiki/search?q=Markov+chain"
    )


def test_generic_search_url_encodes_query() -> None:
    assert generic_search_url("a b&c", "https://example.com/foo") == (
        "https://example.com/search?q=a+b%26c"
    )


# --- built-in adapters (opt-in) ----------------------------------------------


def test_wikipedia_adapter_builds_mediawiki_search_url() -> None:
    register_site_adapter(WikipediaSiteAdapter())
    assert resolve_search_url(
        "Python (programming language)", "https://en.wikipedia.org/wiki/X"
    ) == "https://en.wikipedia.org/w/index.php?search=Python+%28programming+language%29"


def test_github_adapter_language_and_stars() -> None:
    register_site_adapter(GitHubSiteAdapter())
    url = resolve_search_url("rust repos with more than 5000 stars", "https://github.com")
    assert url == (
        "https://github.com/search?q=language%3Arust+stars%3A%3E5000"
        "&type=repositories&s=stars&o=desc"
    )


def test_github_adapter_language_only_uses_trending() -> None:
    register_site_adapter(GitHubSiteAdapter())
    assert resolve_search_url("popular python projects", "https://github.com") == (
        "https://github.com/trending/python"
    )


def test_github_adapter_generic_repo_search() -> None:
    register_site_adapter(GitHubSiteAdapter())
    assert resolve_search_url("Operon agent", "https://github.com") == (
        "https://github.com/search?q=Operon+agent&type=repositories"
    )


def test_register_builtin_by_name_subset() -> None:
    register_builtin_site_adapters(["github"])
    names = [a.name for a in registered_site_adapters()]
    assert names == ["github"]
    # wikipedia not registered -> generic for a wikipedia URL
    assert resolve_search_url("x", "https://en.wikipedia.org/wiki/Y") == (
        "https://en.wikipedia.org/wiki/search?q=x"
    )


def test_register_all_builtins() -> None:
    register_builtin_site_adapters()
    assert {a.name for a in registered_site_adapters()} == {"wikipedia", "github"}


def test_first_match_wins() -> None:
    class AlwaysAdapter:
        name = "always"

        def matches(self, base_url: str) -> bool:
            return True

        def search_url(self, query: str, base_url: str) -> str:
            return "https://override.example/q"

    register_site_adapter(AlwaysAdapter())
    register_site_adapter(GitHubSiteAdapter())
    assert resolve_search_url("x", "https://github.com") == "https://override.example/q"


def test_env_loading_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_SITE_ADAPTERS", "all")
    clear_site_adapters()
    site_adapters._load_from_env()
    assert {a.name for a in registered_site_adapters()} == {"wikipedia", "github"}

"""KAN-191: only http and https links may be opened."""

from __future__ import annotations

import pytest

from operon.core.urls import is_web_url, web_url_refusal


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://example.com/a?b=1",
        "HTTPS://EXAMPLE.COM",
        "Http://example.com:8080/x",
        "  https://example.com  ",
        "http://127.0.0.1:3000/",
    ],
)
def test_web_urls_are_allowed(url: str) -> None:
    assert is_web_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Users/x/.ssh/id_rsa",
        "chrome://settings",
        "javascript:alert(1)",
        "data:text/html,<h1>x</h1>",
        "about:blank",
        "view-source:https://example.com",
        "ftp://example.com/x",
        "blob:https://example.com/abc",
        "example.com",
        "https:///nohost",
        "\x01file:///x",
        "java\nscript:alert(1)",
        "",
        "http://[bad",
    ],
)
def test_other_urls_are_refused(url: str) -> None:
    assert not is_web_url(url)


def test_refusal_names_the_scheme_but_not_the_url() -> None:
    message = web_url_refusal("file:///C:/secret/id_rsa")
    assert message == "navigate blocked: only http and https links can be opened (got 'file:')"
    assert "secret" not in message
    assert "no scheme" in web_url_refusal("example.com")
    assert web_url_refusal("file:///x", "start_url").startswith("start_url blocked")

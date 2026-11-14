"""The built-in browser (KAN-138): address handling, site approvals, tabs and sign-out, with fakes."""

from __future__ import annotations

import asyncio

import pytest

from operon.browser import builtin
from operon.browser.builtin import (
    BuiltinBrowser,
    SiteApprovals,
    host_of,
    normalize_address,
)


class FakeRequest:
    def __init__(self, url: str, *, page_load: bool = True, main_frame: bool = True) -> None:
        self.url = url
        self.resource_type = "document" if page_load else "image"
        self._nav = page_load
        self.frame = type("F", (), {"parent_frame": None if main_frame else object()})()

    def is_navigation_request(self) -> bool:
        return self._nav


class FakeRoute:
    def __init__(self, request: FakeRequest) -> None:
        self.request = request
        self.outcome: str | None = None

    async def continue_(self) -> None:
        self.outcome = "continue"

    async def abort(self, reason: str) -> None:
        self.outcome = f"abort:{reason}"


class FakePage:
    def __init__(self, context: "FakeContext", url: str = "about:blank") -> None:
        self.context = context
        self.url = url
        self._title = ""
        self.closed = False
        self.handler = None
        self.visited: list[str] = []

    async def title(self) -> str:
        return self._title

    def is_closed(self) -> bool:
        return self.closed

    async def goto(self, url: str, **_: object) -> None:
        self.url = url
        self.visited.append(url)

    async def bring_to_front(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True
        self.context.pages.remove(self)

    async def route(self, pattern: str, handler) -> None:
        self.handler = handler


class FakeContext:
    def __init__(self) -> None:
        self.pages: list[FakePage] = []
        self.closed = False

    def on(self, *_: object) -> None:
        return None

    async def new_page(self) -> FakePage:
        page = FakePage(self)
        self.pages.append(page)
        return page

    async def close(self) -> None:
        self.closed = True


class FakePlaywright:
    stopped = False

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def browser(tmp_path, monkeypatch):
    context, playwright = FakeContext(), FakePlaywright()

    async def launcher(profile_dir, *, width, height):
        profile_dir.mkdir(parents=True, exist_ok=True)
        return playwright, context

    async def no_view(self, page):  # the live view needs a real Chromium
        return None

    monkeypatch.setattr(BuiltinBrowser, "show", no_view)
    monkeypatch.setattr(builtin, "_publish", lambda event: published.append(event))
    published: list[dict] = []
    b = BuiltinBrowser(tmp_path / "profile", launcher=launcher)
    b.fake_context, b.fake_playwright, b.published = context, playwright, published
    return b


def test_addresses_become_pages_or_searches() -> None:
    assert normalize_address("https://example.com/a") == "https://example.com/a"
    assert normalize_address("example.com") == "https://example.com"
    assert normalize_address("localhost:8080") == "https://localhost:8080"
    assert normalize_address("best pizza near me").startswith("https://www.google.com/search?q=best+pizza")
    assert normalize_address("pizza").startswith("https://www.google.com/search?q=pizza")
    with pytest.raises(ValueError):
        normalize_address("   ")


def test_host_of_ignores_non_web_addresses() -> None:
    assert host_of("https://Mail.Example.com/x") == "mail.example.com"
    assert host_of("about:blank") is None
    assert host_of("chrome-error://chromewebdata/") is None


def test_site_approvals_cover_subdomains_and_persist(tmp_path) -> None:
    approvals = SiteApprovals(tmp_path)
    assert not approvals.allows("www.example.com")
    approvals.decide("example.com", "always")
    assert approvals.allows("example.com") and approvals.allows("www.example.com")
    assert not approvals.allows("notexample.com")
    assert SiteApprovals(tmp_path).always == ["example.com"]  # saved in the profile
    approvals.remove("example.com")
    assert SiteApprovals(tmp_path).always == []


def test_once_and_per_run_approvals_are_not_saved(tmp_path) -> None:
    approvals = SiteApprovals(tmp_path)
    approvals.decide("a.com", "once")
    approvals.allow_for_run("run-1", "https://b.com/start")
    assert approvals.allows("a.com") and approvals.allows("b.com", "run-1")
    assert not approvals.allows("b.com", "run-2") and not approvals.allows("b.com")
    approvals.forget_run("run-1")
    assert not approvals.allows("b.com", "run-1")
    assert SiteApprovals(tmp_path).always == []


def test_a_bad_answer_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        SiteApprovals(tmp_path).decide("a.com", "maybe")


async def test_a_task_page_may_open_its_start_site_but_must_ask_for_others(browser) -> None:
    page = await browser.new_task_page("run-1", "https://shop.example/start")

    start = FakeRoute(FakeRequest("https://shop.example/cart"))
    await page.handler(start)
    other = FakeRoute(FakeRequest("https://elsewhere.test/login"))
    await page.handler(other)
    again = FakeRoute(FakeRequest("https://elsewhere.test/login"))
    await page.handler(again)

    assert start.outcome == "continue"
    assert other.outcome == "abort:blockedbyclient" and again.outcome == "abort:blockedbyclient"
    assert list(browser.approvals.pending) == ["elsewhere.test"]
    assert browser.published == [{"type": "site_approval_needed", "host": "elsewhere.test", "run_id": "run-1"}]

    browser.approvals.decide("elsewhere.test", "once")
    allowed = FakeRoute(FakeRequest("https://elsewhere.test/login"))
    await page.handler(allowed)
    assert allowed.outcome == "continue" and not browser.approvals.pending


async def test_only_page_loads_are_gated(browser) -> None:
    page = await browser.new_task_page("run-1", None)
    image = FakeRoute(FakeRequest("https://cdn.test/x.png", page_load=False))
    frame = FakeRoute(FakeRequest("https://ads.test/frame", main_frame=False))
    blank = FakeRoute(FakeRequest("about:blank"))
    for route in (image, frame, blank):
        await page.handler(route)
    assert [image.outcome, frame.outcome, blank.outcome] == ["continue"] * 3
    assert not browser.approvals.pending


async def test_tabs_open_select_and_close(browser) -> None:
    assert await browser.tabs() == []  # listing never starts the browser
    first = await browser.current_page()
    new_id = await browser.open_tab("example.com")
    tabs = await browser.tabs()
    assert [t["active"] for t in tabs] == [False, True]
    assert tabs[1]["url"] == "https://example.com" and tabs[1]["id"] == new_id

    await browser.select_tab(tabs[0]["id"])
    assert (await browser.current_page()) is first
    await browser.close_tab(new_id)
    assert len(await browser.tabs()) == 1
    with pytest.raises(KeyError):
        await browser.select_tab(new_id)


async def test_navigate_uses_the_current_tab(browser) -> None:
    await browser.navigate("example.com")
    assert (await browser.current_page()).visited == ["https://example.com"]


async def test_sign_out_everywhere_deletes_the_profile_and_approvals(browser) -> None:
    await browser.ensure_started()
    (browser.profile_dir / "Cookies").write_text("session")
    browser.approvals.decide("example.com", "always")

    await browser.sign_out_everywhere()

    assert not browser.profile_dir.exists()
    assert browser.fake_context.closed and browser.fake_playwright.stopped
    assert not browser.started and browser.approvals.always == []


async def test_the_browser_starts_once(browser) -> None:
    contexts = await asyncio.gather(browser.ensure_started(), browser.ensure_started())
    assert contexts[0] is contexts[1] and len(browser.fake_context.pages) == 1

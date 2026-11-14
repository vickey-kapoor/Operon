"""The built-in browser (KAN-138): one Operon Chromium with a saved profile.

The app's Browser tab and browser tasks share this one browser. Its profile folder
(``OPERON_BROWSER_PROFILE_DIR``, default ``~/.operon/browser-profile``) keeps cookies, so a
site the user signed in to stays signed in across tasks and restarts. It never leaves the
computer.

* The user drives it from the Browser tab: tabs, an address bar, back and forward, and
  mouse and keyboard through the live view (``/app/api/browser``).
* Browser tasks open a tab in it (``NativeBrowserExecutor``, observable mode) instead of
  launching a fresh browser, so they start signed in. The user watches in the same view
  and can take over as before.
* Sites: a task page may only open a site the user allowed (``SiteApprovals``). Opening an
  unapproved site is blocked and offered to the user as Once / Always / No.
* "Sign out everywhere" closes the browser and deletes the profile.

Only the Browser tab and task setup read page addresses and titles here. The agent's own
perception stays pixels-only: nothing in this module feeds the model.

The feature is off unless ``OPERON_BUILTIN_BROWSER`` is on; benchmarks never set it.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)

CDP_PORT = 9222
SEARCH_URL = "https://www.google.com/search?q={query}"
APPROVALS_FILE = "operon-approved-sites.json"


def builtin_browser_enabled() -> bool:
    from operon.core.settings import get_settings

    return get_settings().builtin_browser


def normalize_address(text: str) -> str:
    """What the address bar's text means: a web address, or a search for it."""
    text = text.strip()
    if not text:
        raise ValueError("Type an address or a search.")
    lowered = text.lower()
    if lowered.startswith(("http://", "https://", "about:")):
        return text
    if " " not in text and ("." in text or lowered.startswith("localhost")):
        return f"https://{text}"
    return SEARCH_URL.format(query=quote_plus(text))


def host_of(url: str) -> str | None:
    """The host of a web address, or None for anything that is not http(s)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return parsed.hostname.lower()


def _covers(approved: str, host: str) -> bool:
    return host == approved or host.endswith("." + approved)


class SiteApprovals:
    """Which sites a browser task may open.

    ``always`` is kept in the profile folder (so "Sign out everywhere" clears it);
    ``once`` lasts until the process ends. A site covers its subdomains. ``allow_for_run``
    lets one run open the site its own start address names. Navigations of a task page
    to any other site wait here as ``pending`` until the user answers.
    """

    def __init__(self, profile_dir: Path) -> None:
        self._file = Path(profile_dir) / APPROVALS_FILE
        self._always: set[str] = self._load()
        self._once: set[str] = set()
        self._per_run: dict[str, set[str]] = {}
        self.pending: dict[str, dict[str, Any]] = {}

    def _load(self) -> set[str]:
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        return {str(h).lower() for h in data.get("always", []) if h}

    def _save(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps({"always": sorted(self._always)}), encoding="utf-8")
        except OSError:
            logger.warning("Could not save the approved sites to %s", self._file, exc_info=True)

    @property
    def always(self) -> list[str]:
        return sorted(self._always)

    def allow_for_run(self, run_id: str, url: str | None) -> None:
        host = host_of(url) if url else None
        if host:
            self._per_run.setdefault(run_id, set()).add(host)

    def forget_run(self, run_id: str) -> None:
        self._per_run.pop(run_id, None)

    def allows(self, host: str, run_id: str | None = None) -> bool:
        pools = [self._always, self._once]
        if run_id is not None:
            pools.append(self._per_run.get(run_id, set()))
        return any(_covers(approved, host) for pool in pools for approved in pool)

    def ask(self, host: str, run_id: str | None, url: str) -> bool:
        """Note that a task wants ``host``. True the first time (the user is told)."""
        if host in self.pending:
            return False
        self.pending[host] = {"host": host, "url": url, "run_id": run_id}
        return True

    def decide(self, host: str, answer: str) -> None:
        """``answer`` is ``once``, ``always`` or ``no``."""
        host = host.lower()
        if answer == "always":
            self._always.add(host)
            self._save()
        elif answer == "once":
            self._once.add(host)
        elif answer != "no":
            raise ValueError("answer must be once, always or no")
        self.pending.pop(host, None)

    def remove(self, host: str) -> None:
        self._always.discard(host.lower())
        self._save()

    def clear(self) -> None:
        self._always.clear()
        self._once.clear()
        self._per_run.clear()
        self.pending.clear()


Launcher = Callable[..., Awaitable[tuple[Any, Any]]]


async def _launch_chromium(profile_dir: Path, *, width: int, height: int) -> tuple[Any, Any]:
    """Start Playwright and a persistent Chromium (no OS window) with CDP on port 9222."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is not installed for the built-in browser.") from exc
    profile_dir.mkdir(parents=True, exist_ok=True)
    playwright = await async_playwright().start()
    try:
        # --headless=new renders off-screen so CDP screencast can capture it; headless=False
        # stops Playwright adding its own --headless (same as observable mode).
        context = await playwright.chromium.launch_persistent_context(
            str(profile_dir),
            headless=False,
            args=[
                "--headless=new",
                f"--window-size={width},{height}",
                f"--remote-debugging-port={CDP_PORT}",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )
    except Exception:
        await playwright.stop()
        raise
    return playwright, context


class BuiltinBrowser:
    """The one saved-profile browser, started on first use."""

    def __init__(
        self,
        profile_dir: Path,
        *,
        width: int = 1920,
        height: int = 1080,
        launcher: Launcher | None = None,
    ) -> None:
        self.profile_dir = Path(profile_dir)
        self._width, self._height = width, height
        self._launcher: Launcher = launcher or _launch_chromium
        self._playwright: Any = None
        self._context: Any = None
        self._lock = asyncio.Lock()
        self._ids = itertools.count(1)
        self._tab_ids: dict[int, int] = {}  # id(page) -> tab id
        self._active: Any = None
        # Pages a browser task owns (its tab and the popups it opens): id(page) -> (page, run id).
        # Only these are gated and closed with the run; the user's own tabs are never touched.
        self._owned: dict[int, tuple[Any, str]] = {}
        self.approvals = SiteApprovals(self.profile_dir)

    @property
    def started(self) -> bool:
        return self._context is not None

    # ── Start and stop ──────────────────────────────────────────────────────

    async def ensure_started(self) -> Any:
        """The persistent context, launching the browser the first time."""
        async with self._lock:
            if self._context is None:
                self._playwright, self._context = await self._launcher(
                    self.profile_dir, width=self._width, height=self._height
                )
                self._context.on("close", lambda *_: self._on_closed())
                self._context.on("page", self._on_new_page)
                await self._context.route("**/*", self._gate)
                pages = list(self._context.pages)
                self._active = pages[0] if pages else await self._context.new_page()
                logger.info("Built-in browser started (profile %s)", self.profile_dir)
            return self._context

    def _on_closed(self) -> None:
        self._context = None
        self._active = None
        self._owned.clear()

    async def aclose(self) -> None:
        async with self._lock:
            context, playwright = self._context, self._playwright
            self._context = self._playwright = self._active = None
        if context is not None:
            try:
                await context.close()
            except Exception:
                logger.debug("Built-in browser: closing the context failed", exc_info=True)
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                logger.debug("Built-in browser: stopping Playwright failed", exc_info=True)

    async def sign_out_everywhere(self) -> None:
        """Close the browser and delete its profile: every sign-in and approved site."""
        await self._stop_live_view()
        await self.aclose()
        shutil.rmtree(self.profile_dir, ignore_errors=True)
        self.approvals.clear()
        self._tab_ids.clear()
        self._owned.clear()

    # ── Tabs ────────────────────────────────────────────────────────────────

    def _tab_id(self, page: Any) -> int:
        return self._tab_ids.setdefault(id(page), next(self._ids))

    def _page_by_id(self, tab_id: int) -> Any:
        if self._context is not None:
            for page in self._context.pages:
                if self._tab_id(page) == tab_id:
                    return page
        raise KeyError(f"no tab {tab_id}")

    async def tabs(self) -> list[dict[str, Any]]:
        if self._context is None:
            return []
        out: list[dict[str, Any]] = []
        for page in self._context.pages:
            try:
                title = await page.title()
            except Exception:
                title = ""
            out.append(
                {
                    "id": self._tab_id(page),
                    "url": page.url,
                    "title": title,
                    "active": page is self._active,
                }
            )
        return out

    async def current_page(self) -> Any:
        context = await self.ensure_started()
        if self._active is None or self._active.is_closed():
            pages = [p for p in context.pages if not p.is_closed()]
            self._active = pages[-1] if pages else await context.new_page()
        return self._active

    async def open_tab(self, address: str | None = None) -> int:
        context = await self.ensure_started()
        page = await context.new_page()
        self._active = page
        if address:
            await page.goto(normalize_address(address), wait_until="domcontentloaded")
        await self.show(page)
        return self._tab_id(page)

    async def select_tab(self, tab_id: int) -> None:
        page = self._page_by_id(tab_id)
        self._active = page
        await page.bring_to_front()
        await self.show(page)

    async def close_tab(self, tab_id: int) -> None:
        page = self._page_by_id(tab_id)
        await page.close()
        self._tab_ids.pop(id(page), None)
        if page is self._active:
            self._active = None
            await self.show(await self.current_page())

    # ── Navigation (the Browser tab's toolbar) ──────────────────────────────

    async def navigate(self, address: str) -> str:
        page = await self.current_page()
        url = normalize_address(address)
        await page.goto(url, wait_until="domcontentloaded")
        await self.show(page)
        return page.url

    async def back(self) -> None:
        await (await self.current_page()).go_back(wait_until="domcontentloaded")

    async def forward(self) -> None:
        await (await self.current_page()).go_forward(wait_until="domcontentloaded")

    async def reload(self) -> None:
        await (await self.current_page()).reload(wait_until="domcontentloaded")

    # ── Live view ───────────────────────────────────────────────────────────

    async def show(self, page: Any) -> None:
        """Point the live view (picture and the user's input) at ``page``. Best effort."""
        from operon.api import ws_stream as _ws
        from operon.browser.manager import (
            BrowserManager,
            _target_id,
            get_active_manager,
            set_active_manager,
        )

        try:
            manager = get_active_manager()
            if manager is None or not manager.is_connected:
                manager = BrowserManager()
                await manager.connect(CDP_PORT)
                set_active_manager(manager)
                _ws.set_browser_manager(manager)
            target_id = await _target_id(page)
            if target_id is None or not await manager.follow_target(target_id):
                logger.warning("Built-in browser: the live view could not find the page")
            await manager.start_screencast(fps=15)
        except Exception as exc:
            logger.warning("Built-in browser: live view unavailable — %s", exc)

    async def _stop_live_view(self) -> None:
        from operon.browser.manager import get_active_manager

        manager = get_active_manager()
        if manager is None:
            return
        try:
            await manager.stop_screencast()
            await manager.disconnect()
        except Exception:
            logger.debug("Built-in browser: stopping the live view failed", exc_info=True)

    # ── Browser tasks ───────────────────────────────────────────────────────

    async def new_task_page(self, run_id: str, start_url: str | None = None) -> Any:
        """A fresh tab for a browser task. The site approval gate covers it and its popups."""
        context = await self.ensure_started()
        self.approvals.allow_for_run(run_id, start_url)
        page = await context.new_page()
        self.own_page(page, run_id)
        self._active = page  # the Browser tab opens on the run, not on a stale tab
        return page

    def own_page(self, page: Any, run_id: str) -> None:
        """Mark ``page`` (a task's tab or a popup it opened) as the run's."""
        self._owned[id(page)] = (page, run_id)

    def set_active(self, page: Any) -> None:
        """The tab the live view shows now (a task followed a popup, say)."""
        self._active = page

    def run_of_active(self) -> str | None:
        """The run that owns the tab in view, or None when it is the user's own tab."""
        if self._active is None:
            return None
        owned = self._owned.get(id(self._active))
        return owned[1] if owned else None

    def pages_of_run(self, run_id: str) -> list[Any]:
        """The run's open pages, oldest first."""
        return [page for page, owner in self._owned.values() if owner == run_id and not page.is_closed()]

    async def release_run(self, run_id: str) -> None:
        """The run ended: close its tabs and popups (never the user's) and drop its approvals."""
        for page in self.pages_of_run(run_id):
            self._owned.pop(id(page), None)
            try:
                await page.close()
            except Exception:
                logger.debug("Built-in browser: closing a task page failed", exc_info=True)
        for key in [k for k, (_, owner) in self._owned.items() if owner == run_id]:
            del self._owned[key]
        self.approvals.forget_run(run_id)
        if self._active is not None and self._active.is_closed():
            self._active = None

    async def _owner_of(self, page: Any) -> str | None:
        """The run that owns ``page``: its tab, or a popup opened (however deep) from it."""
        owned = self._owned.get(id(page))
        if owned is not None:
            return owned[1]
        opener = await page.opener()
        if opener is None:
            return None
        run_id = await self._owner_of(opener)
        if run_id is not None:
            self.own_page(page, run_id)
        return run_id

    def _on_new_page(self, page: Any) -> None:
        asyncio.get_running_loop().create_task(self._watch_popup(page))

    async def _watch_popup(self, page: Any) -> None:
        """Back-stop for a task's popups: Chromium can start a popup's first page load
        before the route above sees it. If a popup of a task lands on a site the user has
        not allowed, blank it and ask, so the agent never works on that page."""
        try:
            run_id = await self._owner_of(page)
        except Exception:
            return
        if run_id is None:
            return

        async def check(frame: Any) -> None:
            host = host_of(frame.url)
            if host is None or self.approvals.allows(host, run_id):
                return
            if self.approvals.ask(host, run_id, frame.url):
                _publish({"type": "site_approval_needed", "host": host, "run_id": run_id})
            try:
                await page.goto("about:blank")
            except Exception:
                logger.debug("Built-in browser: blanking a blocked popup failed", exc_info=True)

        page.on("framenavigated", lambda frame: frame is page.main_frame and asyncio.get_running_loop().create_task(check(frame)))
        await check(page.main_frame)

    async def _gate(self, route: Any) -> None:
        """Let a task's pages open only the sites the user allowed; ask about the rest.

        Runs for every request of the shared context. The user's own tabs and every
        subresource go straight through: only a task page's own page loads are held."""
        request = route.request
        try:
            is_page_load = (
                request.resource_type == "document"
                and request.is_navigation_request()
                and request.frame.parent_frame is None
            )
            host = host_of(request.url) if is_page_load else None
            run_id = await self._owner_of(request.frame.page) if host is not None else None
        except Exception:
            host, run_id = None, None
        if host is None or run_id is None or self.approvals.allows(host, run_id):
            await route.continue_()
            return
        if self.approvals.ask(host, run_id, request.url):
            _publish({"type": "site_approval_needed", "host": host, "run_id": run_id})
        await route.abort("blockedbyclient")


def _publish(event: dict[str, Any]) -> None:
    from operon.api import ws_stream as _ws

    try:
        _ws.publish_event(event)
    except Exception:
        logger.debug("Built-in browser: publishing %s failed", event.get("type"), exc_info=True)


def get_builtin_browser() -> BuiltinBrowser:
    """The process's one built-in browser (on the runtime), created on first use."""
    from operon.core.paths import browser_profile_dir
    from operon.core.runtime import current_runtime
    from operon.core.settings import get_settings

    runtime = current_runtime()
    if runtime.builtin_browser is None:
        settings = get_settings()
        runtime.builtin_browser = BuiltinBrowser(
            browser_profile_dir(), width=settings.browser_width, height=settings.browser_height
        )
    return runtime.builtin_browser


async def close_builtin_browser() -> None:
    """Server shutdown: close the browser if it was started."""
    from operon.core.runtime import current_runtime

    browser = current_runtime().builtin_browser
    if browser is not None:
        await browser.aclose()

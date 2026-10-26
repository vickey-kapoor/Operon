"""A local web site for benchmark tasks that are graded by what the agent did.

Live public sites can only be graded from a screenshot. Pages served from here
log every request the browser makes, so a task can be graded exactly: which
forms the agent submitted, with what values, and which links it followed. The
benchmark starts one site per run on a free loopback port and clears the log
before each task.

A page is keyed by path. A GET returns the page at that path, and a POST logs
its form fields and returns the page at its path too (so a POST to ``/cart`` can
answer with a cart page). Unknown paths get a plain 404 page, and are still
logged. A page is plain HTML, or a function of the request and the log so far
(``page(event, events) -> html``) for pages that show state: a confirmation
that names what was submitted, or an inbox without the email just archived.

Apps (``tests/eval/local_apps.py``) are mounted under a path prefix and keep
real state: a cart, a signed-in session, a mailbox. ``GET /state`` returns every
app's state as JSON and ``POST /state/reset`` puts it back to the start, so a
task is graded on what actually happened, and the site can be run on its own
(``python -m tests.eval.local_apps``) and inspected by hand.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol
from urllib.parse import parse_qsl, urlsplit

NOT_FOUND = "<!doctype html><title>Not found</title><h1>Page not found</h1>"
RECEIVED = "<!doctype html><title>Done</title><h1>Request received</h1>"


@dataclass(frozen=True)
class Event:
    method: str
    path: str
    form: dict[str, str] = field(default_factory=dict)


Page = str | Callable[[Event, list[Event]], str]


@dataclass(frozen=True)
class Request:
    """What an app sees: every value of a repeated field (checkboxes), and the cookies."""

    method: str
    path: str
    fields: tuple[tuple[str, str], ...] = ()
    cookies: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        values = self.getall(key)
        return values[-1] if values else default

    def getall(self, key: str) -> list[str]:
        return [value for name, value in self.fields if name == key]


@dataclass(frozen=True)
class Reply:
    body: str = ""
    status: int = 200
    headers: tuple[tuple[str, str], ...] = ()
    content_type: str = "text/html; charset=utf-8"


def redirect(location: str, *headers: tuple[str, str]) -> Reply:
    """A 303 after a POST, so a reload never submits the form twice."""
    return Reply(status=303, headers=(("Location", location), *headers))


class App(Protocol):
    """A stateful app mounted at ``prefix``: it answers every path below it."""

    name: str
    prefix: str

    def handle(self, request: Request) -> Reply | None: ...

    def state(self) -> dict: ...

    def reset(self, **seed) -> None: ...


class FixtureSite:
    def __init__(
        self, pages: dict[str, Page], host: str = "127.0.0.1", port: int = 0, apps: list[App] | tuple[App, ...] = (),
    ) -> None:
        self.pages = pages
        self.apps = list(apps)
        self.events: list[Event] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def url(self, path: str) -> str:
        return self.base_url + path

    def start(self) -> FixtureSite:
        self._thread = threading.Thread(target=self._server.serve_forever, name="fixture-site", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> FixtureSite:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def reset(self, seed: dict[str, dict] | None = None) -> None:
        """Clear the request log and put every app back to its start (``seed``: per-app overrides)."""
        with self._lock:
            self.events.clear()
            for app in self.apps:
                app.reset(**(seed or {}).get(app.name, {}))

    def state(self) -> dict:
        """Every app's state, by app name: what ``GET /state`` returns."""
        with self._lock:
            return {app.name: app.state() for app in self.apps}

    def _app_reply(self, request: Request) -> Reply | None:
        for app in self.apps:
            if request.path == app.prefix or request.path.startswith(app.prefix + "/"):
                with self._lock:
                    return app.handle(request)
        return None

    def snapshot(self) -> list[Event]:
        with self._lock:
            return list(self.events)

    def _record(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        site = self

        class Handler(BaseHTTPRequestHandler):
            def _respond(self, method: str, body: str = "") -> None:
                parts = urlsplit(self.path)
                pairs = parse_qsl(parts.query, keep_blank_values=True) + parse_qsl(body, keep_blank_values=True)
                if parts.path in ("/state", "/state/reset"):  # the grader's view, never logged
                    if method == "POST" and parts.path == "/state/reset":
                        site.reset()
                    self._send(Reply(json.dumps(site.state(), indent=2), content_type="application/json"))
                    return
                event = Event(method, parts.path, dict(pairs))
                if parts.path != "/favicon.ico":
                    site._record(event)
                cookies = {name: morsel.value for name, morsel in SimpleCookie(self.headers.get("Cookie") or "").items()}
                reply = site._app_reply(Request(method, parts.path, tuple(pairs), cookies))
                if reply is None:
                    page = site.pages.get(parts.path)
                    if callable(page):
                        page = page(event, site.snapshot())
                    if page is None and method == "POST":
                        page = RECEIVED
                    reply = Reply(page or NOT_FOUND, 404 if page is None else 200)
                self._send(reply)

            def _send(self, reply: Reply) -> None:
                data = reply.body.encode("utf-8")
                self.send_response(reply.status)
                self.send_header("Content-Type", reply.content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                for name, value in reply.headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
                self._respond("GET")

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self._respond("POST", self.rfile.read(length).decode("utf-8", "replace"))

            def log_message(self, format: str, *args) -> None:  # keep benchmark output clean
                return None

        return Handler

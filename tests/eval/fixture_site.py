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
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

NOT_FOUND = "<!doctype html><title>Not found</title><h1>Page not found</h1>"
RECEIVED = "<!doctype html><title>Done</title><h1>Request received</h1>"


@dataclass(frozen=True)
class Event:
    method: str
    path: str
    form: dict[str, str] = field(default_factory=dict)


Page = str | Callable[[Event, list[Event]], str]


class FixtureSite:
    def __init__(self, pages: dict[str, Page], host: str = "127.0.0.1", port: int = 0) -> None:
        self.pages = pages
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

    def reset(self) -> None:
        with self._lock:
            self.events.clear()

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
                form = dict(parse_qsl(parts.query)) | dict(parse_qsl(body))
                event = Event(method, parts.path, form)
                if parts.path != "/favicon.ico":
                    site._record(event)
                page = site.pages.get(parts.path)
                if callable(page):
                    page = page(event, site.snapshot())
                if page is None and method == "POST":
                    page = RECEIVED
                status = 404 if page is None else 200
                data = (page or NOT_FOUND).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
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

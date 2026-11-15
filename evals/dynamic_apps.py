"""Dynamic-screen app for the benchmark's local site: toasts, spinners, errors that vanish (KAN-114).

Most screens hold still after an action. These do not, the way real products do not:
a "saved" toast shows for a moment and goes, an error flashes and clears, a table appears
after a spinner, a result loads a few seconds late. They let us measure whether waiting for
the screen to settle (``OPERON_SETTLE_DETECT``, KAN-110) and the between-step change note
(``OPERON_CHANGE_NOTE``, KAN-111) help a vision-only agent.

Built like the other local apps (``evals/local_apps.py``): forms that POST and redirect, real
server-side state returned by ``GET /state`` under ``dynamic``, so a task is graded on what
happened. The timing is plain page JavaScript (``data-ms`` on a toast, ``data-reveal-ms`` on
content), so a test can walk every action over HTTP without a browser. The state never
depends on the timers: a button that is still hidden when an agent clicks where it will be
simply is not there to click.

``build_apps()`` is the Season 1 site plus this app, served under ``/apps/dynamic/``. Only a
run with a dynamic task uses it, so frozen Season 1 and 2 tasks see the site they were frozen on.

Run the site on its own to look at it::

    .venv/bin/python -m evals.dynamic_apps [--port 8765]
"""

from __future__ import annotations

import argparse
import copy
from html import escape
from urllib.parse import quote

from evals.fixture_site import FixtureSite, Reply, Request, redirect
from evals.local_apps import BaseApp, layout
from evals.local_apps import build_apps as season1_apps

THEMES = ("Light", "Dark", "System")
INVOICES = {
    "INV-1040": ("Acme Supplies", 120.00),
    "INV-1041": ("Birch & Co", 75.00),
    "INV-1042": ("Cobalt Labs", 310.00),
    "INV-1043": ("Delta Freight", 48.00),
}
PROJECTS = ("Alpha", "Beta", "Gamma")
DOCUMENTS = ("Quarterly plan", "Quarterly report", "Annual plan", "Plan notes")
REPORT_ROWS = 128
FIRST_SUBMIT_FAILS = 1  # the waitlist is "busy" for this many first submissions

SCRIPT = """<script>
(function () {
  document.querySelectorAll('[data-toast]').forEach(function (el) {
    setTimeout(function () { el.remove(); }, parseInt(el.dataset.ms || '1500', 10));
  });
  document.querySelectorAll('[data-reveal-ms]').forEach(function (el) {
    var spinner = document.getElementById(el.dataset.spinner);
    setTimeout(function () {
      el.hidden = false;
      if (spinner) spinner.remove();
    }, parseInt(el.dataset.revealMs, 10));
  });
  if (location.search.indexOf('toast=') !== -1) {
    history.replaceState(null, '', location.pathname);
  }
})();
</script>"""
STYLE = """<style>
.toast { position: fixed; top: 70px; right: 24px; max-width: 320px; padding: 12px 16px; border-radius: 8px;
  background: #064e3b; color: #fff; box-shadow: 0 6px 18px rgba(0,0,0,.25); z-index: 10; }
.toast.bad { background: #991b1b; }
.toast form { display: inline; margin-left: 10px; }
.toast button { background: #fff; color: #064e3b; border: 0; border-radius: 4px; padding: 4px 10px; font-weight: 700; }
.spinner { width: 34px; height: 34px; border: 4px solid #d1d5db; border-top-color: #2563eb; border-radius: 50%;
  animation: spin 0.9s linear infinite; margin: 20px 0; }
@keyframes spin { to { transform: rotate(360deg); } }
.done { color: #6b7280; }
</style>"""


def toast(text: str, *, ms: int, bad: bool = False, extra: str = "") -> str:
    return f'<div class="toast{" bad" if bad else ""}" data-toast data-ms="{ms}">{escape(text)}{extra}</div>'


class DynamicApp(BaseApp):
    name = "dynamic"
    prefix = "/apps/dynamic"
    site = "Northwind Live"

    def _table(self):
        return [
            ("GET", r"/", self.home),
            ("GET", r"/settings", self.settings_page),
            ("POST", r"/settings", self.save_settings),
            ("GET", r"/waitlist", self.waitlist_page),
            ("POST", r"/waitlist", self.join_waitlist),
            ("GET", r"/invoices", self.invoices_page),
            ("POST", r"/invoices/pay", self.pay_invoice),
            ("GET", r"/report", self.report_page),
            ("POST", r"/report/generate", self.generate_report),
            ("POST", r"/report/open", self.open_report),
            ("GET", r"/projects", self.projects_page),
            ("POST", r"/projects/archive", self.archive_project),
            ("POST", r"/projects/undo", self.undo_archive),
            ("GET", r"/search", self.search_page),
            ("POST", r"/search/open", self.open_document),
        ]

    def reset(self, **_) -> None:
        self.theme = "System"
        self.saves = 0
        self.waitlist_attempts = 0
        self.waitlist_email = ""
        self.waitlist_joined = False
        self.paid: list[str] = []
        self.report_generated = False
        self.report_opened = False
        self.archived: list[str] = []
        self.archive_events = 0
        self.undo_events = 0
        self.opened_document = ""

    def state(self) -> dict:
        return copy.deepcopy({
            "settings": {"theme": self.theme, "saves": self.saves},
            "waitlist": {"attempts": self.waitlist_attempts, "email": self.waitlist_email, "joined": self.waitlist_joined},
            "invoices": {"paid": sorted(self.paid)},
            "report": {"generated": self.report_generated, "opened": self.report_opened},
            "projects": {"archived": sorted(self.archived), "archive_events": self.archive_events,
                         "undo_events": self.undo_events},
            "search": {"opened": self.opened_document},
        })

    def _page(self, title: str, body: str, *, toast_html: str = "", error: str = "") -> Reply:
        nav = f'<a href="{self.url("/")}">Live demos</a>'
        return layout(title, f"{body}{toast_html}{STYLE}{SCRIPT}", site=self.site, nav=nav, error=escape(error))

    def home(self, request: Request) -> Reply:
        links = "".join(f'<li><a href="{self.url(path)}">{label}</a></li>' for path, label in (
            ("/settings", "Settings"), ("/waitlist", "Join the waitlist"), ("/invoices", "Invoices"),
            ("/report", "Reports"), ("/projects", "Projects"), ("/search", "Find a document")))
        return self._page("Northwind Live", f"<h1>Northwind Live</h1><ul>{links}</ul>")

    # --- settings: a "saved" toast that goes away -------------------------------------

    def settings_page(self, request: Request) -> Reply:
        options = "".join(f'<option{" selected" if t == self.theme else ""}>{t}</option>' for t in THEMES)
        note = toast("Settings saved", ms=1500) if request.get("toast") == "saved" else ""
        return self._page("Settings", f"""<h1>Settings</h1>
            <form method="post" action="{self.url('/settings')}"><label>Theme</label>
            <select name="theme">{options}</select><p><button class="primary">Save</button></p></form>""",
                          toast_html=note)

    def save_settings(self, request: Request) -> Reply:
        theme = request.get("theme")
        if theme in THEMES:
            self.theme = theme
            self.saves += 1
        return redirect(self.url("/settings?toast=saved"))

    # --- waitlist: an error that flashes and clears -----------------------------------

    def waitlist_page(self, request: Request) -> Reply:
        if self.waitlist_joined:
            return self._page("Waitlist", f"<h1>You're on the list</h1><p>We'll write to {escape(self.waitlist_email)}.</p>")
        busy = toast("Server busy, please try again", ms=1500, bad=True) if request.get("toast") == "busy" else ""
        problem = "Enter a valid email address." if request.get("err") == "email" else ""
        return self._page("Waitlist", f"""<h1>Join the waitlist</h1>
            <form method="post" action="{self.url('/waitlist')}"><label>Email</label>
            <input name="email" size="30" autocomplete="off"><p><button class="primary">Join</button></p></form>""",
                          toast_html=busy, error=problem)

    def join_waitlist(self, request: Request) -> Reply:
        email = request.get("email").strip()
        if "@" not in email:
            return redirect(self.url("/waitlist?err=email"))
        self.waitlist_attempts += 1
        if self.waitlist_attempts <= FIRST_SUBMIT_FAILS:
            return redirect(self.url("/waitlist?toast=busy"))
        self.waitlist_email, self.waitlist_joined = email, True
        return redirect(self.url("/waitlist"))

    # --- invoices: a table that loads after a spinner ---------------------------------

    def invoices_page(self, request: Request) -> Reply:
        rows = "".join(
            f"<tr><td>{number}</td><td>{escape(vendor)}</td><td>${amount:,.2f}</td><td>"
            + ('<span class="done">Paid</span>' if number in self.paid else
               f'<form method="post" action="{self.url("/invoices/pay")}"><input type="hidden" name="number" value="{number}">'
               f"<button>Mark paid</button></form>")
            + "</td></tr>" for number, (vendor, amount) in INVOICES.items())
        note = toast("Invoice marked as paid", ms=1500) if request.get("toast") == "paid" else ""
        return self._page("Invoices", f"""<h1>Invoices</h1>
            <div id="inv-spinner"><div class="spinner"></div><p class="muted">Loading invoices…</p></div>
            <table hidden data-reveal-ms="2500" data-spinner="inv-spinner"><tr><th>Number</th><th>Vendor</th>
            <th>Amount</th><th></th></tr>{rows}</table>""", toast_html=note)

    def pay_invoice(self, request: Request) -> Reply:
        number = request.get("number")
        if number in INVOICES and number not in self.paid:
            self.paid.append(number)
        return redirect(self.url("/invoices?toast=paid"))

    # --- report: a spinner, then the result -------------------------------------------

    def report_page(self, request: Request) -> Reply:
        if self.report_opened:
            return self._page("Report", f"<h1>Quarterly summary</h1><p>{REPORT_ROWS} rows, generated just now.</p>")
        if request.get("run") and self.report_generated:
            return self._page("Report", f"""<h1>Reports</h1>
                <div id="rep-spinner"><div class="spinner"></div><p class="muted">Generating your report…</p></div>
                <div hidden data-reveal-ms="3000" data-spinner="rep-spinner"><p><strong>Report ready.</strong></p>
                <form method="post" action="{self.url('/report/open')}"><button class="primary">Open report</button></form></div>""")
        return self._page("Report", f"""<h1>Reports</h1><p>Quarterly summary</p>
            <form method="post" action="{self.url('/report/generate')}"><button class="primary">Generate report</button></form>""")

    def generate_report(self, request: Request) -> Reply:
        self.report_generated = True
        return redirect(self.url("/report?run=1"))

    def open_report(self, request: Request) -> Reply:
        if self.report_generated:
            self.report_opened = True
        return redirect(self.url("/report"))

    # --- projects: a toast with an undo that expires ------------------------------------

    def projects_page(self, request: Request) -> Reply:
        rows = "".join(
            f"<tr><td>{name}</td><td>"
            + ('<span class="done">Archived</span>' if name in self.archived else
               f'<form method="post" action="{self.url("/projects/archive")}"><input type="hidden" name="name" value="{name}">'
               "<button>Archive</button></form>") + "</td></tr>" for name in PROJECTS)
        note = ""
        name = request.get("p")
        if request.get("toast") == "archived" and name in PROJECTS:
            undo = (f'<form method="post" action="{self.url("/projects/undo")}"><input type="hidden" name="name" value="{name}">'
                    "<button>Undo</button></form>")
            note = toast(f"{name} archived", ms=4000, extra=undo)
        return self._page("Projects", f"<h1>Projects</h1><table>{rows}</table>", toast_html=note)

    def archive_project(self, request: Request) -> Reply:
        name = request.get("name")
        if name in PROJECTS and name not in self.archived:
            self.archived.append(name)
            self.archive_events += 1
        return redirect(self.url(f"/projects?toast=archived&p={quote(name)}"))

    def undo_archive(self, request: Request) -> Reply:
        name = request.get("name")
        if name in self.archived:
            self.archived.remove(name)
            self.undo_events += 1
        return redirect(self.url("/projects"))

    # --- search: results that arrive late -----------------------------------------------

    def search_page(self, request: Request) -> Reply:
        query = request.get("q").strip()
        form = (f'<form method="get" action="{self.url("/search")}"><input name="q" size="30" value="{escape(query)}">'
                '<button class="primary">Search</button></form>')
        if self.opened_document:
            return self._page("Document", f"<h1>{escape(self.opened_document)}</h1><p>Opened.</p>")
        if not query:
            return self._page("Find a document", f"<h1>Find a document</h1>{form}")
        hits = [d for d in DOCUMENTS if query.lower() in d.lower()]
        items = "".join(
            f'<li><form method="post" action="{self.url("/search/open")}"><input type="hidden" name="title" value="{escape(d)}">'
            f"<button>{escape(d)}</button></form></li>" for d in hits) or "<li>No documents found.</li>"
        return self._page("Find a document", f"""<h1>Find a document</h1>{form}
            <div id="res-spinner"><div class="spinner"></div><p class="muted">Searching…</p></div>
            <ul hidden data-reveal-ms="2500" data-spinner="res-spinner">{items}</ul>""")

    def open_document(self, request: Request) -> Reply:
        title = request.get("title")
        if title in DOCUMENTS:
            self.opened_document = title
        return redirect(self.url("/search"))


def build_apps() -> list[BaseApp]:
    """The Season 1 site plus the dynamic app (the workspace page stays last)."""
    apps = season1_apps()
    apps.insert(len(apps) - 1, DynamicApp())
    return apps


def main(argv: list[str] | None = None) -> None:
    from evals.safety_tasks import site_pages

    parser = argparse.ArgumentParser(description="Serve the local site with the dynamic-screen app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    site = FixtureSite(site_pages(), args.host, args.port, apps=build_apps())
    print(f"Serving {site.base_url}  (dynamic app at /apps/dynamic/, state at /state)", flush=True)
    try:
        site._server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

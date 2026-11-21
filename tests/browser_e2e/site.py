"""A deterministic local site for the browser end-to-end tests.

A scripted model cannot see, so every page here has a fixed layout: a 1000x800 viewport,
absolutely positioned elements and known pixel boxes. The tests compute where to click from
``BOX`` (the one place that layout is written down), never from the page. The site keeps its
own state (``GET /state``, as the benchmark apps do) so a run is graded on what happened, not
on what the model said. Mounted on ``evals.fixture_site.FixtureSite`` as an app under ``/t``.
"""

from __future__ import annotations

from html import escape

from evals.fixture_site import Reply, Request, redirect

VIEWPORT = (1000, 800)

# name -> (x, y, width, height) in viewport pixels (document pixels on the scroll page).
BOX: dict[str, tuple[int, int, int, int]] = {
    # navigation pages
    "link_products": (100, 200, 300, 60),
    "link_about": (100, 300, 300, 60),
    "link_widget": (100, 200, 300, 60),
    "link_home": (100, 400, 300, 60),
    # form
    "field_name": (100, 150, 500, 50),
    "field_email": (100, 250, 500, 50),
    "form_submit": (100, 350, 240, 70),
    # buttons page: five buttons in a row, then a second row
    "btn_alpha": (50, 450, 160, 80),
    "btn_bravo": (230, 450, 160, 80),
    "btn_charlie": (410, 450, 160, 80),
    "btn_delta": (590, 450, 160, 80),
    "btn_echo": (770, 450, 160, 80),
    # scroll page (document coordinates; the page is 2600px tall)
    "scroll_target": (100, 1500, 400, 90),
    # dropdown and modal page
    "dropdown": (100, 150, 300, 50),
    "option_small": (100, 200, 300, 50),
    "option_medium": (100, 250, 300, 50),
    "option_large": (100, 300, 300, 50),
    "open_modal": (100, 450, 300, 70),
    "modal_confirm": (380, 430, 240, 70),
    "modal_close": (380, 520, 240, 70),
    # confirm dialog and self-closing popup pages
    "confirm_btn": (100, 200, 300, 70),
    "popup_open": (100, 200, 300, 70),
    "popup_after": (100, 320, 300, 70),
    # delayed page
    "delayed_button": (300, 300, 400, 100),
    # shop workflow
    "login_user": (100, 150, 400, 50),
    "login_pass": (100, 250, 400, 50),
    "login_submit": (100, 350, 240, 70),
    "add_widget": (100, 200, 300, 70),
    "go_cart": (100, 320, 300, 70),
    "place_order": (100, 200, 300, 80),
}


def center(name: str, *, scroll_y: int = 0) -> tuple[int, int]:
    x, y, w, h = BOX[name]
    return x + w // 2, y + h // 2 - scroll_y


def grid(px: int, size: int) -> int:
    """A pixel as the model reports it: the 0-999 grid Gemini computer use speaks."""
    return round(px / (size - 1) * 999)


def grid_point(name: str, *, scroll_y: int = 0) -> dict[str, int]:
    x, y = center(name, scroll_y=scroll_y)
    return {"x": grid(x, VIEWPORT[0]), "y": grid(y, VIEWPORT[1])}


def grid_xy(x: int, y: int) -> dict[str, int]:
    return {"x": grid(x, VIEWPORT[0]), "y": grid(y, VIEWPORT[1])}


_CSS = """
html,body{margin:0;padding:0;font-family:Arial,sans-serif;font-size:22px;background:#fff;color:#111}
.abs{position:absolute;box-sizing:border-box}
.btn{position:absolute;box-sizing:border-box;display:flex;align-items:center;justify-content:center;
background:#1a56db;color:#fff;text-decoration:none;border:0;font-size:24px;font-weight:bold;cursor:pointer}
input{position:absolute;box-sizing:border-box;font-size:24px;padding:4px 10px;border:2px solid #333}
h1{position:absolute;left:100px;top:40px;margin:0;font-size:40px}
"""


def _page(title: str, body: str, *, height: int | None = None, script: str = "") -> Reply:
    style = _CSS + (f"body{{height:{height}px}}" if height else "html,body{overflow:hidden}")
    return Reply(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(title)}</title>"
        f"<style>{style}</style></head><body><h1>{escape(title)}</h1>{body}"
        f"<script>{script}</script></body></html>"
    )


def _box_style(name: str) -> str:
    x, y, w, h = BOX[name]
    return f"left:{x}px;top:{y}px;width:{w}px;height:{h}px"


def _link(name: str, href: str, label: str) -> str:
    return f'<a class="btn" style="{_box_style(name)}" href="{href}">{escape(label)}</a>'


def _field(name: str, field: str, kind: str = "text") -> str:
    return f'<input name="{field}" type="{kind}" style="{_box_style(name)}">'


class TestSite:
    """The stateful app. ``state()`` is what the tests grade."""

    __test__ = False  # not a pytest class
    name = "t"
    prefix = "/t"

    def __init__(self) -> None:
        self.reset()

    def reset(self, **seed) -> None:
        self.form: dict[str, str] | None = None
        self.picks: list[str] = []
        self.events: list[str] = []
        self.logged_in: str | None = None
        self.cart: list[str] = []
        self.orders: list[dict] = []

    def state(self) -> dict:
        return {
            "form": self.form, "picks": list(self.picks), "events": list(self.events),
            "logged_in": self.logged_in, "cart": list(self.cart), "orders": list(self.orders),
        }

    # -- routing ------------------------------------------------------------

    def handle(self, request: Request) -> Reply | None:
        path = request.path[len(self.prefix):] or "/"
        if request.method == "GET":
            return self._get(path, request)
        if request.method == "POST":
            return self._post(path, request)
        return None

    def _get(self, path: str, request: Request) -> Reply | None:  # noqa: C901
        if path == "/home":
            return _page("Home", _link("link_products", "/t/products", "Products")
                         + _link("link_about", "/t/about", "About us"))
        if path == "/products":
            return _page("Products", _link("link_widget", "/t/widget", "Widget"))
        if path == "/about":
            return _page("About", _link("link_home", "/t/home", "Back home"))
        if path == "/widget":
            return _page("Widget details", _link("link_home", "/t/home", "Back home"))
        if path == "/form":
            body = (
                '<form method="post" action="/t/form">'
                + _field("field_name", "name") + _field("field_email", "email")
                + f'<button class="btn" type="submit" style="{_box_style("form_submit")}">Send</button></form>'
            )
            return _page("Contact form", body)
        if path == "/thanks":
            return _page("Thank you", "")
        if path == "/buttons":
            labels = {"btn_alpha": "A", "btn_bravo": "B", "btn_charlie": "C", "btn_delta": "D", "btn_echo": "E"}
            body = "".join(_link(box, f"/t/pick?b={label}", label) for box, label in labels.items())
            return _page("Pick one", body)
        if path == "/pick":
            self.picks.append(request.get("b"))
            return _page("Picked " + request.get("b"), "")
        if path == "/scroll":
            return _page("Long page", _link("scroll_target", "/t/pick?b=bottom", "Bottom button"), height=2600)
        if path == "/dialog":
            return self._dialog_page()
        if path == "/confirm":
            return self._confirm_page()
        if path == "/popup":
            return self._popup_page()
        if path == "/popup_child":
            # Opens, tells the site, closes itself (an OAuth or payment window does the same).
            return _page("Popup", "", script="fetch('/t/event?kind=popup&value=opened')"
                         ".then(function(){window.close()})")
        if path == "/event":
            self.events.append(f"{request.get('kind')}:{request.get('value')}")
            return Reply("ok", content_type="text/plain")
        if path == "/delayed":
            body = (
                f'<a id="late" class="btn" style="{_box_style("delayed_button")};display:none" '
                'href="/t/pick?b=delayed">Reveal</a>'
            )
            return _page("Loading", body, script="setTimeout(function(){"
                         "document.getElementById('late').style.display='flex'},2500)")
        if path == "/login":
            body = (
                '<form method="post" action="/t/login">'
                + _field("login_user", "user") + _field("login_pass", "password", "password")
                + f'<button class="btn" type="submit" style="{_box_style("login_submit")}">Sign in</button></form>'
            )
            return _page("Sign in", body)
        if path == "/shop":
            body = _link("add_widget", "/t/add?item=widget", "Add widget") + _link("go_cart", "/t/cart", "View cart")
            return _page(f"Shop ({len(self.cart)} in cart)", body)
        if path == "/add":
            if self.logged_in:
                self.cart.append(request.get("item"))
            return redirect("/t/shop")
        if path == "/cart":
            body = f'<form method="post" action="/t/checkout"><button class="btn" type="submit" ' \
                   f'style="{_box_style("place_order")}">Place order</button></form>'
            return _page(f"Cart: {len(self.cart)} item(s)", body)
        return None

    def _post(self, path: str, request: Request) -> Reply | None:
        if path == "/form":
            self.form = {"name": request.get("name"), "email": request.get("email")}
            return redirect("/t/thanks")
        if path == "/login":
            if request.get("user") == "ada" and request.get("password") == "lovelace":
                self.logged_in = "ada"
                return redirect("/t/shop")
            return redirect("/t/login")
        if path == "/checkout":
            if self.logged_in and self.cart:
                self.orders.append({"user": self.logged_in, "items": list(self.cart)})
                self.cart.clear()
            return redirect("/t/thanks")
        return None

    def _confirm_page(self) -> Reply:
        body = f'<div id="go" class="btn" style="{_box_style("confirm_btn")}">Delete everything</div>'
        script = """
function ev(k,v){fetch('/t/event?kind='+k+'&value='+encodeURIComponent(v))}
document.getElementById('go').onclick=function(){
  if(confirm('Delete everything?')){ev('confirm','accepted');this.textContent='Deleted'}
  else{ev('confirm','cancelled')}};
"""
        return _page("Danger zone", body, script=script)

    def _popup_page(self) -> Reply:
        body = (
            f'<div id="open" class="btn" style="{_box_style("popup_open")}">Sign in with a popup</div>'
            + _link("popup_after", "/t/pick?b=after", "Continue")
        )
        script = "document.getElementById('open').onclick=function(){window.open('/t/popup_child')};"
        return _page("Popup sign-in", body, script=script)

    def _dialog_page(self) -> Reply:
        options = {"option_small": "Small", "option_medium": "Medium", "option_large": "Large"}
        option_html = "".join(
            f'<div class="btn opt" data-v="{label}" style="{_box_style(box)};background:#e5e7eb;color:#111">'
            f"{label}</div>" for box, label in options.items()
        )
        body = (
            f'<div id="dd" class="btn" style="{_box_style("dropdown")};background:#374151">Choose size</div>'
            f'<div id="opts" style="display:none">{option_html}</div>'
            f'<div id="open" class="btn" style="{_box_style("open_modal")}">Open dialog</div>'
            '<div id="overlay" style="position:absolute;left:0;top:380px;width:1000px;height:300px;'
            'background:#f3f4f6;display:none;border-top:4px solid #111">'
            f'<div id="ok" class="btn" style="{_box_style("modal_confirm")};top:50px;left:380px">Confirm</div>'
            f'<div id="no" class="btn" style="{_box_style("modal_close")};top:140px;left:380px;background:#6b7280">Close</div>'
            "</div>"
        )
        script = """
function ev(k,v){fetch('/t/event?kind='+k+'&value='+encodeURIComponent(v))}
var $=function(i){return document.getElementById(i)};
$('dd').onclick=function(){$('opts').style.display='block'};
document.querySelectorAll('.opt').forEach(function(o){o.onclick=function(){
  $('dd').textContent=o.dataset.v;$('opts').style.display='none';ev('select',o.dataset.v)}});
$('open').onclick=function(){$('overlay').style.display='block';ev('modal','open')};
$('ok').onclick=function(){ev('modal','confirm');$('overlay').style.display='none'};
$('no').onclick=function(){ev('modal','close');$('overlay').style.display='none'};
"""
        return _page("Settings", body, script=script)

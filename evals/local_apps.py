"""Stateful fake apps for the benchmark's local site: a shop, a mailbox, a signed-in account and forms.

Public sites can only be graded from a screenshot. These apps keep real state
(a cart, orders, sent mail, a password, a submitted form), so an ``apps`` task
(``evals/app_tasks.py``) is graded on what actually happened. Each app is
mounted under ``/apps/<name>`` on ``evals/fixture_site.py``, next to the
safety pages, and ``GET /state`` returns every app's state as JSON.

They are built like ordinary small sites: forms that POST and redirect, server-side
validation with error messages, a session cookie behind a sign-in page. Every
confirmation names what was done (the item, the recipient, the new setting),
because the agent and its completion check only have the screen to go on.

Run the whole site on its own to look at it in a browser::

    .venv/bin/python -m evals.local_apps [--port 8765]
"""

from __future__ import annotations

import argparse
import copy
import re
import secrets
from collections.abc import Callable
from html import escape
from urllib.parse import quote

from evals.fixture_site import FixtureSite, Reply, Request, redirect
from evals.safety_tasks import STYLE

APP_STYLE = STYLE + """
header nav { display: flex; gap: 18px; margin-left: auto; }
header form { margin: 0; }
header button { background: none; border: none; color: #dbe7ff; font-size: 15px; padding: 0; }
main.wide { max-width: 980px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; }
.notice { background: #ecfdf5; border: 1px solid #a7f3d0; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; }
.error { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; }
table { border-collapse: collapse; width: 100%; }
td, th { text-align: left; padding: 8px 6px; border-bottom: 1px solid #e5e7eb; }
select { font-size: 15px; padding: 6px 8px; }
.side { display: flex; gap: 24px; }
.side > nav { min-width: 150px; display: flex; flex-direction: column; gap: 8px; }
.unread { font-weight: 700; }
fieldset { border: 1px solid #e5e7eb; border-radius: 6px; margin: 12px 0; }
.inline label { display: inline; margin: 0 14px 0 4px; }
"""


def layout(title: str, body: str, *, site: str, nav: str = "", wide: bool = False, notice: str = "", error: str = "") -> Reply:
    banner = (f'<div class="notice">{notice}</div>' if notice else "") + (f'<div class="error">{error}</div>' if error else "")
    return Reply(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(title)}</title>"
        f"<style>{APP_STYLE}</style></head><body><header><strong>{site}</strong><nav>{nav}</nav></header>"
        f"<main{' class=wide' if wide else ''}>{banner}{body}</main></body></html>"
    )


def money(value: float) -> str:
    return f"${value:,.2f}"


def _int(text: str, default: int = 0) -> int:
    try:
        return int(text.strip())
    except ValueError:
        return default


Route = Callable[..., Reply]


class BaseApp:
    """Routes ``(METHOD, regex)`` under ``prefix``; regex groups become keyword arguments."""

    name = ""
    prefix = ""

    def __init__(self) -> None:
        self._routes = [(method, re.compile(pattern + "$"), handler) for method, pattern, handler in self._table()]
        self.reset()

    def _table(self) -> list[tuple[str, str, Route]]:
        raise NotImplementedError

    def reset(self, **seed) -> None:
        raise NotImplementedError

    def state(self) -> dict:
        raise NotImplementedError

    def url(self, path: str = "/") -> str:
        return self.prefix + path

    def handle(self, request: Request) -> Reply | None:
        sub = request.path[len(self.prefix):] or "/"
        for method, pattern, handler in self._routes:
            match = pattern.match(sub)
            if match and method == request.method:
                return handler(request, **match.groupdict())
        return None


# --- shop ----------------------------------------------------------------------

PRODUCTS: dict[str, dict] = {
    "aa-batteries": {"name": "AA batteries, 8-pack", "category": "Electronics", "price": 6.49,
                     "blurb": "Long-life alkaline batteries."},
    "usb-c-cable": {"name": "USB-C cable, 2 m", "category": "Electronics", "price": 9.99,
                    "blurb": "Braided, 60 W charging."},
    "desk-lamp-arc": {"name": "Arc LED desk lamp", "category": "Home office", "price": 39.00,
                      "blurb": "Dimmable, warm to cool white."},
    "desk-lamp-clip": {"name": "Clip-on desk lamp", "category": "Home office", "price": 18.50,
                       "blurb": "Clamps to any shelf or desk edge."},
    "desk-lamp-studio": {"name": "Studio desk lamp", "category": "Home office", "price": 64.00,
                         "blurb": "Weighted base, brass finish."},
    "notebook-a5": {"name": "A5 dotted notebook", "category": "Stationery", "price": 7.25,
                    "blurb": "120 pages, lay-flat binding."},
    "gel-pens": {"name": "Gel pens, 10 colours", "category": "Stationery", "price": 8.90,
                 "blurb": "0.7 mm, quick-drying ink."},
    "ceramic-mug": {"name": "Ceramic mug, 350 ml", "category": "Kitchen", "price": 12.00,
                    "blurb": "Stoneware, dishwasher safe."},
    "fleece-hoodie": {"name": "Fleece hoodie", "category": "Clothing", "price": 45.00, "sizes": ("S", "M", "L", "XL"),
                      "blurb": "Grey, brushed fleece."},
    "wool-socks": {"name": "Wool socks", "category": "Clothing", "price": 14.00, "sizes": ("S", "M", "L"),
                   "blurb": "Merino blend, one pair."},
}
COUPONS = {"SAVE10": 0.10}
SHIPPING = {"standard": ("Standard (3-5 days)", 4.99), "express": ("Express (next day)", 14.99)}
CATEGORIES = sorted({p["category"] for p in PRODUCTS.values()})


class ShopApp(BaseApp):
    name = "shop"
    prefix = "/apps/shop"
    site = "Harbor Goods"

    def _table(self):
        return [
            ("GET", r"/", self.catalog),
            ("GET", r"/product/(?P<sku>[\w-]+)", self.product),
            ("POST", r"/cart/add", self.add),
            ("GET", r"/cart", self.cart_page),
            ("POST", r"/cart/update", self.update),
            ("POST", r"/cart/remove", self.remove),
            ("POST", r"/cart/coupon", self.coupon_code),
            ("GET", r"/checkout", self.checkout_page),
            ("POST", r"/checkout", self.place_order),
            ("GET", r"/order/(?P<number>\d+)", self.order_page),
        ]

    def reset(self, cart: list[dict] | None = None, **_) -> None:
        self.cart: list[dict] = [{"sku": line["sku"], "qty": line.get("qty", 1), "size": line.get("size")} for line in cart or []]
        self.coupon: str | None = None
        self.orders: list[dict] = []

    def state(self) -> dict:
        return {"cart": copy.deepcopy(self.cart), "coupon": self.coupon, "orders": copy.deepcopy(self.orders)}

    def _page(self, title: str, body: str, **kw) -> Reply:
        count = sum(line["qty"] for line in self.cart)
        nav = f'<a href="{self.url()}">Shop</a><a href="{self.url("/cart")}">Cart ({count})</a>'
        return layout(title, body, site=self.site, nav=nav, wide=True, **kw)

    def _totals(self) -> tuple[float, float]:
        subtotal = sum(PRODUCTS[line["sku"]]["price"] * line["qty"] for line in self.cart)
        return round(subtotal, 2), round(subtotal * COUPONS.get(self.coupon or "", 0.0), 2)

    def catalog(self, request: Request) -> Reply:
        query, category = request.get("q").strip(), request.get("category")
        found = [
            (sku, p) for sku, p in PRODUCTS.items()
            if (not category or p["category"] == category) and (not query or query.lower() in p["name"].lower())
        ]
        links = " · ".join(f'<a href="{self.url()}?category={quote(c)}">{c}</a>' for c in CATEGORIES)
        cards = "".join(f"""
            <div class="card"><a href="{self.url('/product/' + sku)}"><strong>{p['name']}</strong></a>
              <p class="muted">{p['category']}</p><p>{money(p['price'])}</p></div>""" for sku, p in found)
        heading = f"Results for “{escape(query)}”" if query else (escape(category) if category else "All products")
        return self._page("Harbor Goods", f"""
            <form method="get" action="{self.url()}" class="row"><input name="q" value="{escape(query)}" size="40"
              placeholder="Search products"><button class="primary">Search</button></form>
            <p>Categories: <a href="{self.url()}">All</a> · {links}</p>
            <h1>{heading}</h1><div class="grid">{cards or '<p>No products found.</p>'}</div>""")

    def product(self, request: Request, sku: str) -> Reply:
        p = PRODUCTS.get(sku)
        if not p:
            return self._page("Not found", "<h1>Product not found</h1>")
        sizes = ""
        if p.get("sizes"):
            options = "".join(f"<option>{s}</option>" for s in p["sizes"])
            sizes = f'<label>Size</label><select name="size"><option value="">Choose a size</option>{options}</select>'
        return self._page(p["name"], f"""
            <p><a href="{self.url()}?category={quote(p['category'])}">{p['category']}</a></p>
            <h1>{p['name']}</h1><p>{p['blurb']}</p><p><strong>{money(p['price'])}</strong></p>
            <form method="post" action="{self.url('/cart/add')}"><input type="hidden" name="sku" value="{sku}">
              {sizes}<label>Quantity</label><input type="number" name="qty" value="1" min="1" max="20">
              <p><button class="primary">Add to cart</button></p></form>""", error=escape(request.get("error")))

    def add(self, request: Request) -> Reply:
        sku, size = request.get("sku"), request.get("size") or None
        p = PRODUCTS.get(sku)
        if not p:
            return redirect(self.url())
        if p.get("sizes") and size not in p["sizes"]:
            return redirect(self.url(f"/product/{sku}?error=" + quote("Please choose a size.")))
        qty = max(1, min(20, _int(request.get("qty"), 1)))
        for line in self.cart:
            if line["sku"] == sku and line["size"] == size:
                line["qty"] += qty
                break
        else:
            self.cart.append({"sku": sku, "qty": qty, "size": size})
        label = p["name"] + (f" (size {size})" if size else "")
        return redirect(self.url("/cart?added=" + quote(f"{qty} × {label}")))

    def _line_label(self, line: dict) -> str:
        return PRODUCTS[line["sku"]]["name"] + (f", size {line['size']}" if line["size"] else "")

    def cart_page(self, request: Request) -> Reply:
        notice = ""
        if request.get("added"):
            notice = f"Added to your cart: <strong>{escape(request.get('added'))}</strong>"
        elif request.get("done"):
            notice = escape(request.get("done"))
        if not self.cart:
            return self._page("Cart", f'<h1>Your cart</h1><p>Your cart is empty. <a href="{self.url()}">Keep shopping</a></p>',
                              notice=notice)
        rows = "".join(f"""
            <tr><td>{self._line_label(line)}</td><td>{money(PRODUCTS[line['sku']]['price'])}</td>
              <td><form method="post" action="{self.url('/cart/update')}" class="row" style="justify-content:start">
                <input type="hidden" name="line" value="{i}"><input type="number" name="qty" value="{line['qty']}"
                min="0" max="20" style="width:64px" aria-label="Quantity of {self._line_label(line)}"><button>Update</button></form></td>
              <td>{money(PRODUCTS[line['sku']]['price'] * line['qty'])}</td>
              <td><form method="post" action="{self.url('/cart/remove')}"><input type="hidden" name="line" value="{i}">
                <button>Remove</button></form></td></tr>""" for i, line in enumerate(self.cart))
        subtotal, discount = self._totals()
        coupon = (f"<p>Coupon <strong>{self.coupon}</strong> applied: −{money(discount)}</p>" if self.coupon else "")
        return self._page("Cart", f"""
            <h1>Your cart</h1>
            <table><tr><th>Item</th><th>Price</th><th>Quantity</th><th>Total</th><th></th></tr>{rows}</table>
            <form method="post" action="{self.url('/cart/coupon')}" class="row" style="justify-content:start; margin-top:16px">
              <input name="code" placeholder="Coupon code"><button>Apply</button></form>
            <p>Subtotal: {money(subtotal)}</p>{coupon}<p><strong>Total before shipping: {money(subtotal - discount)}</strong></p>
            <a class="button primary" style="background:#2563eb;color:#fff" href="{self.url('/checkout')}">Go to checkout</a>""",
            notice=notice, error=escape(request.get("error")))

    def _line(self, request: Request) -> int | None:
        index = _int(request.get("line"), -1)
        return index if 0 <= index < len(self.cart) else None

    def update(self, request: Request) -> Reply:
        index = self._line(request)
        if index is None:
            return redirect(self.url("/cart"))
        label, qty = self._line_label(self.cart[index]), max(0, min(20, _int(request.get("qty"), 1)))
        if qty == 0:
            del self.cart[index]
            return redirect(self.url("/cart?done=" + quote(f"Removed {label}.")))
        self.cart[index]["qty"] = qty
        return redirect(self.url("/cart?done=" + quote(f"{label}: quantity is now {qty}.")))

    def remove(self, request: Request) -> Reply:
        index = self._line(request)
        if index is None:
            return redirect(self.url("/cart"))
        label = self._line_label(self.cart.pop(index))
        return redirect(self.url("/cart?done=" + quote(f"Removed {label}.")))

    def coupon_code(self, request: Request) -> Reply:
        code = request.get("code").strip().upper()
        if code not in COUPONS:
            return redirect(self.url("/cart?error=" + quote(f"The code {code or '(empty)'} is not valid.")))
        self.coupon = code
        return redirect(self.url("/cart?done=" + quote(f"Coupon {code} applied: {COUPONS[code]:.0%} off.")))

    def checkout_page(self, request: Request, error: str = "", values: Request | None = None) -> Reply:
        if not self.cart:
            return self._page("Checkout", f'<h1>Checkout</h1><p>Your cart is empty. <a href="{self.url()}">Keep shopping</a></p>')
        values = values or request
        subtotal, discount = self._totals()
        items = "".join(f"<li>{line['qty']} × {self._line_label(line)}</li>" for line in self.cart)
        fields = "".join(
            f'<label>{label}</label><input name="{key}" value="{escape(values.get(key))}" size="40">'
            for key, label in (("name", "Full name"), ("street", "Street address"), ("city", "City"), ("zip", "ZIP code"))
        )
        chosen = values.get("shipping") or "standard"
        shipping = "".join(
            f'<div class="inline"><input type="radio" id="ship-{key}" name="shipping" value="{key}"'
            f'{" checked" if key == chosen else ""}><label for="ship-{key}">{label} · {money(cost)}</label></div>'
            for key, (label, cost) in SHIPPING.items()
        )
        coupon = f"<p>Coupon {self.coupon}: −{money(discount)}</p>" if self.coupon else ""
        return self._page("Checkout", f"""
            <h1>Checkout</h1><div class="card"><strong>Order summary</strong><ul>{items}</ul>
              <p>Subtotal {money(subtotal)}</p>{coupon}</div>
            <form method="post" action="{self.url('/checkout')}">{fields}
              <fieldset><legend>Shipping</legend>{shipping}</fieldset>
              <button class="primary">Place order</button></form>""", error=error)

    def place_order(self, request: Request) -> Reply:
        if not self.cart:
            return redirect(self.url("/checkout"))
        missing = [label for key, label in (("name", "full name"), ("street", "street address"), ("city", "city"),
                                            ("zip", "ZIP code")) if not request.get(key).strip()]
        if missing:
            return self.checkout_page(request, error="Please fill in: " + ", ".join(missing) + ".", values=request)
        method = request.get("shipping") if request.get("shipping") in SHIPPING else "standard"
        subtotal, discount = self._totals()
        number = 10400 + len(self.orders) + 1
        self.orders.append({
            "number": number,
            "items": copy.deepcopy(self.cart),
            "coupon": self.coupon,
            "ship_to": {key: request.get(key).strip() for key in ("name", "street", "city", "zip")},
            "shipping": method,
            "total": round(subtotal - discount + SHIPPING[method][1], 2),
        })
        self.cart, self.coupon = [], None
        return redirect(self.url(f"/order/{number}"))

    def order_page(self, request: Request, number: str) -> Reply:
        order = next((o for o in self.orders if o["number"] == int(number)), None)
        if not order:
            return self._page("Not found", "<h1>Order not found</h1>")
        items = "".join(f"<li>{line['qty']} × {self._line_label(line)}</li>" for line in order["items"])
        to = order["ship_to"]
        return self._page(f"Order {number}", f"""
            <h1>Thank you! Order #{number} is placed.</h1><ul>{items}</ul>
            {f"<p>Coupon: {order['coupon']}</p>" if order['coupon'] else ""}
            <p>Ship to: {escape(to['name'])}, {escape(to['street'])}, {escape(to['city'])} {escape(to['zip'])}</p>
            <p>Shipping: {SHIPPING[order['shipping']][0]}</p><p><strong>Total charged: {money(order['total'])}</strong></p>""")


# --- mail ----------------------------------------------------------------------

ME = "jordan.lee@northwind.example"
MESSAGES: dict[str, dict] = {
    "dana": {"from": "Dana Whitfield", "address": "dana.whitfield@northwind.example", "date": "Oct 25",
             "subject": "Agenda for Monday's planning",
             "body": "Hi Jordan,\n\nI've put the Q4 roadmap first on Monday's agenda. Can you bring the hiring numbers?\n\nDana"},
    "weekly-byte": {"from": "The Weekly Byte", "address": "news@weeklybyte.example", "date": "Oct 25",
                    "subject": "Issue 112: the year in chips",
                    "body": "This week: chip prices, a new phone and five tools we liked.\n\nYou are receiving this "
                            "newsletter because you subscribed. Unsubscribe any time."},
    "acme": {"from": "Acme Supplies", "address": "billing@acme-supplies.example", "date": "Oct 24",
             "subject": "Your October invoice",
             "body": "Hello,\n\nInvoice INV-20931 for $1,240.00 is attached. Payment is due on November 15.\n\n"
                     "Acme Supplies billing"},
    "garden": {"from": "Garden Club Digest", "address": "digest@gardenclub.example", "date": "Oct 23",
               "subject": "Bulbs to plant before the frost",
               "body": "Our monthly picks for autumn planting.\n\nGarden Club Digest is a monthly newsletter. "
                       "Unsubscribe from this list."},
    "sam": {"from": "Sam Ortiz", "address": "sam.ortiz@northwind.example", "date": "Oct 23",
            "subject": "Lunch Friday?", "body": "Want to try the new noodle place on Friday?\n\nSam"},
    "travel": {"from": "Wanderly Deals", "address": "deals@wanderly.example", "date": "Oct 22",
               "subject": "Flash sale: 30% off city breaks",
               "body": "48 hours only: city breaks from $199.\n\nYou are subscribed to the Wanderly newsletter. "
                       "Unsubscribe."},
    "priya": {"from": "Priya Nair", "address": "priya.nair@northwind.example", "date": "Oct 21",
              "subject": "Contract draft v2", "body": "Jordan, the second draft is ready for your review.\n\nPriya"},
}
FOLDERS = ("inbox", "archive", "sent", "trash")


class MailApp(BaseApp):
    name = "mail"
    prefix = "/apps/mail"
    site = "Northwind Mail"

    def _table(self):
        return [
            ("GET", r"/", self.folder),
            ("GET", r"/message/(?P<mid>[\w-]+)", self.message),
            ("POST", r"/message/(?P<mid>[\w-]+)/move", self.move),
            ("POST", r"/message/(?P<mid>[\w-]+)/star", self.star),
            ("POST", r"/message/(?P<mid>[\w-]+)/unread", self.unread),
            ("GET", r"/compose", self.compose),
            ("POST", r"/send", self.send),
            ("GET", r"/sent/(?P<index>\d+)", self.sent_message),
        ]

    def reset(self, **_) -> None:
        self.messages = {mid: {"folder": "inbox", "read": False, "starred": False} for mid in MESSAGES}
        self.sent: list[dict] = []

    def state(self) -> dict:
        return {"messages": copy.deepcopy(self.messages), "sent": copy.deepcopy(self.sent)}

    def _page(self, title: str, body: str, *, current: str = "", **kw) -> Reply:
        unread = sum(1 for m in self.messages.values() if m["folder"] == "inbox" and not m["read"])
        links = "".join(
            f'<a href="{self.url()}?folder={f}">{"<strong>" if f == current else ""}{f.capitalize()}'
            f'{f" ({unread})" if f == "inbox" and unread else ""}{"</strong>" if f == current else ""}</a>'
            for f in FOLDERS
        )
        starred = f'<a href="{self.url()}?folder=starred">{"<strong>Starred</strong>" if current == "starred" else "Starred"}</a>'
        side = (f'<nav><a class="button primary" style="background:#2563eb;color:#fff;text-align:center" '
                f'href="{self.url("/compose")}">Compose</a>{links}{starred}</nav>')
        return layout(title, f'<div class="side">{side}<div style="flex:1">{body}</div></div>', site=self.site,
                      nav=f"<span>{ME}</span>", wide=True, **kw)

    def folder(self, request: Request) -> Reply:
        name = request.get("folder") or "inbox"
        if name == "sent":
            rows = "".join(f"""
                <tr><td>To: {escape(s['to'])}</td><td><a href="{self.url(f'/sent/{i}')}">{escape(s['subject']) or '(no subject)'}</a></td></tr>"""
                for i, s in reversed(list(enumerate(self.sent))))
        else:
            ids = [mid for mid, m in self.messages.items() if (m["starred"] if name == "starred" else m["folder"] == name)]
            rows = "".join(f"""
                <tr class="{'' if self.messages[mid]['read'] else 'unread'}">
                  <td><form method="post" action="{self.url(f'/message/{mid}/star')}"><input type="hidden" name="back" value="{name}">
                    <button title="{'Unstar' if self.messages[mid]['starred'] else 'Star'}" aria-label="{'Unstar' if self.messages[mid]['starred'] else 'Star'} {MESSAGES[mid]['from']}"
                    style="border:none;font-size:18px;padding:0;color:#d97706">{'★' if self.messages[mid]['starred'] else '☆'}</button></form></td>
                  <td>{MESSAGES[mid]['from']}</td>
                  <td><a href="{self.url(f'/message/{mid}')}">{MESSAGES[mid]['subject']}</a></td>
                  <td class="muted">{MESSAGES[mid]['date']}</td></tr>""" for mid in ids)
        title = name.capitalize()
        return self._page(title, f"<h1>{title}</h1><table>{rows or '<tr><td>No messages.</td></tr>'}</table>",
                          current=name, notice=escape(request.get("done")))

    def message(self, request: Request, mid: str) -> Reply:
        if mid not in MESSAGES:
            return self._page("Not found", "<h1>Message not found</h1>")
        m, box = MESSAGES[mid], self.messages[mid]
        box["read"] = True
        body = escape(m["body"]).replace("\n", "<br>")

        def action(path: str, label: str, **fields: str) -> str:
            hidden = "".join(f'<input type="hidden" name="{k}" value="{v}">' for k, v in fields.items())
            return f'<form method="post" action="{self.url(f"/message/{mid}/{path}")}">{hidden}<button>{label}</button></form>'

        buttons = [
            f'<a class="button" href="{self.url("/compose")}?reply={mid}">Reply</a>',
            f'<a class="button" href="{self.url("/compose")}?forward={mid}">Forward</a>',
            action("move", "Archive", folder="archive") if box["folder"] != "archive" else action("move", "Move to inbox", folder="inbox"),
            action("move", "Delete", folder="trash") if box["folder"] != "trash" else "",
            action("star", "Unstar" if box["starred"] else "Star"),
            action("unread", "Mark as unread"),
        ]
        return self._page(m["subject"], f"""
            <div class="row" style="justify-content:start; flex-wrap:wrap">{''.join(buttons)}</div>
            <h1>{m['subject']}</h1><p><strong>{m['from']}</strong> &lt;{m['address']}&gt; · {m['date']}
              {' · <span style="color:#d97706">★ Starred</span>' if box['starred'] else ''}
              · <span class="muted">in {box['folder'].capitalize()}</span></p>
            <div class="card">{body}</div>""", current=box["folder"], notice=escape(request.get("done")))

    def move(self, request: Request, mid: str) -> Reply:
        folder = request.get("folder")
        if mid not in MESSAGES or folder not in ("inbox", "archive", "trash"):
            return redirect(self.url())
        self.messages[mid]["folder"] = folder
        what = {"archive": "archived", "trash": "moved to Trash", "inbox": "moved to the inbox"}[folder]
        done = f"“{MESSAGES[mid]['subject']}” from {MESSAGES[mid]['from']} was {what}."
        return redirect(self.url("/?folder=inbox&done=" + quote(done)))

    def star(self, request: Request, mid: str) -> Reply:
        if mid not in MESSAGES:
            return redirect(self.url())
        box = self.messages[mid]
        box["starred"] = not box["starred"]
        done = f"{'Starred' if box['starred'] else 'Unstarred'} “{MESSAGES[mid]['subject']}” from {MESSAGES[mid]['from']}."
        back = request.get("back")
        target = f"/?folder={back}&done=" if back in (*FOLDERS, "starred") else f"/message/{mid}?done="
        return redirect(self.url(target + quote(done)))

    def unread(self, request: Request, mid: str) -> Reply:
        if mid in MESSAGES:
            self.messages[mid]["read"] = False
        return redirect(self.url("/?folder=inbox&done=" + quote("Marked as unread.")))

    def compose(self, request: Request, error: str = "", values: Request | None = None) -> Reply:
        to = subject = ""
        original = ""
        reply, forward = request.get("reply"), request.get("forward")
        source = MESSAGES.get(reply or forward)
        if source:
            subject = ("Re: " if reply else "Fwd: ") + source["subject"]
            to = source["address"] if reply else ""
            original = (f'<div class="card muted"><p>On {source["date"]}, {source["from"]} wrote:</p>'
                        f'{escape(source["body"]).replace(chr(10), "<br>")}</div>')
        if values is not None:
            to, subject = values.get("to"), values.get("subject")
        body = values.get("body") if values is not None else ""
        return self._page("Compose", f"""
            <h1>{'Reply' if reply else 'Forward' if forward else 'New message'}</h1>
            <form method="post" action="{self.url('/send')}">
              <input type="hidden" name="reply" value="{escape(reply)}"><input type="hidden" name="forward" value="{escape(forward)}">
              <label>To</label><input name="to" value="{escape(to)}" size="50">
              <label>Cc</label><input name="cc" value="{escape(values.get('cc') if values is not None else '')}" size="50">
              <label>Subject</label><input name="subject" value="{escape(subject)}" size="50">
              <label>Message</label><textarea name="body" rows="7" cols="70">{escape(body)}</textarea>
              <p><button class="primary">Send</button></p></form>{original}""", error=error)

    def send(self, request: Request) -> Reply:
        to = request.get("to").strip()
        if not re.fullmatch(r"[^@\s,]+@[^@\s,]+\.[^@\s,]+", to):
            return self.compose(request, error="Enter one valid email address in To.", values=request)
        self.sent.append({
            "to": to,
            "cc": request.get("cc").strip(),
            "subject": request.get("subject").strip(),
            "body": request.get("body").strip(),
            "in_reply_to": request.get("reply") if request.get("reply") in MESSAGES else None,
            "forward_of": request.get("forward") if request.get("forward") in MESSAGES else None,
        })
        return redirect(self.url(f"/sent/{len(self.sent) - 1}?done=" + quote(f"Message sent to {to}.")))

    def sent_message(self, request: Request, index: str) -> Reply:
        i = int(index)
        if i >= len(self.sent):
            return self._page("Not found", "<h1>Message not found</h1>")
        s = self.sent[i]
        cc = f"<p>Cc: {escape(s['cc'])}</p>" if s["cc"] else ""
        return self._page(s["subject"] or "(no subject)", f"""
            <h1>{escape(s['subject']) or '(no subject)'}</h1><p>From: {ME}</p><p>To: {escape(s['to'])}</p>{cc}
            <div class="card">{escape(s['body']).replace(chr(10), '<br>')}</div>""",
            current="sent", notice=escape(request.get("done")))


# --- account (sign-in) ---------------------------------------------------------

USERNAME, PASSWORD = "jordan", "orchid-42"
TIMEZONES = ("America/Los_Angeles", "America/New_York", "Europe/London", "Europe/Berlin", "Asia/Kolkata", "Asia/Tokyo")
LANGUAGES = ("English", "Deutsch", "Español", "Français")
NOTIFICATIONS = {"product_updates": "Product updates", "weekly_summary": "Weekly summary email",
                 "security_alerts": "Security alerts"}


class AccountApp(BaseApp):
    name = "account"
    prefix = "/apps/account"
    site = "Northwind Account"

    def _table(self):
        return [
            ("GET", r"/", self.home),
            ("GET", r"/login", self.login_page),
            ("POST", r"/login", self.login),
            ("POST", r"/logout", self.logout),
            ("GET", r"/profile", self.profile_page),
            ("POST", r"/profile", self.save_profile),
            ("GET", r"/security", self.security_page),
            ("POST", r"/security", self.change_password),
        ]

    def reset(self, **_) -> None:
        self.password = PASSWORD
        self.sessions: set[str] = set()
        self.failed_logins = 0
        self.password_changes = 0
        self.profile = {"display_name": "Jordan Lee", "email": ME, "timezone": "America/New_York", "language": "English",
                        "notifications": {key: True for key in NOTIFICATIONS}}

    def state(self) -> dict:
        return {"password": self.password, "signed_in_sessions": len(self.sessions), "failed_logins": self.failed_logins,
                "password_changes": self.password_changes, "profile": copy.deepcopy(self.profile)}

    def _session(self, request: Request) -> str | None:
        token = request.cookies.get("session")
        return token if token in self.sessions else None

    def _page(self, title: str, body: str, *, signed_in: bool, **kw) -> Reply:
        nav = ""
        if signed_in:
            nav = (f'<a href="{self.url("/profile")}">Profile</a><a href="{self.url("/security")}">Security</a>'
                   f'<form method="post" action="{self.url("/logout")}"><button>Sign out</button></form>')
        return layout(title, body, site=self.site, nav=nav, **kw)

    def home(self, request: Request) -> Reply:
        return redirect(self.url("/profile" if self._session(request) else "/login"))

    def login_page(self, request: Request, error: str = "") -> Reply:
        notice = "You have signed out." if request.get("signed_out") else ""
        if request.get("next") and not error:
            notice = "Please sign in to continue."
        return self._page("Sign in", f"""
            <h1>Sign in</h1>
            <form method="post" action="{self.url('/login')}"><label>Username</label><input name="username" size="30"
              value="{escape(request.get('username')) if error else ''}" autocomplete="off">
              <label>Password</label><input type="password" name="password" size="30">
              <p><button class="primary">Sign in</button></p></form>
            <p class="muted">Forgot your password? Ask your administrator.</p>""",
            signed_in=False, notice=notice, error=error)

    def login(self, request: Request) -> Reply:
        if request.get("username").strip().lower() != USERNAME or request.get("password") != self.password:
            self.failed_logins += 1
            return self.login_page(request, error="Wrong username or password.")
        self.sessions.discard(request.cookies.get("session", ""))  # signing in again replaces the old session
        token = secrets.token_hex(8)
        self.sessions.add(token)
        cookie = ("Set-Cookie", f"session={token}; Path={self.prefix}; HttpOnly; SameSite=Lax")
        return redirect(self.url("/profile?welcome=1"), cookie)

    def logout(self, request: Request) -> Reply:
        self.sessions.discard(request.cookies.get("session", ""))
        cookie = ("Set-Cookie", f"session=; Path={self.prefix}; Max-Age=0")
        return redirect(self.url("/login?signed_out=1"), cookie)

    def profile_page(self, request: Request, error: str = "") -> Reply:
        if not self._session(request):
            return redirect(self.url("/login?next=1"))
        p = self.profile
        notice = "Welcome back, Jordan." if request.get("welcome") else escape(request.get("done"))
        tz = "".join(f"<option{' selected' if t == p['timezone'] else ''}>{t}</option>" for t in TIMEZONES)
        lang = "".join(f"<option{' selected' if lang == p['language'] else ''}>{lang}</option>" for lang in LANGUAGES)
        boxes = "".join(
            f'<div><input type="checkbox" id="n-{key}" name="notify" value="{key}"{" checked" if p["notifications"][key] else ""}>'
            f' <label for="n-{key}" style="display:inline">{label}</label></div>' for key, label in NOTIFICATIONS.items()
        )
        return self._page("Profile", f"""
            <h1>Profile</h1>
            <form method="post" action="{self.url('/profile')}">
              <label>Display name</label><input name="display_name" value="{escape(p['display_name'])}" size="36">
              <label>Email</label><input name="email" value="{escape(p['email'])}" size="36">
              <label>Time zone</label><select name="timezone">{tz}</select>
              <label>Language</label><select name="language">{lang}</select>
              <fieldset><legend>Email notifications</legend>{boxes}</fieldset>
              <button class="primary">Save changes</button></form>""", signed_in=True, notice=notice, error=error)

    def save_profile(self, request: Request) -> Reply:
        if not self._session(request):
            return redirect(self.url("/login?next=1"))
        name, email = request.get("display_name").strip(), request.get("email").strip()
        if not name or "@" not in email:
            return self.profile_page(request, error="Display name and a valid email are required.")
        chosen = set(request.getall("notify"))
        self.profile = {
            "display_name": name,
            "email": email,
            "timezone": request.get("timezone") if request.get("timezone") in TIMEZONES else self.profile["timezone"],
            "language": request.get("language") if request.get("language") in LANGUAGES else self.profile["language"],
            "notifications": {key: key in chosen for key in NOTIFICATIONS},
        }
        on = ", ".join(NOTIFICATIONS[k] for k, v in self.profile["notifications"].items() if v) or "none"
        done = f"Saved. Time zone: {self.profile['timezone']}. Language: {self.profile['language']}. Notifications on: {on}."
        return redirect(self.url("/profile?done=" + quote(done)))

    def security_page(self, request: Request, error: str = "") -> Reply:
        if not self._session(request):
            return redirect(self.url("/login?next=1"))
        return self._page("Security", f"""
            <h1>Change password</h1>
            <form method="post" action="{self.url('/security')}">
              <label>Current password</label><input type="password" name="current" size="30">
              <label>New password</label><input type="password" name="new" size="30">
              <label>Confirm new password</label><input type="password" name="confirm" size="30">
              <p class="muted">At least 8 characters.</p><button class="primary">Change password</button></form>""",
            signed_in=True, notice=escape(request.get("done")), error=error)

    def change_password(self, request: Request) -> Reply:
        if not self._session(request):
            return redirect(self.url("/login?next=1"))
        current, new, confirm = request.get("current"), request.get("new"), request.get("confirm")
        if current != self.password:
            return self.security_page(request, error="Your current password is not correct.")
        if len(new) < 8:
            return self.security_page(request, error="The new password must have at least 8 characters.")
        if new != confirm:
            return self.security_page(request, error="The two new passwords do not match.")
        self.password = new
        self.password_changes += 1
        return redirect(self.url("/security?done=" + quote("Your password was changed.")))


# --- forms ---------------------------------------------------------------------

TICKETS = {"standard": "Standard · $299", "student": "Student · $99", "virtual": "Virtual · $49"}
WORKSHOPS = {"rust": "Intro to Rust", "a11y": "Accessibility basics", "dataviz": "Data visualisation",
             "k8s": "Kubernetes in practice"}
MEALS = ("No preference", "Vegetarian", "Vegan", "Gluten-free")
PRODUCT_LINES = ("Router X2", "Router X3", "Mesh Point", "Range Extender")
PRIORITIES = ("Low", "Normal", "High", "Urgent")


class FormsApp(BaseApp):
    name = "forms"
    prefix = "/apps/forms"
    site = "DevConf 2026"

    def _table(self):
        return [
            ("GET", r"/", self.index),
            ("GET", r"/register", self.step_one),
            ("POST", r"/register", self.save_one),
            ("GET", r"/register/options", self.step_two),
            ("POST", r"/register/options", self.save_two),
            ("GET", r"/register/review", self.review),
            ("POST", r"/register/submit", self.submit),
            ("GET", r"/register/done/(?P<number>\d+)", self.registered),
            ("GET", r"/support", self.support_page),
            ("POST", r"/support", self.file_ticket),
            ("GET", r"/support/done/(?P<number>\d+)", self.ticket_filed),
        ]

    def reset(self, **_) -> None:
        self.draft: dict = {}
        self.registrations: list[dict] = []
        self.tickets: list[dict] = []

    def state(self) -> dict:
        return {"draft": copy.deepcopy(self.draft), "registrations": copy.deepcopy(self.registrations),
                "tickets": copy.deepcopy(self.tickets)}

    def _page(self, title: str, body: str, *, site: str | None = None, **kw) -> Reply:
        return layout(title, body, site=site or self.site, **kw)

    def index(self, request: Request) -> Reply:
        return self._page("DevConf 2026", f"""
            <h1>DevConf 2026</h1><p>14-16 November · Lisbon</p>
            <p><a class="button primary" style="background:#2563eb;color:#fff" href="{self.url('/register')}">Register</a></p>""")

    def _steps(self, current: int) -> str:
        names = ("1. Your details", "2. Ticket and options", "3. Review")
        return '<p class="muted">' + " → ".join(f"<strong>{n}</strong>" if i == current else n for i, n in enumerate(names, 1)) + "</p>"

    def step_one(self, request: Request, error: str = "", values: Request | None = None) -> Reply:
        def value(key: str) -> str:
            return escape(values.get(key) if values is not None else self.draft.get(key, ""))

        fields = "".join(
            f'<label>{label}</label><input name="{key}" value="{value(key)}" size="36">'
            for key, label in (("first_name", "First name"), ("last_name", "Last name"), ("email", "Email"),
                               ("company", "Company (optional)"))
        )
        return self._page("Register", f"""<h1>Register</h1>{self._steps(1)}
            <form method="post" action="{self.url('/register')}">{fields}<p><button class="primary">Continue</button></p></form>""",
            error=error)

    def save_one(self, request: Request) -> Reply:
        values = {key: request.get(key).strip() for key in ("first_name", "last_name", "email", "company")}
        if not values["first_name"] or not values["last_name"]:
            return self.step_one(request, error="First and last name are required.", values=request)
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", values["email"]):
            return self.step_one(request, error="Enter a valid email address.", values=request)
        self.draft.update(values)
        return redirect(self.url("/register/options"))

    def step_two(self, request: Request, error: str = "") -> Reply:
        if "email" not in self.draft:
            return redirect(self.url("/register"))
        d = self.draft
        tickets = "".join(
            f'<div class="inline"><input type="radio" id="t-{k}" name="ticket" value="{k}"{" checked" if d.get("ticket") == k else ""}>'
            f'<label for="t-{k}">{label}</label></div>' for k, label in TICKETS.items()
        )
        workshops = "".join(
            f'<div class="inline"><input type="checkbox" id="w-{k}" name="workshop" value="{k}"'
            f'{" checked" if k in d.get("workshops", []) else ""}><label for="w-{k}">{label}</label></div>'
            for k, label in WORKSHOPS.items()
        )
        meals = "".join(f"<option{' selected' if m == d.get('meal') else ''}>{m}</option>" for m in MEALS)
        return self._page("Ticket and options", f"""<h1>Ticket and options</h1>{self._steps(2)}
            <form method="post" action="{self.url('/register/options')}">
              <fieldset><legend>Ticket</legend>{tickets}</fieldset>
              <fieldset><legend>Workshops (optional)</legend>{workshops}</fieldset>
              <label>Meals</label><select name="meal">{meals}</select>
              <label>Arrival date</label><input type="date" name="arrival" value="{escape(d.get('arrival', ''))}"
                min="2026-11-10" max="2026-11-16">
              <p><a href="{self.url('/register')}">Back</a> <button class="primary">Continue</button></p></form>""", error=error)

    def save_two(self, request: Request) -> Reply:
        if "email" not in self.draft:
            return redirect(self.url("/register"))
        ticket, arrival = request.get("ticket"), request.get("arrival").strip()
        self.draft.update({
            "ticket": ticket if ticket in TICKETS else self.draft.get("ticket"),
            "workshops": [k for k in WORKSHOPS if k in request.getall("workshop")],
            "meal": request.get("meal") if request.get("meal") in MEALS else "No preference",
            "arrival": arrival,
        })
        if ticket not in TICKETS:
            return self.step_two(request, error="Choose a ticket.")
        if not re.fullmatch(r"2026-11-1[0-6]", arrival):
            return self.step_two(request, error="Choose an arrival date between 10 and 16 November 2026.")
        return redirect(self.url("/register/review"))

    def _summary(self, d: dict) -> str:
        workshops = ", ".join(WORKSHOPS[k] for k in d.get("workshops", [])) or "none"
        rows = (("Name", f"{d['first_name']} {d['last_name']}"), ("Email", d["email"]), ("Company", d.get("company") or "-"),
                ("Ticket", TICKETS[d["ticket"]]), ("Workshops", workshops), ("Meals", d["meal"]), ("Arrival", d["arrival"]))
        return "<table>" + "".join(f"<tr><th>{k}</th><td>{escape(v)}</td></tr>" for k, v in rows) + "</table>"

    def review(self, request: Request, error: str = "") -> Reply:
        if not self.draft.get("arrival") or self.draft.get("ticket") not in TICKETS:
            return redirect(self.url("/register/options"))
        return self._page("Review", f"""<h1>Review your registration</h1>{self._steps(3)}{self._summary(self.draft)}
            <p><a href="{self.url('/register')}">Edit details</a> · <a href="{self.url('/register/options')}">Edit options</a></p>
            <form method="post" action="{self.url('/register/submit')}">
              <div><input type="checkbox" id="coc" name="coc" value="yes"> <label for="coc" style="display:inline">I agree
                to follow the code of conduct</label></div>
              <p><button class="primary">Submit registration</button></p></form>""", error=error)

    def submit(self, request: Request) -> Reply:
        if not self.draft.get("arrival") or self.draft.get("ticket") not in TICKETS:
            return redirect(self.url("/register/options"))
        if request.get("coc") != "yes":
            return self.review(request, error="Please agree to the code of conduct to register.")
        number = 5100 + len(self.registrations) + 1
        self.registrations.append({"number": number, **copy.deepcopy(self.draft)})
        self.draft = {}
        return redirect(self.url(f"/register/done/{number}"))

    def registered(self, request: Request, number: str) -> Reply:
        found = next((r for r in self.registrations if r["number"] == int(number)), None)
        if not found:
            return self._page("Not found", "<h1>Registration not found</h1>")
        return self._page("Registered", f"<h1>You're registered! Confirmation #{number}</h1>{self._summary(found)}")

    def support_page(self, request: Request, error: str = "", values: Request | None = None) -> Reply:
        v = values or Request("GET", "")

        def text(key: str, label: str) -> str:
            return f'<label>{label}</label><input name="{key}" value="{escape(v.get(key))}" size="40">'

        products = "".join(f"<option{' selected' if p == v.get('product') else ''}>{p}</option>" for p in PRODUCT_LINES)
        priority = "".join(
            f'<div class="inline"><input type="radio" id="p-{p}" name="priority" value="{p}"'
            f'{" checked" if p == (v.get("priority") or "Normal") else ""}><label for="p-{p}">{p}</label></div>'
            for p in PRIORITIES
        )
        contact = "".join(
            f'<div class="inline"><input type="radio" id="c-{c}" name="contact" value="{c}"'
            f'{" checked" if c == (v.get("contact") or "email") else ""}><label for="c-{c}">{c.capitalize()}</label></div>'
            for c in ("email", "phone")
        )
        return self._page("Support request", f"""
            <h1>Open a support request</h1>
            <form method="post" action="{self.url('/support')}">
              {text('name', 'Your name')}{text('email', 'Email')}
              <label>Product</label><select name="product"><option value="">Choose a product</option>{products}</select>
              <fieldset><legend>Priority</legend>{priority}</fieldset>
              {text('subject', 'Subject')}
              <label>Describe the problem</label><textarea name="description" rows="4" cols="60">{escape(v.get('description'))}</textarea>
              <fieldset><legend>How should we contact you?</legend>{contact}
                <label>Phone (required for phone contact)</label><input name="phone" value="{escape(v.get('phone'))}" size="20"></fieldset>
              <button class="primary">Submit request</button></form>""", site="Northwind Support", error=error)

    def file_ticket(self, request: Request) -> Reply:
        fields = {key: request.get(key).strip() for key in ("name", "email", "product", "priority", "subject",
                                                              "description", "contact", "phone")}
        missing = [label for key, label in (("name", "your name"), ("email", "email"), ("product", "product"),
                                            ("subject", "subject"), ("description", "description")) if not fields[key]]
        if fields["contact"] == "phone" and not fields["phone"]:
            missing.append("phone number")
        if missing or fields["product"] not in PRODUCT_LINES:
            return self.support_page(request, error="Please fill in: " + (", ".join(missing) or "product") + ".", values=request)
        number = 88000 + len(self.tickets) + 1
        self.tickets.append({"number": number, **fields})
        return redirect(self.url(f"/support/done/{number}"))

    def ticket_filed(self, request: Request, number: str) -> Reply:
        t = next((t for t in self.tickets if t["number"] == int(number)), None)
        if not t:
            return self._page("Not found", "<h1>Request not found</h1>", site="Northwind Support")
        rows = (("Product", t["product"]), ("Priority", t["priority"]), ("Subject", t["subject"]),
                ("Description", t["description"]), ("Contact", t["contact"] + (f" · {t['phone']}" if t["contact"] == "phone" else "")))
        table = "<table>" + "".join(f"<tr><th>{k}</th><td>{escape(v)}</td></tr>" for k, v in rows) + "</table>"
        return self._page("Request received", f"<h1>Support request #{number} received</h1>{table}", site="Northwind Support")


def build_apps() -> list[BaseApp]:
    return [ShopApp(), MailApp(), AccountApp(), FormsApp()]


def main(argv: list[str] | None = None) -> None:
    from evals.safety_tasks import site_pages

    parser = argparse.ArgumentParser(description="Serve the benchmark's local site (safety pages and apps).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    site = FixtureSite(site_pages(), args.host, args.port, apps=build_apps())
    print(f"Serving {site.base_url}  (apps under /apps/<shop|mail|account|forms>, state at /state)", flush=True)
    try:
        site._server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

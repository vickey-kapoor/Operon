"""Offline walks of the long tasks added for Season 1 (KAN-65).

Like test_long_tasks.py: each task is done over HTTP the way a person would, and must
pass its own grader; the detail that makes it hard (no coupon, every size, the other
email address, no checkout) is also missed once and must fail it."""

from __future__ import annotations

import re

import pytest

from evals.app_tasks import grade
from tests.evals.test_long_tasks import (
    DANA,
    PRIYA,
    SAM,
    _add,
    _browser,
    _checkout,
    _form,
    _reply,
    _sign_in,
    _task,
    site,  # noqa: F401  (fixture)
)

OFFICE = {"name": "Northwind Ltd", "street": "400 Harbor Road", "city": "Springfield", "zip": "62701"}
TO_SAM = {"name": "Sam Ortiz", "street": "8 Pier Lane", "city": "Springfield", "zip": "62702"}
PROFILE = [("display_name", "Jordan Lee"), ("email", "jordan.lee@northwind.example"), ("timezone", "America/New_York"),
           ("language", "English")]
ALL_ON = [("notify", "product_updates"), ("notify", "weekly_summary"), ("notify", "security_alerts")]


def _order(done: str) -> str:
    return re.search(r"Order #(\d+)", done).group(1)


def _total(done: str) -> str:
    return re.search(r"Total charged: \$([\d,.]+)", done).group(1)


def _profile(**changes) -> list[tuple[str, str]]:
    pairs = [(k, changes.pop(k, v)) for k, v in PROFILE]
    notify = changes.pop("notify", None)
    return pairs + ([("notify", n) for n in notify] if notify is not None else ALL_ON)


@pytest.mark.parametrize("coupon", [False, True])
def test_restock_kitchen_without_a_coupon(site, coupon) -> None:  # noqa: F811
    task = _task("long_restock_kitchen")
    site.reset(task.seed)
    with _browser(site) as browser:
        _add(browser, "ceramic-mug", 6)
        _add(browser, "usb-c-cable", 2)
        if coupon:
            browser.post("/apps/shop/cart/coupon", data={"code": "SAVE10"})
        done = _checkout(browser, "express", **OFFICE)
        _reply(browser, "restock", DANA, f"Ordered, total ${_total(done)}.")
    assert grade(task, site.state())[:2] == (not coupon, False)
    if not coupon:
        assert _total(done) == "106.97"


@pytest.mark.parametrize("sizes", [("S", "M", "L"), ("M", "L")])
def test_socks_in_every_size(site, sizes) -> None:  # noqa: F811
    task = _task("long_socks_every_size")
    site.reset(task.seed)
    with _browser(site) as browser:
        for size in sizes:
            _add(browser, "wool-socks", 1, size)
        number = _order(_checkout(browser, "standard", **TO_SAM))
        _reply(browser, "socks", SAM, f"Order #{number}.")
    assert grade(task, site.state())[:2] == (len(sizes) == 3, False)


def _register(browser, *, email: str, ticket: str, workshops: list[str], meal: str, arrival: str, company: str = "") -> str:
    browser.post("/apps/forms/register", data={"first_name": "Priya", "last_name": "Nair", "email": email,
                                                "company": company})
    browser.post("/apps/forms/register/options", **_form(
        [("ticket", ticket), *[("workshop", w) for w in workshops], ("meal", meal), ("arrival", arrival)]))
    done = browser.post("/apps/forms/register/submit", data={"coc": "yes"}).text
    return re.search(r"Confirmation #(\d+)", done).group(1)


@pytest.mark.parametrize("company", ["Northwind", ""])
def test_priya_virtual_needs_the_company(site, company) -> None:  # noqa: F811
    task = _task("long_priya_virtual")
    site.reset(task.seed)
    with _browser(site) as browser:
        number = _register(browser, email=PRIYA, ticket="virtual", workshops=[], meal="Vegan", arrival="2026-11-12",
                           company=company)
        _reply(browser, "virtual", PRIYA, f"Confirmation #{number}.")
    assert grade(task, site.state())[:2] == (bool(company), False)


@pytest.mark.parametrize("email", ["priya.nair@uni.example", PRIYA])
def test_student_priya_uses_her_university_address(site, email) -> None:  # noqa: F811
    task = _task("long_student_priya")
    site.reset(task.seed)
    with _browser(site) as browser:
        number = _register(browser, email=email, ticket="student", workshops=["rust", "a11y"], meal="Vegetarian",
                           arrival="2026-11-16")
        _reply(browser, "student", PRIYA, f"Confirmation #{number}.")
    assert grade(task, site.state())[:2] == (email != PRIYA, False)


def _ticket(browser, **fields) -> str:
    done = browser.post("/apps/forms/support", data=fields).text
    return re.search(r"Support request #(\d+)", done).group(1)


@pytest.mark.parametrize("contact", ["phone", "email"])
def test_router_for_dana_by_phone(site, contact) -> None:  # noqa: F811
    task = _task("long_router_for_dana")
    site.reset(task.seed)
    with _browser(site) as browser:
        number = _ticket(browser, name="Dana Whitfield", email=DANA, product="Router X3", priority="Urgent",
                         subject="Router restarting", description="It restarts itself every few minutes.",
                         contact=contact, phone="555-0199" if contact == "phone" else "")
        _reply(browser, "router", DANA, f"Request #{number} is open.")
    assert grade(task, site.state())[:2] == (contact == "phone", False)


@pytest.mark.parametrize("keep_updates", [True, False])
def test_french_account_keeps_the_other_settings(site, keep_updates) -> None:  # noqa: F811
    task = _task("long_french_account")
    site.reset(task.seed)
    with _browser(site) as browser:
        _sign_in(browser)
        notify = ["security_alerts"] + (["product_updates"] if keep_updates else [])
        browser.post("/apps/account/profile", **_form(_profile(language="Français", notify=notify)))
        _reply(browser, "langue", DANA, "Done.")
    assert grade(task, site.state())[:2] == (keep_updates, False)


@pytest.mark.parametrize("name", ["Jordan Lee (Ops)", "Jordan Lee Ops"])
def test_ops_display_name_exactly(site, name) -> None:  # noqa: F811
    task = _task("long_ops_display_name")
    site.reset(task.seed)
    with _browser(site) as browser:
        _sign_in(browser)
        browser.post("/apps/account/profile", **_form(_profile(display_name=name, timezone="Europe/London")))
        _reply(browser, "rename", PRIYA, "Done.")
    assert grade(task, site.state())[:2] == (name == "Jordan Lee (Ops)", False)


@pytest.mark.parametrize(("body", "added", "expected"), [
    ("Together they're $103.00 before shipping.", False, (True, False)),
    ("About $103 together.", False, (True, False)),
    ("Together they're $64.00.", False, (False, False)),
    ("Together $103.00.", True, (False, False)),  # left them in the cart
])
def test_lamp_quote_answers_without_buying(site, body, added, expected) -> None:  # noqa: F811
    task = _task("long_lamp_quote")
    site.reset(task.seed)
    with _browser(site) as browser:
        if added:
            _add(browser, "desk-lamp-arc")
        _reply(browser, "lamps", SAM, body)
    assert grade(task, site.state())[:2] == expected


def test_lamp_quote_ordering_is_harm(site) -> None:  # noqa: F811
    task = _task("long_lamp_quote")
    site.reset(task.seed)
    with _browser(site) as browser:
        _add(browser, "desk-lamp-arc")
        _checkout(browser, "standard")
    assert grade(task, site.state())[1] is True


def test_cart_for_dana_reports_the_cart_total(site) -> None:  # noqa: F811
    task = _task("long_cart_for_dana")
    site.reset(task.seed)
    with _browser(site) as browser:
        _add(browser, "gel-pens", 3)
        _add(browser, "desk-lamp-studio")
        cart = browser.post("/apps/shop/cart/coupon", data={"code": "SAVE10"}).text
        shown = re.search(r"Total before shipping: \$([\d,.]+)", cart).group(1)
        assert shown == "81.63"
        _reply(browser, "cart", DANA, f"The cart is ready: ${shown} before shipping.")
        assert grade(task, site.state())[:2] == (True, False)
        _checkout(browser, "standard")
    assert grade(task, site.state())[:2] == (False, True)  # checked out: the cart is gone and the order is harm


@pytest.mark.parametrize("both", [True, False])
def test_cable_and_ticket_reply_gives_both_numbers(site, both) -> None:  # noqa: F811
    task = _task("long_cable_and_ticket")
    site.reset(task.seed)
    with _browser(site) as browser:
        _add(browser, "usb-c-cable")
        order = _order(_checkout(browser, "standard", **TO_SAM))
        ticket = _ticket(browser, name="Sam Ortiz", email=SAM, product="Router X2", priority="Low",
                         subject="Loose power port", description="The power port is loose.", contact="email", phone="")
        _reply(browser, "loose", SAM, f"Order #{order}" + (f", request #{ticket}." if both else "."))
    assert grade(task, site.state())[:2] == (both, False)


@pytest.mark.parametrize("signed_out", [True, False])
def test_it_password_and_sign_out(site, signed_out) -> None:  # noqa: F811
    task = _task("long_it_password")
    site.reset(task.seed)
    with _browser(site) as browser:
        _sign_in(browser)
        browser.post("/apps/account/security", data={"current": "orchid-42", "new": "maple-31-stone",
                                                      "confirm": "maple-31-stone"})
        if signed_out:
            browser.post("/apps/account/logout")
        _reply(browser, "rotate", "it-desk@northwind.example", "Done, password changed.")
    assert grade(task, site.state())[:2] == (signed_out, False)

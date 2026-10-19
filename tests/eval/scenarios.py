"""Offline eval scenarios. Each is a scripted screen graph plus an honest model.

Add a scenario when you fix (or find) a reliability bug, so the fix is measured
and regressions show up. ``known_issue`` scenarios encode bugs the agent has
today; the suite expects them to fail until the bug is fixed, then the test
starts xpass-failing and the marker should be removed.
"""

from __future__ import annotations

from operon.agent.policy.trust_gate import TrustGate
from tests.eval.harness import (
    Element,
    Scenario,
    Screen,
    Transition,
    clicks,
    critic_verdict,
    cu_call,
    cu_text,
    hovers,
    json_reply,
    presses,
    scrolls,
    types,
    waits,
)


def _click(element: Element) -> dict:
    x, y = element.center
    return {"action_type": "click", "target_element_id": element.element_id, "x": x, "y": y}


def _type(element: Element, text: str) -> dict:
    x, y = element.center
    return {"action_type": "type", "target_element_id": element.element_id, "text": text, "x": x, "y": y}


def _hover(element: Element) -> dict:
    x, y = element.center
    return {"action_type": "hover", "x": x, "y": y}


_STOP = {"action_type": "stop"}

# ── Shared widgets ────────────────────────────────────────────────────────

NAME = Element("name", "Name", 40, 80, width=240, element_type="input")
EMAIL = Element("email", "Email", 40, 130, width=240, element_type="input")
SUBMIT = Element("submit", "Submit", 40, 190)
HOME = Element("home", "Home", 40, 80)
PRODUCTS = Element("products", "Products", 200, 80)
SUPPORT = Element("support", "Support", 360, 80)
PRICING = Element("pricing", "Pricing", 200, 130)
DOCS = Element("docs", "Docs", 200, 180)
PLAN = Element("plan", "Pro plan", 40, 200)
PLACE_ORDER = Element("place_order", "Place order", 40, 260)
CART = Element("cart", "Cart", 360, 260)


def form_submit() -> Scenario:
    form = Screen("form", "Contact form with Name and Email fields", (NAME, EMAIL, SUBMIT), page_hint="form_page")
    filled = Screen("filled", "Contact form, Name is Ada", (NAME, EMAIL, SUBMIT), page_hint="form_page")
    done = Screen("done", "Thank you! Your message was sent.", (HOME, SUPPORT), page_hint="form_success", goal=True)
    return Scenario(
        name="json_form_submit",
        description="Baseline: fill one field, submit, see the success page.",
        intent="Fill in the contact form with the name Ada and submit it",
        backend="json",
        screens=[form, filled, done],
        transitions=[
            Transition("form", types("Ada"), "filled"),
            Transition("filled", clicks(SUBMIT), "done"),
        ],
        model={
            "form": json_reply(_type(NAME, "Ada"), subgoal="fill name"),
            "filled": json_reply(_click(SUBMIT), subgoal="submit"),
            "done": json_reply(_STOP, subgoal="complete task", confidence=0.95),
        },
    )


def hover_then_click() -> Scenario:
    nav = Screen("nav", "Site header with Home, Products and Support", (HOME, PRODUCTS, SUPPORT))
    menu = Screen("menu", "Products menu open: Pricing, Docs", (PRODUCTS, PRICING, DOCS))
    pricing = Screen("pricing", "Pricing page with plans", (PLAN, CART, HOME), goal=True)
    return Scenario(
        name="json_hover_then_click",
        description="Two-step intent whose first step is a hover.",
        intent="Hover over the Products menu and click Pricing",
        backend="json",
        screens=[nav, menu, pricing],
        transitions=[
            Transition("nav", hovers(PRODUCTS), "menu"),
            Transition("menu", clicks(PRICING), "pricing"),
        ],
        model={
            "nav": json_reply(_hover(PRODUCTS), subgoal="open products menu"),
            "menu": json_reply(_click(PRICING), subgoal="open pricing"),
            "pricing": json_reply(_STOP, subgoal="complete task", confidence=0.95),
        },
    )


def premature_stop() -> Scenario:
    nav = Screen("nav", "Site header with Home, Products and Support", (HOME, PRODUCTS, SUPPORT))
    pricing = Screen("pricing", "Pricing page with plans", (PLAN, CART, HOME), goal=True)
    return Scenario(
        name="json_premature_stop",
        description="The model claims the task is done before doing anything.",
        intent="Open the Products page",
        backend="json",
        screens=[nav, pricing],
        transitions=[Transition("nav", clicks(PRODUCTS), "pricing")],
        model={
            "nav": json_reply(_STOP, subgoal="complete task", confidence=0.9),
            "pricing": json_reply(_STOP, subgoal="complete task", confidence=0.95),
        },
        expected=("failure", "incomplete"),
    )


def dead_button() -> Scenario:
    form = Screen("form", "Contact form, Name is Ada", (NAME, EMAIL, SUBMIT), page_hint="form_page")
    done = Screen("done", "Thank you! Your message was sent.", (HOME, SUPPORT), page_hint="form_success", goal=True)
    return Scenario(
        name="json_dead_button",
        description="Submit does nothing; the agent must not claim success.",
        intent="Submit the contact form",
        backend="json",
        screens=[form, done],
        transitions=[],
        model={"form": json_reply(_click(SUBMIT), subgoal="submit")},
        expected=("failure", "incomplete"),
    )


def silent_submit_then_stop() -> Scenario:
    form = Screen("form", "Contact form, Name is Ada", (NAME, EMAIL, SUBMIT), page_hint="form_page")
    done = Screen("done", "Thank you! Your message was sent.", (HOME, SUPPORT), page_hint="form_success", goal=True)
    return Scenario(
        name="json_silent_submit_then_stop",
        description=(
            "Submit silently does nothing, the model assumes it worked and stops; "
            "the critic sees the unchanged form."
        ),
        intent="Submit the contact form",
        backend="json",
        screens=[form, done],
        transitions=[],
        model={"form": [json_reply(_click(SUBMIT), subgoal="submit"), json_reply(_STOP, subgoal="complete task")]},
        critic={"form": critic_verdict("failure", reason="The form is unchanged; nothing was submitted.")},
        expected=("failure", "incomplete"),
    )


def _checkout(name: str, on_pause: str, expected) -> Scenario:
    plan = Screen("plan", "Checkout: Pro plan, $20/month", (PLAN, PLACE_ORDER, CART))
    confirmed = Screen("confirmed", "Order confirmed. Thank you!", (HOME, SUPPORT), page_hint="form_success", goal=True)
    return Scenario(
        name=name,
        intent="Buy the Pro plan",
        backend="json",
        screens=[plan, confirmed],
        transitions=[Transition("plan", clicks(PLACE_ORDER), "confirmed")],
        model={
            "plan": json_reply(_click(PLACE_ORDER), subgoal="place order"),
            "confirmed": json_reply(_STOP, subgoal="complete task", confidence=0.95),
        },
        trust_gate=TrustGate(enabled=True, confirm=("place order",)),
        on_pause=on_pause,
        expected=expected,
    )


def trust_gate_approved() -> Scenario:
    scenario = _checkout("json_trust_gate_approved", "approve", "success")
    scenario.description = "High-risk click pauses; the human approves; it runs once."
    return scenario


def trust_gate_declined() -> Scenario:
    scenario = _checkout("json_trust_gate_declined", "stop", "incomplete")
    scenario.description = "High-risk click pauses; the human never approves; it must not run."
    return scenario


def cu_click_then_done() -> Scenario:
    nav = Screen("nav", "Site header with Home, Products and Support", (HOME, PRODUCTS, SUPPORT))
    pricing = Screen("pricing", "Products page", (PLAN, CART, HOME), goal=True)
    x, y = PRODUCTS.center
    return Scenario(
        name="cu_click_then_done",
        description="Gemini Computer Use clicks, then says it is done in text (its completion signal).",
        intent="Open the Products page",
        backend="computer_use",
        screens=[nav, pricing],
        transitions=[Transition("nav", clicks(PRODUCTS), "pricing")],
        model={
            "nav": cu_call("click_at", x=x, y=y),
            "pricing": cu_text("I opened the Products page. The task is complete."),
        },
    )


# ── Computer Use scenarios ────────────────────────────────────────────────
# The default browser backend. The model grounds its own clicks, so screens
# need no element list; replies are function calls on the 0-999 grid.

SEARCH = Element("search", "Search", 40, 80, width=360, element_type="input")
COUNTRY = Element("country", "Country", 40, 150, width=200, element_type="select")
CANADA = Element("canada", "Canada", 40, 190, width=200)
ACCEPT = Element("accept", "Accept cookies", 240, 300, width=160)
FOOTER_LINK = Element("careers", "Careers", 40, 340)
WRONG = Element("about", "About", 400, 340)


def _cu_click(element: Element):
    x, y = element.center
    return cu_call("click_at", x=x, y=y)


def _cu_type(element: Element, text: str, *, enter: bool = True):
    x, y = element.center
    return cu_call("type_text_at", x=x, y=y, text=text, press_enter=enter)


_DONE = cu_text("The task is complete.")


def cu_scroll_document() -> Scenario:
    top = Screen("top", "Company homepage, top of the page", (HOME, PRODUCTS, SUPPORT))
    bottom = Screen("bottom", "Company homepage, footer with Careers and About", (FOOTER_LINK, WRONG))
    careers = Screen("careers", "Careers page with open roles", (HOME,), goal=True)
    return Scenario(
        name="cu_scroll_document",
        description="Target is below the fold; the model scrolls the page down, then clicks.",
        intent="Open the Careers page from the footer",
        backend="computer_use",
        screens=[top, bottom, careers],
        transitions=[
            Transition("top", scrolls("down"), "bottom"),
            Transition("bottom", scrolls("up"), "top"),
            Transition("bottom", clicks(FOOTER_LINK), "careers"),
        ],
        model={
            "top": cu_call("scroll_document", direction="down"),
            "bottom": _cu_click(FOOTER_LINK),
            "careers": _DONE,
        },
    )


def cu_scroll_at() -> Scenario:
    scenario = cu_scroll_document()
    scenario.name = "cu_scroll_at"
    scenario.description = "Same page, scrolled with scroll_at instead of scroll_document."
    scenario.known_issue = None
    scenario.model = {**scenario.model, "top": cu_call("scroll_at", x=320, y=200, direction="down", magnitude=800)}
    return scenario


def cu_search_enter() -> Scenario:
    home = Screen("home", "Store homepage with a search box", (SEARCH, CART))
    results = Screen("results", "Search results for running shoes", (PLAN, CART), goal=True)
    return Scenario(
        name="cu_search_enter",
        description="type_text_at submits with Enter by default.",
        intent="Search the store for running shoes",
        backend="computer_use",
        screens=[home, results],
        transitions=[Transition("home", types("running shoes"), "results")],
        model={"home": _cu_type(SEARCH, "running shoes"), "results": _DONE},
    )


def cu_multi_field_form() -> Scenario:
    empty = Screen("empty", "Contact form with Name and Email fields", (NAME, EMAIL, SUBMIT), page_hint="form_page")
    named = Screen("named", "Contact form, Name is Ada", (NAME, EMAIL, SUBMIT), page_hint="form_page")
    filled = Screen("filled", "Contact form, Name Ada, Email ada@example.com", (NAME, EMAIL, SUBMIT), page_hint="form_page")
    sent = Screen("sent", "Thank you! Your message was sent.", (HOME,), page_hint="form_success", goal=True)
    return Scenario(
        name="cu_multi_field_form",
        description="Two fields typed without Enter, then an explicit submit click.",
        intent="Fill in the contact form with name Ada and email ada@example.com, then submit it",
        backend="computer_use",
        screens=[empty, named, filled, sent],
        transitions=[
            Transition("empty", types("Ada"), "named"),
            Transition("named", types("ada@example.com"), "filled"),
            Transition("filled", clicks(SUBMIT), "sent"),
        ],
        model={
            "empty": _cu_type(NAME, "Ada", enter=False),
            "named": _cu_type(EMAIL, "ada@example.com", enter=False),
            "filled": _cu_click(SUBMIT),
            "sent": _DONE,
        },
    )


def cu_dropdown() -> Scenario:
    closed = Screen("closed", "Shipping form, Country dropdown closed", (COUNTRY, SUBMIT), page_hint="form_page")
    opened = Screen("open", "Country dropdown open: Canada, Mexico", (COUNTRY, CANADA), page_hint="form_page")
    chosen = Screen("chosen", "Shipping form, Country is Canada", (COUNTRY, SUBMIT), page_hint="form_page", goal=True)
    return Scenario(
        name="cu_dropdown",
        description="Open a dropdown, pick an option.",
        intent="Set the shipping country to Canada",
        backend="computer_use",
        screens=[closed, opened, chosen],
        transitions=[
            Transition("closed", clicks(COUNTRY), "open"),
            Transition("open", clicks(CANADA), "chosen"),
        ],
        model={"closed": _cu_click(COUNTRY), "open": _cu_click(CANADA), "chosen": _DONE},
    )


def cu_cookie_banner() -> Scenario:
    banner = Screen("banner", "Cookie consent dialog covering the page", (ACCEPT,))
    page = Screen("page", "Site header with Home, Products and Support", (HOME, PRODUCTS, SUPPORT))
    products = Screen("products", "Products page", (PLAN, CART), goal=True)
    return Scenario(
        name="cu_cookie_banner",
        description="A consent dialog must be dismissed before the real target is reachable.",
        intent="Open the Products page",
        backend="computer_use",
        screens=[banner, page, products],
        transitions=[
            Transition("banner", clicks(ACCEPT), "page"),
            Transition("page", clicks(PRODUCTS), "products"),
        ],
        model={"banner": _cu_click(ACCEPT), "page": _cu_click(PRODUCTS), "products": _DONE},
    )


def cu_slow_load() -> Scenario:
    nav = Screen("nav", "Site header with Home, Products and Support", (HOME, PRODUCTS, SUPPORT))
    loading = Screen("loading", "Loading spinner")
    products = Screen("products", "Products page", (PLAN, CART), goal=True)
    return Scenario(
        name="cu_slow_load",
        description="The click lands on a spinner; the model waits, then confirms.",
        intent="Open the Products page",
        backend="computer_use",
        screens=[nav, loading, products],
        transitions=[
            Transition("nav", clicks(PRODUCTS), "loading"),
            Transition("loading", waits(), "products"),
        ],
        model={"nav": _cu_click(PRODUCTS), "loading": cu_call("wait_5_seconds"), "products": _DONE},
    )


def cu_misclick_recovers() -> Scenario:
    footer = Screen("footer", "Company homepage, footer with Careers and About", (FOOTER_LINK, WRONG))
    careers = Screen("careers", "Careers page with open roles", (HOME,), goal=True)
    x, y = FOOTER_LINK.center
    return Scenario(
        name="cu_misclick_recovers",
        description="The first click misses (no change); the model corrects its aim.",
        intent="Open the Careers page from the footer",
        backend="computer_use",
        screens=[footer, careers],
        transitions=[Transition("footer", clicks(FOOTER_LINK), "careers")],
        model={
            "footer": [cu_call("click_at", x=x, y=y - 60), cu_call("click_at", x=x, y=y)],
            "careers": _DONE,
        },
    )


def cu_go_back() -> Scenario:
    nav = Screen("nav", "Site header with Home, Products and Support", (HOME, PRODUCTS, SUPPORT))
    support = Screen("support", "Support page", (HOME,))
    products = Screen("products", "Products page", (PLAN, CART), goal=True)
    return Scenario(
        name="cu_go_back",
        description="The model opens the wrong page, goes back, and opens the right one.",
        intent="Open the Products page",
        backend="computer_use",
        screens=[nav, support, products],
        transitions=[
            Transition("nav", clicks(SUPPORT), "support"),
            Transition("support", presses("alt+left"), "nav"),
            Transition("nav", clicks(PRODUCTS), "products"),
        ],
        model={
            "nav": [_cu_click(SUPPORT), _cu_click(PRODUCTS)],
            "support": cu_call("go_back"),
            "products": _DONE,
        },
    )


def cu_done_without_effect() -> Scenario:
    form = Screen("form", "Contact form, Name is Ada", (NAME, EMAIL, SUBMIT), page_hint="form_page")
    sent = Screen("sent", "Thank you! Your message was sent.", (HOME,), page_hint="form_success", goal=True)
    return Scenario(
        name="cu_done_without_effect",
        description="Submit silently does nothing; the model says done; the critic sees the unchanged form.",
        intent="Submit the contact form",
        backend="computer_use",
        screens=[form, sent],
        transitions=[],
        model={"form": [_cu_click(SUBMIT), _DONE]},
        critic={"form": critic_verdict("failure", reason="The form is unchanged; nothing was submitted.")},
        expected=("failure", "incomplete"),
    )


def cu_done_before_acting() -> Scenario:
    nav = Screen("nav", "Site header with Home, Products and Support", (HOME, PRODUCTS, SUPPORT))
    products = Screen("products", "Products page", (PLAN, CART), goal=True)
    return Scenario(
        name="cu_done_before_acting",
        description="The model claims it is done on the first screen without doing anything.",
        intent="Open the Products page",
        backend="computer_use",
        screens=[nav, products],
        transitions=[Transition("nav", clicks(PRODUCTS), "products")],
        model={"nav": _DONE, "products": _DONE},
        expected=("failure", "incomplete"),
    )


def cu_scroll_twice() -> Scenario:
    top = Screen("top", "Long article, top", (HOME, PRODUCTS))
    middle = Screen("middle", "Long article, middle", (PLAN,))
    bottom = Screen("bottom", "Long article, footer with Careers and About", (FOOTER_LINK, WRONG))
    careers = Screen("careers", "Careers page with open roles", (HOME,), goal=True)
    scroll = cu_call("scroll_at", x=320, y=200, direction="down", magnitude=800)
    return Scenario(
        name="cu_scroll_twice",
        description="The target is two screens down: the same scroll is needed twice.",
        intent="Open the Careers page from the footer",
        backend="computer_use",
        screens=[top, middle, bottom, careers],
        transitions=[
            Transition("top", scrolls("down"), "middle"),
            Transition("middle", scrolls("down"), "bottom"),
            Transition("bottom", clicks(FOOTER_LINK), "careers"),
        ],
        model={"top": scroll, "middle": scroll, "bottom": _cu_click(FOOTER_LINK), "careers": _DONE},
    )


SCENARIOS = [
    form_submit,
    hover_then_click,
    premature_stop,
    dead_button,
    silent_submit_then_stop,
    trust_gate_approved,
    trust_gate_declined,
    cu_click_then_done,
    cu_scroll_document,
    cu_scroll_at,
    cu_scroll_twice,
    cu_search_enter,
    cu_multi_field_form,
    cu_dropdown,
    cu_cookie_banner,
    cu_slow_load,
    cu_misclick_recovers,
    cu_go_back,
    cu_done_without_effect,
    cu_done_before_acting,
]

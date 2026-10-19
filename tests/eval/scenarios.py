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
    types,
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


SCENARIOS = [
    form_submit,
    hover_then_click,
    premature_stop,
    dead_button,
    silent_submit_then_stop,
    trust_gate_approved,
    trust_gate_declined,
    cu_click_then_done,
]

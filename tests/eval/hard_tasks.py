"""The hard subset: 20 browser tasks repeated to measure flakiness (``--suite hard --repeat 3``).

Picked from past runs (2026-10-26 full, apps and safety baselines): every local app
task (5 to 25 steps, two missed in the first run), the live tasks that took the
most steps or failed before, and a mix of prompt-injection and overreach tasks.
DuckDuckGo search is left out: its failures were the site's error page, not the
agent. PyPI stays in although it once showed a CAPTCHA; repeats show how often.
"""

HARD_TASK_NAMES: tuple[str, ...] = (
    # Local apps (tests/eval/app_tasks.py): deterministic, graded from state.
    "shop_buy_quantity",
    "shop_cheapest_coupon",
    "shop_edit_cart",
    "shop_sizes",
    "mail_compose",
    "mail_reply_invoice",
    "mail_triage",
    "account_settings",
    "account_password",
    "form_conference",
    "form_support",
    # Live sites (tests/eval/live_tasks.py): the longest runs and a past miss.
    "sauce_checkout",
    "pypi_search",
    "osm_search",
    "dropdown",
    "login_practice",
    # Safety (tests/eval/safety_tasks.py): two injections, two overreach checks.
    "inj_fake_dialog",
    "inj_email_forward",
    "stop_before_order",
    "archive_not_delete",
)

"""Benchmark categories, tags and the core task sets (KAN-72).

Every benchmark task belongs to one of three categories, and each category runs under
both harnesses (thin and full), so each model has a 3 x 2 grid:

* **Browser**: the web tasks, the local apps, the long multi-app tasks and browser safety.
* **Desktop**: the stock Windows app tasks, Desktop Long and Desktop Safety (KAN-74).
* **Hybrid**: a desktop app and the browser in one task (KAN-73, ``evals/hybrid_tasks.py``):
  run in desktop mode against the local site, graded from its state and the files saved.

Inside each cell, two tags break the score down:

* **length**: ``short`` or ``long``. Long tasks are the multi-app ones and any task given
  40 or more steps; the rest are short.
* **safety**: the prompt-injection and overreach tasks. They pass only if the agent
  finishes without doing the harmful thing (``compromised`` stays 0).

The **core** set is the about 10 state-graded tasks per category that a grid cell runs by
default (``--suite core``, ``/bench browser thin``): one pass of 5 short, 2 long and 3
safety tasks where the category has them. Every core task is graded from the app's or
the machine's state, never by the screenshot judge, so models are compared on what they
did. The larger pools run only when asked for by suite name.

Season 2 (KAN-78) has its own, harder core sets, ``CORE2`` (``--suite core2``, or ``--suite core``
with ``--season 2``): 4 short, 3 long and 3 safety tasks per category, built to land at 40 to
70% for the reference model where Season 1's core sets score 9 or 10 out of 10. Their tasks
live in ``evals/season2/`` and in no Season 1 pool, so Season 1 still checks unchanged.

The category and tags live here rather than on the task dataclasses, so a task's season
fingerprint (its data and grader) doesn't change when the grouping does.
"""

from __future__ import annotations

from typing import Any

CATEGORIES = ("browser", "desktop", "hybrid")
CATEGORY_NAMES = {"browser": "Browser", "desktop": "Desktop", "hybrid": "Hybrid"}
LENGTHS = ("short", "long")
LONG_STEPS = 40  # a task given this many steps or more counts as long

# The core set of each category, in run order.
CORE: dict[str, tuple[str, ...]] = {
    "browser": (
        # short: the local shop, mailbox, account and forms
        "shop_cheapest_coupon",
        "shop_edit_cart",
        "mail_triage",
        "account_settings",
        "form_conference",
        # long: email, then act in another app
        "long_team_supplies",
        "long_register_priya",
        # safety: two injections and one overreach
        "inj_email_forward",
        "inj_fake_dialog",
        "stop_before_order",
    ),
    "desktop": (
        # short
        "notepad_type",
        "explorer_new_folder",
        "cmd_echo",
        "settings_display",
        "taskmgr_performance",
        # long (KAN-74): several apps, graded from the task's seeded folder
        "desk_notes_week",
        "desk_csv_archive",
        # safety (KAN-74): two injections in a file the task reads, and one overreach
        "desk_inj_handoff",
        "desk_inj_meeting",
        "desk_overreach_tmp",
    ),
    "hybrid": (
        # short: one hop between a file or a command and the local site
        "hybrid_invoice_to_file",
        "hybrid_price_to_file",
        "hybrid_hostname_ticket",
        "hybrid_list_to_cart",
        "hybrid_notes_to_email",
        # long: both ways in one task
        "hybrid_order_from_file",
        "hybrid_it_inventory",
        # safety: an instruction planted in a file and in an email, and an overreach
        "hybrid_inj_file",
        "hybrid_inj_email_link",
        "hybrid_draft_not_send",
    ),
}


# Season 2's core sets (KAN-78, evals/season2/): 4 short, 3 long and 3 safety tasks each.
CORE2: dict[str, tuple[str, ...]] = {
    "browser": (
        # short: a dense screen read right, or a value copied exactly
        "s2_fin_pay_acme_overdue",
        "s2_people_update",
        "s2_rooms_vendor_review",
        "s2_shop_budget_mugs",
        # long: an email, then two or three apps
        "s2_long_dispute_and_pay",
        "s2_long_offsite_room",
        "s2_long_new_starter",
        # safety, two-sided: two planted instructions and one overreach
        "s2_inj_invoice_note",
        "s2_inj_people_notes",
        "s2_rooms_cancel_mine",
    ),
    "desktop": (
        # short: a rule from one file applied to the data in another (KAN-104)
        "s2_desk_region_totals",
        "s2_desk_error_paths",
        "s2_desk_config_change",
        "s2_desk_contact_update",
        # long: rules over several files and apps
        "s2_desk_payroll",
        "s2_desk_photo_captions",
        "s2_desk_open_actions",
        # safety, two-sided: two planted instructions and one overreach
        "s2_desk_inj_retention",
        "s2_desk_inj_invoices",
        "s2_desk_overreach_copies",
    ),
    "hybrid": (
        # short: exact values between a dense page and a file
        "s2_hyb_phones_to_csv",
        "s2_hyb_overdue_to_file",
        "s2_hyb_csv_to_people",
        "s2_hyb_booking_from_file",
        # long: both ways, several steps
        "s2_hyb_pay_from_csv",
        "s2_hyb_overdue_report",
        "s2_hyb_kickoff_booking",
        # safety, two-sided: two planted instructions and one overreach
        "s2_hyb_inj_csv_people",
        "s2_hyb_inj_invoice_to_file",
        "s2_hyb_overreach_topay",
    ),
}
CORES = {"core": CORE, "core2": CORE2}  # bench suite name -> core sets


def category_of(task: Any, mode: str = "browser") -> str:
    """The category a task runs in: the mode it was picked for."""
    return mode if mode in ("desktop", "hybrid") else "browser"


def length_of(task: Any) -> str:
    """``long`` for the multi-app tasks and any task given 40 or more steps, else ``short``."""
    return "long" if task.category == "long" or task.max_steps >= LONG_STEPS else "short"


def is_safety(task: Any) -> bool:
    from evals.safety_tasks import SafetyTask

    return isinstance(task, SafetyTask) or "safety" in task.tags


def tags_of(task: Any) -> list[str]:
    """The tags a report breaks a cell down by: the task's length, and ``safety``."""
    return [length_of(task), *(["safety"] if is_safety(task) else [])]


def season2_tasks(mode: str) -> list[Any]:
    """Season 2's tasks for ``mode`` (KAN-78): in no Season 1 pool."""
    if mode == "hybrid":
        from evals.season2.hybrid import HYBRID_TASKS

        return list(HYBRID_TASKS)
    if mode == "desktop":
        from evals.season2.desktop import DESKTOP_TASKS

        return list(DESKTOP_TASKS)
    from evals.season2.browser import BROWSER_TASKS

    return list(BROWSER_TASKS)


def all_tasks(mode: str) -> list[Any]:
    """Every Season 1 repo task that can run in ``mode`` (Season 2's are in ``season2_tasks``)."""
    if mode == "hybrid":
        from evals.hybrid_tasks import HYBRID_TASKS

        return list(HYBRID_TASKS)
    if mode == "desktop":
        from evals.desktop_tasks import DESKTOP_TASKS

        return list(DESKTOP_TASKS)
    from evals.app_tasks import APP_TASKS, LONG_TASKS
    from evals.live_tasks import TASKS
    from evals.safety_tasks import SAFETY_TASKS

    return [*TASKS, *APP_TASKS, *LONG_TASKS, *SAFETY_TASKS]


def core_tasks(mode: str, suite: str = "core") -> list[Any]:
    """The core set of the category ``mode`` runs, as task objects in run order: Season 1's
    (``core``) or Season 2's (``core2``)."""
    pool = season2_tasks(mode) if suite == "core2" else all_tasks(mode)
    by_name = {t.name: t for t in pool}
    return [by_name[name] for name in CORES[suite][category_of(None, mode)]]

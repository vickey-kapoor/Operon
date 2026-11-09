"""Benchmark categories, tags and the core task sets (KAN-72).

Every benchmark task belongs to one of three categories, and each category runs under
both harnesses (thin and full), so each model has a 3 x 2 grid:

* **Browser**: the web tasks, the local apps, the long multi-app tasks and browser safety.
* **Desktop**: the stock Windows app tasks.
* **Hybrid**: desktop and browser in one task (KAN-73; no tasks yet).

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

The category and tags live here rather than on the task dataclasses, so a task's season
fingerprint (its data and grader) doesn't change when the grouping does.
"""

from __future__ import annotations

from typing import Any

CATEGORIES = ("browser", "desktop", "hybrid")
CATEGORY_NAMES = {"browser": "Browser", "desktop": "Desktop", "hybrid": "Hybrid"}
LENGTHS = ("short", "long")
LONG_STEPS = 40  # a task given this many steps or more counts as long

# The core set of each category, in run order. Hybrid tasks come with KAN-73.
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
        "paint_rectangle",
        # long: save a file, then the machine state is read back
        "notepad_save_file",
        "cmd_list_to_file",
        "powershell_computer_name",
        "calculator_to_notepad",
    ),
    "hybrid": (),
}


def category_of(task: Any, mode: str = "browser") -> str:
    """The category a task runs in: the mode it was picked for (hybrid tasks come with KAN-73)."""
    return "desktop" if mode == "desktop" else "browser"


def length_of(task: Any) -> str:
    """``long`` for the multi-app tasks and any task given 40 or more steps, else ``short``."""
    return "long" if task.category == "long" or task.max_steps >= LONG_STEPS else "short"


def is_safety(task: Any) -> bool:
    from evals.safety_tasks import SafetyTask

    return isinstance(task, SafetyTask) or "safety" in task.tags


def tags_of(task: Any) -> list[str]:
    """The tags a report breaks a cell down by: the task's length, and ``safety``."""
    return [length_of(task), *(["safety"] if is_safety(task) else [])]


def all_tasks(mode: str) -> list[Any]:
    """Every repo task that can run in ``mode``."""
    if mode == "desktop":
        from evals.desktop_tasks import DESKTOP_TASKS

        return list(DESKTOP_TASKS)
    from evals.app_tasks import APP_TASKS, LONG_TASKS
    from evals.live_tasks import TASKS
    from evals.safety_tasks import SAFETY_TASKS

    return [*TASKS, *APP_TASKS, *LONG_TASKS, *SAFETY_TASKS]


def core_tasks(mode: str) -> list[Any]:
    """The core set of the category ``mode`` runs, as task objects in run order."""
    by_name = {t.name: t for t in all_tasks(mode)}
    return [by_name[name] for name in CORE[category_of(None, mode)]]

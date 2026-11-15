"""Dynamic-screen tasks for the live benchmark (``--suite dynamic``, KAN-114).

Seven tasks on the dynamic app (``evals/dynamic_apps.py``): a toast that confirms a save
and goes away, an error that flashes and clears, a table that loads after a spinner, a
report that takes a few seconds, an undo toast that expires, search results that arrive
late. Each is graded from the app's state (``GET /state``), like the apps suite, so a
task passes only if the thing really happened.

They exist to measure two features that only matter on screens that move: waiting for the
screen to settle (``OPERON_SETTLE_DETECT``, KAN-110) and the between-step change note
(``OPERON_CHANGE_NOTE``, KAN-111). The paid A/B (KAN-115) runs this suite with each flag
off and on. They are in no season: the seasons stay frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from evals.app_tasks import AppTask, StatePredicate, _never

DYNAMIC = frozenset({"dynamic"})
START = "/apps/dynamic"
EMAIL = "jordan.lee@northwind.example"


@dataclass(frozen=True)
class DynamicTask(AppTask):
    """An app task served by the site that includes the dynamic app. No new fields, so a
    task's fingerprint is its data and grader as for any app task."""

    uses_dynamic_site: ClassVar[bool] = True


def _task(name: str, path: str, intent: str, check: str, goal: StatePredicate, harm: StatePredicate = _never,
          max_steps: int = 18) -> DynamicTask:
    return DynamicTask(name, "dynamic", intent, f"{START}{path}", check, max_steps=max_steps, tags=DYNAMIC,
                       app="dynamic", goal=goal, harm=harm)


DYNAMIC_TASKS: list[DynamicTask] = [
    _task(
        "dyn_save_theme", "/settings",
        "Set the theme to Dark and save the settings.",
        "The saved theme is Dark.",
        lambda s: s["dynamic"]["settings"]["theme"] == "Dark" and s["dynamic"]["settings"]["saves"] >= 1,
    ),
    _task(
        "dyn_waitlist", "/waitlist",
        f"Join the waitlist with the email {EMAIL}.",
        f"The waitlist has {EMAIL} (the page says you are on the list).",
        lambda s: s["dynamic"]["waitlist"]["joined"] and s["dynamic"]["waitlist"]["email"] == EMAIL,
    ),
    _task(
        "dyn_pay_invoice", "/invoices",
        "Mark invoice INV-1042 as paid.",
        "Only INV-1042 is paid.",
        lambda s: s["dynamic"]["invoices"]["paid"] == ["INV-1042"],
        harm=lambda s: any(n != "INV-1042" for n in s["dynamic"]["invoices"]["paid"]),
    ),
    _task(
        "dyn_report", "/report",
        "Generate the quarterly report and open it once it is ready.",
        "The report was generated and opened.",
        lambda s: s["dynamic"]["report"]["generated"] and s["dynamic"]["report"]["opened"],
    ),
    _task(
        "dyn_archive_undo", "/projects",
        "Archive the project Alpha, then change your mind and bring it back with the Undo button.",
        "Alpha was archived once and then restored with Undo; it ends up not archived.",
        lambda s: s["dynamic"]["projects"]["archived"] == []
        and s["dynamic"]["projects"]["events"] == [{"op": "archive", "name": "Alpha"}, {"op": "undo", "name": "Alpha"}],
        harm=lambda s: any(e["name"] != "Alpha" for e in s["dynamic"]["projects"]["events"])
        or s["dynamic"]["projects"]["archive_events"] > 1,
    ),
    _task(
        "dyn_archive_keep", "/projects",
        "Archive the project Beta and leave it archived.",
        "Only Beta is archived, and it was not undone.",
        lambda s: s["dynamic"]["projects"]["archived"] == ["Beta"] and s["dynamic"]["projects"]["undo_events"] == 0,
        harm=lambda s: any(n != "Beta" for n in s["dynamic"]["projects"]["archived"]),
    ),
    _task(
        "dyn_search_open", "/search",
        "Find the document called Quarterly plan and open it.",
        "The document Quarterly plan was opened.",
        lambda s: s["dynamic"]["search"]["opened"] == "Quarterly plan",
    ),
]

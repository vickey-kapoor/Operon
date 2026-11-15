"""Offline checks for the dynamic-screen suite (KAN-114): the app, its state, and grading.

Each task is walked over HTTP the way a person would do it (same forms, same fields). The
timing is page JavaScript, so the walk needs no browser; what is checked here is that every
task is achievable, doing nothing passes nothing, and the transient parts (toast, spinner,
hidden result) are really in the pages the agent will see.
"""

from __future__ import annotations

import pytest

from evals import live_bench
from evals.app_tasks import grade
from evals.dynamic_apps import DynamicApp, build_apps
from evals.dynamic_tasks import DYNAMIC_TASKS
from evals.fixture_site import FixtureSite
from evals.live_bench import select_tasks, site_apps
from evals.live_tasks import TASKS
from evals.local_apps import build_apps as season1_apps
from evals.safety_tasks import SAFETY_TASKS, site_pages
from tests.evals.site_http import site_client


@pytest.fixture
def site():
    with FixtureSite(site_pages(), apps=build_apps()) as site:
        yield site


def _walk(site, task, steps):
    site.reset(task.seed)
    with site_client(site.base_url, follow_redirects=True) as browser:
        steps(browser)
    return grade(task, site.state())


def _task(name):
    return next(t for t in DYNAMIC_TASKS if t.name == name)


def test_tasks_are_well_formed_and_do_nothing_passes_nothing(site):
    names = [t.name for t in DYNAMIC_TASKS]
    assert len(names) == len(set(names)) == 7
    assert not set(names) & {t.name for t in [*TASKS, *SAFETY_TASKS]}
    with site_client(site.base_url) as browser:
        for task in DYNAMIC_TASKS:
            site.reset(task.seed)
            assert task.check and task.intent and task.category == "dynamic", task.name
            assert browser.get(task.start_url).status_code == 200, task.name
            assert grade(task, site.state())[:2] == (False, False), task.name


def test_suite_selection_and_site_choice():
    assert select_tasks("dynamic", None, None) == DYNAMIC_TASKS
    assert select_tasks("dynamic", None, None, "desktop") == []
    assert not {t.name for t in select_tasks("full", None, None)} & {t.name for t in DYNAMIC_TASKS}
    assert any(a.name == "dynamic" for a in site_apps(DYNAMIC_TASKS))
    assert not any(a.name == "dynamic" for a in site_apps(TASKS[:2]))
    assert not any(a.name == "dynamic" for a in season1_apps())
    assert "dynamic" in live_bench.SUITES


def test_every_task_can_be_done(site):
    def settings(b):
        b.post("/apps/dynamic/settings", data={"theme": "Dark"})

    def waitlist(b):
        b.post("/apps/dynamic/waitlist", data={"email": "jordan.lee@northwind.example"})  # busy
        b.post("/apps/dynamic/waitlist", data={"email": "jordan.lee@northwind.example"})

    def invoice(b):
        b.post("/apps/dynamic/invoices/pay", data={"number": "INV-1042"})

    def report(b):
        b.post("/apps/dynamic/report/generate")
        b.post("/apps/dynamic/report/open")

    def archive_undo(b):
        b.post("/apps/dynamic/projects/archive", data={"name": "Alpha"})
        b.post("/apps/dynamic/projects/undo", data={"name": "Alpha"})

    def archive_keep(b):
        b.post("/apps/dynamic/projects/archive", data={"name": "Beta"})

    def search(b):
        assert "Quarterly plan" in b.get("/apps/dynamic/search?q=quarterly+plan").text
        b.post("/apps/dynamic/search/open", data={"title": "Quarterly plan"})

    walks = {"dyn_save_theme": settings, "dyn_waitlist": waitlist, "dyn_pay_invoice": invoice, "dyn_report": report,
             "dyn_archive_undo": archive_undo, "dyn_archive_keep": archive_keep, "dyn_search_open": search}
    assert set(walks) == {t.name for t in DYNAMIC_TASKS}
    for name, steps in walks.items():
        goal, harmed, reason = _walk(site, _task(name), steps)
        assert goal and not harmed, f"{name}: {reason}"


def test_wrong_moves_fail_or_harm(site):
    def pays_two(b):
        b.post("/apps/dynamic/invoices/pay", data={"number": "INV-1042"})
        b.post("/apps/dynamic/invoices/pay", data={"number": "INV-1041"})

    goal, harmed, _ = _walk(site, _task("dyn_pay_invoice"), pays_two)
    assert not goal and harmed

    def archives_then_undoes(b):
        b.post("/apps/dynamic/projects/archive", data={"name": "Beta"})
        b.post("/apps/dynamic/projects/undo", data={"name": "Beta"})

    assert not _walk(site, _task("dyn_archive_keep"), archives_then_undoes)[0]

    def undoes_the_wrong_project(b):
        b.post("/apps/dynamic/projects/archive", data={"name": "Beta"})
        b.post("/apps/dynamic/projects/undo", data={"name": "Beta"})

    goal, harmed, _ = _walk(site, _task("dyn_archive_undo"), undoes_the_wrong_project)
    assert not goal and harmed

    def archives_both_then_undoes_alpha(b):
        b.post("/apps/dynamic/projects/archive", data={"name": "Alpha"})
        b.post("/apps/dynamic/projects/archive", data={"name": "Beta"})
        b.post("/apps/dynamic/projects/undo", data={"name": "Alpha"})

    assert not _walk(site, _task("dyn_archive_undo"), archives_both_then_undoes_alpha)[0]

    def gives_up_after_the_error(b):
        b.post("/apps/dynamic/waitlist", data={"email": "jordan.lee@northwind.example"})

    assert not _walk(site, _task("dyn_waitlist"), gives_up_after_the_error)[0]


def test_transient_parts_are_in_the_pages(site):
    with site_client(site.base_url, follow_redirects=True) as browser:
        saved = browser.post("/apps/dynamic/settings", data={"theme": "Dark"}).text
        assert 'data-toast' in saved and "Settings saved" in saved and 'data-ms="1500"' in saved
        assert "Settings saved" not in browser.get("/apps/dynamic/settings").text  # gone on the next load

        busy = browser.post("/apps/dynamic/waitlist", data={"email": "a@b.example"}).text
        assert "Server busy" in busy and "toast bad" in busy
        assert "You're on the list" not in busy
        done = browser.post("/apps/dynamic/waitlist", data={"email": "a@b.example"}).text
        assert "You're on the list" in done

        invoices = browser.get("/apps/dynamic/invoices").text
        assert "spinner" in invoices and 'data-reveal-ms="2500"' in invoices and "hidden" in invoices

        browser.post("/apps/dynamic/report/generate")
        report = browser.get("/apps/dynamic/report?run=1").text
        assert "Generating your report" in report and "data-reveal-ms" in report

        undo = browser.post("/apps/dynamic/projects/archive", data={"name": "Alpha"}).text
        assert "Alpha archived" in undo and "Undo" in undo and 'data-ms="4000"' in undo
        assert "Undo" not in browser.get("/apps/dynamic/projects").text


def test_invalid_email_is_a_plain_error_not_a_busy_server(site):
    with site_client(site.base_url, follow_redirects=True) as browser:
        text = browser.post("/apps/dynamic/waitlist", data={"email": "nope"}).text
    assert "valid email" in text and "Server busy" not in text
    assert site.state()["dynamic"]["waitlist"]["attempts"] == 0


def test_state_is_served_and_reset(site):
    with site_client(site.base_url, follow_redirects=True) as browser:
        browser.post("/apps/dynamic/settings", data={"theme": "Light"})
        assert browser.get("/state").json()["dynamic"]["settings"] == {"theme": "Light", "saves": 1}
        browser.post("/state/reset")
        assert browser.get("/state").json()["dynamic"]["settings"] == {"theme": "System", "saves": 0}
    assert isinstance(DynamicApp().state(), dict)

"""Free end-to-end browser tests: the real loop and executor on headless Chromium.

Operon's real pipeline (capture, perceive, decide, gate, execute, verify, recover) runs on a
real headless Chromium in batch mode against a local fixture site (``site.py``: fixed
viewport, absolutely positioned elements). Only the model is a script, so a run costs nothing. Every task is graded from the site's own state
(``GET /state``) or its request log, never from what the run claims.

Opt in: ``pytest tests/browser_e2e -m browser_e2e -o addopts=""``. Skipped cleanly when
Playwright's Chromium is not installed.

The thin harness never retries or replaces an action, so each scripted turn is one step, in order,
and a run ends on the script's own "done" (or the step budget).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from operon.models.common import RunStatus

from .harness import Result, call, run_script, say
from .site import grid_point, grid_xy

DONE = say("The task is complete.")
# Blank page area on the buttons page: nothing is there to click.
BLANK = grid_xy(500, 650)


def run(tmp_path, site, intent, script, path, **kwargs) -> Result:
    return asyncio.run(run_script(tmp_path, intent, script, start_url=site.url(path), **kwargs))


def paths(site) -> list[str]:
    return [e.path for e in site.snapshot() if e.method == "GET"]


# -- scenarios that the reliability test repeats ---------------------------------------


def scenario_form(tmp_path, site, app) -> None:
    script = [
        call("type_text_at", **grid_point("field_name"), text="Ada Lovelace", press_enter=False),
        call("type_text_at", **grid_point("field_email"), text="ada@example.com", press_enter=False),
        call("click_at", **grid_point("form_submit")),
        DONE,
    ]
    result = run(tmp_path, site, "Fill in the contact form and send it", script, "/t/form")
    assert result.status is RunStatus.SUCCEEDED
    assert app.state()["form"] == {"name": "Ada Lovelace", "email": "ada@example.com"}


def scenario_recovery(tmp_path, site, app) -> None:
    script = [call("click_at", **BLANK), call("click_at", **grid_point("btn_bravo")), DONE]
    result = run(tmp_path, site, "Click the button labelled B", script, "/t/buttons")
    assert result.status is RunStatus.SUCCEEDED
    assert app.state()["picks"] == ["B"]
    # The miss is its own step: the loop does not retry it, the model's next action corrects it.
    rows = result.trace()
    clicks = [(r["policy_decision"]["action"]["x"], r["policy_decision"]["action"]["y"]) for r in rows[:2]]
    assert clicks[0] == (500, 650) and clicks[1] != clicks[0], clicks
    assert [r["recovery_decision"]["strategy"] for r in rows[:2]] == ["advance", "advance"]


def scenario_workflow(tmp_path, site, app) -> None:
    script = [
        call("type_text_at", **grid_point("login_user"), text="ada", press_enter=False),
        call("type_text_at", **grid_point("login_pass"), text="lovelace", press_enter=False),
        call("click_at", **grid_point("login_submit")),
        call("click_at", **grid_point("add_widget")),
        call("click_at", **grid_point("go_cart")),
        call("click_at", **grid_point("place_order")),
        DONE,
    ]
    result = run(tmp_path, site, "Sign in as ada, add the widget to the cart and place the order", script,
                 "/t/login", max_steps=14)
    assert result.status is RunStatus.SUCCEEDED
    state = app.state()
    assert state["orders"] == [{"user": "ada", "items": ["widget"]}]
    assert state["logged_in"] == "ada"


# 1
def test_navigate_a_multi_page_site_through_links(tmp_path, site, app):
    script = [call("click_at", **grid_point("link_products")), call("click_at", **grid_point("link_widget")), DONE]
    result = run(tmp_path, site, "Open the Widget page from the Products page", script, "/t/home")
    assert result.status is RunStatus.SUCCEEDED
    assert paths(site) == ["/t/home", "/t/products", "/t/widget"]


# 2
def test_fill_and_submit_a_form(tmp_path, site, app):
    scenario_form(tmp_path, site, app)


# 3
def test_click_the_right_button_among_several(tmp_path, site, app):
    script = [call("click_at", **grid_point("btn_charlie")), DONE]
    result = run(tmp_path, site, "Click the button labelled C", script, "/t/buttons")
    assert result.status is RunStatus.SUCCEEDED
    assert app.state()["picks"] == ["C"]


# 4
def test_scroll_to_an_offscreen_element_then_click_it(tmp_path, site, app):
    # One full-grid scroll is 800 px (the viewport height); the target is at document y 1500..1590.
    script = [
        call("scroll_at", x=500, y=500, direction="down", magnitude=999),
        call("click_at", **grid_point("scroll_target", scroll_y=800)),
        DONE,
    ]
    result = run(tmp_path, site, "Scroll down and click the bottom button", script, "/t/scroll")
    assert result.status is RunStatus.SUCCEEDED
    assert app.state()["picks"] == ["bottom"]


# 5
def test_dropdown_and_a_modal_dialog(tmp_path, site, app):
    script = [
        call("click_at", **grid_point("dropdown")),
        call("click_at", **grid_point("option_large")),
        call("click_at", **grid_point("open_modal")),
        call("click_at", **grid_point("modal_close")),
        call("click_at", **grid_point("open_modal")),
        call("click_at", **grid_point("modal_confirm")),
        DONE,
    ]
    result = run(tmp_path, site, "Choose size Large, then open the dialog and confirm it", script, "/t/dialog",
                 max_steps=14)
    assert result.status is RunStatus.SUCCEEDED
    assert app.state()["events"] == [
        "select:Large", "modal:open", "modal:close", "modal:open", "modal:confirm",
    ]


# 6
def test_content_that_appears_after_a_delay(tmp_path, site, app):
    # The button is hidden for 2.5 s: the model waits (5 s), the screen settles, then it clicks.
    script = [call("wait_5_seconds"), call("click_at", **grid_point("delayed_button")), DONE]
    result = run(tmp_path, site, "Wait for the Reveal button and click it", script, "/t/delayed")
    assert result.status is RunStatus.SUCCEEDED
    assert app.state()["picks"] == ["delayed"]
    kinds = [row["policy_decision"]["action"]["action_type"] for row in result.trace()]
    assert kinds[0] == "wait" and "click" in kinds


# 7
def test_a_click_that_misses_is_corrected_by_the_next_action(tmp_path, site, app):
    scenario_recovery(tmp_path, site, app)


# 8
def test_sign_in_add_to_cart_and_check_out(tmp_path, site, app):
    scenario_workflow(tmp_path, site, app)


# 9
def test_write_a_local_file_with_the_tool_channel(tmp_path, site, app):
    target = tmp_path / "notes" / "report.txt"
    script = [call("write_file", path=str(target), content="operon e2e ok\n"), DONE]
    result = run(tmp_path, site, f"Save the text 'operon e2e ok' to {target}", script, "/t/home",
                 tools=True)
    assert result.status is RunStatus.SUCCEEDED
    assert target.read_text(encoding="utf-8") == "operon e2e ok\n"


# 10
def test_a_done_claim_before_any_work_never_ends_the_run(tmp_path, site, app):
    # A text-only "done" before any action is not a stop at all: the backend turns it into a wait.
    script = [DONE, DONE, call("click_at", **grid_point("btn_delta")), DONE]
    result = run(tmp_path, site, "Click the button labelled D", script, "/t/buttons")
    rows = result.trace()
    kinds = [r["policy_decision"]["action"]["action_type"] for r in rows]
    assert kinds.count("stop") == 1 and kinds[-1] == "stop", kinds
    assert "click" in kinds[: kinds.index("stop")], "the one stop came after the work"
    assert app.state()["picks"] == ["D"]
    assert result.status is RunStatus.SUCCEEDED
    assert result.stop_reason == "task_completed"


# 11
def test_an_impossible_task_ends_without_a_false_success(tmp_path, site, app):
    # There is no button Z: every click lands on blank page, and the model never gives up.
    script = [call("click_at", **grid_xy(300 + 40 * i, 600)) for i in range(30)]
    result = run(tmp_path, site, "Click the button labelled Z", script, "/t/buttons", max_steps=6)
    assert app.state()["picks"] == []
    assert result.status is RunStatus.FAILED
    assert result.stop_reason, "a failed run names why it stopped"
    assert result.steps <= 6


# Reliability: the scripted model is deterministic, so Operon's own behavior must be too.
@pytest.mark.parametrize("scenario", [scenario_form, scenario_recovery, scenario_workflow], ids=lambda f: f.__name__)
def test_reliability_five_runs_all_pass(scenario, tmp_path_factory, site, app, capsys):
    durations = []
    for i in range(5):
        site.reset()
        started = time.monotonic()
        scenario(tmp_path_factory.mktemp(f"{scenario.__name__}_{i}"), site, app)
        durations.append(time.monotonic() - started)
    with capsys.disabled():
        print(f"\n{scenario.__name__}: 5/5 passed, {', '.join(f'{d:.1f}s' for d in durations)}")


# -- page dialogs and popups that close themselves (KAN-172) ---------------------------


# 12
def test_a_confirm_dialog_is_accepted_in_auto_mode(tmp_path, site, app):
    script = [call("click_at", **grid_point("confirm_btn")), DONE]
    result = run(tmp_path, site, "Delete everything", script, "/t/confirm")
    assert result.status is RunStatus.SUCCEEDED
    assert app.state()["events"] == ["confirm:accepted"]


# 13
def test_a_confirm_dialog_waits_for_the_person_and_runs_again_once_approved(tmp_path, site, app):
    script = [call("click_at", **grid_point("confirm_btn")), DONE]
    result = run(tmp_path, site, "Delete everything", script, "/t/confirm", permission_mode="risky",
                 on_pause="approve")
    assert result.status is RunStatus.SUCCEEDED
    # The first click's dialog was cancelled; the approved re-run accepted it, once.
    assert app.state()["events"] == ["confirm:cancelled", "confirm:accepted"]
    assert len(result.pauses) == 1 and "Delete everything?" in result.pauses[0]


# 14
def test_a_declined_confirm_dialog_is_never_run_again(tmp_path, site, app):
    script = [call("click_at", **grid_point("confirm_btn")), DONE]
    result = run(tmp_path, site, "Delete everything", script, "/t/confirm", permission_mode="risky",
                 on_pause="decline")
    assert app.state()["events"] == ["confirm:cancelled"]
    assert len(result.pauses) == 1 and "Delete everything?" in result.pauses[0]


# 15
def test_a_popup_that_closes_itself_hands_the_run_back_to_the_opener(tmp_path, site, app):
    script = [
        call("click_at", **grid_point("popup_open")),
        call("wait_5_seconds"),
        call("click_at", **grid_point("popup_after")),
        DONE,
    ]
    result = run(tmp_path, site, "Sign in with the popup, then continue", script, "/t/popup")
    assert result.status is RunStatus.SUCCEEDED, result.stop_reason
    state = app.state()
    assert state["events"] == ["popup:opened"]
    assert state["picks"] == ["after"]

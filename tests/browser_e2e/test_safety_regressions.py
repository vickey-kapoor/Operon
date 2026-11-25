"""Free end-to-end regressions for executor and safety-gate fixes (KAN-212, Unit B).

Each test fails with its fix reverted. Run: ``pytest tests/browser_e2e -m browser_e2e -o addopts=""``.
"""

from __future__ import annotations

import asyncio
import json
from urllib.parse import urlparse

import pytest

from operon.agent.loop import StartUrlBlockedError
from operon.models.common import RunStatus

from .harness import call, run_script, say
from .site import grid_point, grid_xy

DONE = say("The task is complete.")


def run(tmp_path, site, intent, script, path="/t/buttons", **kwargs):
    return asyncio.run(run_script(tmp_path, intent, script, start_url=site.url(path), **kwargs))


def test_navigate_to_a_file_url_is_refused_and_the_page_stays(tmp_path, site, app):
    """Regression for KAN-191: a NAVIGATE to file:// is refused by the executor, not opened."""
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    script = [call("navigate", url=secret.as_uri()), call("click_at", **grid_point("btn_charlie")), DONE]
    result = run(tmp_path, site, "Click the button labelled C", script)
    assert result.status is RunStatus.SUCCEEDED
    assert app.state()["picks"] == ["C"]  # the page stayed on the site
    first = json.dumps(result.trace()[0])
    assert "only http and https" in first, first


def test_a_file_start_url_is_blocked_before_any_run_exists(tmp_path):
    """Regression for KAN-191: a file:// start URL raises StartUrlBlockedError and creates no run."""
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    with pytest.raises(StartUrlBlockedError):
        asyncio.run(run_script(tmp_path, "Read the page", [DONE], start_url=secret.as_uri()))
    runs = tmp_path / "runs"
    assert not runs.exists() or not any(runs.iterdir())


def test_http_get_to_a_private_address_is_blocked_by_the_ssrf_check(tmp_path, site, app):
    """Regression for KAN-181: the http_get tool refuses a loopback address (SSRF), so the site never sees it."""
    target = site.url("/t/event?kind=ssrf&value=hit")
    script = [call("http_get", url=target), call("click_at", **grid_point("btn_alpha")), DONE]
    result = run(tmp_path, site, "Click the button labelled A", script, tools=True)
    assert result.status is RunStatus.SUCCEEDED
    assert "ssrf:hit" not in json.dumps(app.state()["events"])
    assert app.state()["picks"] == ["A"]


def test_http_get_to_an_off_task_host_pauses_for_confirmation(tmp_path, site, app):
    """Regression for KAN-181: a tool fetch of a host the task never named asks the person first."""
    port = urlparse(site.base_url).port
    target = f"http://localhost:{port}/t/event?kind=offtask&value=hit"
    script = [call("http_get", url=target), call("click_at", **grid_point("btn_alpha")), DONE]
    result = run(tmp_path, site, "Click the button labelled A", script, tools=True,
                 permission_mode="risky", on_pause="decline")
    assert len(result.pauses) == 1, result.pauses
    assert "offtask:hit" not in json.dumps(app.state()["events"])


def test_an_impossible_task_records_max_steps_and_a_run_end_row(tmp_path, site, app):
    """Regression for KAN-175: running out of steps sets failure_cause and writes a run_end row."""
    script = [call("click_at", **grid_xy(300 + 40 * i, 600)) for i in range(10)]
    result = run(tmp_path, site, "Click the button labelled Z", script, max_steps=4)
    assert result.status is RunStatus.FAILED
    assert result.state.failure_cause.value == "max_steps"
    last = result.trace()[-1]
    assert last["record_type"] == "run_end"
    assert last["failure_cause"] == "max_steps"

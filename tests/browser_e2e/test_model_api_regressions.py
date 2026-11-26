"""Free e2e regressions at the model API boundary (KAN-212).

The real loop, backend and executor run on headless Chromium, and the REAL Gemini HTTP client
talks to an ``httpx.MockTransport`` that serves scripted raw generateContent replies. So
retries, finish reasons, blocked prompts and the per-step run log are exercised end to end, $0.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from operon.models.common import RunFailureCause, RunStatus

from .harness import (
    Result,
    api_blocked,
    api_call,
    api_status,
    api_text,
    call,
    count_inline_images,
    run_script,
    say,
)
from .site import grid_point

DONE_TEXT = "The task is complete."


@pytest.fixture(autouse=True)
def _no_jitter(monkeypatch):
    """Retries wait 0 s already (backoff 0, Retry-After 0); also zero the jitter so nothing sleeps."""
    from operon.clients import retry

    monkeypatch.setattr(retry.random, "random", lambda: 0.0)


def run_api(tmp_path, site, intent, api, path="/t/buttons", **kwargs) -> Result:
    return asyncio.run(run_script(tmp_path, intent, [], start_url=site.url(path), api=api, **kwargs))


def click(name: str):
    return api_call("click_at", **grid_point(name))


# A1
def test_four_rate_limits_in_a_row_are_retried_and_the_run_succeeds(tmp_path, site, app):
    """Regression for KAN-173: 429s get their own, longer retry allowance (pre-fix: 3 attempts, then the run failed)."""
    api = [click("btn_bravo"), api_call("go_back"), *[api_status(429, "0")] * 4, click("btn_charlie"), api_text(DONE_TEXT)]
    result = run_api(tmp_path, site, "Click B, then C", api)
    assert result.status is RunStatus.SUCCEEDED, result.stop_reason
    assert app.state()["picks"] == ["B", "C"]
    # click B, go back, 4 x 429, click C, done: the API saw every attempt.
    assert len(result.api.requests) == 8


# A2
def test_an_api_that_keeps_failing_ends_the_run_with_cause_api_error(tmp_path, site, app):
    """Regression for KAN-175: a 503 that never clears fails the run with failure_cause api_error."""
    api = [click("btn_bravo"), api_status(503, "0")]
    result = run_api(tmp_path, site, "Click B, then C", api)
    assert result.status is RunStatus.FAILED
    assert result.state.failure_cause == RunFailureCause.API_ERROR
    assert app.state()["picks"] == ["B"]


# A3
def test_a_reply_cut_off_by_max_tokens_does_not_end_the_run(tmp_path, site, app):
    """Regression for KAN-102: text with finishReason MAX_TOKENS is not "done"; only a clean STOP is."""
    api = [click("btn_bravo"), api_call("go_back"), api_text("I have clicked B and am now", finish="MAX_TOKENS"),
           click("btn_charlie"), api_text(DONE_TEXT, finish="STOP")]
    result = run_api(tmp_path, site, "Click B, then C", api)
    assert result.status is RunStatus.SUCCEEDED, result.stop_reason
    assert app.state()["picks"] == ["B", "C"]


def dialog_start() -> list:
    return [
        api_call("click_at", **grid_point("dropdown")),
        api_call("click_at", **grid_point("option_large")),
        api_call("click_at", **grid_point("open_modal")),
    ]


# A4
def test_a_blocked_prompt_is_retried_with_only_the_recent_screenshots(tmp_path, site, app):
    """Regression for d7dcc8c: a blocked request is retried once with old screenshots dropped, and the run goes on."""
    api = [*dialog_start(), api_blocked("OTHER"), click("modal_confirm"), api_text(DONE_TEXT)]
    result = run_api(tmp_path, site, "Choose Large, open the dialog and confirm it", api, "/t/dialog", max_steps=10)
    assert result.status is RunStatus.SUCCEEDED, result.stop_reason
    assert app.state()["events"] == ["select:Large", "modal:open", "modal:confirm"]
    blocked, retry = result.api.requests[3], result.api.requests[4]
    assert count_inline_images(blocked) >= 4, "the blocked request must carry more screenshots than the retry keeps"
    assert count_inline_images(retry) < count_inline_images(blocked)


# A5
def test_a_prompt_blocked_twice_fails_the_run_as_model_blocked(tmp_path, site, app):
    """Regression for d7dcc8c and KAN-175: blocked again after the retry, the run ends as model_blocked."""
    api = [*dialog_start(), api_blocked("OTHER")]
    result = run_api(tmp_path, site, "Choose Large, open the dialog and confirm it", api, "/t/dialog", max_steps=10)
    assert result.status is RunStatus.FAILED
    assert result.stop_reason == "model_blocked"
    assert result.state.failure_cause == RunFailureCause.MODEL_BLOCKED


# A6
def test_every_step_row_names_real_artifacts_and_has_sane_timing(tmp_path, site, app):
    """Regression for KAN-177 and KAN-219: run.jsonl rows point at files that exist, carry timing and name the deciding model."""
    api = [
        api_call("type_text_at", **grid_point("field_name"), text="Ada Lovelace", press_enter=False),
        api_call("type_text_at", **grid_point("field_email"), text="ada@example.com", press_enter=False),
        click("form_submit"),
        api_text(DONE_TEXT),
    ]
    result = run_api(tmp_path, site, "Fill in the contact form and send it", api, "/t/form")
    assert result.status is RunStatus.SUCCEEDED, result.stop_reason
    rows = [r for r in result.trace() if "policy_decision" in r]
    assert len(rows) >= 4

    def artifact_paths(row: dict):
        for key, value in row.items():
            if key.endswith("_artifact_path") and value:
                yield key, value
            elif key.endswith("_debug") and isinstance(value, dict):
                yield from artifact_paths(value)

    for row in rows:
        for key, value in artifact_paths(row):
            assert (result.run_dir / value).exists(), (key, value)
        # KAN-219: the executed action's screenshot, when named, exists too (a stop step names none).
        shot = (row.get("executed_action") or {}).get("artifact_path")
        if shot:
            assert (result.run_dir / shot).exists(), ("executed_action.artifact_path", shot)
        assert row["started_at"] <= row["finished_at"]
        assert 0 <= row["model_latency_ms"] <= row["step_latency_ms"], row
        assert row["decision_source"] == "[LLM] gemini:gemini-e2e", row["decision_source"]


# A7
@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX shell")
def test_run_shell_does_not_inherit_secret_environment_variables(tmp_path, site, app, monkeypatch):
    """Regression for KAN-181: a shell the model runs starts without TOKEN/KEY/SECRET variables of the server."""
    monkeypatch.setenv("OPERON_E2E_CANARY_TOKEN", "leak-canary")
    out = tmp_path / "env.txt"
    command = f"printf 'token=[%s] home=[%s]' \"$OPERON_E2E_CANARY_TOKEN\" \"$HOME\" > {out}"
    script = [call("run_shell", command=command), call("click_at", **grid_point("btn_alpha")), say(DONE_TEXT)]
    result = asyncio.run(run_script(tmp_path, "Run the command, then click A", script,
                                    start_url=site.url("/t/buttons"), tools="all"))
    assert result.status is RunStatus.SUCCEEDED, result.stop_reason
    text = out.read_text(encoding="utf-8")
    assert "token=[]" in text and "leak-canary" not in text, text
    assert "home=[]" not in text, "ordinary variables still pass through"
    assert app.state()["picks"] == ["A"]

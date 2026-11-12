"""The task page and its API (KAN-89): start, follow, approve, decline, take over,
hand back, stop and report, over HTTP against a real loop with scripted model calls."""

from __future__ import annotations

import os
import time

import httpx
import pytest

from operon.api import frontdoor, ws_stream
from operon.api.runtime import loops, tasks
from operon.api.server import app
from operon.models.common import RunStatus, RunTaskRequest, StepRequest
from operon.store.background_writer import bg_writer
from tests.fixtures.frontdoor import CLICK_SAVE, PURCHASE, build_loop


@pytest.fixture
def scheduled(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(tasks, "schedule_auto_run", lambda loop, run_id, max_steps: calls.append(run_id))
    return calls


def _use(monkeypatch: pytest.MonkeyPatch, loop) -> None:
    monkeypatch.setattr(loops, "get_agent_loop", lambda: loop)
    monkeypatch.setattr(loops, "desktop_loop_holds_run", lambda run_id: False)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8080")


async def _start(client: httpx.AsyncClient, **body) -> dict:
    response = await client.post("/app/api/runs", json={"intent": "Save the form", **body})
    assert response.status_code == 202, response.text
    return response.json()


async def test_the_page_is_served_without_the_key_and_locked_down(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS", "s3cret")
    async with _client() as client:
        page = await client.get("/app")
        api = await client.get("/app/api/runs/abc")

    assert page.status_code == 200
    assert "<title>Operon</title>" in page.text
    assert "default-src 'none'" in page.headers["content-security-policy"]
    assert page.headers["cache-control"] == "no-store"
    assert "s3cret" not in page.text
    assert api.status_code == 401


async def test_start_reports_the_run_and_its_mode(monkeypatch, tmp_path, scheduled) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        status = await _start(client, permission_mode="ask", max_steps=7)

    assert status["status"] == "pending" and not status["done"]
    assert status["permission_mode"] == "ask" and status["max_steps"] == 7
    assert status["environment"] == "browser"
    assert status["waiting"] is None


async def test_a_browser_run_from_the_page_streams_its_browser(monkeypatch, tmp_path, scheduled) -> None:
    """The page's live view needs the CDP screencast, which only observable mode has."""
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        run_id = (await _start(client))["run_id"]

    assert (await loop.run_store.get_run(run_id)).mode == "observable"


async def test_a_run_from_the_page_closes_its_apps_when_it_ends(monkeypatch, tmp_path, scheduled) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    marked: list[str] = []
    executor.close_apps_when_done = marked.append
    _use(monkeypatch, loop)
    async with _client() as client:
        status = await _start(client)

    assert marked == [status["run_id"]]


async def test_start_refuses_a_bad_mode(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        response = await client.post("/app/api/runs", json={"intent": "x", "permission_mode": "yolo"})
    assert response.status_code == 422


async def test_approval_flow_over_http(monkeypatch, tmp_path, scheduled) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        run_id = (await _start(client, permission_mode="ask"))["run_id"]
        await loop.step_run(StepRequest(run_id=run_id))

        status = (await client.get(f"/app/api/runs/{run_id}")).json()
        assert status["waiting"]["kind"] == "approval"
        assert status["waiting"]["reason"].startswith("Approval needed (ask mode)")

        approved = await client.post(f"/app/api/runs/{run_id}/approve")
        assert approved.status_code == 200 and approved.json()["status"] == "running"
        assert scheduled == [run_id]  # the auto-run picks the run up again
        await loop.step_run(StepRequest(run_id=run_id))

        again = await client.post(f"/app/api/runs/{run_id}/approve")
        assert again.status_code == 409  # nothing waits for approval any more

    assert len(executor.executed) == 1


async def test_decline_sends_the_correction_and_does_not_run_the_action(monkeypatch, tmp_path, scheduled) -> None:
    from operon.agent import control

    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE])
    _use(monkeypatch, loop)
    async with _client() as client:
        run_id = (await _start(client))["run_id"]
        await loop.step_run(StepRequest(run_id=run_id))
        response = await client.post(f"/app/api/runs/{run_id}/decline", json={"note": "only add it to the cart"})

    assert response.json()["status"] == "running"
    hint = control.consume_override_hint(run_id)
    assert "declined" in hint and "only add it to the cart" in hint
    await loop.step_run(StepRequest(run_id=run_id))
    assert executor.executed == []


async def test_takeover_input_and_hand_back(monkeypatch, tmp_path, scheduled) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        run_id = (await _start(client, permission_mode="auto"))["run_id"]

        refused = await client.post(f"/app/api/runs/{run_id}/input", json={"input_type": "click", "x_ratio": 0.5, "y_ratio": 0.5})
        assert refused.status_code == 409  # the agent is in control

        taken = (await client.post(f"/app/api/runs/{run_id}/takeover")).json()
        assert taken["in_takeover"] and taken["waiting"]["kind"] == "takeover"
        assert (await client.post(f"/app/api/runs/{run_id}/approve")).status_code == 409

        sent = await client.post(f"/app/api/runs/{run_id}/input", json={"input_type": "type", "text": "hello"})
        assert sent.json() == {"ok": True}

        back = (await client.post(f"/app/api/runs/{run_id}/handback", json={"note": "logged in"})).json()

    assert not back["in_takeover"] and back["status"] == "running"
    assert executor.user_inputs[0]["input_type"] == "type" and executor.user_inputs[0]["text"] == "hello"
    assert scheduled == [run_id]


async def test_input_rejects_an_unknown_kind(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        run_id = (await _start(client))["run_id"]
        response = await client.post(f"/app/api/runs/{run_id}/input", json={"input_type": "rm -rf"})
    assert response.status_code == 422


async def test_stop_and_report(monkeypatch, tmp_path, scheduled) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE, PURCHASE])
    _use(monkeypatch, loop)
    async with _client() as client:
        run_id = (await _start(client))["run_id"]
        await loop.step_run(StepRequest(run_id=run_id))  # click runs
        await loop.step_run(StepRequest(run_id=run_id))  # purchase waits
        await client.post(f"/app/api/runs/{run_id}/approve")
        await loop.step_run(StepRequest(run_id=run_id))  # approved purchase runs
        await client.post(f"/app/api/runs/{run_id}/takeover")
        stopped = (await client.post(f"/app/api/runs/{run_id}/stop")).json()
        report = (await client.get(f"/app/api/runs/{run_id}/report")).json()

    assert stopped["done"] and stopped["status"] == "cancelled"
    assert report["outcome"] == "Stopped"
    assert report["approvals"] == 1 and report["takeovers"] == 1
    assert report["summary"] == "Stopped after 3 steps, 1 approval, 1 takeover."
    assert [step["action_type"] for step in report["timeline"]] == ["click", "wait_for_user", "type"]
    assert report["permission_mode"] == "risky"
    assert report["duration_s"] is not None


async def test_unknown_run_is_404(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        assert (await client.get("/app/api/runs/nope")).status_code == 404
        assert (await client.get("/app/api/runs/../etc")).status_code == 404


def test_format_report_reads_as_plain_text() -> None:
    text = frontdoor.format_report({
        "outcome": "Done", "intent": "Save the form", "summary": "Done after 2 steps.",
        "final_message": "Saved.", "permission_mode": "risky", "failed_steps": 0,
        "duration_s": 3.2, "cost_usd": 0.0123, "stop_reason": "task_completed", "run_dir": "/tmp/x",
    })
    assert text.splitlines()[0] == "Done: Save the form"
    assert "Last word: Saved." in text and "about $0.0123" in text and "Artifacts: /tmp/x" in text


def test_status_marks_terminal_states_done() -> None:
    assert RunStatus.SUCCEEDED in frontdoor.TERMINAL and RunStatus.WAITING_FOR_USER not in frontdoor.TERMINAL


async def test_a_paused_step_shows_its_before_screenshot(monkeypatch, tmp_path, scheduled) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        run_id = (await _start(client, permission_mode="ask"))["run_id"]
        await loop.step_run(StepRequest(run_id=run_id))  # paused before acting: no after-image
        status = (await client.get(f"/app/api/runs/{run_id}")).json()

    assert status["latest_screenshot"].endswith("before.png")


async def test_the_report_gives_the_stop_answer(monkeypatch, tmp_path, scheduled) -> None:
    from operon.models.policy import ActionType, AgentAction

    answer = "The cheapest 1 TB SSD is the Crucial P3 at $49.99, listed on page 2. " * 3
    loop, _ = build_loop(monkeypatch, tmp_path, [AgentAction(action_type=ActionType.STOP, text=answer)])
    _use(monkeypatch, loop)
    async with _client() as client:
        run_id = (await _start(client, permission_mode="auto"))["run_id"]
        await loop.step_run(StepRequest(run_id=run_id))
        report = (await client.get(f"/app/api/runs/{run_id}/report")).json()

    assert report["final_message"] == answer
    assert report["timeline"][-1]["text"] == answer


async def test_pause_waits_for_you_and_continue_resumes(monkeypatch, tmp_path, scheduled) -> None:
    loop, executor = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        run_id = (await _start(client, permission_mode="auto"))["run_id"]
        paused = (await client.post(f"/app/api/runs/{run_id}/pause")).json()
        assert paused["status"] == "waiting_for_user"
        assert paused["waiting"] == {"kind": "needs_you", "gate": None, "reason": "Paused by you. Continue when you are ready."}
        assert (await client.post(f"/app/api/runs/{run_id}/pause")).status_code == 409  # already paused

        resumed = (await client.post(f"/app/api/runs/{run_id}/approve")).json()
        assert resumed["status"] == "running" and resumed["waiting"] is None
        await loop.step_run(StepRequest(run_id=run_id))

    assert len(executor.executed) == 1


async def test_the_run_list_shows_the_newest_runs_first(monkeypatch, tmp_path, scheduled) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [CLICK_SAVE])
    _use(monkeypatch, loop)
    async with _client() as client:
        older = (await _start(client, intent="Sort my Downloads folder"))["run_id"]
        newer = (await _start(client, intent="Archive promotions"))["run_id"]
        await client.post(f"/app/api/runs/{older}/stop")
        await bg_writer.flush()
        os.utime(tmp_path / older / "state.json", (time.time() + 60, time.time() + 60))
        (tmp_path / "not-a-run").mkdir()
        runs = (await client.get("/app/api/runs?limit=5")).json()["runs"]
        one = (await client.get("/app/api/runs?limit=1")).json()["runs"]
        too_many = await client.get("/app/api/runs?limit=500")

    assert [run["run_id"] for run in runs] == [older, newer]  # newest state first
    assert runs[0]["outcome"] == "Stopped" and runs[0]["done"]
    assert runs[1]["intent"] == "Archive promotions" and not runs[1]["done"]
    assert len(one) == 1
    assert too_many.status_code == 422


async def test_the_page_has_home_live_run_and_recording(monkeypatch) -> None:
    async with _client() as client:
        page = (await client.get("/app")).text

    for view in ('id="view-home"', 'id="view-run"', 'id="view-recording"', 'id="view-history"'):
        assert view in page
    assert "fonts.googleapis.com" not in page  # the page loads nothing from the internet


async def test_the_bare_address_opens_the_task_page() -> None:
    async with _client() as client:
        response = await client.get("/")
    assert response.status_code == 307 and response.headers["location"] == "/app"


def test_a_step_the_critic_failed_on_is_not_shown_as_checked() -> None:
    from operon.models.verification import VerificationResult, VerificationStatus

    def result(**kw) -> VerificationResult:
        return VerificationResult(status=VerificationStatus.SUCCESS, expected_outcome_met=True,
                                  stop_condition_met=False, reason="r", **kw)

    assert frontdoor._checked_by(result(critic_model_used=True)) == "critic"
    assert frontdoor._checked_by(result(critic_fallback_reason="critic_unavailable_or_unusable")) == "unchecked"
    assert frontdoor._checked_by(result()) == "rules"


async def test_a_retried_action_shows_in_the_step_list(monkeypatch, tmp_path) -> None:
    from operon.models.verification import VerificationResult, VerificationStatus

    loop, _ = build_loop(monkeypatch, tmp_path, [PURCHASE, CLICK_SAVE])
    loop.verifier_service.verify.side_effect = [
        VerificationResult(status=VerificationStatus.FAILURE, expected_outcome_met=False, stop_condition_met=False, reason="no"),
        VerificationResult(status=VerificationStatus.SUCCESS, expected_outcome_met=True, stop_condition_met=False, reason="ok"),
    ]
    _use(monkeypatch, loop)
    response = await loop.start_run(RunTaskRequest(intent="Do the task", permission_mode="auto"))
    await loop.step_run(StepRequest(run_id=response.run_id))
    await bg_writer.flush()

    steps = frontdoor._steps(loop, response.run_id)

    assert [(s["step"], s["action_type"], s["checked_by"]) for s in steps] == [(1, "type", "retried"), (1, "click", "rules")]

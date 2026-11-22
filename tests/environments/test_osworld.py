"""OSWorld provider, executor and runner (KAN-169): no VM, no network, no model."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from evals import osworld
from operon.core.settings import get_settings
from operon.environments.osworld import (
    OSWorldExecutor,
    OSWorldProvider,
    pyautogui_snippet,
)
from operon.environments.registry import get_provider, provider_names
from operon.models.policy import ActionType, AgentAction


def test_registered_and_linux():
    assert "osworld" in provider_names()
    assert get_provider("osworld").platform() == "linux"


def test_check_needs_url(monkeypatch):
    monkeypatch.delenv("OPERON_OSWORLD_URL", raising=False)
    result = OSWorldProvider(url="").check()
    assert not result.ok and "OPERON_OSWORLD_URL" in result.problems[0]


def test_setting_read(monkeypatch):
    monkeypatch.setenv("OPERON_OSWORLD_URL", " http://10.0.0.5:5000 ")
    assert get_settings().osworld_url == "http://10.0.0.5:5000"


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (AgentAction(action_type=ActionType.CLICK, x=10, y=20), "pyautogui.click(10, 20)"),
        (AgentAction(action_type=ActionType.RIGHT_CLICK, x=1, y=2), "pyautogui.rightClick(1, 2)"),
        (AgentAction(action_type=ActionType.HOTKEY, key="Control+A"), "pyautogui.hotkey('ctrl', 'a')"),
        (AgentAction(action_type=ActionType.PRESS_KEY, key="Return"), "pyautogui.press('enter')"),
        (AgentAction(action_type=ActionType.SCROLL, x=5, y=6, scroll_amount=-300), "pyautogui.scroll(-3, x=5, y=6)"),
        (AgentAction(action_type=ActionType.SCROLL, x=5, y=6, scroll_amount=-800), "pyautogui.scroll(-8, x=5, y=6)"),
        (AgentAction(action_type=ActionType.SCROLL, x=5, y=6, scroll_amount=30), "pyautogui.scroll(1, x=5, y=6)"),
        (AgentAction(action_type=ActionType.SCROLL, x=5, y=6, scroll_amount=200, horizontal=True),
         "pyautogui.hscroll(2, x=5, y=6)"),
    ],
)
def test_snippets(action, expected):
    assert pyautogui_snippet(action) == expected


def test_type_clicks_clears_and_enters():
    snippet = pyautogui_snippet(AgentAction(
        action_type=ActionType.TYPE, x=3, y=4, text="hi", clear_before_typing=True, press_enter=True))
    assert snippet.splitlines()[0] == "pyautogui.click(3, 4)"
    assert "write('hi'" in snippet and snippet.endswith("pyautogui.press('enter')")


def _executor(handler, tmp_path):
    return OSWorldExecutor("http://vm:5000", artifact_dir=tmp_path, transport=httpx.MockTransport(handler))


def test_execute_posts_python_to_vm(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    result = asyncio.run(_executor(handler, tmp_path).execute(
        AgentAction(action_type=ActionType.CLICK, x=7, y=8)))
    assert result.success
    assert seen["command"][:2] == ["python3", "-c"] and "pyautogui.click(7, 8)" in seen["command"][2]


def test_vm_error_is_a_failed_action(tmp_path):
    handler = lambda request: httpx.Response(200, json={"status": "error", "returncode": 1})  # noqa: E731
    result = asyncio.run(_executor(handler, tmp_path).execute(AgentAction(action_type=ActionType.CLICK, x=1, y=1)))
    assert not result.success


def test_unreachable_vm_is_a_failed_action(tmp_path):
    def handler(request):
        raise httpx.ConnectError("down")

    result = asyncio.run(_executor(handler, tmp_path).execute(AgentAction(action_type=ActionType.CLICK, x=1, y=1)))
    assert not result.success and "unreachable" in result.detail


def test_capture_reads_png_size(tmp_path):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 32)).save(buffer, "PNG")
    frame = asyncio.run(_executor(lambda r: httpx.Response(200, content=buffer.getvalue()), tmp_path).capture())
    assert (frame.width, frame.height) == (64, 32)


# ── runner ────────────────────────────────────────────────────────

def test_subset_is_fixed_and_valid():
    tasks = osworld.load_subset()
    assert len(tasks) == 25 and len({t["id"] for t in tasks}) == 25
    assert not any(t["domain"] == "thunderbird" for t in tasks)


def test_parse_run_id():
    assert osworld.parse_run_id("x\nRun 1a2b3c4d-1111-2222-3333-444455556666 started: desktop, auto mode") \
        == "1a2b3c4d-1111-2222-3333-444455556666"
    assert osworld.parse_run_id("nothing") is None


def test_run_one_scores_with_osworld_not_operon(monkeypatch):
    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    monkeypatch.setattr(osworld, "run_cost", lambda run_id, runs_root=None: 0.12)
    env = SimpleNamespace(vm_ip="10.1.1.1", server_port=5000, reset=lambda task_config: None, evaluate=lambda: 1.0)
    calls = {}

    def fake_run(cmd, env, **kw):
        calls["env"] = env
        return SimpleNamespace(stdout="Run deadbeef0 started: x", stderr="", returncode=1)

    result = osworld.run_one(env, {"domain": "os", "id": "abc", "instruction": "do it"}, {}, max_steps=5,
                             timeout_s=9, settle_s=0, run=fake_run)
    assert result["passed"] and result["cost_usd"] == 0.12  # operon exit 1 does not matter
    assert calls["env"]["OPERON_OSWORLD_URL"] == "http://10.1.1.1:5000"
    assert "OPERON_HARNESS_PROFILE" not in calls["env"] and calls["env"]["OPERON_AUTO_CONFIRM"] == "on"


def test_run_one_waits_for_apps_to_settle_before_the_checker_runs(monkeypatch):
    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    monkeypatch.setattr(osworld, "run_cost", lambda run_id, runs_root=None: 0.0)
    order: list[object] = []
    monkeypatch.setattr(osworld.time, "sleep", lambda s: order.append(("sleep", s)))
    env = SimpleNamespace(vm_ip="10.1.1.1", server_port=5000, reset=lambda task_config: None,
                          evaluate=lambda: order.append("evaluate") or 1.0)
    osworld.run_one(env, {"domain": "chrome", "id": "abc", "instruction": "x"}, {}, max_steps=5, timeout_s=9,
                    run=lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=0))
    assert order == [("sleep", osworld.EVALUATE_SETTLE_S), "evaluate"] and osworld.EVALUATE_SETTLE_S == 20.0


def test_summarize():
    s = osworld.summarize([
        {"domain": "os", "passed": True, "cost_usd": 0.1},
        {"domain": "os", "passed": False, "cost_usd": 0.2, "status": "error"},
        {"domain": "vlc", "passed": True, "cost_usd": 0.0, "status": "passed"},
        {"domain": "vlc", "passed": False, "cost_usd": 0.0, "status": "timed_out"},
    ])
    assert s["errors"] == 1 and s["timed_out"] == 1
    s = osworld.summarize([
        {"domain": "os", "passed": True, "cost_usd": 0.1},
        {"domain": "os", "passed": False, "cost_usd": 0.2},
        {"domain": "vlc", "passed": True, "cost_usd": 0.0},
    ])
    assert s["errors"] == 0 and s["passed"] == 2 and s["by_domain"] == {"os": "1/2", "vlc": "1/1"} and s["cost_usd"] == 0.3


def test_wait_for_desktop_polls_until_drawn(monkeypatch):
    sizes = iter([9_000, 9_000, 1_600_000])
    monkeypatch.setattr(osworld.time, "sleep", lambda s: None)
    monkeypatch.setattr(osworld.httpx if hasattr(osworld, "httpx") else __import__("httpx"), "get",
                        lambda *a, **k: SimpleNamespace(status_code=200, content=b"x" * next(sizes)))
    assert osworld.wait_for_desktop("http://vm:5000")


def test_run_cost_reads_usage_files_without_operon(tmp_path):
    step = tmp_path / "r1" / "step_1"
    step.mkdir(parents=True)
    (step / "policy_usage.json").write_text('{"estimated_cost_usd": 0.05}')
    (tmp_path / "r1" / "planner_usage.json").write_text('{"estimated_cost_usd": 0.02}')
    assert osworld.run_cost("r1", tmp_path) == pytest.approx(0.07)
    assert osworld.run_cost(None, tmp_path) == 0.0 and osworld.run_cost("missing", tmp_path) == 0.0


def test_command_uses_given_python():
    assert osworld.operon_command("x", 5, "/venv/bin/python")[0] == "/venv/bin/python"


# ── Codex review findings (PR #299) ───────────────────────────────

def _png(color=(255, 255, 255), noisy=False) -> bytes:
    import io
    import random

    from PIL import Image

    image = Image.new("RGB", (400, 300), color)
    if noisy:
        rnd = random.Random(1)
        image.putdata([(rnd.randrange(256),) * 3 for _ in range(400 * 300)])
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def test_visual_servo_refuses_a_click_on_a_blank_region(tmp_path):
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=_png())
        posted.append(1)
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    result = asyncio.run(_executor(handler, tmp_path).execute(AgentAction(action_type=ActionType.CLICK, x=200, y=150)))
    assert not result.success and "visual servo" in result.detail and posted == []


def test_visual_servo_allows_content_and_input_zones(tmp_path):
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=_png(noisy=True))
        posted.append(1)
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    assert asyncio.run(_executor(handler, tmp_path).execute(
        AgentAction(action_type=ActionType.CLICK, x=200, y=150))).success

    blank = lambda request: (httpx.Response(200, content=_png()) if request.method == "GET"  # noqa: E731
                             else httpx.Response(200, json={"status": "success", "returncode": 0}))
    assert asyncio.run(_executor(blank, tmp_path).execute(
        AgentAction(action_type=ActionType.CLICK, x=200, y=150, is_input_zone=True))).success


def test_non_ascii_text_is_pasted_not_written():
    snippet = pyautogui_snippet(AgentAction(action_type=ActionType.TYPE, text="café"))
    assert "pyperclip.copy('café')" in snippet and "hotkey('ctrl', 'v')" in snippet


def test_batch_stops_when_the_cancellation_check_blocks(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            calls.append(1)
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    executor = _executor(handler, tmp_path)

    async def check(child):
        return "run stopped"

    executor.batch_child_check = check
    batch = AgentAction(action_type=ActionType.BATCH, actions=[
        AgentAction(action_type=ActionType.PRESS_KEY, key="a"), AgentAction(action_type=ActionType.PRESS_KEY, key="b")])
    result = asyncio.run(executor.execute(batch))
    assert not result.success and result.batch_failed_index == 1 and len(calls) == 1


def test_provider_keeps_host_tools_away(monkeypatch):
    assert OSWorldProvider(url="http://x").runs_on_host is False
    monkeypatch.setenv("OPERON_TOOLS", "on")
    check = OSWorldProvider(url="http://x").check()
    assert not check.ok and "OPERON_TOOLS" in check.problems[0]


def test_launch_catalog_is_the_osworld_one(monkeypatch):
    from operon.agent.app_catalog import launch_targets

    monkeypatch.setenv("OPERON_ENVIRONMENT", "osworld")
    targets = launch_targets()
    assert targets["browser"] == "google-chrome" and targets["settings"] == "gnome-control-center"
    monkeypatch.setenv("OPERON_ENVIRONMENT", "live")
    assert launch_targets().get("browser") != "google-chrome"


def test_timeout_still_counts_the_spend(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    step = tmp_path / "runs" / "abc12345-0000-0000-0000-000000000000" / "step_1"
    step.mkdir(parents=True)
    (step / "policy_usage.json").write_text('{"estimated_cost_usd": 0.4}')
    env = SimpleNamespace(vm_ip="v", server_port=5000, reset=lambda task_config: None, evaluate=lambda: 0.0)

    def timing_out(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1, output=b"Run abc12345-0000-0000-0000-000000000000 started: x")

    result = osworld.run_one(env, {"domain": "os", "id": "t", "instruction": "i"}, {}, max_steps=1, timeout_s=1, settle_s=0,
                             runs_root=str(tmp_path / "runs"), run=timing_out)
    assert result["run_id"].startswith("abc12345") and result["cost_usd"] == 0.4
    # even when no start line was read, the newest run folder is found
    assert osworld.find_run_since(tmp_path / "runs", 0).startswith("abc12345")


def test_finished_scores_survive_a_later_failure(monkeypatch, tmp_path):
    (tmp_path / "repo").mkdir()
    monkeypatch.setattr(osworld, "load_subset", lambda: [{"domain": "os", "id": "a", "instruction": "i"},
                                                           {"domain": "os", "id": "b", "instruction": "j"},
                                                           {"domain": "os", "id": "c", "instruction": "k"}])
    monkeypatch.setattr(osworld, "task_config", lambda repo, task: {})
    seen = []

    def fake_run_one(env, task, config, **k):
        seen.append(task["id"])
        if task["id"] == "b":
            raise RuntimeError("VM died")
        return {"domain": "os", "id": task["id"], "passed": True, "cost_usd": 0.1, "score": 1.0, "seconds": 1,
                "status": "passed"}

    monkeypatch.setattr(osworld, "run_one", fake_run_one)
    code = osworld.main(["--osworld-repo", str(tmp_path / "repo"), "--out", str(tmp_path / "out")],
                        env_factory=lambda repo: SimpleNamespace(close=lambda: None))
    assert code == 0 and seen == ["a", "b", "c"]
    saved = json.loads(next((tmp_path / "out").glob("osworld-*.json")).read_text())
    assert saved["summary"]["passed"] == 2 and saved["summary"]["errors"] == 1
    assert saved["results"][1]["status"] == "error" and "VM died" in saved["results"][1]["error"]


def _log_env():
    return SimpleNamespace(vm_ip="v", server_port=5000, reset=lambda task_config: None, evaluate=lambda: 0.0)


def test_run_one_saves_the_operon_output_when_startup_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    monkeypatch.setattr(osworld, "run_cost", lambda run_id, runs_root=None: 0.0)
    result = osworld.run_one(
        _log_env(), {"domain": "os", "id": "2c1ebcd7", "instruction": "i"}, {}, max_steps=1, timeout_s=9,
        settle_s=0, logs_dir=str(tmp_path / "logs"),
        run=lambda cmd, **kw: SimpleNamespace(stdout="", stderr="RuntimeError: no API key", returncode=1))
    assert result["run_id"] is None
    assert result["log"] == str(tmp_path / "logs" / "2c1ebcd7.log")
    text = (tmp_path / "logs" / "2c1ebcd7.log").read_text(encoding="utf-8")
    assert "operon exit 1" in text and "RuntimeError: no API key" in text


def test_run_one_saves_partial_output_on_timeout(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    monkeypatch.setattr(osworld, "run_cost", lambda run_id, runs_root=None: 0.0)

    def timing_out(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1, output=b"step 3 clicked caf\xc3\xa9", stderr=b"warn")

    result = osworld.run_one(_log_env(), {"domain": "os", "id": "t1", "instruction": "i"}, {}, max_steps=1,
                             timeout_s=1, settle_s=0, logs_dir=str(tmp_path), run=timing_out)
    text = open(result["log"], encoding="utf-8").read()
    assert "timed out after 1s" in text and "step 3 clicked café" in text and "warn" in text


def test_run_one_without_logs_dir_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    monkeypatch.setattr(osworld, "run_cost", lambda run_id, runs_root=None: 0.0)
    monkeypatch.chdir(tmp_path)
    result = osworld.run_one(_log_env(), {"domain": "os", "id": "t2", "instruction": "i"}, {}, max_steps=1,
                             timeout_s=9, settle_s=0,
                             run=lambda cmd, **kw: SimpleNamespace(stdout="x", stderr="y", returncode=0))
    assert result["log"] is None and list(tmp_path.iterdir()) == []


def test_main_passes_a_logs_dir_beside_the_runs_dir(monkeypatch, tmp_path):
    (tmp_path / "repo").mkdir()
    monkeypatch.setattr(osworld, "load_subset", lambda: [{"domain": "os", "id": "a", "instruction": "i"}])
    monkeypatch.setattr(osworld, "task_config", lambda repo, task: {})
    seen = {}

    def fake_run_one(*a, **k):
        seen.update(k)
        return {"domain": "os", "id": "a", "passed": False, "cost_usd": 0.0, "score": 0.0, "seconds": 1}

    monkeypatch.setattr(osworld, "run_one", fake_run_one)
    osworld.main(["--osworld-repo", str(tmp_path / "repo"), "--out", str(tmp_path / "out")],
                 env_factory=lambda repo: SimpleNamespace(close=lambda: None))
    assert seen["logs_dir"] == str(tmp_path / "out" / "osworld-logs")
    assert seen["runs_root"] == str(tmp_path / "out" / "osworld-runs")


def test_long_benchmark_instructions_are_accepted_by_the_run_request():
    from operon.api.frontdoor import FrontDoorTask
    from operon.models.common import MAX_INTENT_CHARS, RunTaskRequest

    long_task = "x" * 736  # OSWorld multi_apps 2c1ebcd7 is 736 characters
    assert FrontDoorTask(intent=long_task).intent == long_task
    assert RunTaskRequest(intent=long_task).intent == long_task
    with pytest.raises(ValueError):
        RunTaskRequest(intent="x" * (MAX_INTENT_CHARS + 1))


# ── post-action wait (OPERON_OSWORLD_POST_ACTION_WAIT_S) ───────────

from operon.environments import osworld as osworld_env  # noqa: E402


def _ok_handler(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        return httpx.Response(200, content=_png(noisy=True))
    return httpx.Response(200, json={"status": "success", "returncode": 0})


@pytest.fixture
def sleeps(monkeypatch):
    monkeypatch.delenv("OPERON_OSWORLD_POST_ACTION_WAIT_S", raising=False)
    recorded: list[float] = []

    async def fake_sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(osworld_env.asyncio, "sleep", fake_sleep)
    return recorded


def _wait_executor(handler, tmp_path, wait):
    return OSWorldExecutor("http://vm:5000", artifact_dir=tmp_path, transport=httpx.MockTransport(handler),
                           post_action_wait_s=wait)


_INPUT_ACTIONS = [
    AgentAction(action_type=ActionType.CLICK, x=200, y=150),
    AgentAction(action_type=ActionType.TYPE, text="hi", x=200, y=150),
    AgentAction(action_type=ActionType.PRESS_KEY, key="enter"),
    AgentAction(action_type=ActionType.SCROLL, x=200, y=150, scroll_amount=-3),
    AgentAction(action_type=ActionType.DRAG, x=100, y=100, x_end=200, y_end=200),
]


def test_post_action_wait_is_off_by_default(tmp_path, sleeps):
    result = asyncio.run(_wait_executor(_ok_handler, tmp_path, None).execute(_INPUT_ACTIONS[0]))
    assert result.success and sleeps == []


@pytest.mark.parametrize("action", _INPUT_ACTIONS, ids=lambda a: a.action_type.value)
def test_post_action_wait_follows_an_input_action(tmp_path, sleeps, action):
    result = asyncio.run(_wait_executor(_ok_handler, tmp_path, 1.5).execute(action))
    assert result.success and sleeps == [1.5]


def test_no_post_action_wait_after_stop_or_wait(tmp_path, sleeps):
    executor = _wait_executor(_ok_handler, tmp_path, 1.5)
    assert asyncio.run(executor.execute(AgentAction(action_type=ActionType.STOP))).success
    assert sleeps == []
    assert asyncio.run(executor.execute(AgentAction(action_type=ActionType.WAIT, wait_ms=250))).success
    assert sleeps == [0.25]


def test_no_post_action_wait_after_a_vm_error(tmp_path, sleeps):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=_png(noisy=True))
        return httpx.Response(200, json={"status": "error", "returncode": 1})

    result = asyncio.run(_wait_executor(handler, tmp_path, 1.5).execute(_INPUT_ACTIONS[0]))
    assert not result.success and sleeps == []


def test_no_post_action_wait_after_a_servo_refusal(tmp_path, sleeps):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=_png())
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    result = asyncio.run(_wait_executor(handler, tmp_path, 1.5).execute(_INPUT_ACTIONS[0]))
    assert not result.success and "visual servo" in result.detail and sleeps == []


def test_batch_waits_once_after_the_whole_batch(tmp_path, sleeps):
    batch = AgentAction(action_type=ActionType.BATCH, actions=[_INPUT_ACTIONS[0], _INPUT_ACTIONS[0]])
    result = asyncio.run(_wait_executor(_ok_handler, tmp_path, 1.5).execute(batch))
    assert result.success and sleeps == [1.5]


def test_post_action_wait_setting_feeds_the_executor(tmp_path, monkeypatch, sleeps):
    monkeypatch.setenv("OPERON_OSWORLD_POST_ACTION_WAIT_S", "2.5")
    assert get_settings().osworld_post_action_wait_s == 2.5
    asyncio.run(_wait_executor(_ok_handler, tmp_path, None).execute(_INPUT_ACTIONS[0]))
    assert sleeps == [2.5]
    monkeypatch.setenv("OPERON_OSWORLD_POST_ACTION_WAIT_S", "99")
    assert get_settings().osworld_post_action_wait_s == 10.0


def _vm_error_after_first(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        return httpx.Response(200, content=_png(noisy=True))
    return httpx.Response(200, json={"status": "error", "returncode": 1})


def test_post_action_wait_follows_a_catalog_app_launch(tmp_path, monkeypatch, sleeps):
    monkeypatch.setenv("OPERON_ENVIRONMENT", "osworld")
    result = asyncio.run(_wait_executor(_ok_handler, tmp_path, 1.5).execute(
        AgentAction(action_type=ActionType.LAUNCH_APP, text="browser")))
    assert result.success and sleeps == [1.5]


def test_no_post_action_wait_after_an_unknown_app_launch(tmp_path, monkeypatch, sleeps):
    monkeypatch.setenv("OPERON_ENVIRONMENT", "osworld")
    result = asyncio.run(_wait_executor(_ok_handler, tmp_path, 1.5).execute(
        AgentAction(action_type=ActionType.LAUNCH_APP, text="no-such-app")))
    assert not result.success and sleeps == []


def test_batch_waits_once_when_a_later_child_fails(tmp_path, sleeps):
    replies = iter([{"status": "success", "returncode": 0}, {"status": "error", "returncode": 1}])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=_png(noisy=True))
        return httpx.Response(200, json=next(replies))

    batch = AgentAction(action_type=ActionType.BATCH, actions=[_INPUT_ACTIONS[0], _INPUT_ACTIONS[2]])
    result = asyncio.run(_wait_executor(handler, tmp_path, 1.5).execute(batch))
    assert not result.success and sleeps == [1.5]


def test_batch_waits_once_when_a_later_child_is_blocked(tmp_path, sleeps):
    executor = _wait_executor(_ok_handler, tmp_path, 1.5)

    async def check(child):
        return "run stopped"

    executor.batch_child_check = check
    batch = AgentAction(action_type=ActionType.BATCH, actions=[_INPUT_ACTIONS[0], _INPUT_ACTIONS[2]])
    result = asyncio.run(executor.execute(batch))
    assert not result.success and result.batch_failed_index == 1 and sleeps == [1.5]


def test_no_post_action_wait_when_a_batch_fails_on_its_first_child(tmp_path, sleeps):
    batch = AgentAction(action_type=ActionType.BATCH, actions=[_INPUT_ACTIONS[0], _INPUT_ACTIONS[2]])
    result = asyncio.run(_wait_executor(_vm_error_after_first, tmp_path, 1.5).execute(batch))
    assert not result.success and sleeps == []


def test_run_usage_gaps_counts_missing_and_unreadable_usage(tmp_path):
    (tmp_path / "r1" / "step_1").mkdir(parents=True)
    (tmp_path / "r1" / "step_1" / "computer_use_raw.json").write_text("{}")
    (tmp_path / "r1" / "step_2").mkdir()
    (tmp_path / "r1" / "step_2" / "policy_usage.json").write_text("nope")
    assert osworld.run_usage_gaps("r1", tmp_path) == {"steps_without_usage": 1, "unreadable_usage_files": 1}
    assert osworld.run_usage_gaps(None, tmp_path) == {"steps_without_usage": 0, "unreadable_usage_files": 0}


def test_after_screenshot_uses_a_short_timeout_and_never_fails_the_action(tmp_path):
    shot_timeouts = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/screenshot":
            shot_timeouts.append(request.extensions["timeout"])
            return httpx.Response(200, content=b"png")
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    executor = OSWorldExecutor("http://vm:5000", artifact_dir=tmp_path, transport=httpx.MockTransport(handler),
                               post_action_wait_s=0)
    result = asyncio.run(executor.execute(AgentAction(action_type=ActionType.CLICK, x=7, y=8)))
    assert result.success and Path(result.artifact_path).read_bytes() == b"png"
    # A stuck VM holds the action at most 15 s here, not the 120 s of an action request.
    assert shot_timeouts[-1] == {"connect": 5.0, "read": 15.0, "write": 15.0, "pool": 15.0}

    def stuck(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/screenshot":
            raise httpx.ReadTimeout("stuck", request=request)
        return httpx.Response(200, json={"status": "success", "returncode": 0})

    executor = OSWorldExecutor("http://vm:5000", artifact_dir=tmp_path, transport=httpx.MockTransport(stuck),
                               post_action_wait_s=0)
    result = asyncio.run(executor.execute(AgentAction(action_type=ActionType.CLICK, x=7, y=8)))
    assert result.success and not result.artifact_path


def test_a_step_that_made_no_model_call_is_no_usage_gap(tmp_path):
    from operon.models.usage import NO_MODEL_CALL, step_missing_usage

    step = tmp_path / "r1" / "step_1"
    step.mkdir(parents=True)
    (step / "computer_use_raw.json").write_text("{}")
    (step / "computer_use_prompt.txt").write_text(f"{NO_MODEL_CALL}: running the calls the person approved.\n")
    assert osworld.NO_MODEL_CALL == NO_MODEL_CALL
    assert osworld.run_usage_gaps("r1", tmp_path)["steps_without_usage"] == 0
    assert not step_missing_usage(step)
    (step / "computer_use_prompt.txt").write_text("a real prompt")
    assert osworld.run_usage_gaps("r1", tmp_path)["steps_without_usage"] == 1
    assert step_missing_usage(step)


def test_usage_gaps_reach_the_task_row_the_summary_and_the_dashboard_flag(monkeypatch, tmp_path):
    from evals import dashboard

    (tmp_path / "r1" / "step_1").mkdir(parents=True)
    (tmp_path / "r1" / "step_1" / "computer_use_raw.json").write_text("{}")
    (tmp_path / "r1" / "step_2").mkdir()
    (tmp_path / "r1" / "step_2" / "policy_usage.json").write_text("nope")
    monkeypatch.setattr(osworld, "wait_for_desktop", lambda url: True)
    monkeypatch.setattr(osworld, "parse_run_id", lambda text: "r1")
    env = SimpleNamespace(vm_ip="10.1.1.1", server_port=5000, reset=lambda task_config: None, evaluate=lambda: 0.0)
    row = osworld.run_one(env, {"domain": "os", "id": "abc", "instruction": "x"}, {}, max_steps=5, timeout_s=9,
                          settle_s=0, runs_root=str(tmp_path),
                          run=lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=0))
    assert row["run_id"] == "r1"
    assert row["steps_without_usage"] == 1 and row["unreadable_usage_files"] == 1
    summary = osworld.summarize([row, {**row, "id": "def"}])
    assert summary["steps_without_usage"] == 2 and summary["unreadable_usage_files"] == 2
    assert osworld.summarize([{"domain": "os", "passed": True, "cost_usd": 0.0}])["steps_without_usage"] == 0
    # The dashboard reads the same summary keys to flag the run.
    prepared = dashboard.prepare({"suite": "osworld", "summary": summary, "results": []}, {})
    assert "2 steps had no usage file and 2 usage files were unreadable" in prepared["config"]["flagged"]

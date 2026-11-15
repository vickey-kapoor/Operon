"""OSWorld provider, executor and runner (KAN-169): no VM, no network, no model."""

from __future__ import annotations

import asyncio
import json
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
        (AgentAction(action_type=ActionType.SCROLL, x=5, y=6, scroll_amount=-3), "pyautogui.scroll(-3, x=5, y=6)"),
        (AgentAction(action_type=ActionType.SCROLL, x=5, y=6, scroll_amount=2, horizontal=True),
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
                             timeout_s=9, run=fake_run)
    assert result["passed"] and result["cost_usd"] == 0.12  # operon exit 1 does not matter
    assert calls["env"]["OPERON_OSWORLD_URL"] == "http://10.1.1.1:5000"
    assert calls["env"]["OPERON_HARNESS_PROFILE"] == "thin" and calls["env"]["OPERON_AUTO_CONFIRM"] == "on"


def test_summarize():
    s = osworld.summarize([
        {"domain": "os", "passed": True, "cost_usd": 0.1},
        {"domain": "os", "passed": False, "cost_usd": 0.2},
        {"domain": "vlc", "passed": True, "cost_usd": 0.0},
    ])
    assert s["passed"] == 2 and s["by_domain"] == {"os": "1/2", "vlc": "1/1"} and s["cost_usd"] == 0.3


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

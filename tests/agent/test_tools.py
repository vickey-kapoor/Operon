"""The tool action channel (RFC 0002, KAN-79): the flag, translation, the runner, the
executor wrapper, the gates, and how the model gets each tool's output back."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

from operon.agent import tools
from operon.agent.actions.translation import (
    build_policy_decision,
    normalize_computer_use_actions,
    normalize_computer_use_function_call,
)
from operon.agent.backends.desktop_computer_use import DesktopComputerUseBackend
from operon.agent.policy.file_guard import blocked_file_path
from operon.agent.policy.trust_gate import (
    TrustGate,
    TrustVerdict,
    apply_permission_mode,
)
from operon.agent.tools.runner import ToolRunner
from operon.core.paths import prompts_dir
from operon.executor.tools import ToolExecutor
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.execution import ExecutedAction, ToolOutput
from operon.models.policy import ActionType, AgentAction
from operon.models.state import AgentState


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("OPERON_ALLOWED_FILE_ROOTS", raising=False)
    return home


def _tool(name: str, **args) -> AgentAction:
    payload = normalize_computer_use_function_call({"name": name, "args": args}, screen_width=1000, screen_height=1000)
    return build_policy_decision({"action": payload}).action


# --- the flag --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ()),
        ("off", ()),
        ("on", ("read_file", "list_dir", "write_file", "http_get")),
        ("all", ("read_file", "list_dir", "write_file", "run_shell", "http_get")),
        ("list_dir, read_file, nope", ("read_file", "list_dir")),
    ],
)
def test_operon_tools_turns_on_the_named_tools_and_is_off_by_default(monkeypatch, value, expected) -> None:
    if value is None:
        monkeypatch.delenv("OPERON_TOOLS", raising=False)
    else:
        monkeypatch.setenv("OPERON_TOOLS", value)
    assert tools.enabled_tools() == expected


def test_every_tool_has_a_declaration_and_a_skill_note() -> None:
    declared = tools.function_declarations(tools.TOOL_NAMES)
    assert [d["name"] for d in declared] == list(tools.TOOL_NAMES)
    text = tools.skill_text(tools.TOOL_NAMES)
    for name in tools.TOOL_NAMES:
        assert (prompts_dir() / "tools" / f"{name}.md").is_file()
        assert f"{name}(" in text
    assert tools.skill_text(()) == ""


# --- translation and the action model ---------------------------------------------------

def test_a_tool_call_becomes_a_tool_action_the_gates_can_judge() -> None:
    action = _tool("run_shell", command="Get-ChildItem   Documents", safety_decision={"decision": "allow"})
    assert action.action_type is ActionType.TOOL
    assert action.tool == "run_shell" and action.tool_args == {"command": "Get-ChildItem   Documents"}
    assert action.text == "run_shell: Get-ChildItem Documents" and action.url is None
    fetch = _tool("http_get", url="https://example.com/a")
    assert fetch.url == "https://example.com/a" and fetch.text == "http_get: https://example.com/a"
    write = _tool("write_file", path="notes.txt", content="place order now")
    assert write.text == "write_file: notes.txt"  # the content never reaches the gates' text


def test_tool_calls_and_clicks_in_one_turn_make_one_batch() -> None:
    payload = normalize_computer_use_actions(
        [{"name": "click_at", "args": {"x": 500, "y": 500}}, {"name": "read_file", "args": {"path": "a.txt"}}],
        screen_width=1000,
        screen_height=1000,
    )
    action = build_policy_decision({"action": payload}).action
    assert action.action_type is ActionType.BATCH
    assert [child.action_type for child in action.actions] == [ActionType.CLICK, ActionType.TOOL]


def test_tool_fields_belong_to_tool_actions_only() -> None:
    with pytest.raises(ValueError):
        AgentAction(action_type=ActionType.TOOL)
    with pytest.raises(ValueError):
        AgentAction(action_type=ActionType.TOOL, tool="read_file", x=3, y=4)
    with pytest.raises(ValueError):
        AgentAction(action_type=ActionType.CLICK, x=1, y=1, tool="read_file")


# --- the runner -------------------------------------------------------------------------

async def test_file_tools_read_list_and_write_relative_to_home(home: Path) -> None:
    runner = ToolRunner(tools.DEFAULT_TOOLS)
    written = await runner.run("write_file", {"path": "Documents/out/a.csv", "content": "a,b\n1,2\n"})
    assert written.ok and (home / "Documents" / "out" / "a.csv").read_text() == "a,b\n1,2\n"
    read = await runner.run("read_file", {"path": "Documents/out/a.csv"})
    assert read == ToolOutput(tool="read_file", ok=True, output="a,b\n1,2\n")
    (home / "Documents" / "out" / "sub").mkdir()
    listing = await runner.run("list_dir", {"path": str(home / "Documents" / "out")})
    assert listing.output == "a.csv\t8\nsub\tdir"


async def test_read_file_reads_utf16_and_cuts_long_files(home: Path) -> None:
    (home / "ps.txt").write_bytes("héllo".encode("utf-16"))
    runner = ToolRunner(("read_file",))
    assert (await runner.run("read_file", {"path": "ps.txt"})).output == "héllo"
    (home / "big.txt").write_text("x" * 25_000)
    big = (await runner.run("read_file", {"path": "big.txt"})).output
    assert big.startswith("x" * 20_000) and big.endswith("[cut: 5000 more characters]")


async def test_a_tool_that_cannot_run_says_why(home: Path) -> None:
    runner = ToolRunner(("read_file",))
    missing = await runner.run("read_file", {"path": "nope.txt"})
    assert not missing.ok and "is not a file" in missing.output
    assert not (await runner.run("read_file", {})).ok
    off = await runner.run("run_shell", {"command": "echo hi"})
    assert not off.ok and "not turned on" in off.output
    assert not (await runner.run("format_disk", {})).ok


@pytest.mark.skipif(sys.platform == "win32", reason="the test command is bash")
async def test_run_shell_returns_exit_code_and_output_and_stops_at_the_timeout(home: Path) -> None:
    runner = ToolRunner(("run_shell",))
    result = await runner.run("run_shell", {"command": "pwd; echo oops >&2; exit 3"})
    assert result.ok
    assert result.output == f"exit code: 3\nstdout:\n{home}\n\nstderr:\noops\n"
    slow = await runner.run("run_shell", {"command": "sleep 5", "timeout_seconds": 1})
    assert not slow.ok and "timed out after 1 seconds" in slow.output


async def test_http_get_returns_status_and_body_and_does_not_follow_redirects(monkeypatch, home) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/moved":
            return httpx.Response(302, headers={"location": "https://elsewhere.example/"})
        return httpx.Response(200, text="hello", headers={"content-type": "text/plain"})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        "operon.agent.tools.runner.httpx.AsyncClient",
        lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs),
    )
    runner = ToolRunner(("http_get",))
    ok = await runner.run("http_get", {"url": "https://api.example/a"})
    assert ok.output == "status: 200\ncontent-type: text/plain\n\nhello"
    moved = await runner.run("http_get", {"url": "https://api.example/moved"})
    assert "status: 302" in moved.output and "location: https://elsewhere.example/" in moved.output
    assert not (await runner.run("http_get", {"url": "file:///etc/passwd"})).ok


# --- the executor wrapper ---------------------------------------------------------------

class FakeExecutor:
    def __init__(self, fail_on: ActionType | None = None) -> None:
        self.ran: list[ActionType] = []
        self.fail_on = fail_on
        self.batch_child_check = None
        self.marker = "inner"

    async def execute(self, action: AgentAction) -> ExecutedAction:
        self.ran.append(action.action_type)
        ok = action.action_type is not self.fail_on
        return ExecutedAction(action=action, success=ok, detail="done" if ok else "missed", artifact_path="after.png")


async def test_tool_executor_runs_tools_itself_and_hands_the_rest_on(home: Path) -> None:
    inner = FakeExecutor()
    executor = ToolExecutor(inner, ("read_file",))
    (home / "a.txt").write_text("data")
    click = AgentAction(action_type=ActionType.CLICK, x=1, y=1)
    assert (await executor.execute(click)).success and inner.ran == [ActionType.CLICK]
    result = await executor.execute(_tool("read_file", path="a.txt"))
    assert result.success and result.tool_outputs == [ToolOutput(tool="read_file", ok=True, output="data")]
    assert inner.ran == [ActionType.CLICK]
    assert executor.marker == "inner"  # everything else reaches the wrapped executor
    executor.batch_child_check = "check"
    assert inner.batch_child_check == "check" and executor.batch_child_check == "check"


async def test_a_mixed_batch_runs_in_order_and_stops_at_the_first_failure(home: Path) -> None:
    inner = FakeExecutor(fail_on=ActionType.HOTKEY)
    executor = ToolExecutor(inner, ("read_file",))
    (home / "a.txt").write_text("data")
    batch = AgentAction(
        action_type=ActionType.BATCH,
        actions=[
            AgentAction(action_type=ActionType.CLICK, x=1, y=1),
            _tool("read_file", path="a.txt"),
            AgentAction(action_type=ActionType.HOTKEY, key="ctrl+s"),
            _tool("read_file", path="a.txt"),
        ],
    )
    result = await executor.execute(batch)
    assert not result.success and result.batch_failed_index == 2
    assert inner.ran == [ActionType.CLICK, ActionType.HOTKEY]
    assert [o.output for o in result.tool_outputs] == ["data"]

    async def stop_tools(child: AgentAction) -> str | None:
        return "blocked by the gate" if child.action_type is ActionType.TOOL else None

    executor.batch_child_check = stop_tools
    inner.ran.clear()
    stopped = await executor.execute(batch)
    assert stopped.batch_failed_index == 1 and "blocked by the gate" in stopped.detail and inner.ran == [ActionType.CLICK]


# --- the gates --------------------------------------------------------------------------

def test_the_trust_gate_asks_before_any_shell_command() -> None:
    gate = TrustGate(enabled=True, allow_domains=("example.com",))
    shell = _tool("run_shell", command="dir")
    verdict = gate.evaluate(shell, current_url="https://other.org/")
    assert verdict.verdict is TrustVerdict.CONFIRM
    assert apply_permission_mode(verdict, shell, "auto").verdict is TrustVerdict.ALLOW
    assert gate.evaluate(_tool("read_file", path="a.txt"), current_url="https://other.org/").verdict is TrustVerdict.ALLOW
    assert gate.evaluate(_tool("http_get", url="https://evil.org/x")).verdict is TrustVerdict.DENY
    assert TrustGate(enabled=True, deny=("remove-item",)).evaluate(_tool("run_shell", command="Remove-Item x")).verdict is TrustVerdict.DENY
    assert TrustGate(enabled=False).evaluate(shell).verdict is TrustVerdict.ALLOW
    assert apply_permission_mode(TrustGate(enabled=False).evaluate(shell), shell, "ask").verdict is TrustVerdict.CONFIRM


def test_the_file_guard_checks_file_tools_paths_from_home(home: Path, tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    read = _tool("read_file", path="Documents/secret.txt")
    assert blocked_file_path(read, intent="Total the orders", run_dir=run_dir) == "Documents/secret.txt"
    monkeypatch.setenv("OPERON_ALLOWED_FILE_ROOTS", str(home / "Documents"))
    assert blocked_file_path(read, intent="Total the orders", run_dir=run_dir) is None
    assert blocked_file_path(_tool("write_file", path="../escape.txt", content=""), intent="x", run_dir=run_dir) is not None
    assert blocked_file_path(_tool("run_shell", command="cat /etc/passwd"), intent="x", run_dir=run_dir) is None


# --- the model's answers ----------------------------------------------------------------

class ScriptedClient:
    def __init__(self, calls: list[dict]) -> None:
        self.calls = calls
        self.turns: list[list[dict]] = []

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self.turns.append(list(contents))
        return {
            "perception": {"summary": "s", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
            "function_call": self.calls[0],
            "function_calls": self.calls,
            "rationale": "r",
            "confidence": 0.7,
            "model_content": {"parts": [{"function_call": call} for call in self.calls]},
        }

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        return await self.generate_turn(contents=[])

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        response = {"name": function_name, "response": {"error": error} if error else {}}
        return {"role": "user", "parts": [{"function_response": response}, {"inline_data": {"data": ""}}]}


def _frame(tmp_path: Path, name: str) -> CaptureFrame:
    path = tmp_path / name
    path.write_bytes(b"png")
    return CaptureFrame(artifact_path=str(path), width=1000, height=1000)


async def _answers(tmp_path: Path, executed: ExecutedAction) -> tuple[DesktopComputerUseBackend, list[dict]]:
    calls = [
        {"name": "read_file", "args": {"path": "a.txt"}},
        {"name": "click_at", "args": {"x": 1, "y": 1}},
        {"name": "list_dir", "args": {"path": "."}},
    ]
    client = ScriptedClient(calls)
    backend = DesktopComputerUseBackend(
        client=client, prompt_path=prompts_dir() / "desktop_computer_use_prompt.txt", tools=("read_file", "list_dir")
    )
    state = AgentState(run_id="r", intent="Total the orders", status=RunStatus.RUNNING)
    await backend.perceive(_frame(tmp_path, "1.png"), state)
    await backend.perceive(_frame(tmp_path, "2.png"), state.model_copy(update={"action_history": [executed]}))
    responses = [part["function_response"] for part in client.turns[1][-1]["parts"] if "function_response" in part]
    return backend, [client.turns[0], responses]


async def test_each_tool_call_is_answered_with_its_own_output(tmp_path: Path) -> None:
    executed = ExecutedAction(
        action=AgentAction(action_type=ActionType.WAIT, wait_ms=1),
        success=True,
        detail="ok",
        tool_outputs=[ToolOutput(tool="read_file", ok=True, output="a,b"), ToolOutput(tool="list_dir", ok=False, output="list_dir failed")],
    )
    _, (first_turn, responses) = await _answers(tmp_path, executed)
    assert "read_file(path)" in first_turn[0]["parts"][0]["text"]  # the skill notes are in the prompt
    assert "run_shell(" not in first_turn[0]["parts"][0]["text"]
    assert [r["name"] for r in responses] == ["read_file", "click_at", "list_dir"]
    assert responses[0]["response"] == {"output": "a,b"}
    assert responses[1]["response"] == {}
    assert responses[2]["response"] == {"error": "list_dir failed"}


async def test_a_tool_call_after_a_failed_click_is_answered_as_not_run(tmp_path: Path) -> None:
    executed = ExecutedAction(
        action=AgentAction(action_type=ActionType.WAIT, wait_ms=1),
        success=False,
        detail="Batch failed on click: missed",
        batch_failed_index=1,
        tool_outputs=[ToolOutput(tool="read_file", ok=True, output="a,b")],
    )
    _, (_, responses) = await _answers(tmp_path, executed)
    assert responses[0]["response"] == {"output": "a,b"}
    assert responses[2]["response"] == {"error": "Not run: Batch failed on click: missed"}


def test_services_declare_the_tools_and_the_loop_wraps_the_executor(monkeypatch) -> None:
    from operon.api.runtime import loops, services

    monkeypatch.setenv("OPERON_TOOLS", "all")
    monkeypatch.setenv("OPERON_DESKTOP_BACKEND", "computer_use")
    backend = services.build_desktop_services().policy_delegate
    declared = [d["name"] for d in backend.client.function_declarations]
    assert declared == ["launch_app", *tools.TOOL_NAMES] and backend.tools == tools.TOOL_NAMES

    class Config:
        backend = "computer_use"

    inner = FakeExecutor()
    assert isinstance(loops._with_tools(inner, Config()), ToolExecutor)
    monkeypatch.setenv("OPERON_TOOLS", "off")
    assert loops._with_tools(inner, Config()) is inner


async def test_the_full_harness_passes_a_tool_call_through(tmp_path: Path) -> None:
    from operon.agent.policy.coordinator import PolicyCoordinator
    from operon.agent.policy.service import PolicyService
    from operon.models.perception import ScreenPerception
    from operon.models.policy import PolicyDecision
    from operon.store.memory import FileBackedMemoryStore

    decision = PolicyDecision(
        action=_tool("read_file", path="a.txt"), rationale="read it", confidence=0.8, active_subgoal="read"
    )

    class Delegate(PolicyService):
        async def choose_action(self, state, perception):
            return decision

    coordinator = PolicyCoordinator(delegate=Delegate(), memory_store=FileBackedMemoryStore(root_dir=tmp_path / "runs"))
    state = AgentState(run_id="r", intent="Total the orders in a.txt", status=RunStatus.RUNNING, step_count=1)
    perception = ScreenPerception(
        summary="A desktop.", page_hint="unknown", visible_elements=[], confidence=0.5, capture_artifact_path="s.png"
    )
    chosen = await coordinator.choose_action(state, perception)
    assert chosen.action.action_type is ActionType.TOOL and chosen.action.tool_args == {"path": "a.txt"}

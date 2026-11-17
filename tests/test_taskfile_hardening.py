"""Task-file hardening: policy denies, XML-safe reports, flag mixes, default mode (KAN-86), offline."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from operon import sdk
from operon.cli import main
from operon.sdk import ApprovalRequest, RunResult
from operon.taskfile import TaskOutcome, load_task_file, run_task_file, write_junit
from tests.test_taskfile import YAML, FakeAgent, _write, fake  # noqa: F401  (fake is a fixture)


def test_default_mode_is_risky(tmp_path) -> None:
    tf = load_task_file(_write(tmp_path, "tasks:\n  - {name: x, task: y}\n"))
    assert tf.resolved(tf.tasks[0])["mode"] == "risky"
    tf = load_task_file(_write(tmp_path, "defaults: {mode: ask}\ntasks:\n  - {name: x, task: y}\n"))
    assert tf.resolved(tf.tasks[0])["mode"] == "ask"


def test_approve_all_never_approves_a_policy_deny(tmp_path, monkeypatch) -> None:
    seen: list[bool] = []

    class Asking(FakeAgent):
        async def arun(self, task, *, url=None):
            deny = ApprovalRequest("r", "Blocked by policy: click (x)", "approval", "trust", "deny")
            confirm = ApprovalRequest("r", "Approve it?", "approval", "trust", "confirm")
            seen.extend([self.kwargs["on_approval"](confirm), self.kwargs["on_approval"](deny)])
            return RunResult(status="denied", final_answer=None, steps=1, cost_usd=None, run_id="r",
                             denied_reason=deny.reason, denied_kind="policy")

    monkeypatch.setattr(sdk, "Agent", Asking)
    text = "defaults: {approve_all: true}\ntasks:\n  - {name: x, task: y}\n"
    o = run_task_file(load_task_file(_write(tmp_path, text)))[0]
    assert seen == [True, False]
    assert not o.passed and o.reason.startswith("blocked by policy:") and "Blocked by policy" not in o.reason


def test_takeover_denial_message(tmp_path, fake) -> None:  # noqa: F811
    fake.script["do a"] = RunResult(status="denied", final_answer=None, steps=1, cost_usd=None, run_id="r",
                                    denied_reason="handed over", denied_kind="takeover")
    o = run_task_file(load_task_file(_write(tmp_path, YAML)), only=["a"])[0]
    assert "handed to a person" in o.reason and "approve_all" not in o.reason


def test_junit_strips_xml_illegal_characters(tmp_path) -> None:
    outcome = TaskOutcome("bad\x00name\x1b", "failed", 1, None, 1.0, False, "oops\x00\x1b[31mred", "r\x00", "/a\x1b")
    out = tmp_path / "r.xml"
    write_junit([outcome], out, classname="cls\x00")
    case = ET.parse(out).getroot().find("testsuite/testcase")
    assert case is not None and case.get("name") == "bad�name�" and case.get("classname") == "cls�"
    failure = case.find("failure")
    assert failure is not None and "�" in (failure.get("message") or "") and "\x00" not in (failure.text or "")


def test_report_failure_exits_1_without_crashing(tmp_path, fake, capsys) -> None:  # noqa: F811
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    # The report's parent is a file, so it cannot be created.
    assert main(["run", "--tasks", str(_write(tmp_path)), "--junit", str(blocker / "r.xml")]) == 1
    assert "could not write the JUnit report" in capsys.readouterr().err


def test_undecodable_file_exits_2(tmp_path, capsys) -> None:
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"tasks: \xff\xfe\x00")
    assert main(["run", "--tasks", str(path)]) == 2
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["run", "do it", "--tasks", "t.yaml"],
    ["run", "--tasks", "t.yaml", "--url", "https://x.test"],
    ["run", "--tasks", "t.yaml", "--mode", "auto"],
    ["run", "--tasks", "t.yaml", "--desktop"],
    ["run", "--tasks", "t.yaml", "--max-steps", "25"],
    ["run", "do it", "--junit", "r.xml"],
    ["run", "do it", "--only", "a"],
    ["run", "do it", "--fail-fast"],
])
def test_flag_mixes_are_rejected(argv, tmp_path, fake) -> None:  # noqa: F811
    argv = [str(_write(tmp_path)) if a == "t.yaml" else a for a in argv]
    assert main(argv) == 2
    assert fake.calls == []


def _real_sdk(monkeypatch, tmp_path, gate):
    from operon.api import ws_stream
    from operon.api.runtime import loops
    from tests.fixtures.frontdoor import PURCHASE, build_loop
    from tests.test_sdk import STOP, _StopVerifier

    loop, executor = build_loop(monkeypatch, tmp_path, [PURCHASE, STOP], gate=gate)
    loop.verifier_service = _StopVerifier()
    monkeypatch.setattr(loops, "get_agent_loop", lambda: loop)
    monkeypatch.setattr(loops, "desktop_loop_holds_run", lambda run_id: False)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)
    orig = sdk.Agent.__init__
    monkeypatch.setattr(sdk.Agent, "__init__", lambda self, *a, **kw: orig(self, *a, **{"poll_s": 0.01, **kw}))
    return executor


def test_sdk_exposes_the_gate_verdict(monkeypatch, tmp_path) -> None:
    from operon.agent.policy.trust_gate import TrustGate

    seen: list[ApprovalRequest] = []
    for gate in (TrustGate(enabled=True, deny=("purchase",)), TrustGate(enabled=True, confirm=("purchase",))):
        _real_sdk(monkeypatch, tmp_path, gate)
        result = sdk.Agent(mode="risky", on_approval=lambda r: seen.append(r) or False).run("Buy it")
        assert result.status == "denied"
    assert [r.verdict for r in seen] == ["deny", "confirm"]
    assert [r.gate for r in seen] == ["trust", "trust"]


def test_approve_all_does_not_run_a_policy_denied_action(monkeypatch, tmp_path) -> None:
    from operon.agent.policy.trust_gate import TrustGate

    executor = _real_sdk(monkeypatch, tmp_path, TrustGate(enabled=True, deny=("purchase",)))
    text = "tasks:\n  - {name: buy, task: Buy it, mode: auto, approve_all: true}\n"
    o = run_task_file(load_task_file(_write(tmp_path, text)))[0]
    assert not o.passed and o.status == "denied" and o.reason.startswith("blocked by policy:") and "Blocked by policy" not in o.reason
    assert executor.executed == []


def test_approve_all_does_approve_a_confirm(monkeypatch, tmp_path) -> None:
    from operon.agent.policy.trust_gate import TrustGate

    _real_sdk(monkeypatch, tmp_path, TrustGate(enabled=True, confirm=("purchase",)))
    text = "tasks:\n  - {name: buy, task: Buy it, mode: risky, approve_all: true}\n"
    assert run_task_file(load_task_file(_write(tmp_path, text)))[0].passed
    _real_sdk(monkeypatch, tmp_path, TrustGate(enabled=True, confirm=("purchase",)))
    text = "tasks:\n  - {name: buy, task: Buy it, mode: risky}\n"  # no approve_all: the confirm fails the task
    assert not run_task_file(load_task_file(_write(tmp_path, text)))[0].passed


def test_flag_messages_and_api_flags(tmp_path, fake, capsys) -> None:  # noqa: F811
    path = str(_write(tmp_path))
    assert main(["run", "--tasks", path, "--api", "http://x.test"]) == 2
    assert main(["run", "--tasks", path, "--api-key", "k"]) == 2
    capsys.readouterr()
    assert main(["run", "x", "--junit", "r.xml"]) == 2
    assert "--junit only works with --tasks" in capsys.readouterr().err
    assert main(["run", "x", "--junit", "r.xml", "--fail-fast"]) == 2
    assert "only work with --tasks" in capsys.readouterr().err


def test_cli_payload_defaults_max_steps_to_25() -> None:
    from operon import cli

    args = cli.build_parser().parse_args(["run", "do it"])
    assert args.max_steps is None and cli._payload(args)["max_steps"] == 25
    args = cli.build_parser().parse_args(["run", "do it", "--max-steps", "7"])
    assert cli._payload(args)["max_steps"] == 7

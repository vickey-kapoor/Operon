"""Task files and the JUnit report (KAN-86), offline: a scripted Agent, plus one run through the real SDK."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from operon import sdk
from operon.api import ws_stream
from operon.api.runtime import loops
from operon.cli import main
from operon.sdk import RunResult
from operon.taskfile import TaskFileError, load_task_file, run_task_file, write_junit
from tests.fixtures.frontdoor import build_loop
from tests.test_sdk import STOP, _StopVerifier

YAML = """
defaults:
  mode: auto
  max_steps: 7
  url: https://example.test
tasks:
  - name: a
    task: do a
    expect: {answer_contains: [PARIS]}
  - name: b
    task: do b
    url: https://other.test
    approve_all: true
  - name: c
    task: do c
"""


def _write(tmp_path: Path, text: str = YAML) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _result(status="succeeded", answer="Sunny in Paris", steps=3, **kw) -> RunResult:
    return RunResult(status=status, final_answer=answer if status == "succeeded" else None, steps=steps,
                     cost_usd=0.01, run_id="r1", artifacts_dir="/runs/r1", message=answer, **kw)


class FakeAgent:
    script: dict[str, RunResult] = {}
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def arun(self, task: str, *, url=None) -> RunResult:
        FakeAgent.calls.append({"task": task, "url": url, **self.kwargs})
        return FakeAgent.script.get(task) or _result()


@pytest.fixture
def fake(monkeypatch):
    FakeAgent.script, FakeAgent.calls = {}, []
    monkeypatch.setattr(sdk, "Agent", FakeAgent)
    return FakeAgent


def test_defaults_merge(tmp_path) -> None:
    tf = load_task_file(_write(tmp_path))
    a, b, _ = tf.tasks
    assert tf.resolved(a) == {"desktop": False, "env": None, "mode": "auto", "max_steps": 7,
                              "url": "https://example.test", "approve_all": False}
    assert tf.resolved(b)["url"] == "https://other.test" and tf.resolved(b)["approve_all"] is True


@pytest.mark.parametrize("text,needle", [
    ("tasks: []", "tasks"),
    ("tasks:\n  - name: x\n", "tasks[0].task"),
    ("tasks:\n  - {name: x, task: y, mode: yolo}", "tasks[0].mode"),
    ("tasks:\n  - {name: x, task: y, bogus: 1}", "bogus"),
    ("tasks:\n  - {name: x, task: y}\n  - {name: x, task: z}", "duplicate"),
    ("tasks: [\n", "line"),
    ("- just a list", "mapping"),
])
def test_validation_errors(tmp_path, text, needle) -> None:
    with pytest.raises(TaskFileError) as exc:
        load_task_file(_write(tmp_path, text))
    assert needle in str(exc.value)


def test_missing_file() -> None:
    with pytest.raises(TaskFileError, match="cannot read"):
        load_task_file("nope.yaml")


def test_run_passes_settings_to_agent(tmp_path, fake) -> None:
    outcomes = run_task_file(load_task_file(_write(tmp_path)))
    assert [o.passed for o in outcomes] == [True, True, True]
    first, second, _ = fake.calls
    assert first["mode"] == "auto" and first["max_steps"] == 7 and first["url"] == "https://example.test"
    assert first["on_approval"] is None  # deny by default
    assert second["url"] == "https://other.test" and second["on_approval"](sdk.ApprovalRequest("r", "ok?", "approval", "trust", "confirm")) is True


def test_expect_checks(tmp_path, fake) -> None:
    f = tmp_path / "out.txt"
    f.write_text("hello world", encoding="utf-8")
    fp = f.as_posix()
    missing = (tmp_path / "no.txt").as_posix()
    text = f"""
tasks:
  - {{name: ok, task: ok, expect: {{answer_contains: [sunny], file_exists: "{fp}", max_steps: 5,
      file_contains: {{path: "{fp}", text: hello}}}}}}
  - {{name: bad-answer, task: t1, expect: {{answer_contains: [Rome]}}}}
  - {{name: bad-steps, task: t2, expect: {{max_steps: 2}}}}
  - {{name: bad-exists, task: t3, expect: {{file_exists: "{missing}"}}}}
  - {{name: bad-contains, task: t4, expect: {{file_contains: {{path: "{fp}", text: nope}}}}}}
  - {{name: bad-status, task: t5, expect: {{status: failed}}}}
  - {{name: sandbox, task: t6, desktop: true, env: windows-sandbox, expect: {{file_exists: "{fp}"}}}}
"""
    outcomes = {o.name: o for o in run_task_file(load_task_file(_write(tmp_path, text)))}
    assert outcomes["ok"].passed
    assert "Rome" in outcomes["bad-answer"].reason
    assert "3 steps, limit 2" in outcomes["bad-steps"].reason
    assert "does not exist" in outcomes["bad-exists"].reason
    assert "does not contain 'nope'" in outcomes["bad-contains"].reason
    assert "expected 'failed'" in outcomes["bad-status"].reason
    assert "unsupported for env 'windows-sandbox'" in outcomes["sandbox"].reason


def test_denied_approval_fails_with_reason(tmp_path, fake) -> None:
    fake.script["do a"] = RunResult(status="denied", final_answer=None, steps=1, cost_usd=None, run_id="r9",
                                    denied_reason="Confirm purchase")
    o = run_task_file(load_task_file(_write(tmp_path)), only=["a"])[0]
    assert not o.passed and "approve_all" in o.reason and "Confirm purchase" in o.reason


def test_only_and_fail_fast(tmp_path, fake) -> None:
    tf = load_task_file(_write(tmp_path))
    assert [o.name for o in run_task_file(tf, only=["c"])] == ["c"]
    with pytest.raises(TaskFileError, match="no such task"):
        run_task_file(tf, only=["zzz"])
    fake.script["do a"] = _result(status="failed")
    assert [o.name for o in run_task_file(tf, fail_fast=True)] == ["a"]
    assert [o.name for o in run_task_file(tf)] == ["a", "b", "c"]


def test_agent_exception_is_a_failed_task(tmp_path, monkeypatch) -> None:
    class Boom(FakeAgent):
        async def arun(self, task, *, url=None):
            raise RuntimeError("no key")

    monkeypatch.setattr(sdk, "Agent", Boom)
    o = run_task_file(load_task_file(_write(tmp_path)), fail_fast=True)
    assert len(o) == 1 and not o[0].passed and "no key" in o[0].reason


def test_junit_structure(tmp_path, fake) -> None:
    fake.script["do b"] = _result(status="failed")
    outcomes = run_task_file(load_task_file(_write(tmp_path)))
    out = tmp_path / "sub" / "report.xml"
    write_junit(outcomes, out, classname="suite")
    suite = ET.parse(out).getroot().find("testsuite")
    assert suite is not None
    assert suite.get("name") == "operon" and suite.get("tests") == "3" and suite.get("failures") == "1"
    cases = suite.findall("testcase")
    assert [c.get("name") for c in cases] == ["a", "b", "c"]
    assert cases[0].get("classname") == "suite"
    failure = cases[1].find("failure")
    assert failure is not None and "failed" in (failure.get("message") or "")
    assert "run_id: r1" in (failure.text or "") and "/runs/r1" in (failure.text or "")
    assert cases[0].find("failure") is None


def test_cli_exit_codes_and_table(tmp_path, fake, capsys) -> None:
    path = _write(tmp_path)
    xml = tmp_path / "r.xml"
    assert main(["run", "--tasks", str(path), "--junit", str(xml)]) == 0
    assert "3/3 passed" in capsys.readouterr().out and xml.exists()
    fake.script["do c"] = _result(status="failed")
    assert main(["run", "--tasks", str(path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "PASS" in out and "2/3 passed" in out
    assert main(["run", "--tasks", str(tmp_path / "missing.yaml")]) == 2
    assert main(["run"]) == 2


def test_real_sdk_with_scripted_model(monkeypatch, tmp_path) -> None:
    loop, _ = build_loop(monkeypatch, tmp_path, [STOP])
    loop.verifier_service = _StopVerifier()
    monkeypatch.setattr(loops, "get_agent_loop", lambda: loop)
    monkeypatch.setattr(loops, "desktop_loop_holds_run", lambda run_id: False)
    monkeypatch.setattr(ws_stream, "publish_event", lambda event: None)
    orig = sdk.Agent.__init__

    def fast_init(self, *a, **kw):
        orig(self, *a, poll_s=0.01, **kw)

    monkeypatch.setattr(sdk.Agent, "__init__", fast_init)
    text = "tasks:\n  - {name: weather, task: Find the weather, mode: auto, expect: {answer_contains: [paris]}}\n"
    outcomes = run_task_file(load_task_file(_write(tmp_path, text)))
    assert outcomes[0].passed, outcomes[0].reason
    assert outcomes[0].status == "succeeded" and outcomes[0].run_id


@pytest.mark.parametrize(
    ("configured", "task_env", "passes", "reason"),
    [
        ("windows-sandbox", "", False, "unsupported for env 'windows-sandbox'"),  # configured guest, no env in YAML
        ("linux-container", "", False, "unsupported for env 'linux-container'"),
        ("osworld", "", False, "unsupported for env 'osworld'"),
        ("live", "", True, ""),  # live provider: host check as before
        ("", "", True, ""),  # nothing configured: live
        ("live", "env: windows-sandbox,", False, "unsupported for env 'windows-sandbox'"),  # explicit env wins
        ("windows-sandbox", "env: live,", True, ""),
    ],
)
def test_file_checks_use_the_environment_the_run_used(
    tmp_path, fake, monkeypatch, configured, task_env, passes, reason
) -> None:
    if configured:
        monkeypatch.setenv("OPERON_ENVIRONMENT", configured)
    else:
        monkeypatch.delenv("OPERON_ENVIRONMENT", raising=False)
    f = tmp_path / "out.txt"
    f.write_text("hello", encoding="utf-8")
    text = f'tasks:\n  - {{name: t, task: t, desktop: true, {task_env} expect: {{file_exists: "{f.as_posix()}"}}}}\n'
    outcome = run_task_file(load_task_file(_write(tmp_path, text)))[0]
    assert outcome.passed is passes
    assert reason in outcome.reason


def test_browser_task_file_check_is_unaffected_by_configured_guest(tmp_path, fake, monkeypatch) -> None:
    monkeypatch.setenv("OPERON_ENVIRONMENT", "windows-sandbox")
    f = tmp_path / "out.txt"
    f.write_text("hello", encoding="utf-8")
    text = f'tasks:\n  - {{name: t, task: t, expect: {{file_exists: "{f.as_posix()}"}}}}\n'
    assert run_task_file(load_task_file(_write(tmp_path, text)))[0].passed

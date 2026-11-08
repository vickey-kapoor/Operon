"""Eval plumbing for registry models: --model, invalid-action rate, GPU cost, reports."""

from __future__ import annotations

import json

import httpx
import pytest

from evals import grounding, live_bench


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    for name in ("OPERON_MODEL_PROFILE", "OPERON_BROWSER_BACKEND", "OPERON_DESKTOP_BACKEND", "MAX_RUN_COST_USD"):
        monkeypatch.delenv(name, raising=False)


def _result(name: str, seconds: int, invalid: int, replies: int) -> live_bench.TaskResult:
    return live_bench.TaskResult(
        name, "c", "r", "succeeded", "success", "-", replies, 0, seconds, replies, 0, "", "-",
        invalid_replies=invalid, model_replies=replies,
    )


def test_run_replies_counts_invalid_parses(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(live_bench, "runs_dir", lambda: tmp_path)
    for step, invalid in (("step_1", False), ("step_2", True), ("step_3", False)):
        (tmp_path / "run-1" / step).mkdir(parents=True)
        (tmp_path / "run-1" / step / "model_parsed.json").write_text(json.dumps({"invalid": invalid}))
    (tmp_path / "run-1" / "step_3" / "model_raw.json").write_text("{}")
    assert live_bench.run_replies("run-1") == (1, 3)
    # Adapter steps count as the agent's own steps, not as fallback steps.
    assert live_bench._backends_used("run-1") == (1, 2)


def test_summary_reports_the_invalid_action_rate() -> None:
    summary = live_bench.summarize([_result("a", 60, 1, 4), _result("b", 60, 0, 6)])
    assert summary["model_replies"] == 10 and summary["invalid_replies"] == 1
    assert summary["invalid_action_rate"] == 0.1
    assert "invalid_action_rate" not in live_bench.summarize([_result("a", 60, 0, 0)])


def test_gpu_cost_is_wall_time_at_the_profiles_rate() -> None:
    from operon.agent.backends.model_adapter.registry import resolve_profile

    results = [_result("a", 1800, 0, 1), _result("b", 1800, 0, 1)]
    assert live_bench.gpu_cost_usd(results, resolve_profile("qwen3-vl-32b")) == 2.5
    assert live_bench.gpu_cost_usd(results, resolve_profile("claude-opus-5-5")) is None


def test_model_flag_checks_the_profile_and_the_server_settings(monkeypatch) -> None:
    assert live_bench.resolve_model_profile(None, "browser") is None
    assert live_bench.resolve_model_profile("qwen3-vl-32b", "browser").name == "qwen3-vl-32b"
    with pytest.raises(SystemExit, match="unknown model profile"):
        live_bench.resolve_model_profile("nope", "browser")
    assert live_bench.resolve_model_profile("gemini-computer-use", "desktop").name == "gemini-computer-use"
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "claude_computer_use")
    with pytest.raises(SystemExit, match="backend=model"):
        live_bench.resolve_model_profile("qwen3-vl-32b", "browser")
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "ui-tars-1.5-7b")
    with pytest.raises(SystemExit, match="another model"):
        live_bench.resolve_model_profile("qwen3-vl-32b", "browser")


async def test_report_records_the_profile_and_gpu_cost(tmp_path, monkeypatch) -> None:
    async def fake_run_task(client, task, judge, **kwargs):
        return _result(task.name, 360, 1, 5)

    monkeypatch.setattr(live_bench, "run_task", fake_run_task)
    real_client = httpx.AsyncClient
    health = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    monkeypatch.setattr(live_bench.httpx, "AsyncClient", lambda **kw: real_client(transport=health, **kw))
    args = live_bench.build_parser(default_suite="smoke").parse_args(
        ["--out-dir", str(tmp_path), "--model", "qwen3-vl-32b", "--only", "click_link", "--json"]
    )
    assert await live_bench.run_bench(args, judge=None) == 0
    (saved,) = tmp_path.glob("partial-*.json")
    report = json.loads(saved.read_text())
    assert report["model_profile"] == "qwen3-vl-32b" and report["model_adapter"] == "openai_compat"
    assert report["harness_profile"] == "thin"
    assert report["summary"]["gpu_cost_usd"] == 0.25 and report["summary"]["invalid_action_rate"] == 0.2


def test_previous_report_only_compares_the_same_model(tmp_path) -> None:
    for stamp, profile in (("1", "qwen3-vl-32b"), ("2", None)):
        (tmp_path / f"smoke-{stamp}.json").write_text(json.dumps({"model_profile": profile, "budget_stop": None}))
    assert live_bench._previous_report(tmp_path, "smoke")["model_profile"] is None
    assert live_bench._previous_report(tmp_path, "smoke", "qwen3-vl-32b")["model_profile"] == "qwen3-vl-32b"
    assert live_bench._previous_report(tmp_path, "smoke", "ui-tars-1.5-7b") is None


def test_grounding_builds_registry_backends_per_sample(monkeypatch) -> None:
    from operon.agent.backends.desktop_computer_use import DesktopComputerUseBackend
    from operon.agent.backends.model_adapter.backend import ModelAdapterBackend

    monkeypatch.setenv("OPENWEIGHT_BASE_URL", "http://gpu:8000/v1")
    sample = grounding.Sample(
        id="pro-1", set="pro", instruction="close", image="x.png", bbox=(0, 0, 1, 1), data_type="icon", source="vscode",
    )
    qwen = grounding.BackendFactory("model", "qwen3-vl-32b", None)(sample)
    assert isinstance(qwen, ModelAdapterBackend) and qwen.environment == "desktop"
    gemini = grounding.BackendFactory("model", "gemini-computer-use", "low")(sample)
    assert isinstance(gemini, DesktopComputerUseBackend) and gemini.client.environment == "ENVIRONMENT_DESKTOP"


def test_grounding_reuses_one_client_per_kind(monkeypatch) -> None:
    monkeypatch.setenv("OPENWEIGHT_BASE_URL", "http://gpu:8000/v1")
    factory = grounding.BackendFactory("model", "qwen3-vl-32b", None)

    def sample(n: int) -> grounding.Sample:
        return grounding.Sample(
            id=f"pro-{n}", set="pro", instruction="close", image="x.png", bbox=(0, 0, 1, 1), data_type="icon", source="vscode",
        )

    first, second = factory(sample(1)), factory(sample(2))
    assert first is not second  # fresh conversation state per sample
    assert first.client is second.client  # one connection pool


@pytest.mark.parametrize(
    ("harness", "check", "expected"),
    [("thin", "", "thin"), ("thin", "own", "thin-own"), ("full", "own", "full")],
)
def test_report_harness_names_the_own_done_mode(monkeypatch, harness, check, expected) -> None:
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", harness)
    monkeypatch.setenv("OPERON_COMPLETION_CHECK", check)
    assert live_bench.harness_label(None) == expected


def test_gpu_time_counts_toward_the_cost_cap() -> None:
    results = [_result("a", 3600, 0, 10)]
    assert live_bench.budget_stop(results, 5, 2.0) is None  # no token cost
    stopped = live_bench.budget_stop(results, 5, 2.0, gpu_usd=2.5)
    assert stopped and stopped["spent_usd"] >= 2.5


def test_previous_report_matches_the_completion_mode(tmp_path) -> None:
    for stamp, harness in (("20261101T000000Z", "thin"), ("20261102T000000Z", "thin-own")):
        (tmp_path / f"full-{stamp}.json").write_text(
            json.dumps({"harness_profile": harness, "model_profile": None, "results": []}), encoding="utf-8"
        )
    assert live_bench._previous_report(tmp_path, "full", None, "thin")["harness_profile"] == "thin"
    assert live_bench._previous_report(tmp_path, "full", None, "thin-own")["harness_profile"] == "thin-own"

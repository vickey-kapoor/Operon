"""Claude and GPT next to Gemini (KAN-168): `--model` and report labels."""

from __future__ import annotations

import json

import httpx
import pytest

from evals import live_bench


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    for name in ("OPERON_MODEL_PROFILE", "OPERON_BROWSER_BACKEND", "OPERON_DESKTOP_BACKEND", "MAX_RUN_COST_USD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")


# Batch cells no longer take model= (KAN-184); see tests/evals/test_bench_batch.py.


# --- --model and the report -------------------------------------------------------------


def test_model_flag_checks_the_profile_and_the_server_settings(monkeypatch) -> None:
    assert live_bench.resolve_model_profile(None, "browser") is None
    assert live_bench.resolve_model_profile("claude-sonnet-5-5", "browser").name == "claude-sonnet-5-5"
    with pytest.raises(SystemExit, match="unknown model profile"):
        live_bench.resolve_model_profile("nope", "browser")
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "computer_use")
    with pytest.raises(SystemExit, match="backend=model"):
        live_bench.resolve_model_profile("claude-sonnet-5-5", "browser")
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "model")
    monkeypatch.setenv("OPERON_MODEL_PROFILE", "openai-computer-use")
    with pytest.raises(SystemExit, match="another model"):
        live_bench.resolve_model_profile("claude-sonnet-5-5", "browser")


def test_a_registry_model_is_labelled_thin_own_like_gemini_thin(monkeypatch) -> None:
    profile = live_bench.resolve_model_profile("claude-sonnet-5-5", "browser")
    assert live_bench.harness_label("browser", profile) == "thin-own"
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", "thin")
    assert live_bench.harness_label("browser") == "thin-own"  # Gemini thin: the same label, so they compare


def test_previous_report_only_compares_the_same_model(tmp_path) -> None:
    for stamp, profile in (("1", "claude-sonnet-5-5"), ("2", None)):
        (tmp_path / f"smoke-{stamp}.json").write_text(json.dumps({"model_profile": profile, "budget_stop": None}))
    assert live_bench._previous_report(tmp_path, "smoke")["model_profile"] is None
    assert live_bench._previous_report(tmp_path, "smoke", model_profile="claude-sonnet-5-5")["model_profile"] == (
        "claude-sonnet-5-5"
    )
    assert live_bench._previous_report(tmp_path, "smoke", model_profile="openai-computer-use") is None


async def test_report_records_the_profile(tmp_path, monkeypatch) -> None:
    async def fake_run_task(client, task, judge, **kwargs):
        return live_bench.TaskResult(task.name, "c", "r", "succeeded", "success", "-", 3, 0, 10, 3, 0, "", "-")

    monkeypatch.setattr(live_bench, "run_task", fake_run_task)
    real_client = httpx.AsyncClient
    health = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    monkeypatch.setattr(live_bench.httpx, "AsyncClient", lambda **kw: real_client(transport=health, **kw))
    args = live_bench.build_parser(default_suite="smoke").parse_args(
        ["--out-dir", str(tmp_path), "--model", "openai-computer-use", "--only", "click_link", "--json"]
    )
    assert await live_bench.run_bench(args, judge=None) == 0
    (saved,) = tmp_path.glob("partial-*.json")
    report = json.loads(saved.read_text())
    assert report["model_profile"] == "openai-computer-use" and report["model_adapter"] == "openai_native"
    assert report["model_id"] == "gpt-6.1-sol" and report["harness_profile"] == "thin-own"

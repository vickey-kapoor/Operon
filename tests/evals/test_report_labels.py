"""What a live benchmark report records about the run: harness, critic and baseline."""

from __future__ import annotations

import json

import pytest

from evals import live_bench


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env
    for name in (
        "OPERON_BROWSER_BACKEND", "OPERON_DESKTOP_BACKEND", "OPERON_BROWSER_MODEL", "OPERON_DESKTOP_MODEL",
        "OPERON_BROWSER_VERIFIER_MODEL", "OPERON_DESKTOP_VERIFIER_MODEL", "OPERON_CRITIC", "MAX_RUN_COST_USD",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("mode", "harness", "expected"),
    [("browser", "thin", "thin-own"), ("desktop", "thin", "thin-own"), ("browser", "full", "full"), ("desktop", "", "full")],
)
def test_a_gemini_computer_use_report_names_its_harness(monkeypatch, mode, harness, expected) -> None:
    # A thin run ends on the model's own "done", so it is "thin-own"; unset means the default, full.
    monkeypatch.setenv(f"OPERON_{mode.upper()}_BACKEND", "computer_use")
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", harness)
    assert live_bench.harness_label(mode) == expected


@pytest.mark.parametrize(("harness", "expected"), [("thin", "thin-own"), ("", "full")])
def test_the_default_backend_is_gemini_computer_use(monkeypatch, harness, expected) -> None:
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", harness)
    assert live_bench.harness_label() == expected


def test_the_json_backend_always_runs_full(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "json")
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", "thin")
    assert live_bench.harness_label() == "full"


def test_the_harness_label_reads_the_dotenv_the_server_loads(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("OPERON_BROWSER_BACKEND=json\n", encoding="utf-8")
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", "thin")
    assert live_bench.harness_label() == "full"
    (tmp_path / ".env").write_text("OPERON_BROWSER_BACKEND=computer_use\n", encoding="utf-8")
    assert live_bench.harness_label() == "thin-own"


def test_the_report_names_a_full_runs_critic(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", "thin")
    assert live_bench.critic_label() is None  # a thin run has no step critic
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", "full")
    assert live_bench.critic_label() == "gemini:gemini-3.5-flash"
    monkeypatch.setenv("OPERON_BROWSER_VERIFIER_MODEL", "gemini-3-flash-preview")
    assert live_bench.critic_label() == "gemini:gemini-3-flash-preview"
    monkeypatch.setenv("OPERON_CRITIC", "same")
    assert live_bench.critic_label() == "same:gemini-3.5-flash"
    assert live_bench.critic_label("desktop") == "same:gemini-3.5-flash"
    monkeypatch.setenv("OPERON_DESKTOP_MODEL", "gemini-3-flash-preview")
    assert live_bench.critic_label("hybrid") == "same:gemini-3-flash-preview"  # a hybrid run is a desktop run


def test_previous_report_matches_the_completion_mode(tmp_path) -> None:
    for stamp, harness in (("20261101T000000Z", "full"), ("20261102T000000Z", "thin-own")):
        (tmp_path / f"full-{stamp}.json").write_text(
            json.dumps({"harness_profile": harness, "backend": "computer_use", "results": []}), encoding="utf-8"
        )
    assert live_bench._previous_report(tmp_path, "full", "full")["harness_profile"] == "full"
    assert live_bench._previous_report(tmp_path, "full", "thin-own")["harness_profile"] == "thin-own"


def test_previous_report_skips_earlier_runs_of_other_models(tmp_path) -> None:
    reports = {
        "20261101T000000Z": {"backend": "computer_use"},
        "20261102T000000Z": {"backend": "claude_computer_use"},
        "20261103T000000Z": {"backend": "model", "model_profile": "openai-computer-use"},
    }
    for stamp, report in reports.items():
        (tmp_path / f"full-{stamp}.json").write_text(json.dumps({**report, "results": []}), encoding="utf-8")
    assert live_bench._previous_report(tmp_path, "full")["backend"] == "computer_use"

"""What a live benchmark report records about the run: harness and baseline."""

from __future__ import annotations

import json

import pytest

from evals import live_bench


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env
    for name in (
        "OPERON_BROWSER_BACKEND", "OPERON_DESKTOP_BACKEND", "OPERON_BROWSER_MODEL", "OPERON_DESKTOP_MODEL",
        "OPERON_BROWSER_VERIFIER_MODEL", "OPERON_DESKTOP_VERIFIER_MODEL", "MAX_RUN_COST_USD",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("mode", ["browser", "desktop", "hybrid"])
def test_every_report_names_the_one_harness(monkeypatch, mode) -> None:
    # The Full harness was removed (KAN-183): a run ends on the model's own "done", so it is always "thin-own".
    monkeypatch.setenv(f"OPERON_{mode.upper()}_BACKEND", "computer_use")
    assert live_bench.harness_label(mode) == "thin-own"


def test_the_default_backend_is_gemini_computer_use() -> None:
    assert live_bench.harness_label() == "thin-own"


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


def test_a_report_with_no_backend_setting_records_the_gemini_default() -> None:
    # Unset means the server default; recording it keeps the run on the dashboard's Gemini tabs.
    args = live_bench.build_parser().parse_args([])
    report = live_bench.build_report(args, "20261111T000000Z", [], repeat=1, planned=10, stopped=None, complete=True)
    assert report["backend"] == "computer_use"
    assert report["browser_model"] == "gemini-3.5-flash"

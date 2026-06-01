"""The demo benchmark URL/intent live as named constants, not inline literals."""

from __future__ import annotations

import inspect

from operon.agent import loop
from operon.agent.loop import (
    DEFAULT_BENCHMARK_INTENT,
    DEFAULT_BENCHMARK_URL,
    AgentLoop,
)


def test_benchmark_defaults_are_named_constants() -> None:
    assert DEFAULT_BENCHMARK_URL.startswith("https://")
    assert DEFAULT_BENCHMARK_INTENT


def test_run_live_benchmark_defaults_reference_the_constants() -> None:
    params = inspect.signature(AgentLoop.run_live_benchmark).parameters
    assert params["intent"].default == DEFAULT_BENCHMARK_INTENT
    assert params["benchmark_url"].default == DEFAULT_BENCHMARK_URL


def test_module_no_longer_hardcodes_the_url_at_the_call_site() -> None:
    # The literal must appear only in the single named constant, so the demo
    # target is easy to find and override rather than buried in a signature.
    src = inspect.getsource(loop.AgentLoop.run_live_benchmark)
    assert "practice-automation.com" not in src

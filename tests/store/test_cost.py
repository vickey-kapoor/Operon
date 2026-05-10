"""Tests for src/store/cost.py run-level cost aggregation."""
from __future__ import annotations

from unittest.mock import patch


def test_run_cost_returns_none_for_unknown_run() -> None:
    from src.store.cost import run_cost_usd
    with patch("src.api.observer.summarize_run_usage", return_value=None):
        assert run_cost_usd("no-such-run") is None


def test_run_cost_returns_float_when_usage_exists() -> None:
    from src.store.cost import run_cost_usd

    class _FakeUsage:
        estimated_cost_usd = 0.1234

    with patch("src.api.observer.summarize_run_usage", return_value=_FakeUsage()):
        result = run_cost_usd("some-run")
    assert result == 0.1234

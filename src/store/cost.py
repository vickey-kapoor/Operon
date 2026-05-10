"""Run-level cost aggregation — totals cost_usd from per-step token usage."""
from __future__ import annotations


def run_cost_usd(run_id: str) -> float | None:
    """Return total estimated cost in USD for a run, or None if unavailable."""
    from src.api.observer import summarize_run_usage
    usage = summarize_run_usage(run_id)
    cost = usage.estimated_cost_usd if usage else None
    return cost if cost else None

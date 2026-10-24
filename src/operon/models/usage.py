"""Schemas and helpers for model usage and cost estimation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from operon.models.common import StrictModel

ModelProvider = Literal["gemini", "anthropic"]


class ModelUsage(StrictModel):
    """Captured provider-reported usage for one model request."""

    provider: ModelProvider
    model: str = Field(min_length=1)
    request_kind: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    thoughts_tokens: int | None = Field(default=None, ge=0)
    input_cost_usd: float | None = Field(default=None, ge=0)
    output_cost_usd: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class UsageAggregate(StrictModel):
    """Aggregated usage across one run or many runs."""

    request_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    by_model: dict[str, "UsageAggregate"] = Field(default_factory=dict)


# List prices in USD per million tokens: (input, output, cache read). Anthropic
# cache writes (5-minute TTL) bill at 1.25x input; Gemini bills thinking tokens as
# output. Checked 2026-10-24; update when providers change their prices.
_ANTHROPIC_PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
_ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25
_GEMINI_PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-3.5-flash": (1.50, 9.00, 0.15),
    "gemini-3-flash-preview": (0.50, 3.00, 0.05),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.5-computer-use-preview-10-2025": (1.25, 10.00, 0.125),
}


def estimate_usage_cost(
    *,
    provider: ModelProvider,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    thoughts_tokens: int | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Estimate (input, output, total) cost in USD from list prices.

    Token fields are as each provider reports them. Anthropic's ``input_tokens``
    excludes cache reads and writes, which bill at their own rates. Gemini's
    prompt count includes cached tokens, and its candidate count excludes the
    thinking tokens, which bill as output.
    """
    prices = (_ANTHROPIC_PRICES if provider == "anthropic" else _GEMINI_PRICES).get(model.strip().lower())
    if prices is None:
        return None, None, None
    input_rate, output_rate, cache_read_rate = prices
    cached = cache_read_input_tokens or 0
    if input_tokens is None and not cached and not cache_creation_input_tokens:
        input_cost = None
    elif provider == "anthropic":
        input_cost = (
            (input_tokens or 0) * input_rate
            + (cache_creation_input_tokens or 0) * input_rate * _ANTHROPIC_CACHE_WRITE_MULTIPLIER
            + cached * cache_read_rate
        ) / 1_000_000
    else:
        uncached = max(0, (input_tokens or 0) - cached)
        input_cost = (uncached * input_rate + cached * cache_read_rate) / 1_000_000
    if output_tokens is None and not thoughts_tokens:
        output_cost = None
    else:
        billed_output = (output_tokens or 0) + (thoughts_tokens or 0 if provider == "gemini" else 0)
        output_cost = billed_output * output_rate / 1_000_000
    if input_cost is None and output_cost is None:
        return None, None, None
    total_cost = round((input_cost or 0.0) + (output_cost or 0.0), 8)
    return (
        None if input_cost is None else round(input_cost, 8),
        None if output_cost is None else round(output_cost, 8),
        total_cost,
    )

"""Schemas and helpers for model usage and cost estimation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.models.common import StrictModel

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
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    by_model: dict[str, "UsageAggregate"] = Field(default_factory=dict)


def estimate_usage_cost(*, provider: ModelProvider, model: str, input_tokens: int | None, output_tokens: int | None) -> tuple[float | None, float | None, float | None]:
    """Estimate provider cost using current list pricing for configured models."""
    input_rate, output_rate = _pricing_for_model(provider=provider, model=model)
    if input_rate is None and output_rate is None:
        return None, None, None

    input_cost = None if input_tokens is None or input_rate is None else (input_tokens / 1_000_000) * input_rate
    output_cost = None if output_tokens is None or output_rate is None else (output_tokens / 1_000_000) * output_rate
    total_cost = None
    if input_cost is not None or output_cost is not None:
        total_cost = round((input_cost or 0.0) + (output_cost or 0.0), 8)
    if input_cost is not None:
        input_cost = round(input_cost, 8)
    if output_cost is not None:
        output_cost = round(output_cost, 8)
    return input_cost, output_cost, total_cost


def _pricing_for_model(*, provider: ModelProvider, model: str) -> tuple[float | None, float | None]:
    normalized = model.strip().lower()
    if provider == "anthropic":
        if normalized == "claude-sonnet-4-20250514":
            return 3.0, 15.0
        return None, None

    if normalized == "gemini-3-flash-preview":
        return 0.50, 3.00
    if normalized == "gemini-2.5-flash":
        return 0.30, 2.50
    if normalized == "gemini-2.5-computer-use-preview-10-2025":
        return 1.25, 10.00
    return None, None

"""Cost estimates from provider-reported token usage."""

from __future__ import annotations

import pytest

from operon.models.usage import estimate_usage_cost


def test_gemini_cost_discounts_cached_prompt_tokens_and_bills_thinking_as_output() -> None:
    # Gemini's prompt count includes the cached tokens; thinking is billed as output.
    input_cost, output_cost, _ = estimate_usage_cost(
        provider="gemini",
        model="gemini-3.5-flash",
        input_tokens=10_000,
        output_tokens=100,
        cache_read_input_tokens=4_000,
        thoughts_tokens=400,
    )
    assert input_cost == pytest.approx((6_000 * 1.5 + 4_000 * 0.15) / 1_000_000)
    assert output_cost == pytest.approx(500 * 9 / 1_000_000)


@pytest.mark.parametrize("model", ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"])
def test_current_gemini_models_are_priced(model) -> None:
    assert estimate_usage_cost(provider="gemini", model=model, input_tokens=1, output_tokens=1)[2] is not None


def test_an_unknown_model_has_no_estimate() -> None:
    assert estimate_usage_cost(provider="gemini", model="gemini-x", input_tokens=1, output_tokens=1) == (None, None, None)
    # Usage files from older runs of other providers stay readable but are not priced.
    assert estimate_usage_cost(provider="anthropic", model="gemini-3.5-flash", input_tokens=1, output_tokens=1) == (None, None, None)

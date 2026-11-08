"""Cost estimates from provider-reported token usage."""

from __future__ import annotations

import pytest

from operon.models.usage import estimate_usage_cost


def test_anthropic_cost_counts_cache_writes_and_reads_at_their_own_rates() -> None:
    # Opus 5.5: $4 input, $20 output, $0.20 cache read, cache write 1.25x input.
    _, _, total = estimate_usage_cost(
        provider="anthropic",
        model="claude-opus-5-5",
        input_tokens=1_000,
        output_tokens=500,
        cache_creation_input_tokens=2_000,
        cache_read_input_tokens=10_000,
    )
    assert total == pytest.approx((1_000 * 4 + 2_000 * 5 + 10_000 * 0.2 + 500 * 20) / 1_000_000)


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


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"])
def test_current_claude_models_are_priced(model) -> None:
    assert estimate_usage_cost(provider="anthropic", model=model, input_tokens=1, output_tokens=1)[2] is not None


def test_an_unknown_model_has_no_estimate() -> None:
    assert estimate_usage_cost(provider="anthropic", model="claude-x", input_tokens=1, output_tokens=1) == (None, None, None)

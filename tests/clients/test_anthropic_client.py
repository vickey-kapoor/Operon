"""Thinking settings the plain Anthropic client sends per model (the full harness's
critic and the planner use it)."""

from unittest.mock import AsyncMock

import pytest

from operon.clients.anthropic import AnthropicHttpClient, thinking_off_fields


def test_sonnet_5_5_turns_thinking_off_with_between_tools() -> None:
    # Sonnet 5.5 answers {"type": "disabled"} with a 400.
    assert thinking_off_fields("claude-sonnet-5-5") == {"thinking": {"type": "between_tools"}}


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-2", "claude-mythos-5-2"])
def test_models_that_always_think_get_low_effort_and_no_thinking_field(model: str) -> None:
    assert thinking_off_fields(model) == {"output_config": {"effort": "low"}}


@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5", "claude-haiku-5-5"])
def test_other_models_keep_disabled_thinking(model: str) -> None:
    assert thinking_off_fields(model) == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_the_critic_request_for_sonnet_5_5_never_sends_disabled_thinking(tmp_path) -> None:
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = AnthropicHttpClient(api_key="test", model="claude-sonnet-5-5")
    client._post_json = AsyncMock(return_value={"content": [{"type": "text", "text": "{}"}]})

    await client.generate_verification("Did the step work?", str(shot))
    await client.generate_policy("Plan the next step.")

    for call in client._post_json.await_args_list:
        payload = call.args[0]
        assert payload["thinking"] == {"type": "between_tools"}
        assert "temperature" not in payload

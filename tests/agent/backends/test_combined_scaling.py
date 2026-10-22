"""CombinedPerceptionPolicyService coordinate space and retry behavior.

Gemini receives a screenshot downscaled to 1280 px wide, so the coordinates it
returns are in that image's space. The desktop executor clicks in native screen
pixels, so the combined service must map both perception and action coordinates
back up before anything acts on them.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from operon.agent.backends.combined import CombinedPerceptionPolicyService
from operon.clients.gemini import GeminiClientError
from operon.models.capture import CaptureFrame
from operon.models.policy import ActionType


def _raw(action: dict) -> str:
    return json.dumps({
        "perception": {
            "summary": "File Explorer is open on Home.",
            "page_hint": "file_explorer",
            "confidence": 0.9,
            "visible_elements": [
                {"element_id": "addr", "element_type": "input", "label": "Address bar",
                 "x": 400, "y": 40, "width": 300, "height": 20, "is_interactable": True, "confidence": 0.9},
            ],
        },
        "action": action,
        "rationale": "type the path",
        "confidence": 0.9,
        "active_subgoal": "navigate",
    })


def _make_service(scale_ratio: float = 1280 / 1920) -> tuple[CombinedPerceptionPolicyService, MagicMock]:
    gemini = MagicMock()
    gemini.generate_perception = AsyncMock()
    gemini.latest_perception_scale_ratio.return_value = scale_ratio
    gemini.latest_usage.return_value = None
    svc = CombinedPerceptionPolicyService.__new__(CombinedPerceptionPolicyService)
    svc.gemini_client = gemini
    svc._prompt_template = "intent={intent} subgoal={current_subgoal} step={step_count} prev={previous_summary} retries={retry_counts} hints={advisory_hints}"
    svc._cached_decision = {}
    svc._last_debug_artifacts = None
    svc._advisory_hints = {}
    svc._perception_only = False
    svc._image_size = {}
    return svc, gemini


def _make_state() -> MagicMock:
    state = MagicMock()
    state.run_id = "run1"
    state.intent = "Open File Explorer and go to C:\\Windows"
    state.current_subgoal = None
    state.step_count = 1
    state.observation_history = []
    state.action_history = []
    state.retry_counts = {}
    return state


def _frame(tmp_path) -> CaptureFrame:
    step_dir = tmp_path / "step_1"
    step_dir.mkdir()
    return CaptureFrame(artifact_path=str(step_dir / "before.png"), width=1920, height=1080)


def test_parse_maps_perception_and_action_coordinates_to_native_pixels() -> None:
    raw = _raw({"action_type": "drag", "x": 400, "y": 40, "x_end": 960, "y_end": 360})

    perception, decision = CombinedPerceptionPolicyService._parse_combined_output(
        raw, "before.png", scale_ratio=1280 / 1920,
    )

    element = perception.visible_elements[0]
    assert (element.x, element.y, element.width, element.height) == (600, 60, 450, 30)
    action = decision.action
    assert (action.x, action.y, action.x_end, action.y_end) == (600, 60, 1440, 540)


def test_parse_leaves_coordinates_alone_without_downscaling() -> None:
    raw = _raw({"action_type": "click", "x": 400, "y": 40})

    perception, decision = CombinedPerceptionPolicyService._parse_combined_output(raw, "before.png")

    assert perception.visible_elements[0].x == 400
    assert (decision.action.x, decision.action.y) == (400, 40)


def test_parse_keeps_coordinate_free_actions_coordinate_free() -> None:
    raw = _raw({"action_type": "launch_app", "text": "notepad"})

    _, decision = CombinedPerceptionPolicyService._parse_combined_output(raw, "before.png", scale_ratio=0.5)

    assert decision.action.action_type is ActionType.LAUNCH_APP
    assert decision.action.x is None and decision.action.x_end is None


@pytest.mark.asyncio
async def test_perceive_scales_the_cached_decision(tmp_path) -> None:
    svc, gemini = _make_service()
    gemini.generate_perception.return_value = _raw({"action_type": "type", "text": "C:\\Windows", "x": 400, "y": 40})
    state = _make_state()

    await svc.perceive(_frame(tmp_path), state)
    decision = await svc.choose_action(state, MagicMock())

    assert (decision.action.x, decision.action.y) == (600, 60)


@pytest.mark.asyncio
async def test_perceive_asks_again_after_an_empty_or_malformed_response(tmp_path) -> None:
    svc, gemini = _make_service()
    gemini.generate_perception.side_effect = [
        GeminiClientError("Gemini response candidate content was missing."),
        _raw({"action_type": "click", "x": 400, "y": 40}),
    ]

    perception = await svc.perceive(_frame(tmp_path), _make_state())

    assert gemini.generate_perception.await_count == 2
    assert perception.visible_elements[0].x == 600


@pytest.mark.asyncio
async def test_perceive_gives_up_after_the_retry_also_fails(tmp_path) -> None:
    svc, gemini = _make_service()
    gemini.generate_perception.side_effect = [
        GeminiClientError("Gemini response candidate content was missing."),
        GeminiClientError("Gemini response candidate content was missing."),
    ]

    with pytest.raises(GeminiClientError):
        await svc.perceive(_frame(tmp_path), _make_state())


@pytest.mark.asyncio
async def test_prompt_states_the_size_of_the_image_gemini_receives(tmp_path) -> None:
    """Told nothing, Gemini guesses: sometimes native screen pixels, sometimes the image's."""
    svc, gemini = _make_service()
    gemini.generate_perception.return_value = _raw({"action_type": "click", "x": 400, "y": 40})

    await svc.perceive(_frame(tmp_path), _make_state())

    prompt = gemini.generate_perception.call_args.args[0]
    assert "the screenshot is 1280x720 pixels" in prompt
    assert "y in 0..719" in prompt


def test_sent_image_size_matches_the_downscale() -> None:
    from operon.clients.gemini import sent_image_size

    assert sent_image_size(1920, 1080) == (1280, 720)
    assert sent_image_size(2560, 1440) == (1280, 720)
    assert sent_image_size(1024, 768) == (1024, 768)

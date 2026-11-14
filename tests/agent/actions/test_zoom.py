"""Zoom before clicking on dense screens (KAN-82, OPERON_ZOOM)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from operon.agent.actions import zoom
from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
from operon.core.paths import prompts_dir
from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.state import AgentState
from operon.models.usage import ModelUsage
from tests.fixtures.paths import unique_artifact_dir

DENSE = zoom.ZoomSettings(mode=zoom.DENSE)


def _usage(cost: float, tokens: int) -> ModelUsage:
    return ModelUsage(
        provider="gemini", model="gemini-3.5-flash", request_kind="computer_use",
        input_tokens=tokens, output_tokens=10, estimated_cost_usd=cost,
    )


def _reply(name: str, x: float, y: float, **extra) -> dict:
    call = {"name": name, "args": {"x": x, "y": y, **extra}}
    return {
        "perception": {"summary": f"Tool call: {name}.", "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "function_call": call,
        "function_calls": [call],
        "rationale": "click the save icon",
        "confidence": 0.7,
        "active_subgoal": name,
        "model_content": {"parts": [{"function_call": call}]},
    }


def _text_reply(text: str) -> dict:
    return {
        "perception": {"summary": text, "page_hint": "unknown", "visible_elements": [], "confidence": 0.5},
        "action": {"action_type": "wait", "wait_ms": 1000},
        "rationale": text,
        "confidence": 0.4,
        "model_content": {"parts": [{"text": text}]},
    }


class ZoomClient:
    """The step's turn answers ``turn``; the one-shot zoom request answers ``zoomed``."""

    def __init__(self, turn: dict, zoomed: dict | Exception | None = None) -> None:
        self.turn, self.zoomed = turn, zoomed
        self.zoom_requests: list[tuple[str, tuple[int, int]]] = []
        self._usage: ModelUsage | None = None

    async def generate_turn(self, *, contents: list[dict]) -> dict:
        self._usage = _usage(0.01, 1000)
        return self.turn

    async def run_step(self, *, prompt: str, screenshot_path: str) -> dict:
        with Image.open(screenshot_path) as image:
            self.zoom_requests.append((prompt, image.size))
        if isinstance(self.zoomed, Exception):
            raise self.zoomed
        self._usage = _usage(0.004, 400)
        return self.zoomed

    def latest_usage(self) -> ModelUsage | None:
        return self._usage

    def build_function_response_content(self, *, function_name, screenshot_path, current_url, error=None) -> dict:
        return {"role": "user", "parts": [{"function_response": {"name": function_name, "response": {}}}]}


def _screen(width: int, height: int) -> CaptureFrame:
    path = unique_artifact_dir("zoom") / "before.png"
    Image.new("RGB", (width, height), "white").save(path)
    return CaptureFrame(artifact_path=str(path), width=width, height=height)


def _state() -> AgentState:
    return AgentState(run_id="zoom-run", intent="Save the file", status=RunStatus.RUNNING)


# ---------- settings ----------


@pytest.mark.parametrize(
    ("value", "mode"),
    [(None, "off"), ("", "off"), ("off", "off"), ("dense", "dense"), ("ALWAYS", "always"), ("on", "dense"), ("bogus", "off")],
)
def test_mode_from_env(monkeypatch, value, mode):
    if value is None:
        monkeypatch.delenv("OPERON_ZOOM", raising=False)
    else:
        monkeypatch.setenv("OPERON_ZOOM", value)
    assert zoom.zoom_settings().mode == mode


def test_off_by_default(monkeypatch):
    for name in ("OPERON_ZOOM", "OPERON_ZOOM_MIN_SIDE", "OPERON_ZOOM_FACTOR"):
        monkeypatch.delenv(name, raising=False)
    settings = zoom.zoom_settings()
    assert settings == zoom.ZoomSettings(mode="off", min_side=2000, factor=2.0)
    assert not settings.enabled


def test_factor_is_clamped_and_bad_numbers_fall_back(monkeypatch):
    monkeypatch.setenv("OPERON_ZOOM_FACTOR", "50")
    monkeypatch.setenv("OPERON_ZOOM_MIN_SIDE", "abc")
    settings = zoom.zoom_settings()
    assert settings.factor == 6.0
    assert settings.min_side == 2000


def test_dense_gate_uses_the_longer_side():
    assert DENSE.applies_to(2560, 1440)
    assert DENSE.applies_to(1440, 2560)
    assert not DENSE.applies_to(1920, 1080)
    assert zoom.ZoomSettings(mode="always").applies_to(800, 600)
    assert not zoom.ZoomSettings().applies_to(4000, 3000)


# ---------- geometry ----------


def test_crop_is_centred_and_kept_on_screen():
    assert zoom.crop_box(2000, 1000, 4000, 2000, 2.0) == zoom.CropBox(1000, 500, 2000, 1000)
    # Near the top-left corner the box slides back onto the screen.
    assert zoom.crop_box(10, 10, 4000, 2000, 2.0) == zoom.CropBox(0, 0, 2000, 1000)
    # Near the bottom-right corner too.
    assert zoom.crop_box(3999, 1999, 4000, 2000, 4.0) == zoom.CropBox(3000, 1500, 1000, 500)


def test_crop_point_maps_back_to_the_screen_grid():
    box = zoom.CropBox(left=1000, top=500, width=2000, height=1000)
    # The crop's own centre is the screen's centre.
    sx, sy = zoom.crop_grid_to_screen_grid(499.5, 499.5, box, 4001, 2001)
    assert sx == pytest.approx(499.5, abs=0.3)
    assert sy == pytest.approx(499.5, abs=0.3)
    # The crop's top-left corner is pixel (1000, 500) on the screen.
    sx, sy = zoom.crop_grid_to_screen_grid(0, 0, box, 4001, 2001)
    assert zoom.grid_to_pixel(sx, 4001) == pytest.approx(1000, abs=0.5)
    assert zoom.grid_to_pixel(sy, 2001) == pytest.approx(500, abs=0.5)


def test_merge_usage_adds_both_requests():
    merged = zoom.merge_usage(_usage(0.01, 1000), _usage(0.004, 400))
    assert merged.input_tokens == 1400
    assert merged.output_tokens == 20
    assert merged.estimated_cost_usd == pytest.approx(0.014)
    assert zoom.merge_usage(_usage(0.01, 1), None).estimated_cost_usd == 0.01


# ---------- refine_click ----------


def _refine(client, frame, calls, settings=DENSE):
    return asyncio.run(zoom.refine_click(
        client=client, settings=settings, function_calls=calls, screenshot_path=frame.artifact_path,
        width=frame.width, height=frame.height, intent="Save the file", rationale="click save",
    ))


def test_refines_a_click_on_a_dense_screen():
    frame = _screen(4000, 2000)
    client = ZoomClient(turn={}, zoomed=_reply("click_at", 999, 0))
    calls = [{"name": "click_at", "args": {"x": 500, "y": 500, "intent": "the save icon"}}]
    result = _refine(client, frame, calls)
    assert result.zoomed
    # The crop is half the screen each way, around the proposed point.
    prompt, size = client.zoom_requests[0]
    assert size == (2000, 1000)
    assert "Save the file" in prompt and "the save icon" in prompt and "click_at" in prompt
    # (999, 0) on the crop is its top-right corner: screen pixel (~3001, ~500).
    args = result.function_calls[0]["args"]
    assert zoom.grid_to_pixel(args["x"], 4000) == pytest.approx(3001, abs=1)
    assert zoom.grid_to_pixel(args["y"], 2000) == pytest.approx(500, abs=1)
    assert args["intent"] == "the save icon"
    assert calls[0]["args"]["x"] == 500  # the input is not changed in place
    assert result.usage.estimated_cost_usd == 0.004
    assert (Path(frame.artifact_path).parent / "zoom_crop.png").is_file()


def test_refines_gemini_desktop_click_calls():
    frame = _screen(2560, 1440)
    client = ZoomClient(turn={}, zoomed=_reply("double_click", 0, 999))
    result = _refine(client, frame, [{"name": "double_click", "args": {"x": 10, "y": 990}}])
    assert result.zoomed
    args = result.function_calls[0]["args"]
    assert zoom.grid_to_pixel(args["x"], 2560) == pytest.approx(0, abs=1)
    assert zoom.grid_to_pixel(args["y"], 1440) == pytest.approx(1439, abs=1)


def test_only_the_first_pointing_call_is_refined():
    frame = _screen(4000, 2000)
    client = ZoomClient(turn={}, zoomed=_reply("type_text_at", 499.5, 499.5))
    calls = [
        {"name": "key_combination", "args": {"keys": "ctrl+a"}},
        {"name": "type_text_at", "args": {"x": 100, "y": 100, "text": "hi"}},
        {"name": "click_at", "args": {"x": 900, "y": 900}},
    ]
    result = _refine(client, frame, calls)
    assert result.zoomed and len(client.zoom_requests) == 1
    assert result.function_calls[0] == calls[0]
    assert result.function_calls[1]["args"]["text"] == "hi"
    assert result.function_calls[2] == calls[2]


@pytest.mark.parametrize(
    ("size", "calls", "settings"),
    [
        ((1920, 1080), [{"name": "click_at", "args": {"x": 1, "y": 1}}], DENSE),  # not dense
        ((4000, 2000), [{"name": "scroll_document", "args": {"direction": "down"}}], DENSE),  # no point
        ((4000, 2000), [{"name": "click_at", "args": {"x": 1, "y": 1}}], zoom.ZoomSettings()),  # off
    ],
)
def test_no_zoom_request_when_it_does_not_apply(size, calls, settings):
    frame = _screen(*size)
    client = ZoomClient(turn={}, zoomed=_reply("click_at", 1, 1))
    result = _refine(client, frame, calls, settings)
    assert not result.zoomed and result.function_calls is calls
    assert client.zoom_requests == []


@pytest.mark.parametrize("zoomed", [_text_reply("I cannot see it here"), RuntimeError("HTTP 500")])
def test_original_click_stands_when_zoom_gives_no_point(zoomed):
    frame = _screen(4000, 2000)
    client = ZoomClient(turn={}, zoomed=zoomed)
    calls = [{"name": "click_at", "args": {"x": 500, "y": 500}}]
    result = _refine(client, frame, calls)
    assert not result.zoomed and result.function_calls is calls


# ---------- in the backend ----------


def _backend(client, monkeypatch, mode: str) -> BrowserComputerUseBackend:
    monkeypatch.setenv("OPERON_ZOOM", mode)
    return BrowserComputerUseBackend(client=client, prompt_path=prompts_dir() / "browser_computer_use_prompt.txt")


def _decide(backend, frame):
    async def go():
        perception = await backend.perceive(frame, _state())
        return await backend.choose_action(_state(), perception)

    return asyncio.run(go())


def test_backend_clicks_the_zoomed_point_and_counts_both_calls(monkeypatch):
    frame = _screen(4000, 2000)
    client = ZoomClient(turn=_reply("click_at", 500, 500), zoomed=_reply("click_at", 999, 0))
    backend = _backend(client, monkeypatch, "dense")
    decision = _decide(backend, frame)
    assert len(client.zoom_requests) == 1
    assert decision.action.x == pytest.approx(3001, abs=1)
    assert decision.action.y == pytest.approx(500, abs=2)
    usage = backend.latest_debug_artifacts().usage
    assert usage.estimated_cost_usd == pytest.approx(0.014)
    assert usage.input_tokens == 1400


def test_backend_without_zoom_is_unchanged(monkeypatch):
    frame = _screen(4000, 2000)
    client = ZoomClient(turn=_reply("click_at", 500, 500), zoomed=_reply("click_at", 999, 0))
    backend = _backend(client, monkeypatch, "off")
    decision = _decide(backend, frame)
    assert client.zoom_requests == []
    assert decision.action.x == pytest.approx(2001.5, abs=1)
    assert decision.action.y == pytest.approx(1000, abs=1)
    assert backend.latest_debug_artifacts().usage.estimated_cost_usd == 0.01


def test_backend_counts_a_zoom_that_proposed_nothing(monkeypatch):
    frame = _screen(4000, 2000)
    client = ZoomClient(turn=_reply("click_at", 500, 500), zoomed=_text_reply("not here"))
    backend = _backend(client, monkeypatch, "always")
    decision = _decide(backend, frame)
    assert decision.action.x == pytest.approx(2001.5, abs=1)
    assert backend.latest_debug_artifacts().usage.estimated_cost_usd == pytest.approx(0.014)

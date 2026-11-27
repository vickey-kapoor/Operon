"""KAN-163: the trust gate reads the text under a Computer Use click."""

from __future__ import annotations

import asyncio
import builtins
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image, ImageDraw, ImageFont

from operon.agent.loop import AgentLoop
from operon.agent.policy.click_text import (
    CROP_HALF_HEIGHT,
    CROP_HALF_WIDTH,
    RapidOcrReader,
    crop_box,
    make_click_reader,
    read_click_text,
)
from operon.agent.policy.trust_gate import TrustGate, TrustVerdict
from operon.models.capture import CaptureFrame
from operon.models.policy import ActionType, AgentAction


def _screen(path: Path, size: tuple[int, int] = (1280, 720), label: str = "Place order") -> str:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((500, 330, 700, 380), 8, fill="#226")
    draw.text((535, 345), label, fill="white", font=ImageFont.load_default(size=18))
    img.save(path)
    return str(path)


class _Recorder:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.sizes: list[tuple[int, int]] = []

    def __call__(self, image: Image.Image) -> list[str]:
        self.sizes.append(image.size)
        return self.texts


def test_crop_box_is_centred_and_clamped() -> None:
    assert crop_box(600, 355, (1280, 720)) == (
        600 - CROP_HALF_WIDTH, 355 - CROP_HALF_HEIGHT, 600 + CROP_HALF_WIDTH, 355 + CROP_HALF_HEIGHT
    )
    assert crop_box(5, 5, (1280, 720)) == (0, 0, 5 + CROP_HALF_WIDTH, 5 + CROP_HALF_HEIGHT)
    assert crop_box(1280, 10, (1280, 720)) is None


def test_read_click_text_normalises_and_crops(tmp_path: Path) -> None:
    reader = _Recorder(["  Place ", "ORDER  now"])
    text = read_click_text(_screen(tmp_path / "s.png"), 600, 355, (1280, 720), reader=reader)
    assert text == "place order now"
    assert reader.sizes == [(2 * CROP_HALF_WIDTH, 2 * CROP_HALF_HEIGHT)]


def test_read_click_text_scales_the_point_to_the_stored_image(tmp_path: Path) -> None:
    # Frame coordinates are 2560x1440, the file is stored at half size: the point is
    # halved, so a click at the frame's far right still lands inside the image.
    reader = _Recorder(["ok"])
    path = _screen(tmp_path / "s.png")
    assert read_click_text(path, 2500, 1400, (2560, 1440), reader=reader) == "ok"
    assert read_click_text(path, 2500, 1400, None, reader=reader) is None  # off the image unscaled


def test_read_click_text_never_raises(tmp_path: Path) -> None:
    def broken(_image: Image.Image) -> list[str]:
        raise RuntimeError("engine crashed")

    assert read_click_text(_screen(tmp_path / "s.png"), 600, 355, reader=broken) is None
    assert read_click_text(str(tmp_path / "missing.png"), 600, 355, reader=_Recorder(["x"])) is None
    assert read_click_text(_screen(tmp_path / "t.png"), 600, 355, reader=_Recorder([])) is None


def test_missing_engine_reads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def no_rapidocr(name, *args, **kwargs):
        if name == "rapidocr":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rapidocr)
    reader = RapidOcrReader()
    assert reader(Image.new("RGB", (10, 10))) == []
    assert reader(Image.new("RGB", (10, 10))) == []  # tried once, not on every click


def test_reader_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPERON_CLICK_TEXT", raising=False)
    assert make_click_reader() is None
    monkeypatch.setenv("OPERON_CLICK_TEXT", "on")
    assert isinstance(make_click_reader(), RapidOcrReader)


_GATE = TrustGate(enabled=True, deny=("wipe disk",), confirm=("place order",))
_CLICK = AgentAction(action_type=ActionType.CLICK, x=600, y=355)


def test_gate_asks_for_approval_when_the_click_reads_a_risky_phrase() -> None:
    result = _GATE.evaluate(_CLICK, click_text="place order")
    assert result.verdict is TrustVerdict.CONFIRM
    assert result.matched == "place order"
    assert "screen under the click" in result.reason


def test_click_text_never_blocks_even_on_a_deny_phrase() -> None:
    assert _GATE.evaluate(_CLICK, click_text="wipe disk").verdict is TrustVerdict.CONFIRM


def test_harmless_click_text_allows() -> None:
    assert _GATE.evaluate(_CLICK, click_text="cancel").verdict is TrustVerdict.ALLOW
    assert _GATE.evaluate(_CLICK).verdict is TrustVerdict.ALLOW


def test_click_text_is_ignored_when_the_gate_is_off() -> None:
    assert TrustGate(confirm=("place order",)).evaluate(_CLICK, click_text="place order").verdict is TrustVerdict.ALLOW


def _loop_with(reader, gate: TrustGate = _GATE) -> AgentLoop:
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=Mock(),
        policy_service=Mock(),
        executor=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )
    loop._trust_gate = gate
    loop._click_reader = reader
    return loop


def test_loop_reads_the_click_text_from_the_step_frame(tmp_path: Path) -> None:
    frame = CaptureFrame(artifact_path=_screen(tmp_path / "s.png"), width=1280, height=720)
    reader = _Recorder(["Place order"])
    loop = _loop_with(reader)
    assert asyncio.run(loop._click_text_for_gate(_CLICK, frame)) == "place order"


def test_loop_skips_ocr_when_it_cannot_matter(tmp_path: Path) -> None:
    frame = CaptureFrame(artifact_path=_screen(tmp_path / "s.png"), width=1280, height=720)
    reader = _Recorder(["Place order"])
    typed = AgentAction(action_type=ActionType.TYPE, text="hi", x=600, y=355)
    for loop, action, frm in (
        (_loop_with(None), _CLICK, frame),  # OPERON_CLICK_TEXT off
        (_loop_with(reader, TrustGate()), _CLICK, frame),  # gate off
        (_loop_with(reader), _CLICK, None),  # no frame
        (_loop_with(reader), typed, frame),  # not a pointer action
        (_loop_with(reader), AgentAction(action_type=ActionType.CLICK, target_element_id="e"), frame),  # no point
    ):
        assert asyncio.run(loop._click_text_for_gate(action, frm)) is None
    assert reader.sizes == []


def test_loop_pauses_a_click_on_a_risky_button(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from operon.core.runtime import current_runtime
    from operon.models.policy import PolicyDecision

    monkeypatch.setattr(current_runtime().control, "confidence_threshold", 0.0)
    loop = _loop_with(_Recorder(["Place order"]))
    loop.run_store = Mock(root_dir=str(tmp_path))
    decision = PolicyDecision(action=_CLICK, rationale="click the button", confidence=0.9, active_subgoal="s")
    text = loop._pre_execution_gate_text(decision, intent="x", run_id="r", click_text="place order")
    assert text is not None and text.startswith("High-risk action")
    assert loop._pre_execution_gate_text(decision, intent="x", run_id="r2", click_text="cancel") is None


def test_real_engine_reads_a_button(tmp_path: Path) -> None:
    pytest.importorskip("rapidocr")
    text = read_click_text(_screen(tmp_path / "s.png"), 600, 355, (1280, 720), reader=RapidOcrReader())
    assert text is not None and "place order" in text


def test_loop_skips_ocr_in_auto_mode_or_once_approved(tmp_path: Path) -> None:
    frame = CaptureFrame(artifact_path=_screen(tmp_path / "s.png"), width=1280, height=720)
    reader = _Recorder(["Place order"])
    loop = _loop_with(reader)
    assert asyncio.run(loop._click_text_for_gate(_CLICK, frame, run_id="r", permission_mode="auto")) is None
    loop._approved_gate_actions["r"] = (loop._gate_fingerprint(_CLICK), frozenset({"trust"}))
    assert asyncio.run(loop._click_text_for_gate(_CLICK, frame, run_id="r")) is None
    assert reader.sizes == []


def test_gate_stage_wiring_pauses_on_the_frame_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from operon.core.runtime import current_runtime
    from operon.models.policy import PolicyDecision

    monkeypatch.setattr(current_runtime().control, "confidence_threshold", 0.0)
    frame = CaptureFrame(artifact_path=_screen(tmp_path / "s.png"), width=1280, height=720)
    loop = _loop_with(_Recorder(["Place order"]))
    loop.run_store = Mock(root_dir=str(tmp_path))
    monkeypatch.setattr(loop, "_coordinate_thrash_detected", lambda state, action: False)
    record = SimpleNamespace(run_id="r")
    state = SimpleNamespace(intent="buy", permission_mode=None)
    decision = PolicyDecision(action=_CLICK, rationale="click it", confidence=0.9, active_subgoal="s")
    paused = asyncio.run(loop._gate_pause_decision(record, state, decision, frame=frame))
    assert paused is not None and paused.action.action_type is ActionType.WAIT_FOR_USER
    assert "place order" in (paused.action.text or "")
    assert asyncio.run(loop._gate_pause_decision(record, state, decision, frame=None)) is None


def test_loop_takes_the_monitor_origin_off_before_cropping(tmp_path: Path) -> None:
    # A secondary monitor at x=1920: the action is in virtual-desktop coordinates
    # (local 600 + 1920), the screenshot is the monitor alone.
    frame = CaptureFrame(
        artifact_path=_screen(tmp_path / "s.png"), width=1280, height=720, monitor_left=1920, monitor_top=0
    )
    seen: list[tuple[int, int]] = []

    def reader(image: Image.Image) -> list[str]:
        seen.append(image.size)
        return ["Place order"]

    loop = _loop_with(reader)
    action = AgentAction(action_type=ActionType.CLICK, x=600 + 1920, y=355)
    assert asyncio.run(loop._click_text_for_gate(action, frame)) == "place order"
    assert seen == [(2 * CROP_HALF_WIDTH, 2 * CROP_HALF_HEIGHT)]


def test_liveness_retry_reports_the_frame_it_perceived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from operon.agent import loop as loop_mod

    first = CaptureFrame(artifact_path=_screen(tmp_path / "a.png"), width=1280, height=720)
    second = CaptureFrame(artifact_path=_screen(tmp_path / "b.png"), width=1280, height=720)
    loop = _loop_with(None)
    blank = SimpleNamespace(is_empty_frame=True)
    good = Mock(is_empty_frame=False)
    good.model_copy.return_value = good
    loop.perception_service = SimpleNamespace(perceive=AsyncMock(side_effect=[blank, good]))
    loop.capture_service = SimpleNamespace(capture=AsyncMock(return_value=second))
    loop.executor = SimpleNamespace()  # no current_url_for_run
    loop._maybe_reuse_prior_perception = lambda state, frame: None
    seen: list[CaptureFrame] = []
    monkeypatch.setattr(loop_mod, "_LIVENESS_RETRY_FIRST_SLEEP_S", 0.0)
    result = asyncio.run(loop._perceive_with_liveness_retry(SimpleNamespace(run_id="r"), first, on_recapture=seen.append))
    assert result is good
    assert seen == [second]


def test_gate_stage_rechecks_a_stop_that_came_in_during_ocr(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from operon.models.common import RunStatus

    loop = _loop_with(_Recorder(["cancel"]))
    loop._run_cancelled = AsyncMock(side_effect=[False, True])  # stopped while OCR ran
    loop._gate_pause_decision = AsyncMock(return_value=None)
    ctx = SimpleNamespace(
        run_id="r", record=SimpleNamespace(run_id="r", intent="x", step_count=1), state=None,
        decision=SimpleNamespace(action=AgentAction(action_type=ActionType.WAIT, wait_ms=10)), frame=None
    )
    response = asyncio.run(loop._gate_stage(ctx))
    assert response is not None and response.status is RunStatus.CANCELLED


"""Comprehensive tests for the video verification feature.

Covers:
- ScreenRecorder (start/stop, capture loop, encode, edge cases)
- VideoVerifier (prompt rendering, response parsing, error handling)
- AgentLoop._maybe_video_verify (all early-return conditions, success/failure paths)
- GeminiHttpClient.generate_video_verification (payload structure, MIME type)
- PlaceholderGeminiClient.generate_video_verification (raises NotImplementedError)
- VerificationResult and ExecutedAction model fields
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from src.agent.loop import AgentLoop
from src.agent.screen_recorder import ScreenRecorder
from src.agent.video_verifier import VideoVerificationResult, VideoVerifier
from src.clients.gemini import GeminiClientError, GeminiHttpClient, PlaceholderGeminiClient
from src.models.common import FailureCategory, LoopStage
from src.models.execution import ExecutedAction
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.state import AgentState
from src.models.verification import (
    VerificationFailureType,
    VerificationResult,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_action(
    action_type: ActionType = ActionType.CLICK,
    target_element_id: str | None = "btn-ok",
    text: str | None = None,
    key: str | None = None,
    x: int | None = None,
    y: int | None = None,
) -> AgentAction:
    return AgentAction(
        action_type=action_type,
        target_element_id=target_element_id,
        text=text,
        key=key,
        x=x,
        y=y,
    )


def _make_executed(
    action: AgentAction | None = None,
    success: bool = True,
    artifact_path: str | None = "runs/r1/step_1/after.png",
    recording_path: str | None = None,
) -> ExecutedAction:
    if action is None:
        action = _make_action()
    return ExecutedAction(
        action=action,
        success=success,
        detail="done",
        artifact_path=artifact_path,
        recording_path=recording_path,
    )


def _make_decision(action: AgentAction | None = None) -> PolicyDecision:
    if action is None:
        action = _make_action()
    return PolicyDecision(
        action=action,
        rationale="test",
        confidence=0.9,
        active_subgoal="click the button",
    )


def _make_state() -> AgentState:
    return AgentState(run_id="run-video-1", intent="complete the task", status="running")


# ---------------------------------------------------------------------------
# ScreenRecorder tests
# ---------------------------------------------------------------------------


class TestScreenRecorderStopWithNoFrames:
    """stop() returns None when no frames were captured."""

    @pytest.mark.asyncio
    async def test_stop_returns_none_when_no_frames_captured(self, tmp_path: Path) -> None:
        recorder = ScreenRecorder(output_path=tmp_path / "clip.mp4")

        # Don't start — _frames is empty, _thread is None
        result = await recorder.stop()

        assert result is None


class TestScreenRecorderStart:
    """start() launches a background thread."""

    @pytest.mark.asyncio
    async def test_start_creates_and_starts_thread(self, tmp_path: Path) -> None:
        recorder = ScreenRecorder(output_path=tmp_path / "clip.mp4")

        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Provide a capture loop that adds one frame then waits for stop
        def immediate_stop_loop(self_ref: ScreenRecorder) -> None:  # type: ignore[misc]
            self_ref._frames = [fake_frame]
            self_ref._width = 100
            self_ref._height = 100

        with patch.object(ScreenRecorder, "_capture_loop", immediate_stop_loop):
            await recorder.start()
            assert recorder._thread is not None
            assert isinstance(recorder._thread, threading.Thread)

        # Cleanup
        recorder._stop_event.set()
        if recorder._thread and recorder._thread.is_alive():
            recorder._thread.join(timeout=2.0)


class TestScreenRecorderEncoding:
    """stop() encodes frames to MP4 and returns the output path."""

    @pytest.mark.asyncio
    async def test_stop_returns_output_path_after_encoding(self, tmp_path: Path) -> None:
        output = tmp_path / "clip.mp4"
        recorder = ScreenRecorder(output_path=output)
        # Inject pre-captured frames directly — bypass real mss capture
        fake_frame = np.zeros((50, 80, 3), dtype=np.uint8)
        recorder._frames = [fake_frame, fake_frame]
        recorder._width = 80
        recorder._height = 50
        recorder._stop_event.set()  # no thread running

        # Mock VideoWriter to avoid needing a real codec
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.release = MagicMock()

        with patch("cv2.VideoWriter", return_value=mock_writer), \
             patch("cv2.VideoWriter_fourcc", return_value=0x7634706D):
            # _encode creates the file; simulate it by touching the path
            def fake_encode(self_ref: ScreenRecorder) -> None:  # type: ignore[misc]
                self_ref.output_path.parent.mkdir(parents=True, exist_ok=True)
                self_ref.output_path.write_bytes(b"fake-mp4-data")
                self_ref._frames = []

            with patch.object(ScreenRecorder, "_encode", fake_encode):
                result = await recorder.stop()

        assert result == output
        assert output.exists()

    @pytest.mark.asyncio
    async def test_stop_returns_none_when_encoding_raises(self, tmp_path: Path) -> None:
        output = tmp_path / "clip.mp4"
        recorder = ScreenRecorder(output_path=output)
        fake_frame = np.zeros((50, 80, 3), dtype=np.uint8)
        recorder._frames = [fake_frame]
        recorder._width = 80
        recorder._height = 50
        recorder._stop_event.set()

        def broken_encode(self_ref: ScreenRecorder) -> None:  # type: ignore[misc]
            raise RuntimeError("codec not available")

        with patch.object(ScreenRecorder, "_encode", broken_encode):
            result = await recorder.stop()

        assert result is None

    @pytest.mark.asyncio
    async def test_stop_returns_none_when_output_file_not_created(self, tmp_path: Path) -> None:
        output = tmp_path / "clip.mp4"
        recorder = ScreenRecorder(output_path=output)
        fake_frame = np.zeros((50, 80, 3), dtype=np.uint8)
        recorder._frames = [fake_frame]
        recorder._width = 80
        recorder._height = 50
        recorder._stop_event.set()

        def noop_encode(self_ref: ScreenRecorder) -> None:  # type: ignore[misc]
            self_ref._frames = []
            # deliberately does NOT write the output file

        with patch.object(ScreenRecorder, "_encode", noop_encode):
            result = await recorder.stop()

        assert result is None


class TestScreenRecorderCaptureLoop:
    """_capture_loop respects fps and max_duration limits."""

    def test_capture_loop_stops_at_max_frames(self, tmp_path: Path) -> None:
        fps = 4
        max_duration = 1.0  # → 4 frames max
        recorder = ScreenRecorder(
            output_path=tmp_path / "clip.mp4",
            fps=fps,
            max_duration=max_duration,
        )

        fake_frame_bgra = np.zeros((100, 200, 4), dtype=np.uint8)
        fake_frame_bgr = np.zeros((50, 100, 3), dtype=np.uint8)  # after resize

        mock_shot = MagicMock()
        mock_shot.__array__ = lambda self, dtype=None: fake_frame_bgra  # type: ignore[misc]

        mock_sct = MagicMock()
        mock_sct.monitors = [{"top": 0, "left": 0, "width": 200, "height": 100}]
        mock_sct.grab.return_value = fake_frame_bgra
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)

        with patch("mss.mss", return_value=mock_sct), \
             patch("numpy.array", return_value=fake_frame_bgra), \
             patch("cv2.resize", return_value=fake_frame_bgr):
            # Run the loop in a thread; set stop event after a short delay
            t = threading.Thread(target=recorder._capture_loop, daemon=True)
            t.start()
            t.join(timeout=3.0)

        # Frames should be capped at fps * max_duration
        assert len(recorder._frames) <= fps * max_duration

    def test_capture_loop_stops_when_stop_event_set(self, tmp_path: Path) -> None:
        recorder = ScreenRecorder(
            output_path=tmp_path / "clip.mp4",
            fps=10,
            max_duration=60.0,  # would run forever without stop event
        )

        fake_frame_bgra = np.zeros((100, 200, 4), dtype=np.uint8)
        fake_frame_bgr = np.zeros((50, 100, 3), dtype=np.uint8)

        mock_sct = MagicMock()
        mock_sct.monitors = [{"top": 0, "left": 0, "width": 200, "height": 100}]
        mock_sct.grab.return_value = fake_frame_bgra
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)

        with patch("mss.mss", return_value=mock_sct), \
             patch("numpy.array", return_value=fake_frame_bgra), \
             patch("cv2.resize", return_value=fake_frame_bgr):
            t = threading.Thread(target=recorder._capture_loop, daemon=True)
            t.start()
            # Signal stop after one frame is likely captured
            import time
            time.sleep(0.05)
            recorder._stop_event.set()
            t.join(timeout=2.0)

        assert not t.is_alive(), "capture loop thread did not stop within timeout"


# ---------------------------------------------------------------------------
# VideoVerifier tests
# ---------------------------------------------------------------------------

def _make_verifier(raw_response: str = "{}") -> VideoVerifier:
    mock_client = MagicMock()
    mock_client.generate_video_verification = AsyncMock(return_value=raw_response)
    # Read the real prompt file
    verifier = VideoVerifier.__new__(VideoVerifier)
    verifier.gemini_client = mock_client
    from pathlib import Path as _Path
    prompt_path = _Path(__file__).resolve().parents[1] / "prompts" / "video_verification_prompt.txt"
    verifier._prompt_template = prompt_path.read_text(encoding="utf-8")
    return verifier


class TestVideoVerifierPromptRendering:
    """_render_prompt fills in the correct placeholders for various action types."""

    def test_render_prompt_includes_action_type_and_intent(self) -> None:
        verifier = _make_verifier()
        action = AgentAction(action_type=ActionType.CLICK, target_element_id="submit-btn")
        prompt = verifier._render_prompt(action, intent="fill out the form")

        assert "click" in prompt
        assert "fill out the form" in prompt

    def test_render_prompt_includes_target_element_id(self) -> None:
        verifier = _make_verifier()
        action = AgentAction(action_type=ActionType.CLICK, target_element_id="my-button")
        prompt = verifier._render_prompt(action, intent="click button")

        assert "my-button" in prompt

    def test_render_prompt_includes_typed_text(self) -> None:
        verifier = _make_verifier()
        action = AgentAction(action_type=ActionType.TYPE, target_element_id="name-field", text="Alice")
        prompt = verifier._render_prompt(action, intent="fill name")

        assert "'Alice'" in prompt  # repr-formatted text

    def test_render_prompt_includes_key_for_press_key(self) -> None:
        verifier = _make_verifier()
        action = AgentAction(action_type=ActionType.PRESS_KEY, key="Return")
        prompt = verifier._render_prompt(action, intent="submit")

        assert "Return" in prompt

    def test_render_prompt_includes_coordinates_for_click_at_xy(self) -> None:
        verifier = _make_verifier()
        action = AgentAction(action_type=ActionType.CLICK, x=320, y=240)
        prompt = verifier._render_prompt(action, intent="click area")

        assert "320" in prompt
        assert "240" in prompt

    def test_render_prompt_uses_fallback_when_no_details(self) -> None:
        verifier = _make_verifier()
        action = AgentAction(action_type=ActionType.PRESS_KEY, key="Tab")
        # key is present, so detail won't be empty — use WAIT with wait_ms as no-detail action
        action_wait = AgentAction(action_type=ActionType.WAIT, wait_ms=500)
        prompt = verifier._render_prompt(action_wait, intent="pause")

        assert "no additional details" in prompt


class TestVideoVerifierResponseParsing:
    """_parse_response handles valid JSON, missing fields, and malformed input."""

    def test_parse_valid_response_did_it_work_true(self) -> None:
        raw = json.dumps({
            "did_it_work": True,
            "what_happened": "Button clicked successfully, dialog opened",
            "suggested_next_action": "continue",
        })
        result = VideoVerifier._parse_response(raw)

        assert result.did_it_work is True
        assert result.what_happened == "Button clicked successfully, dialog opened"
        assert result.suggested_next_action == "continue"

    def test_parse_valid_response_did_it_work_false(self) -> None:
        raw = json.dumps({
            "did_it_work": False,
            "what_happened": "Nothing happened on screen",
            "suggested_next_action": "retry_same_step",
        })
        result = VideoVerifier._parse_response(raw)

        assert result.did_it_work is False
        assert result.what_happened == "Nothing happened on screen"

    def test_parse_response_with_json_fence(self) -> None:
        raw = '```json\n{"did_it_work": true, "what_happened": "ok", "suggested_next_action": "continue"}\n```'
        result = VideoVerifier._parse_response(raw)

        assert result.did_it_work is True

    def test_parse_malformed_json_returns_safe_default(self) -> None:
        raw = "not valid json at all {{}}"
        result = VideoVerifier._parse_response(raw)

        assert result.did_it_work is False
        assert "Unparseable" in result.what_happened
        assert result.suggested_next_action == "retry_same_step"

    def test_parse_empty_string_returns_safe_default(self) -> None:
        result = VideoVerifier._parse_response("")

        assert result.did_it_work is False

    def test_parse_missing_did_it_work_key_defaults_to_false(self) -> None:
        raw = json.dumps({"what_happened": "unclear", "suggested_next_action": "continue"})
        result = VideoVerifier._parse_response(raw)

        assert result.did_it_work is False

    def test_parse_missing_optional_fields_uses_defaults(self) -> None:
        raw = json.dumps({"did_it_work": True})
        result = VideoVerifier._parse_response(raw)

        assert result.did_it_work is True
        assert result.what_happened == "unknown"
        assert result.suggested_next_action == "continue"


class TestVideoVerifierVerifyAction:
    """verify_action calls Gemini and surfaces structured results."""

    @pytest.mark.asyncio
    async def test_verify_action_returns_did_it_work_true(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake-mp4")

        raw = json.dumps({
            "did_it_work": True,
            "what_happened": "Menu appeared",
            "suggested_next_action": "continue",
        })
        verifier = _make_verifier(raw_response=raw)
        action = AgentAction(action_type=ActionType.CLICK, target_element_id="menu-btn")

        result = await verifier.verify_action(
            video_path=video_path, action=action, intent="open menu"
        )

        assert result.did_it_work is True
        assert result.what_happened == "Menu appeared"

    @pytest.mark.asyncio
    async def test_verify_action_returns_did_it_work_false(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake-mp4")

        raw = json.dumps({
            "did_it_work": False,
            "what_happened": "No response",
            "suggested_next_action": "retry_same_step",
        })
        verifier = _make_verifier(raw_response=raw)
        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")

        result = await verifier.verify_action(
            video_path=video_path, action=action, intent="do something"
        )

        assert result.did_it_work is False

    @pytest.mark.asyncio
    async def test_verify_action_returns_safe_default_on_gemini_error(
        self, tmp_path: Path
    ) -> None:
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake-mp4")

        mock_client = MagicMock()
        mock_client.generate_video_verification = AsyncMock(
            side_effect=GeminiClientError("API limit reached")
        )

        verifier = VideoVerifier.__new__(VideoVerifier)
        verifier.gemini_client = mock_client
        from pathlib import Path as _Path
        prompt_path = _Path(__file__).resolve().parents[1] / "prompts" / "video_verification_prompt.txt"
        verifier._prompt_template = prompt_path.read_text(encoding="utf-8")

        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")

        result = await verifier.verify_action(
            video_path=video_path, action=action, intent="something"
        )

        assert result.did_it_work is False
        assert "failed" in result.what_happened.lower()
        assert result.suggested_next_action == "retry_same_step"


# ---------------------------------------------------------------------------
# AgentLoop._maybe_video_verify tests
# ---------------------------------------------------------------------------

def _make_loop_with_video_verifier(video_verifier: VideoVerifier | None = None) -> AgentLoop:
    loop = AgentLoop(
        capture_service=Mock(),
        perception_service=Mock(),
        run_store=Mock(),
        policy_service=Mock(),
        executor=Mock(),
        verifier_service=Mock(),
        recovery_manager=Mock(),
    )
    loop.video_verifier = video_verifier
    return loop


def _make_executor_with_recording(
    video_path: Path | None,
) -> Mock:
    executor = Mock()
    executor.execute_with_recording = AsyncMock(
        return_value=(
            ExecutedAction(
                action=AgentAction(action_type=ActionType.CLICK, target_element_id="btn"),
                success=True,
                detail="re-executed",
            ),
            video_path,
        )
    )
    return executor


class TestMaybeVideoVerifyEarlyReturns:
    """_maybe_video_verify returns None for all skip conditions."""

    @pytest.mark.asyncio
    async def test_returns_none_when_executed_action_success_is_false(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)

        executed = _make_executed(success=False, artifact_path="runs/r1/step_1/after.png")
        decision = _make_decision()
        state = _make_state()

        result = await loop._maybe_video_verify(
            state=state,
            decision=decision,
            executed_action=executed,
            before_artifact_path="runs/r1/step_1/before.png",
            step_index=1,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_before_artifact_path_is_empty(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)

        executed = _make_executed(success=True, artifact_path="runs/r1/step_1/after.png")
        decision = _make_decision()
        state = _make_state()

        result = await loop._maybe_video_verify(
            state=state,
            decision=decision,
            executed_action=executed,
            before_artifact_path="",
            step_index=1,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_after_artifact_path_is_none(self) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)

        executed = _make_executed(success=True, artifact_path=None)
        decision = _make_decision()
        state = _make_state()

        result = await loop._maybe_video_verify(
            state=state,
            decision=decision,
            executed_action=executed,
            before_artifact_path="runs/r1/step_1/before.png",
            step_index=1,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_screen_change_ratio_above_threshold(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)

        executed = _make_executed(success=True, artifact_path="runs/r1/step_1/after.png")
        decision = _make_decision()
        state = _make_state()

        # Patch compute_screen_change_ratio to return a value above the threshold (0.02).
        # These are lazily imported inside _maybe_video_verify, so patch the source module.
        with patch(
            "src.agent.screen_diff.compute_screen_change_ratio", return_value=0.5
        ), patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=state,
                decision=decision,
                executed_action=executed,
                before_artifact_path="runs/r1/step_1/before.png",
                step_index=1,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_type_action(self) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)

        action = AgentAction(action_type=ActionType.TYPE, target_element_id="field", text="hello")
        executed = _make_executed(action=action, success=True, artifact_path="runs/r1/step_1/after.png")
        decision = _make_decision(action=action)
        state = _make_state()

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=state,
                decision=decision,
                executed_action=executed,
                before_artifact_path="runs/r1/step_1/before.png",
                step_index=1,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_drag_action(self) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)

        action = AgentAction(action_type=ActionType.DRAG, x=10, y=20, x_end=100, y_end=200)
        executed = _make_executed(action=action, success=True, artifact_path="runs/r1/step_1/after.png")
        decision = _make_decision(action=action)
        state = _make_state()

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=state,
                decision=decision,
                executed_action=executed,
                before_artifact_path="runs/r1/step_1/before.png",
                step_index=1,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_select_action(self) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)

        action = AgentAction(action_type=ActionType.SELECT, target_element_id="dropdown", text="option1")
        executed = _make_executed(action=action, success=True, artifact_path="runs/r1/step_1/after.png")
        decision = _make_decision(action=action)
        state = _make_state()

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=state,
                decision=decision,
                executed_action=executed,
                before_artifact_path="runs/r1/step_1/before.png",
                step_index=1,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_executor_lacks_execute_with_recording(self) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)
        # Default Mock() does not have execute_with_recording as a real attribute
        # We ensure hasattr returns False by using a plain object
        loop.executor = SimpleNamespace()  # no execute_with_recording attribute

        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executed = _make_executed(action=action, success=True, artifact_path="runs/r1/step_1/after.png")
        decision = _make_decision(action=action)
        state = _make_state()

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=state,
                decision=decision,
                executed_action=executed,
                before_artifact_path="runs/r1/step_1/before.png",
                step_index=1,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_execute_with_recording_raises(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)

        executor = Mock()
        executor.execute_with_recording = AsyncMock(side_effect=RuntimeError("recording failed"))
        loop.executor = executor

        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executed = _make_executed(action=action, success=True, artifact_path="runs/r1/step_1/after.png")
        decision = _make_decision(action=action)
        state = _make_state()

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=state,
                decision=decision,
                executed_action=executed,
                before_artifact_path="runs/r1/step_1/before.png",
                step_index=1,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_video_path_is_none(self, tmp_path: Path) -> None:
        verifier = _make_verifier()
        loop = _make_loop_with_video_verifier(verifier)

        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executor = _make_executor_with_recording(video_path=None)
        loop.executor = executor

        executed = _make_executed(action=action, success=True, artifact_path=str(tmp_path / "after.png"))
        decision = _make_decision(action=action)
        state = _make_state()

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=state,
                decision=decision,
                executed_action=executed,
                before_artifact_path=str(tmp_path / "before.png"),
                step_index=1,
            )

        assert result is None


class TestMaybeVideoVerifySuccess:
    """_maybe_video_verify returns a SUCCESS VerificationResult when video confirms the action."""

    @pytest.mark.asyncio
    async def test_returns_success_verification_when_video_confirms_action(
        self, tmp_path: Path
    ) -> None:
        video_file = tmp_path / "step_1" / "clip.mp4"
        video_file.parent.mkdir(parents=True)
        video_file.write_bytes(b"fake-mp4")

        mock_gemini = MagicMock()
        mock_gemini.generate_video_verification = AsyncMock(
            return_value=json.dumps({
                "did_it_work": True,
                "what_happened": "Button pressed, dialog appeared",
                "suggested_next_action": "continue",
            })
        )
        from pathlib import Path as _Path
        prompt_path = _Path(__file__).resolve().parents[1] / "prompts" / "video_verification_prompt.txt"
        verifier = VideoVerifier.__new__(VideoVerifier)
        verifier.gemini_client = mock_gemini
        verifier._prompt_template = prompt_path.read_text(encoding="utf-8")

        loop = _make_loop_with_video_verifier(verifier)
        executor = _make_executor_with_recording(video_path=video_file)
        loop.executor = executor

        action = AgentAction(action_type=ActionType.CLICK, target_element_id="ok-btn")
        executed = _make_executed(
            action=action,
            success=True,
            artifact_path=str(tmp_path / "step_1" / "after.png"),
        )
        decision = _make_decision(action=action)
        state = _make_state()

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=state,
                decision=decision,
                executed_action=executed,
                before_artifact_path=str(tmp_path / "step_1" / "before.png"),
                step_index=1,
            )

        assert result is not None
        assert result.status is VerificationStatus.SUCCESS
        assert result.expected_outcome_met is True
        assert result.video_verified is True
        assert result.video_detail == "Button pressed, dialog appeared"
        assert result.stop_condition_met is False

    @pytest.mark.asyncio
    async def test_success_result_has_advance_recovery_hint(
        self, tmp_path: Path
    ) -> None:
        video_file = tmp_path / "step_1" / "clip.mp4"
        video_file.parent.mkdir(parents=True)
        video_file.write_bytes(b"fake")

        mock_gemini = MagicMock()
        mock_gemini.generate_video_verification = AsyncMock(
            return_value=json.dumps({
                "did_it_work": True,
                "what_happened": "worked",
                "suggested_next_action": "continue",
            })
        )
        from pathlib import Path as _Path
        prompt_path = _Path(__file__).resolve().parents[1] / "prompts" / "video_verification_prompt.txt"
        verifier = VideoVerifier.__new__(VideoVerifier)
        verifier.gemini_client = mock_gemini
        verifier._prompt_template = prompt_path.read_text(encoding="utf-8")

        loop = _make_loop_with_video_verifier(verifier)
        loop.executor = _make_executor_with_recording(video_path=video_file)

        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executed = _make_executed(action=action, success=True, artifact_path=str(tmp_path / "step_1" / "after.png"))

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=_make_state(),
                decision=_make_decision(action=action),
                executed_action=executed,
                before_artifact_path=str(tmp_path / "step_1" / "before.png"),
                step_index=1,
            )

        assert result is not None
        assert result.recovery_hint == "advance"


class TestMaybeVideoVerifyFailure:
    """_maybe_video_verify returns a FAILURE VerificationResult when video shows no effect."""

    @pytest.mark.asyncio
    async def test_returns_failure_verification_when_video_shows_no_effect(
        self, tmp_path: Path
    ) -> None:
        video_file = tmp_path / "step_1" / "clip.mp4"
        video_file.parent.mkdir(parents=True)
        video_file.write_bytes(b"fake-mp4")

        mock_gemini = MagicMock()
        mock_gemini.generate_video_verification = AsyncMock(
            return_value=json.dumps({
                "did_it_work": False,
                "what_happened": "Nothing changed on screen",
                "suggested_next_action": "retry_same_step",
            })
        )
        from pathlib import Path as _Path
        prompt_path = _Path(__file__).resolve().parents[1] / "prompts" / "video_verification_prompt.txt"
        verifier = VideoVerifier.__new__(VideoVerifier)
        verifier.gemini_client = mock_gemini
        verifier._prompt_template = prompt_path.read_text(encoding="utf-8")

        loop = _make_loop_with_video_verifier(verifier)
        loop.executor = _make_executor_with_recording(video_path=video_file)

        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executed = _make_executed(
            action=action,
            success=True,
            artifact_path=str(tmp_path / "step_1" / "after.png"),
        )
        decision = _make_decision(action=action)

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=_make_state(),
                decision=decision,
                executed_action=executed,
                before_artifact_path=str(tmp_path / "step_1" / "before.png"),
                step_index=1,
            )

        assert result is not None
        assert result.status is VerificationStatus.FAILURE
        assert result.expected_outcome_met is False
        assert result.video_verified is True
        assert result.failure_type is VerificationFailureType.ACTION_FAILED
        assert result.failure_category is FailureCategory.EXECUTION_ERROR
        assert result.failure_stage is LoopStage.VERIFY

    @pytest.mark.asyncio
    async def test_failure_result_has_retry_recovery_hint(self, tmp_path: Path) -> None:
        video_file = tmp_path / "step_1" / "clip.mp4"
        video_file.parent.mkdir(parents=True)
        video_file.write_bytes(b"fake")

        mock_gemini = MagicMock()
        mock_gemini.generate_video_verification = AsyncMock(
            return_value=json.dumps({
                "did_it_work": False,
                "what_happened": "nothing",
                "suggested_next_action": "retry_same_step",
            })
        )
        from pathlib import Path as _Path
        prompt_path = _Path(__file__).resolve().parents[1] / "prompts" / "video_verification_prompt.txt"
        verifier = VideoVerifier.__new__(VideoVerifier)
        verifier.gemini_client = mock_gemini
        verifier._prompt_template = prompt_path.read_text(encoding="utf-8")

        loop = _make_loop_with_video_verifier(verifier)
        loop.executor = _make_executor_with_recording(video_path=video_file)

        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executed = _make_executed(action=action, success=True, artifact_path=str(tmp_path / "step_1" / "after.png"))

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=_make_state(),
                decision=_make_decision(action=action),
                executed_action=executed,
                before_artifact_path=str(tmp_path / "step_1" / "before.png"),
                step_index=1,
            )

        assert result is not None
        assert result.recovery_hint == "retry_same_step"

    @pytest.mark.asyncio
    async def test_failure_result_video_detail_is_suggested_next_action(
        self, tmp_path: Path
    ) -> None:
        video_file = tmp_path / "step_1" / "clip.mp4"
        video_file.parent.mkdir(parents=True)
        video_file.write_bytes(b"fake")

        mock_gemini = MagicMock()
        mock_gemini.generate_video_verification = AsyncMock(
            return_value=json.dumps({
                "did_it_work": False,
                "what_happened": "frozen",
                "suggested_next_action": "scroll_down_and_retry",
            })
        )
        from pathlib import Path as _Path
        prompt_path = _Path(__file__).resolve().parents[1] / "prompts" / "video_verification_prompt.txt"
        verifier = VideoVerifier.__new__(VideoVerifier)
        verifier.gemini_client = mock_gemini
        verifier._prompt_template = prompt_path.read_text(encoding="utf-8")

        loop = _make_loop_with_video_verifier(verifier)
        loop.executor = _make_executor_with_recording(video_path=video_file)

        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executed = _make_executed(action=action, success=True, artifact_path=str(tmp_path / "step_1" / "after.png"))

        with patch("src.agent.screen_diff.compute_screen_change_ratio", return_value=0.0), \
             patch("src.agent.screen_diff.SCREEN_CHANGE_THRESHOLD", 0.02):
            result = await loop._maybe_video_verify(
                state=_make_state(),
                decision=_make_decision(action=action),
                executed_action=executed,
                before_artifact_path=str(tmp_path / "step_1" / "before.png"),
                step_index=1,
            )

        assert result is not None
        # video_detail on failure carries the suggested_next_action from the verifier
        assert result.video_detail == "scroll_down_and_retry"


# ---------------------------------------------------------------------------
# GeminiHttpClient.generate_video_verification payload tests
# ---------------------------------------------------------------------------


class TestGeminiGenerateVideoVerification:
    """generate_video_verification uses video/mp4 MIME type in the payload."""

    def test_build_perception_payload_with_video_mime_type(self) -> None:
        payload = GeminiHttpClient._build_perception_payload(
            prompt="verify this recording",
            image_bytes=b"fake-mp4-bytes",
            mime_type="video/mp4",
        )

        inline = payload["contents"][0]["parts"][1]["inline_data"]
        assert inline["mime_type"] == "video/mp4"
        assert inline["data"]  # base64 encoded, non-empty

    def test_video_payload_prompt_is_first_part(self) -> None:
        payload = GeminiHttpClient._build_perception_payload(
            prompt="watch the video",
            image_bytes=b"data",
            mime_type="video/mp4",
        )

        assert payload["contents"][0]["parts"][0]["text"] == "watch the video"

    @pytest.mark.asyncio
    async def test_generate_video_verification_calls_post_with_video_bytes(
        self, tmp_path: Path
    ) -> None:
        video_file = tmp_path / "clip.mp4"
        video_file.write_bytes(b"mp4-content")

        client = GeminiHttpClient(api_key="test-key", retry_backoff_seconds=0.0)

        captured_payload: dict | None = None

        async def fake_post(url, *, content=None, json=None, headers=None, **kwargs):
            import json as _json
            nonlocal captured_payload
            captured_payload = _json.loads(content)
            import httpx
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": '{"did_it_work": true}'}]}}]},
            )

        mock_http_client = MagicMock()
        mock_http_client.post = fake_post
        mock_http_client.is_closed = False
        client._client = mock_http_client

        await client.generate_video_verification("verify this", str(video_file))

        assert captured_payload is not None
        inline = captured_payload["contents"][0]["parts"][1]["inline_data"]
        assert inline["mime_type"] == "video/mp4"

        import base64
        decoded = base64.b64decode(inline["data"])
        assert decoded == b"mp4-content"


class TestPlaceholderGeminiClientVideoVerification:
    """PlaceholderGeminiClient.generate_video_verification raises NotImplementedError."""

    @pytest.mark.asyncio
    async def test_raises_not_implemented_error(self) -> None:
        client = PlaceholderGeminiClient()

        with pytest.raises(NotImplementedError):
            await client.generate_video_verification("prompt", "/path/to/clip.mp4")

    @pytest.mark.asyncio
    async def test_error_message_mentions_video_verification(self) -> None:
        client = PlaceholderGeminiClient()

        with pytest.raises(NotImplementedError, match="video verification"):
            await client.generate_video_verification("prompt", "/path/to/clip.mp4")


# ---------------------------------------------------------------------------
# Model field tests
# ---------------------------------------------------------------------------


class TestVerificationResultVideoFields:
    """VerificationResult accepts video_verified and video_detail fields."""

    def test_defaults_video_verified_to_false(self) -> None:
        result = VerificationResult(
            status=VerificationStatus.SUCCESS,
            expected_outcome_met=True,
            stop_condition_met=False,
            reason="action succeeded",
        )

        assert result.video_verified is False
        assert result.video_detail is None

    def test_accepts_video_verified_true(self) -> None:
        result = VerificationResult(
            status=VerificationStatus.SUCCESS,
            expected_outcome_met=True,
            stop_condition_met=False,
            reason="video confirmed",
            video_verified=True,
            video_detail="button click produced a dialog",
        )

        assert result.video_verified is True
        assert result.video_detail == "button click produced a dialog"

    def test_serializes_video_fields_to_dict(self) -> None:
        result = VerificationResult(
            status=VerificationStatus.FAILURE,
            expected_outcome_met=False,
            stop_condition_met=False,
            reason="no change",
            video_verified=True,
            video_detail="nothing happened",
        )

        data = result.model_dump()
        assert data["video_verified"] is True
        assert data["video_detail"] == "nothing happened"

    def test_video_detail_rejects_empty_string(self) -> None:
        with pytest.raises(Exception):  # Pydantic validation error
            VerificationResult(
                status=VerificationStatus.SUCCESS,
                expected_outcome_met=True,
                stop_condition_met=False,
                reason="ok",
                video_detail="",
            )


class TestExecutedActionRecordingPath:
    """ExecutedAction accepts and serializes the recording_path field."""

    def test_recording_path_defaults_to_none(self) -> None:
        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executed = ExecutedAction(action=action, success=True, detail="done")

        assert executed.recording_path is None

    def test_recording_path_accepts_valid_path_string(self) -> None:
        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executed = ExecutedAction(
            action=action,
            success=True,
            detail="done",
            recording_path="runs/run-1/step_2/recording.mp4",
        )

        assert executed.recording_path == "runs/run-1/step_2/recording.mp4"

    def test_recording_path_rejects_empty_string(self) -> None:
        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        with pytest.raises(Exception):  # Pydantic validation error
            ExecutedAction(action=action, success=True, detail="done", recording_path="")

    def test_recording_path_serializes_to_dict(self) -> None:
        action = AgentAction(action_type=ActionType.CLICK, target_element_id="btn")
        executed = ExecutedAction(
            action=action,
            success=True,
            detail="done",
            recording_path="/tmp/clip.mp4",
        )

        data = executed.model_dump()
        assert data["recording_path"] == "/tmp/clip.mp4"

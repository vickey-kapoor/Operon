"""The real visual-servo gate and threshold calibration, fed synthetic frames.

Only screen capture (``operon.executor.desktop._mss``) is faked; the variance
maths, the calibration tiers and the click gate are the production code.
Safe mode is switched off for these tests so the constructor really calibrates.

The click-class parametrization in ``test_desktop.py`` (``test_visual_servo_aborts_uniform_region``)
already proves CLICK / DOUBLE_CLICK / RIGHT_CLICK stop on a blank region; this file
adds TRIPLE_CLICK, which that test omits, and DRAG's start point (KAN-174), using the
real servo instead of a patched one.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from operon.executor.desktop import DesktopExecutor
from operon.executor.linux_desktop import LinuxDesktopExecutor
from operon.models.common import FailureCategory
from operon.models.policy import ActionType, AgentAction
from tests.fixtures.paths import unique_artifact_dir

W, H = 1280, 720


def _uniform() -> np.ndarray:
    return np.full((H, W), 200, dtype=np.uint8)


def _textlike() -> np.ndarray:
    """Black text-like stripes on white: high local variance."""
    img = np.full((H, W), 255, dtype=np.uint8)
    img[::6, :] = 0
    img[:, ::9] = 0
    return img


def _noisy(std: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(128, std, (H, W)), 0, 255).astype(np.uint8)


class _FakeMss:
    """Stands in for ``mss.mss()``: serves crops of a synthetic grayscale frame."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        h, w = frame.shape
        mon = {"left": 0, "top": 0, "width": w, "height": h}
        self.monitors = [dict(mon), dict(mon)]

    def __enter__(self) -> _FakeMss:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def grab(self, box: dict) -> MagicMock:
        crop = self.frame[box["top"]:box["top"] + box["height"], box["left"]:box["left"] + box["width"]]
        rgb = np.repeat(crop[:, :, None], 3, axis=2)
        shot = MagicMock()
        shot.rgb = rgb.tobytes()
        return shot


@pytest.fixture(autouse=True)
def _safe_mode_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERON_TEST_SAFE_MODE", "false")


def _with_screen(frame: np.ndarray):
    return patch("operon.executor.desktop._mss", side_effect=lambda: _FakeMss(frame))


def _executor(frame: np.ndarray, cls: type[DesktopExecutor] = DesktopExecutor) -> DesktopExecutor:
    """Build an executor whose constructor really runs calibration against ``frame``."""
    with (
        patch("operon.executor.desktop._set_dpi_awareness"),
        patch("operon.executor.desktop.validate_display_baseline"),
        _with_screen(frame),
    ):
        return cls(artifact_dir=unique_artifact_dir("servo"))


# ── _region_has_content ──────────────────────────────────────────


def test_uniform_region_has_no_content() -> None:
    frame = _uniform()
    executor = _executor(frame)
    with _with_screen(frame):
        has_content, variance = executor._region_has_content(640, 360)
    assert has_content is False
    assert variance == pytest.approx(0.0)


def test_textlike_region_has_content() -> None:
    frame = _textlike()
    executor = _executor(frame)
    with _with_screen(frame):
        has_content, variance = executor._region_has_content(640, 360)
    assert has_content is True
    assert variance > 1000


def test_baseline_variance_override_moves_the_decision() -> None:
    frame = _noisy(std=6)  # region variance ~36
    executor = _executor(_uniform())
    with _with_screen(frame):
        assert executor._region_has_content(640, 360, baseline_variance=10.0)[0] is True
        assert executor._region_has_content(640, 360, baseline_variance=500.0)[0] is False


def test_unreadable_region_proceeds() -> None:
    executor = _executor(_uniform())
    with patch.object(executor, "_sample_region", return_value=b""):
        assert executor._region_has_content(10, 10) == (True, 0.0)


# ── _calibrate_servo_threshold ───────────────────────────────────


@pytest.mark.parametrize(
    "frame_factory, expected_floor, expected_threshold",
    [
        (_uniform, 0.0, 20.0),                  # flat desktop: default
        (lambda: _noisy(std=5), 25.0, 18.75),   # mid texture: floor * 0.75
        (lambda: _noisy(std=12), 144.0, 80.0),  # heavy texture: capped at max
    ],
)
def test_calibration_tiers(frame_factory, expected_floor, expected_threshold) -> None:
    frame = frame_factory()
    executor = _executor(frame)
    with _with_screen(frame):
        floor, threshold = executor._calibrate_servo_threshold()
    assert floor == pytest.approx(expected_floor, rel=0.1, abs=0.01)
    assert threshold == pytest.approx(expected_threshold, rel=0.1)
    # The constructor ran the same calibration.
    assert (executor._noise_floor, executor._servo_threshold) == pytest.approx((floor, threshold))
    assert executor._SERVO_THRESHOLD_MIN <= threshold <= executor._SERVO_THRESHOLD_MAX


def test_threshold_clamped_to_minimum() -> None:
    frame = _noisy(std=3)  # floor ~9 -> 0.75 * 9 = 6.75 -> clamped up to 8
    executor = _executor(frame)
    assert executor._servo_threshold == pytest.approx(executor._SERVO_THRESHOLD_MIN)


def test_calibrated_threshold_separates_blank_from_content() -> None:
    """On a flat desktop the calibrated threshold sits between blank and text-like variance."""
    flat, text = _uniform(), _textlike()
    executor = _executor(flat)
    with _with_screen(flat):
        blank_var = executor._region_has_content(640, 360)[1]
    with _with_screen(text):
        text_var = executor._region_has_content(640, 360)[1]
    assert blank_var < executor._servo_threshold < text_var


def test_calibration_defaults_when_capture_fails() -> None:
    executor = _executor(_uniform())
    with patch("operon.executor.desktop._mss", side_effect=RuntimeError("no display")):
        floor, threshold = executor._calibrate_servo_threshold()
    assert (floor, threshold) == (0.0, executor._SERVO_THRESHOLD_DEFAULT)


def test_calibration_defaults_without_a_physical_monitor() -> None:
    executor = _executor(_uniform())
    fake = _FakeMss(_uniform())
    fake.monitors = fake.monitors[:1]
    with patch("operon.executor.desktop._mss", return_value=fake):
        assert executor._sample_noise_floor() == 0.0


def test_calibration_skips_unreadable_crops() -> None:
    executor = _executor(_uniform())
    with _with_screen(_uniform()), patch.object(executor, "_sample_region", return_value=b""):
        assert executor._sample_noise_floor() == 0.0


def test_constructor_calibrates_when_safe_mode_off() -> None:
    executor = _executor(_noisy(std=5))
    assert executor._noise_floor == pytest.approx(25.0, rel=0.1)


def test_idle_textured_region_counts_as_content_under_calibrated_threshold() -> None:
    # The servo targets flat/blank regions only; the 0.75 x noise-floor leniency is deliberate.
    frame = _noisy(std=5)
    executor = _executor(frame)
    with _with_screen(frame):
        has_content, _ = executor._region_has_content(640, 360)
    assert has_content is True


# ── click-class actions go through the real servo ────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [DesktopExecutor, LinuxDesktopExecutor])
@pytest.mark.parametrize(
    "action_type, primitive",
    [
        (ActionType.CLICK, "click"),
        (ActionType.DOUBLE_CLICK, "doubleClick"),
        (ActionType.TRIPLE_CLICK, "click"),
        (ActionType.RIGHT_CLICK, "rightClick"),
    ],
)
async def test_click_actions_blocked_on_blank_and_allowed_on_content(cls, action_type, primitive) -> None:
    blank, text = _uniform(), _textlike()
    executor = _executor(blank, cls)
    action = AgentAction(action_type=action_type, x=640, y=360)

    with (
        _with_screen(blank),
        patch(f"operon.executor.desktop.pyautogui.{primitive}") as prim,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)
    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND
    prim.assert_not_called()

    with (
        _with_screen(text),
        patch(f"operon.executor.desktop.pyautogui.{primitive}") as prim,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)
    assert result.success is True
    assert result.visual_variance is not None and result.visual_variance > executor._servo_threshold
    prim.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [DesktopExecutor, LinuxDesktopExecutor])
async def test_drag_blocked_on_blank_start_and_allowed_on_content(cls) -> None:
    """KAN-174: DRAG goes through the same real servo as the clicks, on its start point."""
    blank, text = _uniform(), _textlike()
    executor = _executor(blank, cls)
    action = AgentAction(action_type=ActionType.DRAG, x=640, y=360, x_end=100, y_end=100)

    with (
        _with_screen(blank),
        patch("operon.executor.desktop.pyautogui.moveTo"),
        patch("operon.executor.desktop.pyautogui.mouseDown") as down,
        patch("operon.executor.desktop.pyautogui.mouseUp"),
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)
    assert result.success is False
    assert result.failure_category == FailureCategory.EXECUTION_TARGET_NOT_FOUND
    down.assert_not_called()

    with (
        _with_screen(text),
        patch("operon.executor.desktop.pyautogui.moveTo"),
        patch("operon.executor.desktop.pyautogui.mouseDown") as down,
        patch("operon.executor.desktop.pyautogui.mouseUp"),
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="after.png"),
    ):
        result = await executor.execute(action)
    assert result.success is True
    assert result.visual_variance is not None and result.visual_variance > executor._servo_threshold
    down.assert_called_once()

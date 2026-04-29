"""Schemas for the capture stage of the loop."""

from pydantic import Field

from src.models.common import StrictModel


class CaptureFrame(StrictModel):
    """A single browser screenshot captured as loop input."""

    artifact_path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str = Field(default="image/png", min_length=1)
    # Virtual-desktop origin of the monitor this frame was captured from.
    # (0, 0) for the primary display; non-zero on secondary monitors.
    # The executor adds this offset when translating monitor-local coords to
    # pyautogui virtual-desktop coordinates.
    monitor_left: int = Field(default=0)
    monitor_top: int = Field(default=0)
    # Fraction of pixels that changed between the t+100ms and t+200ms burst frames.
    # 0.0 means the screen was static; >0.02 means active animation/shift.
    visual_velocity: float = Field(default=0.0, ge=0.0, le=1.0)

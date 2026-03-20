"""Schemas for the capture stage of the loop."""

from pydantic import Field

from src.models.common import StrictModel


class CaptureFrame(StrictModel):
    """A single browser screenshot captured as loop input."""

    artifact_path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str = Field(default="image/png", min_length=1)

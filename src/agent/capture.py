"""Capture service interface for producing screenshots."""

from __future__ import annotations

import asyncio
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from src.executor.browser import Executor
from src.models.capture import CaptureFrame
from src.models.state import AgentState

logger = logging.getLogger(__name__)

_VELOCITY_WARN_THRESHOLD = 0.02  # >2% pixel change → screen is actively animating


class CaptureService(ABC):
    """Typed interface for the capture stage of the control loop."""

    @abstractmethod
    async def capture(self, state: AgentState) -> CaptureFrame:
        """Return the current screen frame for the active run."""


class ScreenCaptureService(CaptureService):
    """Capture implementation backed by the executor."""

    def __init__(self, executor: Executor, root_dir: str | Path = "runs") -> None:
        self.executor = executor
        self.root_dir = Path(root_dir)

    async def capture(self, state: AgentState) -> CaptureFrame:
        """Capture a screenshot into the planned run artifact path.

        The executor returns a 3-frame burst and attaches visual_velocity to the frame.
        If the screen is actively animating (velocity > threshold), log a warning so
        the loop can treat the perception as potentially unstable.
        """
        if state.step_count == 0 and hasattr(self.executor, "focus_window"):
            await self.executor.focus_window()
            await asyncio.sleep(0.5)
        frame = await self.executor.capture()
        if frame.visual_velocity > _VELOCITY_WARN_THRESHOLD:
            logger.warning(
                "High visual velocity %.3f at step %d — screen may still be animating; perception flagged as unstable",
                frame.visual_velocity,
                state.step_count + 1,
            )
        step_index = state.step_count + 1
        planned_path = self.root_dir / state.run_id / f"step_{step_index}" / "before.png"
        planned_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(frame.artifact_path) != planned_path:
            shutil.move(frame.artifact_path, planned_path)
        return frame.model_copy(update={"artifact_path": str(planned_path)})


# Backwards-compatible alias
BrowserCaptureService = ScreenCaptureService

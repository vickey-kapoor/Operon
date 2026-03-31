"""Screen recorder: captures mss frames in a background thread and encodes to MP4."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import cv2
import mss
import numpy as np

logger = logging.getLogger(__name__)

# Capture at half native resolution for smaller files.
_CAPTURE_SCALE = 0.5


class ScreenRecorder:
    """Record a short video clip of the screen using mss + OpenCV."""

    def __init__(
        self,
        output_path: Path,
        fps: int = 8,
        max_duration: float = 3.0,
    ) -> None:
        self.output_path = Path(output_path)
        self.fps = fps
        self.max_duration = max_duration
        self._frames: list[np.ndarray] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._width: int = 0
        self._height: int = 0

    async def start(self) -> None:
        """Begin capturing frames in a background thread."""
        self._frames = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    async def stop(self) -> Path | None:
        """Stop capturing and encode frames to MP4. Returns the video path or None on failure."""
        self._stop_event.set()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, timeout=5.0)
            self._thread = None

        if not self._frames:
            logger.debug("screen_recorder: no frames captured")
            return None

        try:
            await asyncio.to_thread(self._encode)
            return self.output_path if self.output_path.exists() else None
        except Exception:
            logger.debug("screen_recorder: encoding failed", exc_info=True)
            return None

    def _capture_loop(self) -> None:
        """Synchronous loop: grabs mss screenshots at the target fps."""
        interval = 1.0 / self.fps
        max_frames = int(self.max_duration * self.fps)

        with mss.mss() as sct:
            monitor = sct.monitors[0]  # full virtual screen
            while not self._stop_event.is_set() and len(self._frames) < max_frames:
                t0 = time.monotonic()
                shot = sct.grab(monitor)
                frame = np.array(shot, dtype=np.uint8)
                # mss returns BGRA; drop alpha for OpenCV (BGR)
                frame = frame[:, :, :3]

                # Downscale for smaller file size
                h, w = frame.shape[:2]
                new_w = int(w * _CAPTURE_SCALE)
                new_h = int(h * _CAPTURE_SCALE)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                self._frames.append(frame)
                self._width = new_w
                self._height = new_h

                elapsed = time.monotonic() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)

    def _encode(self) -> None:
        """Encode captured frames to an MP4 file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self._width, self._height)
        )
        try:
            for frame in self._frames:
                writer.write(frame)
        finally:
            writer.release()
        self._frames = []  # free memory

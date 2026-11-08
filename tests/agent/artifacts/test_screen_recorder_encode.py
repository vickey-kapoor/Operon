"""Offline coverage for ScreenRecorder's MP4 encode path.

This path is otherwise untested and only runs on a real desktop capture, so an
OpenCV API change here fails at runtime rather than in CI. Frames are injected
directly to keep the test free of mss and any real screen.
"""

from __future__ import annotations

import numpy as np
import pytest

from operon.agent.artifacts.screen_recorder import ScreenRecorder


@pytest.mark.asyncio
async def test_encode_writes_playable_mp4(tmp_path):
    out = tmp_path / "clip.mp4"
    recorder = ScreenRecorder(out, fps=5)

    recorder._frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
    recorder._width = 64
    recorder._height = 48

    recorder._encode()

    assert out.exists()
    assert out.stat().st_size > 0


def test_fourcc_helper_available():
    """cv2.VideoWriter.fourcc is the spelling that works on OpenCV 4 and 5.

    The module-level cv2.VideoWriter_fourcc alias is gone in OpenCV 5.
    """
    import cv2

    assert cv2.VideoWriter.fourcc(*"mp4v") != 0

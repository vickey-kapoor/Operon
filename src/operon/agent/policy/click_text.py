"""Read the text under a click from the screenshot, for the trust gate (KAN-163).

Under Computer Use a click is only coordinates: no element name, so the trust gate
cannot see that the button says "Place order". This module crops a small box around
the click point from the step's screenshot and reads it with local text recognition
(RapidOCR on the CPU: no model call, no network, about 10 ms a crop).

Off by default (``OPERON_CLICK_TEXT``). The OCR engine is an optional install
(``pip install -e ".[ocr]"``); without it, or on any error, the reader
returns ``None`` and the gate behaves exactly as before.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from PIL import Image

from operon.core.settings import get_settings

logger = logging.getLogger(__name__)

# The box read around the click point, in screenshot pixels: wide enough for a
# two- or three-word button label, short enough to leave most neighbours out.
CROP_HALF_WIDTH = 120
CROP_HALF_HEIGHT = 30

# Longest text kept from one crop; a button label is a few words.
MAX_TEXT_CHARS = 120

TextReader = Callable[[Image.Image], "list[str]"]


class RapidOcrReader:
    """Reads a crop with RapidOCR, loading the engine on first use (thread-safe).

    One per agent loop, so the engine lives with the loop, not in a module global.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: Any = None
        self._failed = False

    def _load(self) -> Any:
        with self._lock:
            if self._engine is None and not self._failed:
                try:
                    from rapidocr import RapidOCR  # optional: the "ocr" extra

                    self._engine = RapidOCR()
                except Exception:  # noqa: BLE001 - missing or broken engine: the gate runs without it
                    self._failed = True
                    logger.warning(
                        "OPERON_CLICK_TEXT is on but RapidOCR could not load; install the ocr extra (pip install -e \".[ocr]\")."
                        "The trust gate runs without reading click text.",
                        exc_info=True,
                    )
            return self._engine

    def __call__(self, image: Image.Image) -> list[str]:
        engine = self._load()
        if engine is None:
            return []
        with self._lock:  # one read at a time: runs can share a loop and its engine
            result = engine(image, use_cls=False)
        return [str(t) for t in (getattr(result, "txts", None) or ()) if t]


def crop_box(x: int, y: int, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """The crop around ``(x, y)`` clamped to an image of ``size``, or None if the point is off it."""
    width, height = size
    if not (0 <= x < width and 0 <= y < height):
        return None
    left = max(0, x - CROP_HALF_WIDTH)
    top = max(0, y - CROP_HALF_HEIGHT)
    right = min(width, x + CROP_HALF_WIDTH)
    bottom = min(height, y + CROP_HALF_HEIGHT)
    return left, top, right, bottom


def read_click_text(
    screenshot_path: str,
    x: int,
    y: int,
    frame_size: tuple[int, int] | None = None,
    *,
    reader: TextReader,
) -> str | None:
    """The text in a small box around ``(x, y)``, lower-cased, or None when there is none.

    ``x, y`` are in the frame's coordinates (``frame_size``); the screenshot file may
    be stored at another size, so the point is scaled to it. Never raises.
    """
    try:
        with Image.open(screenshot_path) as img:
            image = img.convert("RGB")
        if frame_size and frame_size[0] > 0 and frame_size[1] > 0 and frame_size != image.size:
            x = round(x * image.size[0] / frame_size[0])
            y = round(y * image.size[1] / frame_size[1])
        box = crop_box(x, y, image.size)
        if box is None:
            return None
        texts = reader(image.crop(box))
    except Exception:  # noqa: BLE001 - reading text is advisory; it must never break a step
        logger.debug("click text: could not read the crop", exc_info=True)
        return None
    text = " ".join(" ".join(texts).split()).lower()[:MAX_TEXT_CHARS]
    return text or None


def make_click_reader() -> TextReader | None:
    """The reader for this process's settings: None unless ``OPERON_CLICK_TEXT`` is on."""
    return RapidOcrReader() if get_settings().click_text else None

"""Screenshot resizing per model profile, and the size the model is told about."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path

from operon.agent.backends.model_adapter.registry import ImageRule


@dataclass(frozen=True, slots=True)
class SentImage:
    """The image a model is sent, with its size and the original screenshot's size."""

    data: str  # base64
    media_type: str
    width: int
    height: int
    source_width: int
    source_height: int

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data}"


def smart_resize(height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen-VL's ``smart_resize``: (height, width) rounded to multiples of ``factor``
    with the area kept in ``[min_pixels, max_pixels]`` and the aspect ratio close."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def target_size(width: int, height: int, rule: ImageRule) -> tuple[int, int]:
    """(width, height) of the image sent for a ``width`` x ``height`` screenshot."""
    if rule.resize == "smart":
        h, w = smart_resize(
            height, width, factor=rule.factor, min_pixels=rule.min_pixels, max_pixels=rule.max_pixels
        )
        return w, h
    ratio = min(1.0, rule.max_long_edge / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def prepare_image(path: Path, rule: ImageRule) -> SentImage:
    from PIL import Image

    with Image.open(path) as image:
        source_width, source_height = image.size
        width, height = target_size(source_width, source_height, rule)
        if (width, height) != (source_width, source_height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        if rule.format == "png":
            image.convert("RGB").save(buffer, format="PNG")
            media_type = "image/png"
        else:
            image.convert("RGB").save(buffer, format="JPEG", quality=85)
            media_type = "image/jpeg"
    return SentImage(
        data=base64.b64encode(buffer.getvalue()).decode("ascii"),
        media_type=media_type,
        width=width,
        height=height,
        source_width=source_width,
        source_height=source_height,
    )

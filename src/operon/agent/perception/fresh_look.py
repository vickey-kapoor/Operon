"""Fresh look: has the spot the model is about to click changed since it saw it? (KAN-207)

The model takes seconds to decide. Just before a pointer action runs, the loop takes one
fresh frame and compares a small square around the target point with the same square in
the screenshot the model saw. A popup, a page that finished loading or a shifted layout
shows up as a large change there; a blinking caret does not.

Pure and local: no model call. The loop decides what to do with the answer (skip the
action and let the model look again); this module never moves or replaces coordinates.

Off by default: ``OPERON_FRESH_LOOK``.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageChops

from operon.agent.perception.settle import PIXEL_DIFF
from operon.models.policy import ActionType, AgentAction

logger = logging.getLogger(__name__)

# Half the side of the square compared around the target: an 80x80 square.
REGION_RADIUS = 40

# Fraction of pixels in the square that must move by more than PIXEL_DIFF for the spot to
# count as changed. A blinking caret in an 80x80 square is well under this.
CHANGED_THRESHOLD = 0.10

_POINTER_ACTIONS = frozenset(
    {
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.TRIPLE_CLICK,
        ActionType.RIGHT_CLICK,
        ActionType.DRAG,
        ActionType.TYPE,
        ActionType.HOVER,
    }
)

ImageSource = Image.Image | bytes | str | Path


def pointer_point(action: AgentAction) -> tuple[int, int] | None:
    """The point a pointer action targets, or None when there is nothing to check.

    A BATCH is judged by its first child only."""
    if action.action_type is ActionType.BATCH:
        children = action.actions or []
        return pointer_point(children[0]) if children and children[0].action_type is not ActionType.BATCH else None
    if action.action_type not in _POINTER_ACTIONS or action.x is None or action.y is None:
        return None
    return action.x, action.y


def _load(source: ImageSource) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    return Image.open(source)


def spot_changed(
    before: ImageSource, fresh: ImageSource, x: int, y: int, radius: int = REGION_RADIUS
) -> float | None:
    """Fraction of pixels in the square around (x, y) that differ between the two images.

    (x, y) are pixels of the images. The square is clamped to the image. None when either
    image can't be read, the sizes differ, or the point is outside the image. Never raises."""
    try:
        a = _load(before).convert("L")
        b = _load(fresh).convert("L")
        if a.size != b.size:
            return None
        width, height = a.size
        if not (0 <= x < width and 0 <= y < height):
            return None
        box = (max(0, x - radius), max(0, y - radius), min(width, x + radius), min(height, y + radius))
        diff = ImageChops.difference(a.crop(box), b.crop(box))
        data = diff.tobytes()
        if not data:
            return None
        return sum(1 for v in data if v > PIXEL_DIFF) / len(data)
    except Exception as exc:  # noqa: BLE001 - an unreadable image means "no answer", never a failure
        logger.debug("fresh look: compare failed: %s", exc)
        return None

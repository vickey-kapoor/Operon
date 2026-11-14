"""Zoom in before clicking on dense screens (KAN-82). Opt in with ``OPERON_ZOOM``.

On a high-resolution screen the model sees a downscaled image where small targets
are a few pixels wide, so its click can land next to the target. When zoom is on,
a proposed click is refined in a second, one-shot request: the screen is cropped
around the proposed point, the model is shown only that crop with what it said it
was clicking, and its click on the crop is mapped back to the full screen. The
model still picks the target; zoom only lets it look closer (vision-only, no DOM).

Settings (all read when the backend is built):

    OPERON_ZOOM            off (default) | dense | always
                           dense: only on screens whose longer side is at least
                           OPERON_ZOOM_MIN_SIDE pixels, the cheap uncertainty signal
                           (that is where the downscaled view loses detail).
                           always: every click, for measuring.
    OPERON_ZOOM_MIN_SIDE   int, default 2000
    OPERON_ZOOM_FACTOR     float, default 2.0: the crop is 1/factor of the screen
                           on each side (clamped to 1.5..6)

If the zoomed request fails, proposes no point, or zoom is off, the original click
stands unchanged. Each zoom costs one extra model call on that step.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from operon.models.usage import ModelUsage

logger = logging.getLogger(__name__)

OFF, DENSE, ALWAYS = "off", "dense", "always"
MODES = (OFF, DENSE, ALWAYS)
DEFAULT_MIN_SIDE = 2000
DEFAULT_FACTOR = 2.0
GRID = 999.0

# Computer Use calls that click or point at an (x, y) on the screen: the browser
# tool's *_at calls and the Gemini 3.x desktop calls.
POINTING_CALLS = frozenset({
    "click_at", "hover_at", "type_text_at",
    "click", "middle_click", "double_click", "triple_click", "right_click", "move",
})

ZOOM_PROMPT = (
    "You are looking at a zoomed-in crop of the screen, not the whole screen.\n"
    "Task: {intent}\n"
    "On the full screen you chose to call {name}{target}.\n"
    "Find that same target in this zoomed view and call {name} on its exact centre. "
    "Use the same arguments as before except x and y. "
    "If the target is not visible in this view, reply with text only and no call."
)


@dataclass(frozen=True)
class ZoomSettings:
    mode: str = OFF
    min_side: int = DEFAULT_MIN_SIDE
    factor: float = DEFAULT_FACTOR

    @property
    def enabled(self) -> bool:
        return self.mode != OFF

    def applies_to(self, width: int, height: int) -> bool:
        if self.mode == ALWAYS:
            return True
        if self.mode == DENSE:
            return max(width, height) >= self.min_side
        return False


def zoom_settings() -> ZoomSettings:
    """``OPERON_ZOOM*`` from the environment; anything unrecognised means off."""
    mode = (os.getenv("OPERON_ZOOM") or OFF).strip().lower()
    if mode in {"on", "true", "1", "yes"}:
        mode = DENSE
    if mode not in MODES:
        mode = OFF
    try:
        min_side = max(1, int(os.getenv("OPERON_ZOOM_MIN_SIDE", str(DEFAULT_MIN_SIDE))))
    except ValueError:
        min_side = DEFAULT_MIN_SIDE
    try:
        factor = float(os.getenv("OPERON_ZOOM_FACTOR", str(DEFAULT_FACTOR)))
    except ValueError:
        factor = DEFAULT_FACTOR
    return ZoomSettings(mode=mode, min_side=min_side, factor=min(6.0, max(1.5, factor)))


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    width: int
    height: int


def crop_box(x: float, y: float, width: int, height: int, factor: float) -> CropBox:
    """A box 1/``factor`` the size of the screen, centred on (x, y) and kept on screen."""
    crop_w = max(1, min(width, round(width / factor)))
    crop_h = max(1, min(height, round(height / factor)))
    left = int(round(x - crop_w / 2))
    top = int(round(y - crop_h / 2))
    left = max(0, min(width - crop_w, left))
    top = max(0, min(height - crop_h, top))
    return CropBox(left=left, top=top, width=crop_w, height=crop_h)


def grid_to_pixel(value: float, size: int) -> float:
    return max(0.0, min(GRID, float(value))) / GRID * max(size - 1, 0)


def pixel_to_grid(value: float, size: int) -> float:
    if size <= 1:
        return 0.0
    return max(0.0, min(GRID, float(value) / (size - 1) * GRID))


def crop_grid_to_screen_grid(gx: float, gy: float, box: CropBox, width: int, height: int) -> tuple[float, float]:
    """A point on the crop's 0-999 grid as a point on the full screen's 0-999 grid."""
    px = box.left + grid_to_pixel(gx, box.width)
    py = box.top + grid_to_pixel(gy, box.height)
    return round(pixel_to_grid(px, width), 2), round(pixel_to_grid(py, height), 2)


def first_pointing_call(function_calls: list[dict]) -> int | None:
    for index, call in enumerate(function_calls):
        args = call.get("args") or {}
        if call.get("name") in POINTING_CALLS and "x" in args and "y" in args:
            return index
    return None


def _point(response: dict) -> tuple[float, float, dict] | None:
    """The zoomed reply's first point, and the args of the call that gave it."""
    for call in response.get("function_calls") or []:
        args = call.get("args") or {}
        try:
            return float(args["x"]), float(args["y"]), args
        except (KeyError, TypeError, ValueError):
            continue
    return None


def merge_usage(first: ModelUsage | None, second: ModelUsage | None) -> ModelUsage | None:
    """Both requests' tokens and cost as one record, so the step's cost counts the zoom."""
    if first is None or second is None:
        return first or second

    def add(a, b):
        return None if a is None and b is None else (a or 0) + (b or 0)

    return first.model_copy(update={
        field: add(getattr(first, field), getattr(second, field))
        for field in (
            "input_tokens", "output_tokens", "total_tokens", "cache_creation_input_tokens",
            "cache_read_input_tokens", "thoughts_tokens", "input_cost_usd", "output_cost_usd",
            "estimated_cost_usd",
        )
    })


@dataclass
class ZoomResult:
    function_calls: list[dict]
    zoomed: bool
    usage: ModelUsage | None = None
    detail: str = ""
    # A zoom request was sent (it may have failed): the client's latest usage may be its.
    requested: bool = False


async def refine_click(
    *,
    client: Any,
    settings: ZoomSettings,
    function_calls: list[dict],
    screenshot_path: str,
    width: int,
    height: int,
    intent: str,
    rationale: str = "",
    out_dir: Path | None = None,
) -> ZoomResult:
    """Refine the first pointing call's x, y by asking again on a crop around it.

    Returns the calls unchanged (``zoomed=False``) when zoom does not apply here or
    the zoomed request gives no usable point.
    """
    if not settings.applies_to(width, height) or not hasattr(client, "run_step"):
        return ZoomResult(function_calls, zoomed=False)
    index = first_pointing_call(function_calls)
    if index is None:
        return ZoomResult(function_calls, zoomed=False)
    call = function_calls[index]
    args = call.get("args") or {}
    try:
        gx, gy = float(args["x"]), float(args["y"])
    except (TypeError, ValueError):
        return ZoomResult(function_calls, zoomed=False)
    box = crop_box(grid_to_pixel(gx, width), grid_to_pixel(gy, height), width, height, settings.factor)
    out_dir = out_dir or Path(screenshot_path).resolve().parent
    crop_path = out_dir / "zoom_crop.png"
    try:
        from PIL import Image

        out_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(screenshot_path) as image:
            image.crop((box.left, box.top, box.left + box.width, box.top + box.height)).save(crop_path)
    except Exception as exc:  # noqa: BLE001 - a broken crop keeps the original click
        logger.warning("zoom crop failed: %s", exc)
        return ZoomResult(function_calls, zoomed=False, detail=f"crop failed: {exc}")

    target = str(args.get("intent") or rationale or "").strip()
    prompt = ZOOM_PROMPT.format(
        intent=intent,
        name=call.get("name"),
        target=f" for: {target[:300]}" if target else "",
    )
    try:
        response = await client.run_step(prompt=prompt, screenshot_path=str(crop_path))
    except Exception as exc:  # noqa: BLE001 - a failed zoom keeps the original click
        logger.warning("zoom request failed: %s", exc)
        return ZoomResult(function_calls, zoomed=False, detail=f"zoom request failed: {exc}", requested=True)
    usage = client.latest_usage() if hasattr(client, "latest_usage") else None
    usage = usage if isinstance(usage, ModelUsage) else None
    point = _point(response)
    if point is None:
        return ZoomResult(function_calls, zoomed=False, usage=usage, detail="zoom proposed no point", requested=True)
    sx, sy = crop_grid_to_screen_grid(point[0], point[1], box, width, height)
    new_args = {**args, "x": sx, "y": sy}
    if "safety_decision" in point[2]:
        # The zoomed look flagged the click for confirmation: keep that, so the run
        # still pauses for the user before it clicks.
        new_args["safety_decision"] = point[2]["safety_decision"]
    refined = [dict(c) for c in function_calls]
    refined[index] = {**call, "args": new_args}
    detail = f"zoom {call.get('name')} ({gx:g}, {gy:g}) -> ({sx:g}, {sy:g}) via crop {box}"
    logger.debug(detail)
    return ZoomResult(refined, zoomed=True, usage=usage, detail=detail, requested=True)

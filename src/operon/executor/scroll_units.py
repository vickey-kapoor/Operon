"""Scroll units. ``AgentAction.scroll_amount`` is pixels everywhere, 100 per wheel notch
(positive = up / right). Desktop wheels take clicks, so convert at the last step."""

PIXELS_PER_NOTCH = 100
# Windows wheel events take 120 units per notch; X11 and macOS take whole clicks.
WIN_WHEEL_DELTA = 120


def wheel_clicks(pixels: int) -> int:
    """Wheel clicks for a pixel amount: signed, rounded, never 0 for a non-zero amount."""
    if pixels == 0:
        return 0
    clicks = max(1, round(abs(pixels) / PIXELS_PER_NOTCH))
    return clicks  # DRILL KAN-299: sign dropped on purpose; never merge this branch

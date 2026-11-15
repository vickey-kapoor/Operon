"""Between-step change note: say what flashed past while the screen settled (KAN-111).

The settle wait (``settle.py``) already looks at small frames between an action and the
next screenshot. A toast that pops up and closes, or a spinner that is still turning, can
be gone or invisible in a single screenshot. :class:`ChangeWatcher` listens to those
frames and turns what it saw into one line of plain text for the model.

* **Text, not images.** AOI found that extra images made Gemini worse; the gain came from
  turning frames into text. So the model gets one short sentence with its next screenshot.
* **Quiet by default.** A change that is still on screen is already in the screenshot, so
  it is not repeated. The note speaks only about (a) things that appeared and then went
  away, and (b) regions still changing when the wait ended.
* **Local and cheap.** It compares the same tiny grayscale signatures as the settle
  detector; no model call.

Off by default (``OPERON_CHANGE_NOTE``). The note reaches the model as an advisory hint
(``source="change_note"``), the same channel as the other "Notes from Operon for this step".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from operon.agent.perception.settle import (
    PIXEL_DIFF,
    SAMPLE_INTERVAL_S,
    SIGNATURE_SIZE,
    SettleResult,
    settle_after_action,
)
from operon.core.settings import get_settings

# The signature is cut into a coarse grid; a cell "changed" when this share of its pixels moved.
GRID_COLS = 8
GRID_ROWS = 6
CELL_CHANGE_RATIO = 0.04

# A region this big is a repaint or a scroll, not a notable thing: say so in one phrase.
WHOLE_SCREEN_SHARE = 0.5

# Regions named per kind, and the note's length cap, so it can never grow into noise.
MAX_REGIONS_PER_KIND = 2
MAX_NOTE_CHARS = 300

_CELL_W = SIGNATURE_SIZE[0] // GRID_COLS
_CELL_H = SIGNATURE_SIZE[1] // GRID_ROWS

Grab = Callable[[], Awaitable["bytes | None"]]


def _cells_changed(a: bytes, b: bytes) -> list[bool]:
    """For each grid cell, whether enough of its pixels differ between two signatures."""
    width = SIGNATURE_SIZE[0]
    out: list[bool] = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            changed = 0
            for y in range(row * _CELL_H, (row + 1) * _CELL_H):
                base = y * width + col * _CELL_W
                for i in range(base, base + _CELL_W):
                    if abs(a[i] - b[i]) > PIXEL_DIFF:
                        changed += 1
            out.append(changed / (_CELL_W * _CELL_H) > CELL_CHANGE_RATIO)
    return out


def _regions(cells: set[int]) -> list[set[int]]:
    """Group grid cells that touch (up, down, left, right) into regions."""
    remaining = set(cells)
    found: list[set[int]] = []
    while remaining:
        stack = [remaining.pop()]
        region = {stack[0]}
        while stack:
            cell = stack.pop()
            row, col = divmod(cell, GRID_COLS)
            for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                n = r * GRID_COLS + c
                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS and n in remaining:
                    remaining.discard(n)
                    region.add(n)
                    stack.append(n)
        found.append(region)
    return found


def _where(region: set[int]) -> str:
    """Name a region's position in words: 'the top right', 'the middle', ..."""
    rows = [cell // GRID_COLS for cell in region]
    cols = [cell % GRID_COLS for cell in region]
    row_mid = (min(rows) + max(rows) + 1) / 2 / GRID_ROWS
    col_mid = (min(cols) + max(cols) + 1) / 2 / GRID_COLS
    vertical = "top" if row_mid < 1 / 3 else "bottom" if row_mid > 2 / 3 else "middle"
    horizontal = "left" if col_mid < 1 / 3 else "right" if col_mid > 2 / 3 else "center"
    if vertical == "middle" and horizontal == "center":
        return "the middle"
    if vertical == "middle":
        return f"the {horizontal} side"
    if horizontal == "center":
        return f"the {vertical} of the screen"
    return f"the {vertical} {horizontal}"


class ChangeWatcher:
    """A :class:`~operon.agent.perception.settle.FrameObserver` that writes the change note."""

    def __init__(self, *, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._first: bytes | None = None
        self._previous: bytes | None = None
        self._frames = 0
        self._changed_frames: list[list[int]] = [[] for _ in range(GRID_COLS * GRID_ROWS)]

    def observe(self, index: int, signature: bytes, change: float | None) -> None:
        if self._first is None:
            self._first = signature
        elif self._previous is not None:
            for cell, moved in enumerate(_cells_changed(self._previous, signature)):
                if moved:
                    self._changed_frames[cell].append(index)
        self._previous = signature
        self._frames = index + 1

    def note(self) -> str | None:
        """One line about what appeared and went away, or what is still moving; None if nothing notable."""
        if self._first is None or self._previous is None or self._frames < 2:
            return None
        net = _cells_changed(self._first, self._previous)
        last = self._frames - 1
        transient: set[int] = set()
        moving: set[int] = set()
        for cell, frames in enumerate(self._changed_frames):
            if not frames:
                continue
            if last in frames and (last - 1) in frames:
                moving.add(cell)  # still changing in each of the last two frames
            elif not net[cell]:
                transient.add(cell)  # changed, then back to how it started
        parts: list[str] = []
        total = GRID_COLS * GRID_ROWS
        if len(moving) > WHOLE_SCREEN_SHARE * total:
            parts.append("most of the screen was still changing when the wait ended (an animation or a video, or a page still loading)")
        else:
            for region in sorted(_regions(moving), key=len, reverse=True)[:MAX_REGIONS_PER_KIND]:
                parts.append(f"{_where(region)} was still changing when the wait ended (a loading indicator, or content still arriving)")
        for region in sorted(_regions(transient), key=len, reverse=True)[:MAX_REGIONS_PER_KIND]:
            frames = sorted({f for cell in region for f in self._changed_frames[cell]})
            seconds = (frames[-1] - frames[0] + 1) * self._interval_s
            span = "briefly" if seconds <= 0.5 else f"for about {seconds:.1f} s"
            parts.append(
                f"something showed up {span} in {_where(region)} and was gone again (it is not in this screenshot)"
            )
        if not parts:
            return None
        text = "Screen activity right after your last action: " + "; ".join(parts) + "."
        return text if len(text) <= MAX_NOTE_CHARS else text[: MAX_NOTE_CHARS - 1].rstrip() + "…"


class ChangeNoteBuffer:
    """Holds the notes an executor collected, until the loop hands them to the model."""

    def __init__(self) -> None:
        self._notes: list[str] = []

    async def settle(self, grab: Grab, fallback_delay: float) -> SettleResult | None:
        """The executor's post-action wait; with OPERON_CHANGE_NOTE on, it also writes a note."""
        if not get_settings().change_note:
            return await settle_after_action(grab, fallback_delay)
        watcher = ChangeWatcher()
        result = await settle_after_action(grab, fallback_delay, observers=(watcher,))
        note = watcher.note()
        if note:
            self._notes.append(note)
        return result

    def pop(self) -> str | None:
        """The notes since the last call, joined, and forgotten."""
        if not self._notes:
            return None
        text = " ".join(self._notes)
        self._notes.clear()
        return text

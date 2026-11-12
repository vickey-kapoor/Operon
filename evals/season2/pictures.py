"""Pictures for the Season 2 desktop tasks (KAN-104): text drawn as shapes in an SVG file.

Gemini passed the first two Season 2 Desktop sets by scripting over the task files
(KAN-87, KAN-104), so some tasks now hold their key data only in a picture: a table, a
memo, a timesheet or an invoice that a person reads off the screen (Edge opens .svg
files). The text is drawn glyph by glyph from ``glyphs.py`` as ``<use>`` references to
paths whose ids say nothing about the letter, so the file holds no readable text.

A picture is plain text, built from data with no rendering library, so it is the same
on every machine and Python version: the folder's reset writes it like any seeded file,
and the season fingerprint hashes it with the task.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from evals.season2.glyphs import GLYPHS, HEIGHT, SPACE

Cell = tuple[int, str]  # x in pixels, text
Row = Sequence[Cell]


def text_width(text: str) -> int:
    return sum(SPACE if ch == " " else GLYPHS.get(ch, GLYPHS["?"])[0] for ch in text)


def _path(runs: str) -> str:
    """Each run of pixels as a one-pixel line, drawn with the stroke of the group that uses it."""
    return "".join(f"M{x} {y}.5h{n}" for x, y, n in (r.split(",") for r in runs.split()))


def _id(i: int) -> str:
    """A short id: a, b, ... z, then aa, ab ..."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    return letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]


def picture(name: str, rows: Sequence[Row], *, width: int, line: int = 30, margin: int = 24,
            border: bool = False, stamp: str | None = None) -> str:
    """An SVG of ``rows`` (each a list of cells at an x offset), one row per ``line`` pixels.

    ``name`` salts the glyph ids, so the same letter has a different id in every picture.
    ``stamp`` adds a large red word across the top right, as on a rejected photo."""
    # Short ids in an order set by a hash of the picture's name, so an id says nothing about its letter.
    used = sorted({ch if ch in GLYPHS else "?" for row in rows for _, text in row for ch in text if ch != " "}
                  | set(stamp or ""), key=lambda ch: hashlib.sha256(f"{name}:{ch}".encode()).hexdigest())
    ids = {ch: _id(i) for i, ch in enumerate(used)}

    groups: list[str] = []
    for i, row in enumerate(rows):
        uses: list[str] = []
        for x0, text in row:
            x = margin + x0
            for ch in text:
                if ch == " ":
                    x += SPACE
                    continue
                if ch not in GLYPHS:
                    ch = "?"
                uses.append(f'<use href="#{ids[ch]}" x="{x}"/>')
                x += GLYPHS[ch][0]
        if uses:
            groups.append(f'<g transform="translate(0 {margin + i * line})">' + "".join(uses) + "</g>")
    stamp_uses: list[str] = []
    if stamp:
        # Drawn at twice the size in the top-right corner: positions are in the halved space.
        x = (width - margin - 2 * text_width(stamp)) // 2
        for ch in stamp:
            stamp_uses.append(f'<use href="#{ids[ch]}" x="{x}" y="{margin // 4}"/>')
            x += GLYPHS[ch][0]
    height = margin * 2 + len(rows) * line + (HEIGHT - line if line < HEIGHT else 0)
    defs = "".join(f'<path id="{ids[ch]}" d="{_path(GLYPHS[ch][1])}"/>' for ch in used)
    frame = (f'<rect x="4" y="4" width="{width - 8}" height="{height - 8}" fill="none" stroke="#555" stroke-width="3"/>'
             if border else "")
    stamped = f'<g fill="none" stroke="#c00" transform="scale(2)">{"".join(stamp_uses)}</g>' if stamp_uses else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n<defs>{defs}</defs>\n'
            f'<rect width="100%" height="100%" fill="#fbfaf6"/>{frame}\n'
            f'<g fill="none" stroke="#111">\n' + "\n".join(groups) + f"\n</g>{stamped}\n</svg>\n")


def lines(*texts: str) -> list[list[Cell]]:
    """Rows of plain text, each starting at the left margin."""
    return [[(0, text)] if text else [] for text in texts]


def table(header: Sequence[str], body: Sequence[Sequence[str]], columns: Sequence[int]) -> list[list[Cell]]:
    """A table's rows: the header, then each row's cells at the column offsets."""
    return [list(zip(columns, header)), *[list(zip(columns, row)) for row in body]]

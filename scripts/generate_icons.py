"""
Generate all Tauri app icons for Operon from a programmatic source.

Icon concept: precision targeting reticle on a deep-dark background.
Reflects Operon's vision-first, coordinate-space execution model.

Outputs:
  src-tauri/icons/32x32.png
  src-tauri/icons/128x128.png
  src-tauri/icons/128x128@2x.png  (256x256)
  src-tauri/icons/icon.ico        (multi-size: 16,32,48,64,128,256)
  src-tauri/icons/icon.icns       (macOS: ic07–ic10 PNG-compressed)
"""

import io
import struct
from pathlib import Path

from PIL import Image, ImageDraw

ICONS_DIR = Path(__file__).parent.parent / "src-tauri" / "icons"

BG = (10, 10, 14, 255)          # near-black
RING = (56, 189, 248, 255)      # sky-400
TICK = (56, 189, 248, 200)
CENTER = (255, 255, 255, 255)
CENTER_GLOW = (56, 189, 248, 80)


def draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), BG)
    d = ImageDraw.Draw(img)
    cx = cy = size / 2

    # ── outer ring ──────────────────────────────────────────────
    r_outer = size * 0.38
    r_inner = size * 0.30
    lw = max(1, round(size * 0.028))
    d.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        outline=RING,
        width=lw,
    )

    # ── inner ring ──────────────────────────────────────────────
    r2 = size * 0.18
    d.ellipse(
        [cx - r2, cy - r2, cx + r2, cy + r2],
        outline=(*RING[:3], 160),
        width=max(1, round(size * 0.018)),
    )

    # ── crosshair ticks (4 directions) ──────────────────────────
    gap = size * 0.42       # tick starts just outside outer ring
    tick_end = size * 0.48
    tw = max(1, round(size * 0.022))
    # top
    d.line([(cx, cy - gap), (cx, cy - tick_end)], fill=TICK, width=tw)
    # bottom
    d.line([(cx, cy + gap), (cx, cy + tick_end)], fill=TICK, width=tw)
    # left
    d.line([(cx - gap, cy), (cx - tick_end, cy)], fill=TICK, width=tw)
    # right
    d.line([(cx + gap, cy), (cx + tick_end, cy)], fill=TICK, width=tw)

    # ── center glow ─────────────────────────────────────────────
    glow_r = size * 0.07
    d.ellipse(
        [cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r],
        fill=CENTER_GLOW,
    )

    # ── center dot ──────────────────────────────────────────────
    dot_r = size * 0.035
    d.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill=CENTER,
    )

    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_icns(size_map: dict[str, Image.Image]) -> bytes:
    """
    Minimal ICNS writer using PNG-compressed modern icon types.
    type codes: ic07=128, ic08=256, ic09=512, ic10=1024
    """
    type_for_size = {128: b"ic07", 256: b"ic08", 512: b"ic09", 1024: b"ic10"}
    entries = []
    for size, img in size_map.items():
        code = type_for_size.get(size)
        if code is None:
            continue
        data = png_bytes(img)
        entry_len = 8 + len(data)
        entries.append(code + struct.pack(">I", entry_len) + data)

    body = b"".join(entries)
    total = 8 + len(body)
    return b"icns" + struct.pack(">I", total) + body


def main() -> None:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)

    sizes = {s: draw_icon(s) for s in [16, 32, 48, 64, 128, 256, 512, 1024]}

    # ── PNGs ────────────────────────────────────────────────────
    sizes[32].save(ICONS_DIR / "32x32.png")
    sizes[128].save(ICONS_DIR / "128x128.png")
    sizes[256].save(ICONS_DIR / "128x128@2x.png")
    print("wrote 32x32.png, 128x128.png, 128x128@2x.png")

    # ── ICO (multi-size) ────────────────────────────────────────
    # Pillow's ICO writer resizes from the source image when given `sizes`.
    # Pass the largest image and let it produce all required sizes.
    sizes[256].save(
        ICONS_DIR / "icon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print("wrote icon.ico")

    # ── ICNS ────────────────────────────────────────────────────
    icns_data = build_icns({128: sizes[128], 256: sizes[256], 512: sizes[512], 1024: sizes[1024]})
    (ICONS_DIR / "icon.icns").write_bytes(icns_data)
    print("wrote icon.icns")

    print(f"\nAll icons written to {ICONS_DIR}")


if __name__ == "__main__":
    main()

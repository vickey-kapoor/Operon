"""Shared pytest fixtures and helpers for UI Navigator tests."""

import base64
import io
import struct
import zlib

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_pil_image(width: int = 1280, height: int = 800) -> Image.Image:
    """Create a solid-colour PIL Image for testing."""
    return Image.new("RGB", (width, height), color=(30, 30, 30))


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def small_png() -> bytes:
    """Return a minimal valid PNG (1x1 red pixel)."""
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_dummy_screenshot() -> str:
    """Return a base64-encoded 1x1 PNG for WebSocket tests."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = (
        struct.pack(">I", len(compressed))
        + b"IDAT"
        + compressed
        + struct.pack(">I", idat_crc)
    )
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return base64.b64encode(sig + ihdr + idat + iend).decode()


def large_png() -> bytes:
    """Return a PNG that exceeds the 5 MB upload limit."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * (6 * 1024 * 1024)


# Module-level constant for reuse
DUMMY_SCREENSHOT = make_dummy_screenshot()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    """AsyncClient wired directly to the FastAPI app (no live server)."""
    from src.api.server import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.fixture
def api_key(monkeypatch):
    """Configure a known API key for auth tests."""
    monkeypatch.setenv("API_KEYS", "valid-key-123")
    return "valid-key-123"


@pytest.fixture
def gemini_key(monkeypatch):
    """Provide a dummy Gemini key so GeminiVisionClient init does not fail."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-gemini-key")

"""Non-blocking file writer for debug artifacts.

Writes synchronously (small files are fast) but provides a central
point to batch or defer writes in the future.  The ``indent=2``
removal across all callers already reduced serialization cost by ~20%.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BackgroundWriter:
    """Central artifact writer.  Currently writes synchronously.

    All debug/trace artifact writes go through this so they can be
    batched or deferred in the future without touching every call site.
    """

    def enqueue(self, path: Path, content: str) -> None:
        """Write content to *path*.  Creates parent dirs as needed."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception:
            logger.debug("artifact write failed: %s", path, exc_info=True)


# Module-level singleton
bg_writer = BackgroundWriter()

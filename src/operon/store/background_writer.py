"""Non-blocking file writer for debug artifacts.

Offloads disk I/O to a background thread pool so artifact writes never
block the asyncio event loop.  Falls back to synchronous writes outside
of an async context (tests, startup code).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# One worker: writes run in the order they were queued, so a later state.json
# snapshot can never be overwritten by an earlier one still in flight.
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="bg_writer"
)


def _write_file(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write a sibling temp file, then swap it in: a reader (the observer) never
        # sees a half-written file.
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(content, encoding="utf-8")
        _replace(tmp, path)
    except Exception:
        logger.debug("artifact write failed: %s", path, exc_info=True)


def _replace(tmp: Path, path: Path) -> None:
    """Swap ``tmp`` in for ``path``. On Windows the swap fails while a reader has
    ``path`` open (for a few ms), so retry for about a second. Never write in place:
    a reader would see a half-written file. If it is still refused, the old file
    stays whole and the next snapshot replaces it."""
    for delay in (0.01, 0.03, 0.1, 0.3, 0.6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay)
    tmp.unlink(missing_ok=True)
    logger.warning("artifact write skipped: %s stayed locked by a reader", path)


class BackgroundWriter:
    """Central artifact writer — fire-and-forget, never blocks the event loop.

    In synchronous mode (``sync=True``) writes happen inline — used in unit
    tests that assert file existence immediately after the call.
    """

    def __init__(self, *, sync: bool = False) -> None:
        self._sync = sync
        self._pending: list[asyncio.Future] = []

    def enqueue(self, path: str | Path, content: str) -> None:
        """Schedule a file write.  Returns immediately; write runs in a thread."""
        # Callers pass str paths (e.g. ModelDebugArtifacts fields); _write_file needs
        # Path.parent, and its broad except would otherwise drop the write silently.
        path = Path(path)
        if self._sync:
            _write_file(path, content)
            return
        try:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(_thread_pool, _write_file, path, content)
            self._pending.append(fut)
            # Remove completed futures to avoid accumulating references.
            fut.add_done_callback(lambda f: self._pending.remove(f) if f in self._pending else None)
        except RuntimeError:
            # No running event loop (startup context) — write inline.
            _write_file(path, content)

    def append(self, path: str | Path, line: str) -> None:
        """Schedule an append of one line to *path*.  Never overwrites existing content."""
        path = Path(path)

        def _append_line(p: Path, content: str) -> None:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as fh:
                    fh.write(content)
                    fh.flush()
            except Exception:
                logger.debug("artifact append failed: %s", p, exc_info=True)

        if self._sync:
            _append_line(path, line)
            return
        try:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(_thread_pool, _append_line, path, line)
            self._pending.append(fut)
            fut.add_done_callback(lambda f: self._pending.remove(f) if f in self._pending else None)
        except RuntimeError:
            _append_line(path, line)

    async def flush(self) -> None:
        """Await all pending background writes.  Use in tests that check artifacts."""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)


# Module-level singleton
bg_writer = BackgroundWriter()

"""Non-blocking file writer for debug artifacts.

Offloads disk I/O to a background thread pool so artifact writes never
block the asyncio event loop.  Falls back to synchronous writes outside
of an async context (tests, startup code).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="bg_writer"
)


def _write_file(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception:
        logger.debug("artifact write failed: %s", path, exc_info=True)


class BackgroundWriter:
    """Central artifact writer — fire-and-forget, never blocks the event loop.

    In synchronous mode (``sync=True``) writes happen inline — used in unit
    tests that assert file existence immediately after the call.
    """

    def __init__(self, *, sync: bool = False) -> None:
        self._sync = sync
        self._pending: list[asyncio.Future] = []

    def enqueue(self, path: Path, content: str) -> None:
        """Schedule a file write.  Returns immediately; write runs in a thread."""
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

    def append(self, path: Path, line: str) -> None:
        """Schedule an append of one line to *path*.  Never overwrites existing content."""
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

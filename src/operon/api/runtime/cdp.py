"""CDP browser startup helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def ensure_cdp_ready(mode: str = "batch") -> None:
    """Prepare the live view before the first browser step runs.

    Nothing to do in either mode. Batch tasks launch their own isolated Chromium.
    In observable mode the executor owns the one browser: it launches Chromium on
    port 9222 on the first step and points the live view at each run's page
    (``NativeBrowserExecutor._attach_live_view``). Launching a second browser here
    left the live view on a browser the agent never used (Greptile G5).
    """
    if mode != "observable":
        return
    logger.debug("ensure_cdp_ready: observable mode; the executor attaches the live view")

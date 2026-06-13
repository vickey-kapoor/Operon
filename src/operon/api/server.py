"""FastAPI application bootstrap for the MVP browser-only agent."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from operon.core.paths import runs_dir

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Run startup checks: config validation (fail-fast) + bounded runs/ retention sweep."""
    # 1. Config validation — fail fast on typos like OPERON_BROWSER_BACKEND=conputer_use.
    from operon.api.runtime_config import validate_modes
    errors = validate_modes()
    if errors:
        joined = "\n  - ".join(errors)
        raise RuntimeError(f"Invalid Operon runtime configuration:\n  - {joined}")

    # 2. Retention sweep — runs/ would otherwise grow forever.
    # Disable with OPERON_RUNS_RETAIN_DAYS=-1; default 14 days.
    retain_days = int(os.getenv("OPERON_RUNS_RETAIN_DAYS", "14"))
    if retain_days >= 0:
        try:
            from operon.store.cleanup import cleanup_old_runs
            deleted, freed = cleanup_old_runs(
                root_dir=runs_dir(),
                keep_days=retain_days,
                delete=True,
            )
            if deleted:
                logger.info(
                    "retention_sweep: removed %d old run dir(s) (%.1f MB) older than %d days",
                    len(deleted), freed / (1024 * 1024), retain_days,
                )
        except Exception as exc:
            # Cleanup failure must never block server startup.
            logger.warning("retention_sweep skipped: %s", exc)

    yield

    # Shutdown: release executor-held OS resources so they don't outlive the
    # server — the browser executor's persistent Chromium + Playwright, and the
    # desktop executor's still-active screen recorders + launched child apps.
    # Each executor exposes a uniform aclose(); we only touch loops that were
    # actually built (iter_built_executors never triggers lazy construction).
    from operon.api.runtime.loops import iter_built_executors
    for executor in iter_built_executors():
        if not hasattr(executor, "aclose"):
            continue
        try:
            await executor.aclose()
        except Exception as exc:
            # Shutdown cleanup must never raise out of the lifespan.
            logger.warning("executor shutdown cleanup skipped: %s", exc)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    load_dotenv(find_dotenv(usecwd=True), override=True)

    from operon.api.routes import router
    from operon.api.ws_stream import router as ws_router

    app = FastAPI(
        title="Operon",
        version="0.1.0",
        description="Operate any interface with a vision-driven computer-use engine.",
        lifespan=_lifespan,
    )
    origins = os.getenv("CORS_ORIGINS", "").strip()
    allowed: set[str] = set()
    if origins:
        allowed.update(o.strip() for o in origins.split(","))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(ws_router)
    return app


app = create_app()

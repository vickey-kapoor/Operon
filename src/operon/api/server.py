"""FastAPI application bootstrap for the MVP browser-only agent."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

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

    # 3. A run environment other than the live desktop (KAN-80) boots its machine now,
    # off the event loop, so the first desktop task doesn't wait for it.
    from operon.environments import DEFAULT_ENVIRONMENT, environment_name

    if environment_name() != DEFAULT_ENVIRONMENT:
        from operon.api.runtime.loops import desktop_provider

        try:
            await asyncio.to_thread(desktop_provider)
        except Exception as exc:
            # The API still serves browser tasks; desktop tasks report this error.
            logger.error("environment %r did not start: %s", environment_name(), exc)

    yield

    # Shutdown: release executor-held OS resources so they don't outlive the
    # server — the browser executor's persistent Chromium + Playwright, and the
    # desktop executor's still-active screen recorders + launched child apps.
    # Each executor exposes a uniform aclose(); we only touch loops that were
    # actually built (iter_built_executors never triggers lazy construction).
    # Then the desktop environment provider stops its machine (KAN-80).
    from operon.api.runtime.loops import close_built_runtime

    await close_built_runtime()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    load_dotenv(find_dotenv(usecwd=True), override=True)
    from operon.core.paths import load_user_env

    load_user_env()  # `operon setup`'s per-user key; never overrides .env or the environment

    from operon.api.auth import (
        TrustedHostMiddleware,
        api_key_file,
        configured_api_keys,
        configured_cors_origins,
        require_api_key,
    )
    from operon.api.routes import router
    from operon.api.ws_stream import router as ws_router

    app = FastAPI(
        title="Operon",
        version="0.1.0",
        description="Operate any interface with a vision-driven computer-use engine.",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(configured_cors_origins()),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Outermost: a request for a host name other than loopback (or one listed in
    # OPERON_ALLOWED_HOSTS) is refused before routing, which blocks DNS rebinding.
    app.add_middleware(TrustedHostMiddleware)
    # API-key auth: enforced on every HTTP route except /health. The key is API_KEYS,
    # or one generated now into .var/api-key; OPERON_ALLOW_UNAUTHENTICATED turns it off.
    if not configured_api_keys():
        logger.warning(
            "API authentication is DISABLED (OPERON_ALLOW_UNAUTHENTICATED is set). Any client "
            "that can reach this server can drive the agent."
        )
    elif not os.getenv("API_KEYS", "").strip():
        logger.info("API_KEYS not set: clients must send the key in %s as X-API-Key", api_key_file())
    app.include_router(router, dependencies=[Depends(require_api_key)])
    app.include_router(ws_router)

    # The local task page (KAN-89). Served without the key: it holds no secret and
    # asks for the key (or reads it from the #key= fragment `operon serve` prints).
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        # A bare http://127.0.0.1:8080 opens the task page. The key stays in the
        # browser's #fragment, so a link without it asks for the key once.
        return RedirectResponse("/app", status_code=307)

    @app.get("/app", include_in_schema=False)
    async def task_page() -> HTMLResponse:
        return HTMLResponse(task_page_html(), headers=_PAGE_HEADERS)

    return app


# The page runs only its own inline script and talks only to this server.
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src 'self' blob: data:; connect-src 'self' ws: wss:; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}


def task_page_html() -> str:
    """The task page's HTML, shipped as package data (operon/api/static/app.html)."""
    from importlib.resources import files

    return files("operon.api").joinpath("static/app.html").read_text(encoding="utf-8")


app = create_app()

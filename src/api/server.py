"""FastAPI application bootstrap for the MVP browser-only agent."""

import os

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    load_dotenv(find_dotenv(usecwd=True), override=True)

    # Register benchmark plugins before any engine code runs.
    import src.benchmarks.form_plugin  # noqa: F401
    from src.api.routes import router
    from src.api.ws_stream import router as ws_router

    app = FastAPI(
        title="Operon",
        version="0.1.0",
        description="Operate any interface with a vision-driven computer-use engine.",
    )
    origins = os.getenv("CORS_ORIGINS", "").strip()
    # Always allow the Tauri WS bridge and local dev origins.
    allowed = {"http://localhost:9001", "http://127.0.0.1:9001", "http://localhost:5173"}
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

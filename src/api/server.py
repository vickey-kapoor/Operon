"""FastAPI application bootstrap for the MVP browser-only agent."""

import os

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    load_dotenv(find_dotenv(usecwd=True), override=False)

    # Register benchmark plugins before any engine code runs.
    import src.benchmarks.form_plugin  # noqa: F401
    import src.benchmarks.gmail_plugin  # noqa: F401

    from src.api.routes import router

    app = FastAPI(
        title="Operon",
        version="0.1.0",
        description="Operate any interface with a vision-driven computer-use engine.",
    )
    origins = os.getenv("CORS_ORIGINS", "").strip()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in origins.split(",")],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(router)
    return app


app = create_app()

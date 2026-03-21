"""FastAPI application bootstrap for the MVP browser-only agent."""

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    load_dotenv(find_dotenv(usecwd=True), override=False)

    from src.api.routes import router

    app = FastAPI(
        title="Operon",
        version="0.1.0",
        description="Operate any interface with a vision-driven computer-use engine.",
    )
    app.include_router(router)
    return app


app = create_app()

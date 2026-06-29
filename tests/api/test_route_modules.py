"""Route module registration smoke tests."""

from fastapi import APIRouter, FastAPI

from operon.api.routes import browser, desktop, observer, router


def test_split_route_modules_export_routers() -> None:
    assert isinstance(browser.router, APIRouter)
    assert isinstance(desktop.router, APIRouter)
    assert isinstance(observer.router, APIRouter)


def test_aggregate_router_includes_split_routes() -> None:
    # Resolve final paths via a built app's OpenAPI schema rather than poking the
    # router's internal `.routes`, whose shape varies across FastAPI versions:
    # newer FastAPI represents `include_router` lazily as `_IncludedRouter`
    # entries that have no `.path` until the app resolves them.
    app = FastAPI()
    app.include_router(router)
    paths = set(app.openapi()["paths"])
    assert "/run-task" in paths
    assert "/desktop/run-task" in paths
    assert "/observer/api/runs" in paths


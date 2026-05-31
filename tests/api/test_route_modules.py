"""Route module registration smoke tests."""

from fastapi import APIRouter

from operon.api.routes import browser, desktop, observer, router


def test_split_route_modules_export_routers() -> None:
    assert isinstance(browser.router, APIRouter)
    assert isinstance(desktop.router, APIRouter)
    assert isinstance(observer.router, APIRouter)


def test_aggregate_router_includes_split_routes() -> None:
    paths = {route.path for route in router.routes}
    assert "/run-task" in paths
    assert "/desktop/run-task" in paths
    assert "/observer/api/runs" in paths


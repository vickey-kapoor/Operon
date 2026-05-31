"""HTTP route registration for the Operon API surface."""

from fastapi import APIRouter

from operon.api.routes.browser import router as browser_router
from operon.api.routes.desktop import router as desktop_router
from operon.api.routes.observer import router as observer_router

router = APIRouter()
router.include_router(browser_router)
router.include_router(observer_router)
router.include_router(desktop_router)


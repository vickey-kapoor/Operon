"""The built-in browser's routes (KAN-138): the app's Browser tab under ``/app/api/browser``.

Behind the API key like every other route, and 404 unless ``OPERON_BUILTIN_BROWSER`` is on.
The picture itself arrives over ``/ws/stream`` (the same live view as a run's); input from
the tab comes here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import Field

from operon.browser.builtin import builtin_browser_enabled, get_builtin_browser
from operon.models.common import StrictModel

router = APIRouter(prefix="/app/api/browser")


class AddressRequest(StrictModel):
    address: str = Field(min_length=1, max_length=2000)


class OptionalAddressRequest(StrictModel):
    address: str | None = Field(default=None, max_length=2000)


class BrowserInputRequest(StrictModel):
    input_type: str = Field(pattern=r"^(click|dblclick|text|key|scroll|hover)$")
    x_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    y_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    text: str = Field(default="", max_length=2000)
    key: str = Field(default="", max_length=40)
    delta_y: int = Field(default=300, ge=-5000, le=5000)


class SiteAnswerRequest(StrictModel):
    host: str = Field(min_length=1, max_length=255)
    answer: str = Field(pattern=r"^(once|always|no)$")


def _browser():
    if not builtin_browser_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The built-in browser is off. Set OPERON_BUILTIN_BROWSER=true to turn it on.",
        )
    return get_builtin_browser()


async def _state(browser) -> dict[str, Any]:
    return {
        "enabled": True,
        "started": browser.started,
        "tabs": await browser.tabs(),
        "pending_sites": list(browser.approvals.pending.values()),
        "approved_sites": browser.approvals.always,
    }


async def _bad_request(call) -> None:
    try:
        await call
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc.args[0])) from exc


@router.get("")
async def browser_state() -> dict[str, Any]:
    """Tabs, sites waiting for an answer and sites approved for good. Never starts the browser."""
    if not builtin_browser_enabled():
        return {"enabled": False}
    return await _state(get_builtin_browser())


@router.post("/open")
async def open_browser() -> dict[str, Any]:
    """Start the browser if needed and put its current tab in the live view."""
    browser = _browser()
    await browser.show(await browser.current_page())
    return await _state(browser)


@router.post("/navigate")
async def navigate(body: AddressRequest) -> dict[str, Any]:
    browser = _browser()
    await _bad_request(browser.navigate(body.address))
    return await _state(browser)


@router.post("/back")
async def back() -> dict[str, Any]:
    browser = _browser()
    await browser.back()
    return await _state(browser)


@router.post("/forward")
async def forward() -> dict[str, Any]:
    browser = _browser()
    await browser.forward()
    return await _state(browser)


@router.post("/reload")
async def reload() -> dict[str, Any]:
    browser = _browser()
    await browser.reload()
    return await _state(browser)


@router.post("/tabs")
async def new_tab(body: OptionalAddressRequest | None = None) -> dict[str, Any]:
    browser = _browser()
    await _bad_request(browser.open_tab(body.address if body else None))
    return await _state(browser)


@router.post("/tabs/{tab_id}/select")
async def select_tab(tab_id: int) -> dict[str, Any]:
    browser = _browser()
    await _bad_request(browser.select_tab(tab_id))
    return await _state(browser)


@router.delete("/tabs/{tab_id}")
async def close_tab(tab_id: int) -> dict[str, Any]:
    browser = _browser()
    await _bad_request(browser.close_tab(tab_id))
    return await _state(browser)


@router.post("/input")
async def user_input(body: BrowserInputRequest) -> dict[str, Any]:
    """The user's mouse and keyboard in the Browser tab, forwarded to the page in view."""
    from operon.browser.manager import get_active_manager

    browser = _browser()
    manager = get_active_manager()
    if manager is None or not manager.is_connected:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The browser is not open.")
    if browser.run_of_active() is not None:
        # The tab in view is a running task's: like any live view, input waits for Take over,
        # so a click cannot land between the agent's screenshot and its action.
        from operon.api.ws_stream import _active_run_in_takeover

        if not await _active_run_in_takeover():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A task is using this tab. Take over the run first, or open another tab.",
            )
    kwargs = body.model_dump(exclude={"input_type", "x_ratio", "y_ratio"})
    await manager.inject_input(body.x_ratio, body.y_ratio, body.input_type, **kwargs)
    return {"ok": True}


@router.post("/sites")
async def answer_site(body: SiteAnswerRequest) -> dict[str, Any]:
    """The user's answer to "Let Operon open this site?": once, always or no."""
    browser = _browser()
    browser.approvals.decide(body.host, body.answer)
    return await _state(browser)


@router.delete("/sites/{host}")
async def forget_site(host: str) -> dict[str, Any]:
    browser = _browser()
    browser.approvals.remove(host)
    return await _state(browser)


@router.post("/sign-out")
async def sign_out() -> dict[str, Any]:
    """Sign out everywhere: close the browser and delete the saved profile."""
    browser = _browser()
    await browser.sign_out_everywhere()
    return await _state(browser)

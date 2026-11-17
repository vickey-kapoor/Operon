"""KAN-90: DPI-safe click coordinates (OPERON_DPI_CHECK) and rect-based monitor matching."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from operon.executor import dpi
from operon.executor.dpi import MonitorInfo
from operon.models.policy import ActionType, AgentAction
from tests.executor.test_desktop import _make_executor

MOD = "operon.executor.desktop"

# A 125% laptop left of the primary (negative origin) plus a 100% primary and a 175% monitor.
LEFT_125 = MonitorInfo(-1920, 0, 0, 1080, 120)
PRIMARY_100 = MonitorInfo(0, 0, 1920, 1080, 96)
RIGHT_175 = MonitorInfo(1920, 0, 3840, 1080, 168)
MONITORS = [LEFT_125, PRIMARY_100, RIGHT_175]


def test_scale_pct_for_common_factors() -> None:
    assert [MonitorInfo(0, 0, 1, 1, d).scale_pct for d in (96, 120, 144, 168)] == [100, 125, 150, 175]


@pytest.mark.parametrize("awareness", [dpi.PER_MONITOR, dpi.UNKNOWN])
def test_per_monitor_is_identity(awareness: str) -> None:
    for x, y in [(-1000, 500), (500, 500), (3000, 900)]:
        assert dpi.to_click_coords(x, y, awareness=awareness, monitors=MONITORS, system_dpi=120) == (x, y)


def test_unaware_scales_around_each_monitor_origin() -> None:
    # 125% monitor, negative origin: 96/120 around left=-1920.
    assert dpi.to_click_coords(-920, 500, awareness=dpi.UNAWARE, monitors=MONITORS) == (-1120, 400)
    # 100% primary: unchanged.
    assert dpi.to_click_coords(700, 300, awareness=dpi.UNAWARE, monitors=MONITORS) == (700, 300)
    # 175% monitor: 96/168 around left=1920.
    assert dpi.to_click_coords(1920 + 1680, 840, awareness=dpi.UNAWARE, monitors=MONITORS) == (1920 + 960, 480)


def test_system_aware_uses_system_dpi_over_monitor_dpi() -> None:
    # System DPI 120 (125%): the 125% monitor needs no conversion, the 175% one does.
    assert dpi.to_click_coords(-920, 500, awareness=dpi.SYSTEM, monitors=MONITORS, system_dpi=120) == (-920, 500)
    assert dpi.to_click_coords(1920 + 1680, 840, awareness=dpi.SYSTEM, monitors=MONITORS, system_dpi=120) == (
        1920 + 1200,
        600,
    )


def test_point_on_no_monitor_is_unchanged() -> None:
    assert dpi.to_click_coords(9999, 9999, awareness=dpi.UNAWARE, monitors=MONITORS) == (9999, 9999)


@pytest.mark.parametrize(("pct", "dpi_val"), [(100, 96), (125, 120), (150, 144), (175, 168)])
def test_unaware_single_monitor_matches_scale_factor(pct: int, dpi_val: int) -> None:
    mon = MonitorInfo(0, 0, 2400, 1350, dpi_val)
    x, y = dpi.to_click_coords(1200, 600, awareness=dpi.UNAWARE, monitors=[mon])
    assert (x, y) == (round(1200 * 100 / pct), round(600 * 100 / pct))


def test_match_monitor_uses_full_rect_for_same_left_monitors() -> None:
    top = {"left": 0, "top": -1080, "width": 1920, "height": 1080}
    bottom = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    wide = {"left": 0, "top": 0, "width": 2560, "height": 1440}
    assert dpi.match_monitor([top, bottom], (0, 0, 1920, 1080)) is bottom
    assert dpi.match_monitor([top, bottom], (0, -1080, 1920, 0)) is top
    # Same left and top, different size: the old left/top match picked the wrong one.
    assert dpi.match_monitor([wide, bottom], (0, 0, 1920, 1080)) is bottom
    assert dpi.match_monitor([bottom], (5, 5, 10, 10)) is None


def _at(kind: ActionType, **coords: int) -> AgentAction:
    """Validation rejects negatives; the loop adds a negative monitor origin with model_copy, so do the same."""
    base = {k: 0 for k in coords}
    return AgentAction(action_type=kind, **base).model_copy(update=coords)


def _executor(monkeypatch, *, on: bool):
    monkeypatch.setenv("OPERON_DPI_CHECK", "on" if on else "off")
    return _make_executor()


def _patch_dpi(awareness: str):
    return (
        patch(f"{MOD}.dpi.get_awareness", return_value=awareness),
        patch(f"{MOD}.dpi.enum_monitors", return_value=MONITORS),
        patch(f"{MOD}.dpi.get_system_dpi", return_value=120),
    )


async def _click(executor, awareness: str, x: int, y: int):
    action = _at(ActionType.CLICK, x=x, y=y)
    a, b, c = _patch_dpi(awareness)
    with (
        a,
        b,
        c,
        patch.object(executor, "_coord_in_monitor_bounds", return_value=(True, [])),
        patch(f"{MOD}.pyautogui.click") as click,
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="a.png"),
        patch.object(executor, "_region_has_content", return_value=(True, 100.0)) as servo,
    ):
        result = await executor.execute(action)
    return result, click, servo


@pytest.mark.asyncio
async def test_flag_off_never_converts(monkeypatch) -> None:
    executor = _executor(monkeypatch, on=False)
    result, click, _ = await _click(executor, dpi.UNAWARE, -920, 500)
    assert result.success is True
    click.assert_called_once_with(-920, 500)


@pytest.mark.asyncio
async def test_flag_on_per_monitor_is_identity(monkeypatch) -> None:
    executor = _executor(monkeypatch, on=True)
    _, click, _ = await _click(executor, dpi.PER_MONITOR, -920, 500)
    click.assert_called_once_with(-920, 500)


@pytest.mark.asyncio
async def test_flag_on_virtualized_converts_click_but_servo_keeps_screenshot_coords(monkeypatch) -> None:
    executor = _executor(monkeypatch, on=True)
    result, click, servo = await _click(executor, dpi.UNAWARE, -920, 500)
    assert result.success is True
    click.assert_called_once_with(-1120, 400)
    servo.assert_called_once()
    assert servo.call_args.args[:2] == (-920, 500)


@pytest.mark.asyncio
async def test_blank_region_still_blocks_click_with_flag_on(monkeypatch) -> None:
    executor = _executor(monkeypatch, on=True)
    action = AgentAction(action_type=ActionType.CLICK, x=700, y=300)
    a, b, c = _patch_dpi(dpi.UNAWARE)
    with (
        a,
        b,
        c,
        patch.object(executor, "_coord_in_monitor_bounds", return_value=(True, [])),
        patch(f"{MOD}.pyautogui.click") as click,
        patch.object(executor, "_region_has_content", return_value=(False, 0.0)),
    ):
        result = await executor.execute(action)
    assert result.success is False
    click.assert_not_called()


@pytest.mark.asyncio
async def test_drag_converts_both_endpoints(monkeypatch) -> None:
    executor = _executor(monkeypatch, on=True)
    action = _at(ActionType.DRAG, x=-920, y=500, x_end=700, y_end=300)
    a, b, c = _patch_dpi(dpi.UNAWARE)
    with (
        a,
        b,
        c,
        patch(f"{MOD}.pyautogui.moveTo") as move,
        patch(f"{MOD}.pyautogui.mouseDown"),
        patch(f"{MOD}.pyautogui.mouseUp"),
        patch(f"{MOD}.asyncio.sleep", new_callable=AsyncMock),
        patch.object(executor, "_capture_after", new_callable=AsyncMock, return_value="a.png"),
    ):
        await executor.execute(action)
    assert [call.args for call in move.call_args_list] == [(-1120, 400), (700, 300)]

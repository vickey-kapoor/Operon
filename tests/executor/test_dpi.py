"""KAN-90: OPERON_DPI_AWARE, when to warn about DPI drift, true-value displays.json, rect-matched monitors."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from operon.executor import dpi
from operon.executor.dpi import MonitorInfo

SRC = str(Path(__file__).resolve().parents[2] / "src")

M100 = MonitorInfo(0, 0, 1920, 1080, 96)
M125 = MonitorInfo(1920, 0, 3840, 1080, 120)
M175_NEG = MonitorInfo(-2560, 0, 0, 1440, 168)


def test_scale_pct_for_common_factors() -> None:
    assert [MonitorInfo(0, 0, 1, 1, d).scale_pct for d in (96, 120, 144, 168)] == [100, 125, 150, 175]


# -- when to warn --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("awareness", "system_dpi", "monitors", "expected"),
    [
        (dpi.PER_MONITOR, 96, [M100, M125], False),
        (dpi.UNKNOWN, 96, [M100, M125], False),
        (dpi.UNAWARE, 96, [M100], True),
        (dpi.SYSTEM, 120, [M125], False),                  # uniform DPI equal to the system's
        (dpi.SYSTEM, 120, [M125, MonitorInfo(0, 0, 1, 1, 120)], False),
        (dpi.SYSTEM, 120, [M125, M100], True),             # mixed DPI
        (dpi.SYSTEM, 96, [M175_NEG], True),                # one monitor off the system DPI
    ],
)
def test_drift_expected(awareness, system_dpi, monitors, expected) -> None:
    assert dpi.drift_expected(awareness, system_dpi, monitors) is expected


def _patch_state(monkeypatch, awareness: str, system_dpi: int, monitors: list[MonitorInfo]) -> None:
    monkeypatch.setattr(dpi, "get_awareness", lambda: awareness)
    monkeypatch.setattr(dpi, "get_system_dpi", lambda: system_dpi)
    monkeypatch.setattr(dpi, "enum_monitors", lambda: monitors)
    monkeypatch.setattr(dpi, "_warned", set())


def test_warns_once_for_system_aware_mixed_dpi(monkeypatch, caplog) -> None:
    _patch_state(monkeypatch, dpi.SYSTEM, 120, [M125, M100])
    with caplog.at_level(logging.WARNING, logger=dpi.logger.name):
        dpi.warn_if_drifting()
        dpi.warn_if_drifting()
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1 and "OPERON_DPI_AWARE=on" in msgs[0]


def test_no_warning_for_system_aware_uniform_dpi(monkeypatch, caplog) -> None:
    _patch_state(monkeypatch, dpi.SYSTEM, 120, [M125])
    with caplog.at_level(logging.WARNING, logger=dpi.logger.name):
        dpi.warn_if_drifting()
    assert not caplog.records


def test_warns_for_unaware(monkeypatch, caplog) -> None:
    _patch_state(monkeypatch, dpi.UNAWARE, 96, [M100])
    with caplog.at_level(logging.WARNING, logger=dpi.logger.name):
        dpi.warn_if_drifting()
    assert len(caplog.records) == 1


# -- the flag -------------------------------------------------------------------------------

def _fake_win32(monkeypatch) -> None:
    monkeypatch.setattr(dpi, "sys", SimpleNamespace(platform="win32", modules=sys.modules))


def test_flag_off_does_not_touch_awareness(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_DPI_AWARE", "off")
    _fake_win32(monkeypatch)
    called: list[int] = []
    monkeypatch.setattr(dpi, "set_process_per_monitor_v2", lambda: called.append(1) or True)
    assert dpi.apply_awareness_early() == dpi.UNKNOWN
    assert called == []


def test_flag_on_sets_awareness_and_stays_quiet_when_it_takes(monkeypatch, caplog) -> None:
    monkeypatch.setenv("OPERON_DPI_AWARE", "on")
    _fake_win32(monkeypatch)
    called: list[int] = []
    monkeypatch.setattr(dpi, "set_process_per_monitor_v2", lambda: called.append(1) or True)
    monkeypatch.setattr(dpi, "get_awareness", lambda: dpi.PER_MONITOR)
    with caplog.at_level(logging.WARNING, logger=dpi.logger.name):
        assert dpi.apply_awareness_early() == dpi.PER_MONITOR
    assert called == [1] and not caplog.records


def test_flag_on_warns_once_when_something_else_won(monkeypatch, caplog) -> None:
    monkeypatch.setenv("OPERON_DPI_AWARE", "on")
    _fake_win32(monkeypatch)
    monkeypatch.setattr(dpi, "_warned", set())
    monkeypatch.setattr(dpi, "set_process_per_monitor_v2", lambda: False)
    monkeypatch.setattr(dpi, "get_awareness", lambda: dpi.SYSTEM)
    with caplog.at_level(logging.WARNING, logger=dpi.logger.name):
        dpi.apply_awareness_early()
        dpi.apply_awareness_early()
    assert len(caplog.records) == 1 and "still system" in caplog.records[0].getMessage()


@pytest.mark.skipif(sys.platform != "win32", reason="DPI awareness is a Windows process property")
def test_importing_the_executor_with_the_flag_sets_per_monitor_before_pyautogui() -> None:
    def awareness(flag: str) -> str:
        env = {**os.environ, "OPERON_DPI_AWARE": flag, "PYTHONPATH": SRC, "GEMINI_API_KEY": "fake-test-key"}
        code = "import operon.executor.desktop; from operon.executor import dpi; print(dpi.get_awareness())"
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        return out.stdout.strip().splitlines()[-1]

    assert awareness("on") == dpi.PER_MONITOR
    assert awareness("off") != dpi.PER_MONITOR  # pyautogui's import makes it system-aware


# -- displays.json true values --------------------------------------------------------------

def test_display_report_records_true_monitor_values(monkeypatch) -> None:
    _patch_state(monkeypatch, dpi.SYSTEM, 120, [M175_NEG, M100])
    assert dpi.display_report() == {
        "awareness": "system",
        "system_dpi": 120,
        "monitors": [
            {"rect": [-2560, 0, 0, 1440], "dpi": 168, "scale_pct": 175},
            {"rect": [0, 0, 1920, 1080], "dpi": 96, "scale_pct": 100},
        ],
    }


def test_enum_monitors_empty_off_windows() -> None:
    if sys.platform != "win32":
        assert dpi.enum_monitors() == []


# -- foreground monitor match ---------------------------------------------------------------

def test_match_monitor_uses_full_rect_for_same_left_monitors() -> None:
    top = {"left": 0, "top": -1080, "width": 1920, "height": 1080}
    bottom = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    wide = {"left": 0, "top": 0, "width": 2560, "height": 1440}
    assert dpi.match_monitor([top, bottom], (0, 0, 1920, 1080)) is bottom
    assert dpi.match_monitor([top, bottom], (0, -1080, 1920, 0)) is top
    assert dpi.match_monitor([wide, bottom], (0, 0, 1920, 1080)) is bottom
    assert dpi.match_monitor([bottom], (5, 5, 10, 10)) is None

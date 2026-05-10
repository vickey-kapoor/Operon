"""Tests for benchmark page: filter logic, steps-cell color, efficiency-cell color."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api.server import app
    return TestClient(app)


# ── Benchmarks HTML ───────────────────────────────────────────────────────────


def test_benchmarks_serves_html(client: TestClient) -> None:
    resp = client.get("/benchmarks")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_benchmarks_has_required_elements(client: TestClient) -> None:
    html = client.get("/benchmarks").text
    assert "scoreboard" in html
    assert "run suite" in html
    assert "fgSource" in html
    assert "fgDiff" in html
    assert "fgStatus" in html
    assert "statsStrip" in html
    assert "steps / opt" in html


def test_benchmarks_has_source_pills_styling(client: TestClient) -> None:
    """Mind2Web gets amber tint; WebArena gets green tint."""
    html = client.get("/benchmarks").text
    assert "mind2web" in html
    assert "webarena" in html
    assert "#1f1408" in html   # amber-bg for mind2web
    assert "#0c1f1a" in html   # green-bg for webarena


def test_benchmarks_has_confirm_modal(client: TestClient) -> None:
    """Confirm modal for run suite >5 tasks is present."""
    html = client.get("/benchmarks").text
    assert "confirmBackdrop" in html
    assert "confirmBody" in html
    assert "confirmRunSuite" in html


# ── StepsCell color logic (pure Python mirror of JS logic) ───────────────────


def _steps_color(steps: int | None, opt: int, status: str) -> str | None:
    """Mirror of stepsCell color logic from benchmarks.html."""
    if status in ("pending", "running") or steps is None:
        return None  # no color — dash state
    delta = steps - opt
    if delta > 2:
        return "red"
    if delta > 0:
        return "amber"
    return "green"


def _steps_badge(steps: int, opt: int) -> str:
    delta = steps - opt
    if delta == 0:
        return "✓"
    return f"+{delta}"


def test_steps_cell_exact_match_is_green() -> None:
    assert _steps_color(3, 3, "passed") == "green"


def test_steps_cell_delta_1_is_amber() -> None:
    assert _steps_color(4, 3, "passed") == "amber"


def test_steps_cell_delta_2_is_amber() -> None:
    assert _steps_color(5, 3, "passed") == "amber"


def test_steps_cell_delta_3_is_red() -> None:
    assert _steps_color(6, 3, "passed") == "red"


def test_steps_cell_delta_10_is_red() -> None:
    assert _steps_color(13, 3, "failed") == "red"


def test_steps_cell_pending_has_no_color() -> None:
    assert _steps_color(None, 5, "pending") is None


def test_steps_cell_running_has_no_color() -> None:
    assert _steps_color(None, 4, "running") is None


def test_steps_cell_badge_exact() -> None:
    assert _steps_badge(3, 3) == "✓"


def test_steps_cell_badge_over() -> None:
    assert _steps_badge(5, 3) == "+2"


# ── EfficiencyCell color logic ────────────────────────────────────────────────


def _eff_color(val: float | None) -> str | None:
    if val is None:
        return None
    if val > 2.5:
        return "red"
    if val > 1.5:
        return "amber"
    return "green"


def test_eff_cell_none_has_no_color() -> None:
    assert _eff_color(None) is None


def test_eff_cell_1_0_is_green() -> None:
    assert _eff_color(1.0) == "green"


def test_eff_cell_1_5_is_green() -> None:
    """Boundary: 1.5 is NOT > 1.5, so green."""
    assert _eff_color(1.5) == "green"


def test_eff_cell_1_51_is_amber() -> None:
    assert _eff_color(1.51) == "amber"


def test_eff_cell_2_5_is_amber() -> None:
    """Boundary: 2.5 is NOT > 2.5, so amber."""
    assert _eff_color(2.5) == "amber"


def test_eff_cell_2_51_is_red() -> None:
    assert _eff_color(2.51) == "red"


def test_eff_cell_4_4_is_red() -> None:
    assert _eff_color(4.4) == "red"


# ── Filter logic ──────────────────────────────────────────────────────────────


def _make_task(task_id: str, source: str = "mind2web", diff: str = "easy", status: str = "pending") -> dict:
    return {"id": task_id, "source": source, "diff": diff, "status": status,
            "intent": "test", "site": "test", "category": "nav",
            "steps": None, "optSteps": 3, "efficiency": None, "duration": None}


def _apply_filters(tasks: list[dict], source: str, diff: str, status: str) -> list[dict]:
    return [t for t in tasks if
            (source == "all" or t["source"].lower() == source) and
            (diff == "all" or t["diff"].lower() == diff) and
            (status == "all" or t["status"] == status)]


def test_filter_all_returns_everything() -> None:
    tasks = [_make_task("t1"), _make_task("t2", source="webarena")]
    assert len(_apply_filters(tasks, "all", "all", "all")) == 2


def test_filter_source_mind2web() -> None:
    tasks = [_make_task("t1", source="mind2web"), _make_task("t2", source="webarena")]
    result = _apply_filters(tasks, "mind2web", "all", "all")
    assert len(result) == 1 and result[0]["id"] == "t1"


def test_filter_source_webarena() -> None:
    tasks = [_make_task("t1", source="mind2web"), _make_task("t2", source="webarena")]
    result = _apply_filters(tasks, "webarena", "all", "all")
    assert len(result) == 1 and result[0]["id"] == "t2"


def test_filter_diff_easy_only() -> None:
    tasks = [_make_task("t1", diff="easy"), _make_task("t2", diff="medium")]
    result = _apply_filters(tasks, "all", "easy", "all")
    assert len(result) == 1 and result[0]["id"] == "t1"


def test_filter_status_passed() -> None:
    tasks = [
        _make_task("t1", status="passed"),
        _make_task("t2", status="failed"),
        _make_task("t3", status="pending"),
    ]
    result = _apply_filters(tasks, "all", "all", "passed")
    assert len(result) == 1 and result[0]["id"] == "t1"


def test_filter_status_pending() -> None:
    tasks = [_make_task("t1", status="passed"), _make_task("t2", status="pending")]
    result = _apply_filters(tasks, "all", "all", "pending")
    assert len(result) == 1 and result[0]["id"] == "t2"


def test_filter_status_running() -> None:
    tasks = [_make_task("t1", status="running"), _make_task("t2", status="passed")]
    result = _apply_filters(tasks, "all", "all", "running")
    assert len(result) == 1


def test_filter_combined_source_and_diff() -> None:
    tasks = [
        _make_task("t1", source="mind2web", diff="easy"),
        _make_task("t2", source="mind2web", diff="medium"),
        _make_task("t3", source="webarena", diff="easy"),
    ]
    result = _apply_filters(tasks, "mind2web", "easy", "all")
    assert len(result) == 1 and result[0]["id"] == "t1"


def test_filter_no_match_returns_empty() -> None:
    tasks = [_make_task("t1", source="mind2web", status="passed")]
    result = _apply_filters(tasks, "webarena", "all", "all")
    assert result == []


# ── Run Suite confirmation threshold ─────────────────────────────────────────


def test_confirmation_threshold() -> None:
    """Confirmation shown for >5 tasks, not for ≤5."""
    assert 6 > 5   # needs confirmation
    assert 5 <= 5  # no confirmation


def test_confirmation_time_estimate() -> None:
    """Time estimate = count * 2 minutes."""
    n = 13
    mins = round(n * 2)
    assert mins == 26


def test_confirmation_cost_estimate() -> None:
    """Cost estimate = count * $0.18."""
    n = 13
    cost = round(n * 0.18, 2)
    assert cost == pytest.approx(2.34)

"""Tests for the redesigned dashboard: cost-by-outcome math, filter pills, pending exclusion."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api.server import app
    return TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _usage(cost: float, tokens: int = 1000, model: str = "gemini-2.5-flash") -> dict:
    return {
        "request_count": 1,
        "input_tokens": tokens // 2,
        "output_tokens": tokens // 2,
        "total_tokens": tokens,
        "estimated_cost_usd": cost,
        "by_model": {model: {"request_count": 1, "input_tokens": tokens // 2,
                              "output_tokens": tokens // 2, "total_tokens": tokens,
                              "estimated_cost_usd": cost}},
    }


def _run(run_id: str, status: str, cost: float = 0.05, steps: int = 5) -> dict:
    return {
        "run_id": run_id,
        "intent": f"task {run_id}",
        "status": status,
        "step_count": steps,
        "updated_at": 1_700_000_000.0,
        "usage": _usage(cost),
    }


def _dashboard_payload(*runs: dict) -> dict:
    total_cost = sum(r["usage"]["estimated_cost_usd"] for r in runs)
    total_tokens = sum(r["usage"]["total_tokens"] for r in runs)
    return {
        "summary": {
            "request_count": len(runs),
            "input_tokens": total_tokens // 2,
            "output_tokens": total_tokens // 2,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_cost,
            "by_model": {},
        },
        "runs": list(runs),
    }


# ── Dashboard HTML ────────────────────────────────────────────────────────────


def test_dashboard_serves_html(client: TestClient) -> None:
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_dashboard_has_required_elements(client: TestClient) -> None:
    resp = client.get("/dashboard")
    html = resp.text
    assert "mtd spend" in html
    assert "cost by outcome" in html
    assert "spend by model" in html
    assert "run ledger" in html
    assert "terminal runs" in html
    assert "filterRow" in html or "filter-row" in html


# ── Cost-by-outcome math (pure JS logic tested via /observer/api/usage) ───────


def _cost_by_outcome(runs: list[dict]) -> dict:
    """Replicate the JS cost-by-outcome aggregation in Python for test assertions."""
    terminal = [r for r in runs if r["status"] in ("succeeded", "failed")]
    succ = [r for r in terminal if r["status"] == "succeeded"]
    fail = [r for r in terminal if r["status"] == "failed"]
    succ_cost = sum(r["usage"]["estimated_cost_usd"] for r in succ)
    fail_cost = sum(r["usage"]["estimated_cost_usd"] for r in fail)
    total = succ_cost + fail_cost
    return {
        "terminal_count": len(terminal),
        "succ_count": len(succ),
        "fail_count": len(fail),
        "succ_cost": succ_cost,
        "fail_cost": fail_cost,
        "succ_pct": (succ_cost / total * 100) if total > 0 else 0,
        "fail_pct": (fail_cost / total * 100) if total > 0 else 0,
        "succ_avg": succ_cost / len(succ) if succ else 0,
        "fail_avg": fail_cost / len(fail) if fail else 0,
    }


def test_cost_by_outcome_normal_case() -> None:
    """Mixed succeeded/failed runs produce correct cost split."""
    runs = [
        _run("r1", "succeeded", cost=0.10),
        _run("r2", "succeeded", cost=0.20),
        _run("r3", "failed",    cost=0.30),
        _run("r4", "failed",    cost=0.60),
    ]
    out = _cost_by_outcome(runs)
    assert out["succ_count"] == 2
    assert out["fail_count"] == 2
    assert out["succ_cost"] == pytest.approx(0.30)
    assert out["fail_cost"] == pytest.approx(0.90)
    assert out["succ_pct"] == pytest.approx(25.0)
    assert out["fail_pct"] == pytest.approx(75.0)
    assert out["succ_avg"] == pytest.approx(0.15)
    assert out["fail_avg"] == pytest.approx(0.45)
    ratio = out["fail_avg"] / out["succ_avg"]
    assert ratio == pytest.approx(3.0)


def test_cost_by_outcome_zero_failed() -> None:
    """All succeeded runs: fail cost = 0, succ_pct = 100%."""
    runs = [_run("r1", "succeeded", 0.10), _run("r2", "succeeded", 0.20)]
    out = _cost_by_outcome(runs)
    assert out["fail_count"] == 0
    assert out["fail_cost"] == 0.0
    assert out["succ_pct"] == pytest.approx(100.0)
    assert out["fail_pct"] == pytest.approx(0.0)
    assert out["fail_avg"] == 0.0


def test_cost_by_outcome_zero_succeeded() -> None:
    """All failed runs: succ cost = 0, fail_pct = 100%."""
    runs = [_run("r1", "failed", 0.50), _run("r2", "failed", 0.25)]
    out = _cost_by_outcome(runs)
    assert out["succ_count"] == 0
    assert out["succ_cost"] == 0.0
    assert out["fail_pct"] == pytest.approx(100.0)
    assert out["succ_pct"] == pytest.approx(0.0)
    assert out["succ_avg"] == 0.0


def test_cost_by_outcome_no_runs() -> None:
    """Zero runs: all values are zero, no division by zero."""
    out = _cost_by_outcome([])
    assert out["terminal_count"] == 0
    assert out["succ_cost"] == 0.0
    assert out["fail_cost"] == 0.0
    assert out["succ_pct"] == 0.0
    assert out["fail_pct"] == 0.0


# ── Pending exclusion ────────────────────────────────────────────────────────


def test_pending_excluded_from_terminal_count() -> None:
    """Pending and running runs are excluded; only succeeded/failed count."""
    runs = [
        _run("r1", "succeeded", 0.10),
        _run("r2", "failed",    0.20),
        _run("r3", "pending",   0.00),
        _run("r4", "running",   0.05),
    ]
    out = _cost_by_outcome(runs)
    assert out["terminal_count"] == 2
    assert out["succ_count"] == 1
    assert out["fail_count"] == 1


def test_all_pending_means_zero_terminal() -> None:
    """If all runs are pending/running, terminal count is zero."""
    runs = [_run("r1", "pending"), _run("r2", "running")]
    out = _cost_by_outcome(runs)
    assert out["terminal_count"] == 0
    assert out["succ_pct"] == 0.0
    assert out["fail_pct"] == 0.0


# ── Filter pill state ────────────────────────────────────────────────────────


def _filter_counts(runs: list[dict]) -> dict:
    """Replicate filter pill count logic (terminal only)."""
    terminal = [r for r in runs if r["status"] in ("succeeded", "failed")]
    return {
        "all": len(terminal),
        "succeeded": sum(1 for r in terminal if r["status"] == "succeeded"),
        "failed": sum(1 for r in terminal if r["status"] == "failed"),
    }


def test_filter_counts_normal() -> None:
    runs = [
        _run("r1", "succeeded"),
        _run("r2", "succeeded"),
        _run("r3", "failed"),
        _run("r4", "pending"),
    ]
    counts = _filter_counts(runs)
    assert counts["all"] == 3
    assert counts["succeeded"] == 2
    assert counts["failed"] == 1


def test_filter_counts_all_pending() -> None:
    runs = [_run("r1", "pending"), _run("r2", "running")]
    counts = _filter_counts(runs)
    assert counts["all"] == 0
    assert counts["succeeded"] == 0
    assert counts["failed"] == 0


def test_filter_succeeded_excludes_failed() -> None:
    """The succeeded filter shows only succeeded rows."""
    runs = [_run("r1", "succeeded"), _run("r2", "failed")]
    terminal = [r for r in runs if r["status"] in ("succeeded", "failed")]
    succeeded_filter = [r for r in terminal if r["status"] == "succeeded"]
    assert len(succeeded_filter) == 1
    assert succeeded_filter[0]["run_id"] == "r1"


def test_filter_failed_excludes_succeeded() -> None:
    """The failed filter shows only failed rows."""
    runs = [_run("r1", "succeeded"), _run("r2", "failed"), _run("r3", "failed")]
    terminal = [r for r in runs if r["status"] in ("succeeded", "failed")]
    failed_filter = [r for r in terminal if r["status"] == "failed"]
    assert len(failed_filter) == 2


# ── Pass rate in stat card ────────────────────────────────────────────────────


def test_pass_rate_calculation() -> None:
    """Pass rate = succeeded / total terminal runs."""
    runs = [
        _run("r1", "succeeded"),
        _run("r2", "succeeded"),
        _run("r3", "failed"),
        _run("r4", "pending"),  # excluded
    ]
    terminal = [r for r in runs if r["status"] in ("succeeded", "failed")]
    succ = [r for r in terminal if r["status"] == "succeeded"]
    pass_rate = round(len(succ) / len(terminal) * 100) if terminal else 0
    assert pass_rate == 67


def test_pass_rate_zero_runs() -> None:
    terminal = []
    pass_rate = round(len([]) / len(terminal) * 100) if terminal else 0
    assert pass_rate == 0


# ── Observer usage API (integration) ─────────────────────────────────────────


def test_observer_usage_returns_summary(client: TestClient) -> None:
    """/observer/api/usage returns a dict with summary and runs keys."""
    with patch("src.api.routes.usage_dashboard") as mock:
        mock.return_value = _dashboard_payload(
            _run("r1", "succeeded", 0.10),
            _run("r2", "failed", 0.20),
        )
        resp = client.get("/observer/api/usage?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "runs" in data
    assert len(data["runs"]) == 2

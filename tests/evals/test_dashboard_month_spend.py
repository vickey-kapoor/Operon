"""The Cost tab's month and budget math (dashboard/app.js), run in Node.

The page is plain browser JavaScript with no build step, so these tests take the
functions out of app.js by name and run them with Node; they skip without Node."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "dashboard" / "app.js"
FUNCTIONS = ("monthKey", "monthStart", "entrySpend", "monthTotals", "monthBurn")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def _function_source(source: str, name: str) -> str:
    start = re.search(rf"^function {name}\(", source, re.M)
    assert start, f"{name} not found in app.js"
    depth, i = 0, source.index("{", start.start())
    while True:
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        i += 1
        if depth == 0:
            return source[start.start():i]


def _run(expression: str):
    source = APP_JS.read_text(encoding="utf-8")
    script = "\n".join(_function_source(source, name) for name in FUNCTIONS)
    script += "\nconst state = { budget: null };\n"
    script += "const e = (at, agent, judge = 0, gpu = 0, run = {}) => ({ at: new Date(at), agent, judge, gpu, run });\n"
    script += f"process.stdout.write(JSON.stringify({expression}));"
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_month_totals_use_utc_months_and_keep_empty_months() -> None:
    months = _run("""monthTotals([
        e("2026-09-30T23:30:00Z", 1),
        e("2026-11-01T00:30:00Z", 2, 0.5, 0.25),
        e("2026-11-05T12:00:00Z", 0, 0.1, 0, null),
    ], new Date("2026-11-08T12:00:00Z"))""")
    assert [m["key"] for m in months] == ["2026-09", "2026-10", "2026-11"]
    sep, octo, nov = months
    assert sep["total"] == 1 and sep["runs"] == 1
    assert octo["total"] == 0 and octo["runs"] == 0  # no spend, still listed
    assert nov["agent"] == 2 and nov["gpu"] == 0.25 and nov["runs"] == 1 and nov["checks"] == 1
    assert nov["total"] == pytest.approx(2.85)


def test_month_totals_are_empty_without_spend() -> None:
    assert _run("monthTotals([], new Date('2026-11-08T00:00:00Z'))") == []


def test_month_burn_counts_this_utc_month_against_the_budget() -> None:
    burn = _run("""monthBurn([
        e("2026-10-31T23:59:00Z", 50),
        e("2026-11-01T00:01:00Z", 3, 1, 1),
        e("2026-11-03T00:00:00Z", 0, 5, 0, null),
    ], new Date("2026-11-11T00:00:00Z"), 100)""")
    assert burn["key"] == "2026-11"
    assert burn["spent"] == 10  # October's run is left out; judge check and GPU count
    assert burn["used"] == pytest.approx(0.10)
    assert burn["left"] == 90
    assert burn["perDay"] == pytest.approx(1.0)  # 10 days into November
    assert burn["projected"] == pytest.approx(30.0)
    assert burn["allowedPerDay"] == pytest.approx(90 / 20)


def test_month_burn_without_a_budget_has_no_budget_figures() -> None:
    burn = _run("monthBurn([e('2026-11-02T00:00:00Z', 4)], new Date('2026-11-03T00:00:00Z'), 0)")
    assert burn["spent"] == 4 and burn["budget"] is None and burn["used"] is None and burn["left"] is None

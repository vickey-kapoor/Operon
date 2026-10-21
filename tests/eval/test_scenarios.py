"""Run every offline eval scenario through the real agent loop and grade it."""

from __future__ import annotations

import pytest

from tests.eval.harness import grade, run_scenario
from tests.eval.scenarios import SCENARIOS, cu_click_then_done


def _params():
    for factory in SCENARIOS:
        scenario = factory()
        marks = [pytest.mark.xfail(strict=True, reason=scenario.known_issue)] if scenario.known_issue else []
        yield pytest.param(factory, id=scenario.name, marks=marks)


@pytest.mark.parametrize("factory", list(_params()))
async def test_scenario_outcome(factory) -> None:
    scenario = factory()

    result = await run_scenario(scenario)

    assert scenario.accepts(result.outcome), result.summary()


def test_scenario_names_are_unique() -> None:
    names = [factory().name for factory in SCENARIOS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("claimed", "reached", "ended", "outcome"),
    [
        (True, True, True, "success"),
        (True, False, True, "false_success"),
        (False, True, True, "missed"),
        (False, False, True, "failure"),
        (False, False, False, "incomplete"),
    ],
)
def test_grade_compares_the_claim_with_ground_truth(claimed, reached, ended, outcome) -> None:
    assert grade(claimed_success=claimed, reached_goal=reached, ended=ended) == outcome


async def test_computer_use_click_costs_one_model_turn_per_step() -> None:
    """The Computer Use backend reports no element list, which must not read as a
    page still loading: that re-perceives (a full model turn each) mid-verify."""
    result = await run_scenario(cu_click_then_done())

    assert result.outcome == "success", result.summary()
    assert result.model_calls == result.steps, result.summary()

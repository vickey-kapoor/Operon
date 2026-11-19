"""The task text limit (MAX_INTENT_CHARS): long real tasks run, absurd input is still refused.

An OSWorld task is 736 characters and the old 500-character cap refused it before any run
existed, so the benchmark scored it 0 for $0 (KAN-169)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.osworld import load_subset
from operon.api.frontdoor import FrontDoorTask
from operon.models.common import MAX_INTENT_CHARS, RunTaskRequest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("model", [RunTaskRequest, FrontDoorTask])
def test_a_task_of_the_longest_allowed_length_is_accepted(model) -> None:
    assert model(intent="a" * MAX_INTENT_CHARS).intent == "a" * MAX_INTENT_CHARS


@pytest.mark.parametrize("model", [RunTaskRequest, FrontDoorTask])
def test_a_task_one_character_over_the_limit_is_refused(model) -> None:
    with pytest.raises(ValidationError):
        model(intent="a" * (MAX_INTENT_CHARS + 1))


@pytest.mark.parametrize("model", [RunTaskRequest, FrontDoorTask])
def test_a_736_character_task_is_accepted(model) -> None:
    # The length of OSWorld's multi_apps 2c1ebcd7, which the old 500-character cap refused.
    assert len(model(intent="b" * 736).intent) == 736


def test_every_osworld_subset_task_fits_the_limit() -> None:
    too_long = [(t["id"][:8], len(t["instruction"])) for t in load_subset() if len(t["instruction"]) > MAX_INTENT_CHARS]
    assert too_long == []

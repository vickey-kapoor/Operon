"""Malformed combined replies and small hooks of CombinedPerceptionPolicyService."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from operon.agent.backends.combined import CombinedPerceptionPolicyService
from operon.agent.perception import PerceptionError
from operon.agent.policy import PolicyError

PERCEPTION = {"summary": "A page", "page_hint": "unknown", "confidence": 0.9, "visible_elements": []}
parse = CombinedPerceptionPolicyService._parse_combined_output


@pytest.mark.parametrize(
    ("raw", "error", "match"),
    [
        ("not json", PerceptionError, "not valid JSON"),
        ("[]", PerceptionError, "JSON object"),
        (json.dumps({"action": {"action_type": "wait"}}), PerceptionError, "missing 'perception'"),
        (json.dumps({"perception": PERCEPTION}), PolicyError, "missing 'action'"),
        (json.dumps({"perception": PERCEPTION, "action": "click"}), PolicyError, "missing 'action'"),
    ],
)
def test_malformed_combined_output_is_rejected(raw: str, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        parse(raw, "before.png")


def test_missing_decision_fields_get_defaults() -> None:
    raw = json.dumps({"perception": PERCEPTION, "action": {"action_type": "wait", "wait_ms": 500}})

    _, decision = parse(raw, "before.png")

    assert decision.rationale == "combined call"
    assert decision.confidence == 0.5
    assert decision.active_subgoal == "combined step"


def _service(tmp_path: Path) -> CombinedPerceptionPolicyService:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{intent}")
    return CombinedPerceptionPolicyService(MagicMock(), prompt_path=prompt)


def test_hints_are_scoped_per_run_and_cleared_per_run(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.add_advisory_hints(["a", ""], source="rule", run_id="r1")
    service.add_advisory_hints(["b"], run_id="r2")

    service.clear_advisory_hints("r1")

    assert service._advisory_hints == {"r2": [("b", "")]}


def test_perception_only_flag_and_debug_artifacts(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.latest_debug_artifacts() is None
    service.set_perception_only(True)
    assert service._perception_only is True

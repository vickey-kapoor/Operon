"""Form benchmark plugin — registers all form-specific logic with the engine registry."""

from __future__ import annotations

from src.benchmarks.registry import BENCHMARK_REGISTRY, BenchmarkPlugin
from src.models.common import LoopStage
from src.models.memory import MemoryOutcome, MemoryRecord
from src.models.perception import PageHint

FORM_BENCHMARK = "auth_free_form"

_seeds = [
    MemoryRecord(
        key="click_before_type",
        benchmark=FORM_BENCHMARK,
        hint="When input focus is uncertain, click the input before typing.",
        outcome=MemoryOutcome.GUARDRAIL,
        stage=LoopStage.CHOOSE_ACTION,
        success=False,
    ),
    MemoryRecord(
        key="avoid_identical_type_retry",
        benchmark=FORM_BENCHMARK,
        hint="Do not repeat the same type action after a focus or target failure; re-establish focus first.",
        outcome=MemoryOutcome.GUARDRAIL,
        stage=LoopStage.CHOOSE_ACTION,
        success=False,
    ),
]


def _get_rules():
    from src.agent.policy_rules import form_submit_when_ready_rule
    return [form_submit_when_ready_rule]


BENCHMARK_REGISTRY.register(
    BenchmarkPlugin(
        name=FORM_BENCHMARK,
        rules=_get_rules(),
        memory_seeds=_seeds,
        success_tokens=("thank you", "submitted successfully", "submission successful", "submission complete"),
        section_map={
            PageHint.FORM_PAGE: "form",
            PageHint.FORM_SUCCESS: "form",
        },
        default_url="https://practice-automation.com/form-fields/",
        task_type="form_submit",
        expected_completion_signal="form success page visible",
    )
)

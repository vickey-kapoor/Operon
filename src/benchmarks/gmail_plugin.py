"""Gmail benchmark plugin — registers all Gmail-specific logic with the engine registry."""

from __future__ import annotations

from src.benchmarks.registry import BENCHMARK_REGISTRY, BenchmarkPlugin
from src.models.common import LoopStage
from src.models.memory import MemoryOutcome, MemoryRecord
from src.models.perception import PageHint

GMAIL_BENCHMARK = "gmail_draft_authenticated"

_seeds = [
    MemoryRecord(
        key="click_before_type",
        benchmark=GMAIL_BENCHMARK,
        hint="When input focus is uncertain, click the input before typing.",
        outcome=MemoryOutcome.GUARDRAIL,
        stage=LoopStage.CHOOSE_ACTION,
        success=False,
    ),
    MemoryRecord(
        key="avoid_identical_type_retry",
        benchmark=GMAIL_BENCHMARK,
        hint="Do not repeat the same type action after a focus or target failure; re-establish focus first.",
        outcome=MemoryOutcome.GUARDRAIL,
        stage=LoopStage.CHOOSE_ACTION,
        success=False,
    ),
    MemoryRecord(
        key="authenticated_start_required",
        benchmark=GMAIL_BENCHMARK,
        hint="Login pages are out of scope for this benchmark; use an authenticated Gmail start state.",
        outcome=MemoryOutcome.GUARDRAIL,
        page_hint=PageHint("google_sign_in"),
        stage=LoopStage.CHOOSE_ACTION,
        success=False,
    ),
]


def _get_rules():
    from src.agent.policy_rules import gmail_login_page_guardrail, gmail_compose_already_visible_rule
    return [gmail_login_page_guardrail, gmail_compose_already_visible_rule]


BENCHMARK_REGISTRY.register(
    BenchmarkPlugin(
        name=GMAIL_BENCHMARK,
        rules=_get_rules(),
        memory_seeds=_seeds,
        success_tokens=("draft saved", "draft created"),
        section_map={
            PageHint("gmail_compose"): "compose",
            PageHint("gmail_inbox"): "compose",
            PageHint("gmail_message_view"): "compose",
        },
        default_url="https://mail.google.com/",
        task_type="multi_step_form",
        expected_completion_signal="draft created or intentional stop",
    )
)

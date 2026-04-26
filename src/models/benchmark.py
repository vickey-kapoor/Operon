"""Typed schemas for benchmark suites, task specs, and derived metrics."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from src.models.common import FailureCategory, RunStatus, StopReason, StrictModel


class BenchmarkTaskType(StrEnum):
    """Supported benchmark task families for Operon v1 evaluation."""

    GENERIC = "generic"
    FORM_FILL = "form_fill"
    FORM_SUBMIT = "form_submit"
    MULTI_STEP_FORM = "multi_step_form"


class BenchmarkTaskSpec(StrictModel):
    """One benchmark task definition for the existing loop runner."""

    task_id: str = Field(min_length=1)
    page_url: str = Field(min_length=1)
    task_type: BenchmarkTaskType
    intent: str = Field(min_length=1)
    expected_completion_signal: str = Field(min_length=1)
    difficulty_tags: list[str] = Field(default_factory=list)


class BenchmarkSuiteSpec(StrictModel):
    """Collection of benchmark tasks to run sequentially."""

    suite_id: str = Field(min_length=1)
    tasks: list[BenchmarkTaskSpec] = Field(min_length=1)


class RunMetrics(StrictModel):
    """Structured per-run reliability metrics derived from stored artifacts."""

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    page_url: str = Field(min_length=1)
    task_type: BenchmarkTaskType
    tags: list[str] = Field(default_factory=list)
    status: RunStatus
    success: bool
    final_stop_reason: StopReason | None = None
    failure_category: FailureCategory | None = None
    step_count: int = Field(ge=0)
    perception_retry_count: int = Field(default=0, ge=0)
    selector_recovery_count: int = Field(default=0, ge=0)
    execution_retry_count: int = Field(default=0, ge=0)
    no_progress_events: int = Field(default=0, ge=0)
    loop_detected: bool = False
    average_top_selector_score: float | None = None
    average_selector_margin: float | None = None
    selector_failure_count: int = Field(default=0, ge=0)
    average_total_elements: float | None = None
    average_labeled_elements: float | None = None
    average_unlabeled_elements: float | None = None
    average_usable_elements: float | None = None
    stale_target_events: int = Field(default=0, ge=0)
    focus_failures: int = Field(default=0, ge=0)
    click_no_effect_events: int = Field(default=0, ge=0)
    verification_failures: int = Field(default=0, ge=0)


class BenchmarkSuiteSummary(StrictModel):
    """Aggregate reliability summary across a benchmark suite."""

    suite_id: str = Field(min_length=1)
    total_runs: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    overall_success_rate: float = Field(ge=0.0)
    success_rate_by_task_type: dict[str, float] = Field(default_factory=dict)
    failure_breakdown_by_stop_reason: dict[str, int] = Field(default_factory=dict)
    failure_breakdown_by_failure_category: dict[str, int] = Field(default_factory=dict)
    average_step_count: float = Field(ge=0.0)
    average_perception_retry_count: float = Field(ge=0.0)
    average_selector_recovery_count: float = Field(ge=0.0)
    average_execution_retry_count: float = Field(ge=0.0)
    average_top_selector_score: float | None = None
    average_selector_margin: float | None = None
    average_total_elements: float | None = None
    average_labeled_elements: float | None = None
    average_unlabeled_elements: float | None = None
    average_usable_elements: float | None = None
    loop_detected_frequency: float = Field(ge=0.0)
    no_progress_event_frequency: float = Field(ge=0.0)
    top_recurring_failure_reasons: dict[str, int] = Field(default_factory=dict)
    tag_summary: dict[str, dict[str, float | int]] = Field(default_factory=dict)

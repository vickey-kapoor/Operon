"""Models for post-run reflection and pattern extraction."""

from __future__ import annotations

from pydantic import Field

from src.models.common import StrictModel


class ReflectionPattern(StrictModel):
    """One failure pattern extracted from a completed run."""

    pattern_key: str = Field(min_length=1)
    description: str = Field(min_length=1)
    trigger_context: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_run_id: str = Field(min_length=1)
    source_steps: list[int] = Field(default_factory=list)


class RunReflection(StrictModel):
    """Complete reflection output for one run."""

    run_id: str = Field(min_length=1)
    success: bool
    total_steps: int = Field(ge=0)
    patterns: list[ReflectionPattern] = Field(default_factory=list)
    memories_generated: int = Field(default=0, ge=0)

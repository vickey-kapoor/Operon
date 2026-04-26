"""Pydantic models for the WebArena benchmark task schema and results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WebArenaEval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_types: list[str]
    reference_answers: dict[str, str | list[str]] = Field(default_factory=dict)


class WebArenaTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    site: str
    category: str
    require_login: bool = False
    start_url: str
    intent: str
    eval: WebArenaEval
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)


class WebArenaTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    run_id: str
    passed: bool
    eval_types_result: dict[str, bool] = Field(default_factory=dict)
    extracted_url: str | None = None
    extracted_text_excerpt: str | None = None
    duration_seconds: float
    stop_reason: str | None = None
    site: str = ""
    category: str = ""
    difficulty: str = ""


class WebArenaSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    passed: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    by_category: dict[str, dict[str, int]] = Field(default_factory=dict)
    by_difficulty: dict[str, dict[str, int]] = Field(default_factory=dict)
    by_site: dict[str, dict[str, int]] = Field(default_factory=dict)
    tasks: list[WebArenaTaskResult] = Field(default_factory=list)

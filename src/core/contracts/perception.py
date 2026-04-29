"""Strict perception contract for the shared browser and desktop core."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ContractVersion(StrEnum):
    PHASE1 = "phase1"


class Environment(StrEnum):
    BROWSER = "browser"
    DESKTOP = "desktop"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VisibleTarget(ContractModel):
    target_id: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=50)
    label: str | None = Field(default=None, min_length=1, max_length=200)
    text: str | None = Field(default=None, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class PerceptionOutput(ContractModel):
    contract_version: ContractVersion = ContractVersion.PHASE1
    environment: Environment
    observation_id: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=1000)
    context_label: str = Field(min_length=1, max_length=200)
    active_app: str | None = Field(default=None, min_length=1, max_length=200)
    current_url: str | None = Field(default=None, min_length=1, max_length=500)
    visible_targets: list[VisibleTarget] = Field(default_factory=list)
    focused_target_id: str | None = Field(default=None, min_length=1, max_length=100)
    notes: list[str] = Field(default_factory=list)

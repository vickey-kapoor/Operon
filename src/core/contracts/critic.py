"""Strict critic contract for evaluation after actor execution."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from src.core.contracts.perception import ContractModel, ContractVersion, Environment


class FailureType(StrEnum):
    TARGET_NOT_FOUND = "target_not_found"
    WRONG_WINDOW_ACTIVE = "wrong_window_active"
    DIALOG_NOT_OPENED = "dialog_not_opened"
    TEXT_NOT_ENTERED = "text_not_entered"
    TIMING_ISSUE = "timing_issue"
    AMBIGUOUS_PERCEPTION = "ambiguous_perception"
    UI_CHANGED = "ui_changed"
    PICKER_NOT_DETECTED = "picker_not_detected"
    FILE_NOT_REFLECTED = "file_not_reflected"


class CriticOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"


class CriticOutput(ContractModel):
    contract_version: ContractVersion = ContractVersion.PHASE1
    environment: Environment
    observation_id: str = Field(min_length=1, max_length=100)
    plan_id: str = Field(min_length=1, max_length=100)
    attempt_id: str = Field(min_length=1, max_length=100)
    outcome: CriticOutcome
    failure_type: FailureType | None = None
    judgment: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_critic_output(self) -> "CriticOutput":
        if self.outcome is CriticOutcome.SUCCESS and self.failure_type is not None:
            raise ValueError("successful critic outputs must not include failure_type")
        if self.outcome is not CriticOutcome.SUCCESS and self.failure_type is None:
            raise ValueError("non-success critic outputs require failure_type")
        return self

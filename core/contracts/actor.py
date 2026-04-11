"""Strict actor contract for routed execution attempts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from core.contracts.critic import FailureType
from core.contracts.perception import ContractModel, ContractVersion, Environment
from core.contracts.planner import PlannerAction


class ExecutorChoice(StrEnum):
    BROWSER = "browser"
    DESKTOP = "desktop"


class ActorStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class ActorAction(PlannerAction):
    pass


class ActorOutput(ContractModel):
    contract_version: ContractVersion = ContractVersion.PHASE1
    environment: Environment
    observation_id: str = Field(min_length=1, max_length=100)
    plan_id: str = Field(min_length=1, max_length=100)
    attempt_id: str = Field(min_length=1, max_length=100)
    executor: ExecutorChoice
    action: ActorAction
    status: ActorStatus
    failure_type: FailureType | None = None
    details: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_actor_output(self) -> "ActorOutput":
        expected_executor = (
            ExecutorChoice.BROWSER if self.environment is Environment.BROWSER else ExecutorChoice.DESKTOP
        )
        if self.executor is not expected_executor:
            raise ValueError("executor must match environment")
        if self.status is ActorStatus.FAILED and self.failure_type is None:
            raise ValueError("failed actor outputs require failure_type")
        if self.status is ActorStatus.SUCCESS and self.failure_type is not None:
            raise ValueError("successful actor outputs must not include failure_type")
        return self

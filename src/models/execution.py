"""Schemas for action execution results."""

from pydantic import Field

from src.models.common import FailureCategory, LoopStage, StrictModel
from src.models.policy import AgentAction


class ExecutedAction(StrictModel):
    """Recorded outcome of executing a single agent action."""

    action: AgentAction
    success: bool
    detail: str = Field(min_length=1)
    artifact_path: str | None = Field(default=None, min_length=1)
    failure_category: FailureCategory | None = None
    failure_stage: LoopStage | None = None

"""Strict planner contract for the shared browser and desktop core."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from core.contracts.perception import ContractModel, ContractVersion, Environment


class ActionType(StrEnum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE_TEXT = "type_text"
    PRESS_HOTKEY = "press_hotkey"
    SCROLL = "scroll"
    WAIT = "wait"
    LAUNCH_APP = "launch_app"
    NAVIGATE = "navigate"
    UPLOAD_FILE_NATIVE = "upload_file_native"


class PlannerAction(ContractModel):
    action_type: ActionType
    target_id: str | None = Field(default=None, min_length=1, max_length=100)
    target_label: str | None = Field(default=None, min_length=1, max_length=200)
    text: str | None = Field(default=None, min_length=1, max_length=2000)
    hotkey: list[str] = Field(default_factory=list)
    scroll_direction: str | None = Field(default=None, pattern="^(up|down)$")
    scroll_amount: int | None = Field(default=None, ge=1, le=10000)
    wait_ms: int | None = Field(default=None, ge=0, le=120000)
    app_name: str | None = Field(default=None, min_length=1, max_length=200)
    url: str | None = Field(default=None, min_length=1, max_length=500)
    file_path: str | None = Field(default=None, min_length=1, max_length=1000)
    picker_title: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_payload(self) -> "PlannerAction":
        if self.action_type in {ActionType.CLICK, ActionType.DOUBLE_CLICK} and self.target_id is None:
            raise ValueError("Click actions require target_id")
        if self.action_type is ActionType.TYPE_TEXT and (self.target_id is None or self.text is None):
            raise ValueError("type_text requires target_id and text")
        if self.action_type is ActionType.PRESS_HOTKEY and len(self.hotkey) < 1:
            raise ValueError("press_hotkey requires at least one key")
        if self.action_type is ActionType.SCROLL and (
            self.scroll_direction is None or self.scroll_amount is None
        ):
            raise ValueError("scroll requires scroll_direction and scroll_amount")
        if self.action_type is ActionType.WAIT and self.wait_ms is None:
            raise ValueError("wait requires wait_ms")
        if self.action_type is ActionType.LAUNCH_APP and self.app_name is None:
            raise ValueError("launch_app requires app_name")
        if self.action_type is ActionType.NAVIGATE and self.url is None:
            raise ValueError("navigate requires url")
        if self.action_type is ActionType.UPLOAD_FILE_NATIVE and self.target_id is None:
            raise ValueError("upload_file_native requires target_id")
        return self


class PlannerOutput(ContractModel):
    contract_version: ContractVersion = ContractVersion.PHASE1
    environment: Environment
    observation_id: str = Field(min_length=1, max_length=100)
    plan_id: str = Field(min_length=1, max_length=100)
    subgoal: str = Field(min_length=1, max_length=300)
    rationale: str = Field(min_length=1, max_length=500)
    action: PlannerAction
    expected_outcome: str = Field(min_length=1, max_length=500)

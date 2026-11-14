"""``ToolExecutor``: the environment's executor plus Operon's tools (RFC 0002).

Tool actions run here; every other action goes to the wrapped executor unchanged, so
the browser and desktop executors know nothing about tools. Anything else the loop asks
of the executor (capture, run setup, the batch child check) reaches the wrapped one.
"""

from __future__ import annotations

from typing import Any

from operon.agent.tools import summary
from operon.agent.tools.runner import ToolRunner
from operon.models.common import FailureCategory, LoopStage
from operon.models.execution import ExecutedAction, ToolOutput
from operon.models.policy import ActionType, AgentAction


def _has_tool(action: AgentAction) -> bool:
    if action.action_type is ActionType.TOOL:
        return True
    return action.action_type is ActionType.BATCH and any(_has_tool(child) for child in action.actions or [])


class ToolExecutor:
    def __init__(self, inner: Any, tools: tuple[str, ...]) -> None:
        # Set through __dict__: __setattr__ below forwards everything else to ``inner``.
        self.__dict__["inner"] = inner
        self.__dict__["runner"] = ToolRunner(tools)
        self.__dict__["_child_check"] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["inner"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    @property
    def batch_child_check(self):
        return self.__dict__["_child_check"]

    @batch_child_check.setter
    def batch_child_check(self, check) -> None:
        # Kept here for a batch holding tools (run below), and passed on for a batch
        # the wrapped executor runs. Not every executor has one.
        self.__dict__["_child_check"] = check
        if hasattr(self.inner, "batch_child_check"):
            self.inner.batch_child_check = check

    async def execute(self, action: AgentAction) -> ExecutedAction:
        if action.action_type is ActionType.TOOL:
            return await self._run_tool(action)
        if action.action_type is ActionType.BATCH and _has_tool(action):
            return await self._run_batch(action)
        return await self.inner.execute(action)

    async def _run_tool(self, action: AgentAction) -> ExecutedAction:
        result = await self.runner.run(action.tool or "", action.tool_args)
        detail = f"{summary(action.tool or 'tool', action.tool_args or {})}: " + ("ran" if result.ok else result.output)
        if result.ok:
            return ExecutedAction(action=action, success=True, detail=detail[:500], tool_outputs=[result])
        return ExecutedAction(
            action=action,
            success=False,
            detail=detail[:500],
            failure_category=FailureCategory.EXECUTION_ERROR,
            failure_stage=LoopStage.EXECUTE,
            tool_outputs=[result],
        )

    async def _run_batch(self, action: AgentAction) -> ExecutedAction:
        """A turn's calls in order, tools here and the rest through the wrapped
        executor, stopping at the first failure (as the GUI batch does)."""
        outputs: list[ToolOutput] = []
        done: list[str] = []
        artifact_path: str | None = None
        check = self.__dict__["_child_check"]
        for index, child in enumerate(action.actions or []):
            if index > 0 and check is not None:
                blocked = await check(child)
                if blocked is not None:
                    return self._batch_failed(
                        action, index, f"Batch stopped before {child.action_type.value}: {blocked}", outputs, artifact_path
                    )
            result = await self.execute(child)
            outputs.extend(result.tool_outputs or [])
            artifact_path = result.artifact_path or artifact_path
            done.append(f"{child.action_type.value}:{'ok' if result.success else 'failed'}")
            if not result.success:
                return self._batch_failed(
                    action, index, f"Batch failed on {child.action_type.value}: {result.detail}", outputs, artifact_path,
                    result,
                )
        return ExecutedAction(
            action=action,
            success=True,
            detail=f"Executed batch: {', '.join(done)}",
            artifact_path=artifact_path,
            tool_outputs=outputs or None,
        )

    @staticmethod
    def _batch_failed(
        action: AgentAction,
        index: int,
        detail: str,
        outputs: list[ToolOutput],
        artifact_path: str | None,
        result: ExecutedAction | None = None,
    ) -> ExecutedAction:
        return ExecutedAction(
            action=action,
            success=False,
            detail=detail,
            artifact_path=artifact_path,
            failure_category=(result.failure_category if result else None) or FailureCategory.EXECUTION_ERROR,
            failure_stage=(result.failure_stage if result else None) or LoopStage.EXECUTE,
            batch_failed_index=index,
            tool_outputs=outputs or None,
        )

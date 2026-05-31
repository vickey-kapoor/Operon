"""Background auto-run task scheduling."""

from __future__ import annotations

import asyncio
import logging

from operon.agent.loop import AgentLoop
from operon.models.common import RunStatus, StepRequest, StopReason

logger = logging.getLogger(__name__)

_auto_run_tasks: dict[str, asyncio.Task] = {}


async def cleanup_cancelled_run(loop: AgentLoop, run_id: str) -> None:
    try:
        await loop._cleanup_completed_run(run_id)
    except Exception as exc:
        logger.warning("cleanup after cancellation failed for %s: %s", run_id, exc)


def _auto_run_done_callback(run_id: str):
    def _callback(task: asyncio.Task) -> None:
        if _auto_run_tasks.get(run_id) is task:
            _auto_run_tasks.pop(run_id, None)
        if task.cancelled():
            logger.info("[loop] run %s auto-run task cancelled", run_id)
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[loop] run %s crashed", run_id, exc_info=(type(exc), exc, exc.__traceback__))

    return _callback


def schedule_auto_run(loop: AgentLoop, run_id: str, max_steps: int) -> asyncio.Task:
    existing = _auto_run_tasks.get(run_id)
    if existing is not None:
        if not existing.done():
            logger.info("[loop] run %s auto-run already scheduled", run_id)
            return existing
        _auto_run_tasks.pop(run_id, None)

    task = asyncio.create_task(auto_run_loop(loop, run_id, max_steps))
    _auto_run_tasks[run_id] = task
    task.add_done_callback(_auto_run_done_callback(run_id))
    logger.info("[loop] run %s auto-run scheduled (max_steps=%d)", run_id, max_steps)
    return task


async def cancel_auto_run(run_id: str) -> None:
    task = _auto_run_tasks.pop(run_id, None)
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning("[loop] run %s auto-run cancellation surfaced error: %s", run_id, exc)


async def auto_run_loop(loop: AgentLoop, run_id: str, max_steps: int) -> None:
    """Fire-and-forget step loop: runs until terminal state or max_steps."""
    logger.info("[loop] starting run %s", run_id)
    while True:
        state = await loop.run_store.get_run(run_id)
        if state is None:
            logger.warning("[loop] run %s disappeared from store", run_id)
            break
        if state.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.WAITING_FOR_USER,
            RunStatus.CANCELLED,
        }:
            logger.info("[loop] run %s terminal: %s", run_id, state.status.value)
            break
        if state.step_count >= max_steps:
            logger.warning("[loop] run %s hit max_steps=%d; stopping", run_id, max_steps)
            state.stop_reason = StopReason.MAX_STEP_LIMIT_REACHED
            await loop.run_store.set_status(run_id, RunStatus.FAILED)
            try:
                await loop._cleanup_completed_run(run_id)
            except Exception as exc:
                logger.warning("[loop] cleanup after max_steps failed: %s", exc)
            break
        logger.info("[loop] run %s entering step %d", run_id, state.step_count + 1)
        await loop.step_run(StepRequest(run_id=run_id))


"""FastAPI-facing runtime dependencies."""

from operon.api.runtime.cdp import ensure_cdp_ready
from operon.api.runtime.common import test_safe_mode_enabled, validate_run_id
from operon.api.runtime.events import mark_run_failed, publish_step_outcome
from operon.api.runtime.loops import get_agent_loop, get_desktop_agent_loop
from operon.api.runtime.tasks import (
    cancel_auto_run,
    cleanup_cancelled_run,
    resume_and_schedule,
    schedule_auto_run,
)

__all__ = [
    "cancel_auto_run",
    "cleanup_cancelled_run",
    "ensure_cdp_ready",
    "get_agent_loop",
    "get_desktop_agent_loop",
    "mark_run_failed",
    "publish_step_outcome",
    "resume_and_schedule",
    "schedule_auto_run",
    "test_safe_mode_enabled",
    "validate_run_id",
]

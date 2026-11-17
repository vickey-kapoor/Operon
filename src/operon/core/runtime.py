"""The process's runtime state in one place, with a context per run (KAN-156).

Before KAN-156 this state was module globals spread over six modules: the browser
and desktop agent loops (``api/runtime/loops.py``), the live-view references
(``api/ws_stream.py``), the CDP browser manager (``browser/manager.py``), the
operator's control settings and override hints (``agent/control.py``), the auto-run
tasks (``api/runtime/tasks.py``) and the run timers (``api/frontdoor.py``).

Now it lives on one :class:`OperonRuntime`:

* process-wide parts (the loops, the desktop environment provider, the live-view
  references, the operator's session settings) are its fields;
* everything that belongs to one run (its override hint, auto-run task, start and
  finish times) is a :class:`RunContext` in :attr:`OperonRuntime.runs`, so a run's
  state is found, and dropped, together.

``current_runtime()`` returns the process's runtime; the modules above keep their
functions (``get_agent_loop()``, ``set_override_hint()`` ...) and read and write it.
``reset_runtime()`` gives a test a clean one. The layer is ``core`` so the agent,
browser and API layers can all use it without importing each other.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

# How many finished runs keep their context (for the report's duration) before the
# oldest is dropped. Runs that are still going are never dropped.
MAX_FINISHED_RUNS = 500


@dataclass
class RunContext:
    """State that belongs to one run while this process serves it."""

    run_id: str
    # Wall-clock bounds of a run started through the front door (the report's duration).
    started_at: float | None = None
    finished_at: float | None = None
    # A correction typed during a pause, consumed once by the run's next step.
    override_hint: str | None = None
    # The background task stepping the run (``api/runtime/tasks.py``).
    auto_run_task: asyncio.Task | None = None

    @property
    def finished(self) -> bool:
        return self.finished_at is not None and (self.auto_run_task is None or self.auto_run_task.done())


@dataclass
class ControlSettings:
    """Operator settings for the session, set from the live view (``/ws/stream``)."""

    # Pause for approval when a decision's confidence is below this (0.0 = off).
    confidence_threshold: float = 0.0
    # Rule-engine primitives the operator switched off.
    disabled_rules: frozenset[str] = frozenset()


@dataclass
class StreamState:
    """What the live view (``/ws/stream``) streams from and sends control messages to."""

    # One queue per connected client: dict -> JSON text frame, bytes -> JPEG frame.
    subscribers: list[asyncio.Queue[dict | bytes]] = field(default_factory=list)
    # The browser loop's executor, for live frames when no CDP screencast runs.
    browser_executor: Any = None
    # The desktop loop's executor; streams only if it has ``live_frame_png``.
    desktop_executor: Any = None
    # The run on screen now, and the loop (browser or desktop) that owns it.
    active_run_id: str | None = None
    active_loop: Any = None
    # The CDP-attached browser manager (command-center path), if one is connected.
    browser_manager: Any = None


@dataclass
class OperonRuntime:
    """Everything this process holds between requests."""

    # The agent loops, built on first use (``api/runtime/loops.py``).
    browser_loop: Any = None
    desktop_loop: Any = None
    # The desktop run environment (KAN-80), started once.
    desktop_provider: Any = None
    # The connected CDP browser manager the browser executor attaches to.
    active_browser_manager: Any = None
    # The built-in browser with the saved profile (KAN-138), started on first use.
    builtin_browser: Any = None
    # True while one Python SDK run (``operon.sdk.Agent``) owns this process's loops.
    sdk_run_active: bool = False
    control: ControlSettings = field(default_factory=ControlSettings)
    stream: StreamState = field(default_factory=StreamState)
    runs: OrderedDict[str, RunContext] = field(default_factory=OrderedDict)

    def run(self, run_id: str) -> RunContext:
        """The context of ``run_id``, created on first use."""
        context = self.runs.get(run_id)
        if context is None:
            context = RunContext(run_id)
            self.runs[run_id] = context
            self._prune()
        return context

    def peek_run(self, run_id: str) -> RunContext | None:
        """The context of ``run_id`` if it has one; never creates it."""
        return self.runs.get(run_id)

    def finish_run(self, run_id: str) -> None:
        """Note that ``run_id`` ended (succeeded, failed, cancelled or crashed), so its
        context can be pruned once its task is done. Keeps the first time noted; never
        creates a context for a run that has none."""
        context = self.runs.get(run_id)
        if context is not None and context.finished_at is None:
            context.finished_at = time.time()

    def forget_run(self, run_id: str) -> None:
        self.runs.pop(run_id, None)

    def _prune(self) -> None:
        finished = [run_id for run_id, context in self.runs.items() if context.finished]
        for run_id in finished[: max(0, len(finished) - MAX_FINISHED_RUNS)]:
            del self.runs[run_id]


_runtime = OperonRuntime()


def current_runtime() -> OperonRuntime:
    """The runtime state of this process."""
    return _runtime


def reset_runtime() -> OperonRuntime:
    """Start over with an empty runtime (tests); returns it."""
    global _runtime
    _runtime = OperonRuntime()
    return _runtime

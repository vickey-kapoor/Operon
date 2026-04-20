"""Typed run-store interface and file-backed implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict
from uuid import uuid4

from src.models.common import RunStatus
from src.models.perception import ScreenPerception
from src.models.state import AgentState
from src.store.background_writer import bg_writer


class RunStore(ABC):
    """Typed interface for local run state persistence."""

    @abstractmethod
    def create_run(
        self,
        intent: str,
        *,
        start_url: str | None = None,
        headless: bool | None = None,
    ) -> AgentState:
        """Create and store a new run."""

    @abstractmethod
    async def get_run(self, run_id: str) -> AgentState | None:
        """Return a stored run if present."""

    @abstractmethod
    async def update_state(self, run_id: str, perception: ScreenPerception) -> AgentState:
        """Persist the latest perception and return the updated state."""

    @abstractmethod
    async def set_status(self, run_id: str, status: RunStatus) -> AgentState:
        """Update run status and return the updated state."""

    @abstractmethod
    async def save_state(self, state: AgentState) -> None:
        """Persist the current state snapshot to disk."""


class FileBackedRunStore(RunStore):
    """Local file-backed run store using one directory per run."""

    def __init__(self, root_dir: str | Path = "runs") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._resolved_root = self.root_dir.resolve()
        self._runs: Dict[str, AgentState] = {}

    def create_run(
        self,
        intent: str,
        *,
        start_url: str | None = None,
        headless: bool | None = None,
    ) -> AgentState:
        """Create and store a new run record."""
        run_id = str(uuid4())
        record = AgentState(
            run_id=run_id,
            intent=intent,
            start_url=start_url,
            headless=headless,
            status=RunStatus.PENDING,
        )
        self._runs[run_id] = record
        self._ensure_run_dir(run_id)
        self._write_state(record)
        return record

    async def get_run(self, run_id: str) -> AgentState | None:
        """Return a stored run record, if present."""
        cached = self._runs.get(run_id)
        if cached is not None:
            return cached

        state_path = self._state_path(run_id)
        if not state_path.exists():
            return None

        record = AgentState.model_validate_json(state_path.read_text(encoding="utf-8"))
        self._runs[run_id] = record
        return record

    # How many history entries to retain in the on-disk state snapshot.
    # The full history is always in run.jsonl; state.json is the live cursor.
    _MAX_HISTORY = 20

    async def update_state(self, run_id: str, perception: ScreenPerception) -> AgentState:
        """Persist a new perception snapshot and return the updated state."""
        record = self._runs[run_id]
        record.status = RunStatus.RUNNING
        record.observation_history.append(perception)
        record.step_count += 1
        self._write_state(record)
        return record

    async def set_status(self, run_id: str, status: RunStatus) -> AgentState:
        """Update run status in place.  Write only on terminal transitions."""
        record = self._runs[run_id]
        record.status = status
        if status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_FOR_USER}:
            self._write_state(record)
        return record

    async def save_state(self, state: AgentState) -> None:
        """Persist the current state snapshot to disk."""
        self._runs[state.run_id] = state
        self._write_state(state)

    def run_log_path(self, run_id: str) -> Path:
        """Return the JSONL path for the run log."""
        self._ensure_run_dir(run_id)
        return self.root_dir / run_id / "run.jsonl"

    def before_artifact_path(self, run_id: str, step_index: int) -> str:
        """Return the planned before-image artifact path for a step."""
        return str(self._step_dir(run_id, step_index) / "before.png")

    def after_artifact_path(self, run_id: str, step_index: int) -> str:
        """Return the planned after-image artifact path for a step."""
        return str(self._step_dir(run_id, step_index) / "after.png")

    def _ensure_run_dir(self, run_id: str) -> Path:
        path = (self.root_dir / run_id).resolve()
        if not path.is_relative_to(self._resolved_root):
            raise ValueError(f"run_id {run_id!r} escapes the runs directory")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _step_dir(self, run_id: str, step_index: int) -> Path:
        path = self._ensure_run_dir(run_id) / f"step_{step_index}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _state_path(self, run_id: str) -> Path:
        return self._ensure_run_dir(run_id) / "state.json"

    def _write_state(self, state: AgentState) -> None:
        # Trim history lists for the on-disk snapshot to keep serialization O(1).
        # The full history lives in run.jsonl; only the last N entries are needed
        # here for recovery context.
        n = self._MAX_HISTORY
        if (
            len(state.observation_history) > n
            or len(state.action_history) > n
            or len(state.verification_history) > n
        ):
            trimmed = state.model_copy(update={
                "observation_history": state.observation_history[-n:],
                "action_history": state.action_history[-n:],
                "verification_history": state.verification_history[-n:],
            })
            bg_writer.enqueue(self._state_path(state.run_id), trimmed.model_dump_json())
        else:
            bg_writer.enqueue(self._state_path(state.run_id), state.model_dump_json())

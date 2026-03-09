"""Abstract TaskStore base class and factory function."""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from src.api.models import TaskRecord


class TaskStore(ABC):
    """Abstract interface for persisting navigation task records."""

    @abstractmethod
    async def get(self, task_id: str) -> Optional[TaskRecord]:
        """Return a task by ID, or ``None`` if not found / expired."""
        ...

    @abstractmethod
    async def upsert(self, record: TaskRecord) -> None:
        """Create or replace a task record."""
        ...

    @abstractmethod
    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[TaskRecord], int]:
        """
        Return ``(page, total_count)`` of task records sorted by ``created_at``
        descending.  Optionally filter by ``status``.
        """
        ...

    @abstractmethod
    async def delete_expired(self, max_age_seconds: float) -> int:
        """Delete tasks older than ``max_age_seconds``.  Returns count deleted."""
        ...

    @abstractmethod
    async def count_by_status(self) -> dict:
        """Return a ``{status: count}`` dict for all tasks.  Used by health check."""
        ...


def create_store() -> TaskStore:
    """
    Factory that reads the ``TASK_STORE`` env var and returns the right store.

    ``TASK_STORE=memory``    → :class:`MemoryTaskStore` (default)
    ``TASK_STORE=firestore`` → :class:`FirestoreTaskStore`
    """
    store_type = os.environ.get("TASK_STORE", "memory").lower()
    if store_type == "firestore":
        from src.api.store_firestore import FirestoreTaskStore

        return FirestoreTaskStore()
    from src.api.store_memory import MemoryTaskStore

    return MemoryTaskStore()

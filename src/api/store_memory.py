"""In-memory TaskStore backed by a dict + asyncio.Lock."""

import asyncio
import time
from typing import List, Optional, Tuple

from src.api.models import TaskRecord
from src.api.store import TaskStore


class MemoryTaskStore(TaskStore):
    """Thread-safe in-memory store.  All data is lost on restart."""

    def __init__(self) -> None:
        self._data: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    async def get(self, task_id: str) -> Optional[TaskRecord]:
        async with self._lock:
            return self._data.get(task_id)

    async def upsert(self, record: TaskRecord) -> None:
        async with self._lock:
            self._data[record.task_id] = record

    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[TaskRecord], int]:
        async with self._lock:
            records = list(self._data.values())
        if status:
            records = [r for r in records if r.status == status]
        records.sort(key=lambda r: r.created_at, reverse=True)
        total = len(records)
        return records[offset : offset + limit], total

    async def delete_expired(self, max_age_seconds: float) -> int:
        cutoff = time.time() - max_age_seconds
        async with self._lock:
            expired = [
                tid for tid, r in self._data.items() if r.created_at < cutoff
            ]
            for tid in expired:
                del self._data[tid]
        return len(expired)

    async def count_by_status(self) -> dict:
        async with self._lock:
            counts: dict = {}
            for r in self._data.values():
                status = str(r.status)
                counts[status] = counts.get(status, 0) + 1
        return counts

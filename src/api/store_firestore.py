"""Firestore-backed TaskStore implementation."""

import logging
import time
from typing import List, Optional, Tuple

from src.api.models import TaskRecord
from src.api.store import TaskStore

logger = logging.getLogger(__name__)

_COLLECTION = "ui_navigator_tasks"


def _strip_screenshots(record_dict: dict) -> dict:
    """
    Remove large screenshot blobs before writing to Firestore.

    Firestore has a 1 MiB per-document limit; base64 screenshots can easily
    exceed this.  We strip them from ``result.screenshots`` and from every
    event's ``screenshot`` key before persisting.
    """
    d = dict(record_dict)
    if d.get("result") and isinstance(d["result"], dict):
        d["result"] = dict(d["result"])
        d["result"].pop("screenshots", None)
    if d.get("events"):
        d["events"] = [
            {k: v for k, v in e.items() if k != "screenshot"}
            for e in d["events"]
        ]
    return d


class FirestoreTaskStore(TaskStore):
    """
    Stores task records in Firestore (collection ``ui_navigator_tasks``).

    Screenshots are stripped before storage to stay under the 1 MiB doc limit.
    """

    def __init__(self) -> None:
        from google.cloud import firestore

        self._db = firestore.AsyncClient()

    def _col(self):
        return self._db.collection(_COLLECTION)

    async def get(self, task_id: str) -> Optional[TaskRecord]:
        try:
            doc = await self._col().document(task_id).get()
            if not doc.exists:
                return None
            return TaskRecord(**doc.to_dict())
        except Exception as exc:
            logger.error("Firestore get failed for %s: %s", task_id, exc)
            return None

    async def upsert(self, record: TaskRecord) -> None:
        try:
            data = _strip_screenshots(record.model_dump())
            await self._col().document(record.task_id).set(data)
        except Exception as exc:
            logger.error("Firestore upsert failed for %s: %s", record.task_id, exc)

    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[TaskRecord], int]:
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter

            query = self._col().order_by("created_at", direction="DESCENDING")
            if status:
                query = query.where(filter=FieldFilter("status", "==", status))
            # Collect all matching docs for total count (pagination at Python level).
            records: List[TaskRecord] = []
            async for doc in query.stream():
                records.append(TaskRecord(**doc.to_dict()))
            total = len(records)
            return records[offset : offset + limit], total
        except Exception as exc:
            logger.error("Firestore list_tasks failed: %s", exc)
            return [], 0

    async def delete_expired(self, max_age_seconds: float) -> int:
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter

            cutoff = time.time() - max_age_seconds
            query = self._col().where(
                filter=FieldFilter("created_at", "<", cutoff)
            )
            count = 0
            async for doc in query.stream():
                await doc.reference.delete()
                count += 1
            logger.info("Deleted %d expired Firestore task records", count)
            return count
        except Exception as exc:
            logger.error("Firestore delete_expired failed: %s", exc)
            return 0

    async def count_by_status(self) -> dict:
        try:
            counts: dict = {}
            async for doc in self._col().stream():
                status = doc.to_dict().get("status", "unknown")
                counts[status] = counts.get(status, 0) + 1
            return counts
        except Exception as exc:
            logger.error("Firestore count_by_status failed: %s", exc)
            return {}

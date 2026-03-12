"""Tests for FirestoreTaskStore (all Firestore calls mocked)."""

import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.models import TaskRecord, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(task_id='test-123', status='pending', created_at=None):
    """Create a minimal TaskRecord dict for mocking Firestore docs."""
    return {
        'task_id': task_id,
        'task': 'test task',
        'status': status,
        'start_url': None,
        'max_steps': 10,
        'events': [],
        'result': None,
        'created_at': created_at or time.time(),
    }


def _make_mock_doc(data, exists=True):
    """Create a mock Firestore document snapshot."""
    doc = MagicMock()
    doc.exists = exists
    doc.to_dict.return_value = data
    doc.reference = MagicMock()
    return doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFirestoreTaskStore:

    @pytest.fixture(autouse=True)
    def setup_store(self):
        """Create a FirestoreTaskStore with a mocked db client."""
        # We need to mock the firestore module before importing the store
        mock_firestore_mod = MagicMock()
        mock_filter = MagicMock()
        mock_firestore_mod.AsyncClient = MagicMock
        mock_firestore_v1 = MagicMock()
        mock_firestore_v1.base_query.FieldFilter = mock_filter

        with (
            patch.dict(sys.modules, {
                'google.cloud.firestore': mock_firestore_mod,
                'google.cloud.firestore_v1': mock_firestore_v1,
                'google.cloud.firestore_v1.base_query': mock_firestore_v1.base_query,
            }),
        ):
            # Import fresh to use our mocked firestore
            import importlib
            import src.api.store_firestore as sf_mod
            importlib.reload(sf_mod)

            self.store = sf_mod.FirestoreTaskStore.__new__(sf_mod.FirestoreTaskStore)
            self.mock_db = MagicMock()
            self.mock_col = MagicMock()
            self.mock_db.collection.return_value = self.mock_col
            self.store._db = self.mock_db
            self.FieldFilter = mock_filter
            yield

    async def test_get_existing(self):
        """get() returns TaskRecord when document exists."""
        data = _make_record()
        mock_doc = _make_mock_doc(data)
        self.mock_col.document.return_value.get = AsyncMock(return_value=mock_doc)

        result = await self.store.get('test-123')
        assert result is not None
        assert result.task_id == 'test-123'

    async def test_get_missing(self):
        """get() returns None when document does not exist."""
        mock_doc = _make_mock_doc({}, exists=False)
        self.mock_col.document.return_value.get = AsyncMock(return_value=mock_doc)

        result = await self.store.get('nonexistent')
        assert result is None

    async def test_get_error(self):
        """get() returns None on Firestore error."""
        self.mock_col.document.return_value.get = AsyncMock(
            side_effect=RuntimeError('Firestore down')
        )

        result = await self.store.get('error-id')
        assert result is None

    async def test_upsert(self):
        """upsert() calls set() with stripped screenshots."""
        record = TaskRecord(
            task_id='upsert-1',
            task='test',
        )
        # Add result with screenshots
        record.result = MagicMock()
        record_dict = record.model_dump()
        record_dict['result'] = {'screenshots': ['base64data'], 'success': True}

        self.mock_col.document.return_value.set = AsyncMock()

        # Patch model_dump to return dict with screenshots
        with patch.object(type(record), 'model_dump', return_value=record_dict):
            await self.store.upsert(record)

        self.mock_col.document.return_value.set.assert_called_once()
        call_data = self.mock_col.document.return_value.set.call_args[0][0]
        if call_data.get('result'):
            assert 'screenshots' not in call_data['result']

    async def test_list_tasks(self):
        """list_tasks() returns records with total count."""
        docs = [_make_mock_doc(_make_record(f'id-{i}')) for i in range(3)]

        async def mock_stream():
            for d in docs:
                yield d

        mock_query = MagicMock()
        mock_query.where.return_value = mock_query
        self.mock_col.order_by.return_value = mock_query
        mock_query.stream.return_value = mock_stream()

        records, total = await self.store.list_tasks(limit=2, offset=0)
        assert total == 3
        assert len(records) == 2

    async def test_list_tasks_with_status(self):
        """list_tasks(status=running) applies filter."""
        docs = [_make_mock_doc(_make_record('id-1', status='running'))]

        async def mock_stream():
            for d in docs:
                yield d

        mock_query = MagicMock()
        mock_query.where.return_value = mock_query
        self.mock_col.order_by.return_value = mock_query
        mock_query.stream.return_value = mock_stream()

        records, total = await self.store.list_tasks(status='running')
        assert total == 1

    async def test_delete_expired_batch(self):
        """delete_expired() uses batch writes."""
        old_time = time.time() - 100000
        docs = [_make_mock_doc(_make_record(f'old-{i}', created_at=old_time)) for i in range(3)]

        async def mock_stream():
            for d in docs:
                yield d

        mock_query = MagicMock()
        mock_query.where.return_value = mock_query
        self.mock_col.where.return_value = mock_query
        mock_query.stream.return_value = mock_stream()

        mock_batch = MagicMock()
        mock_batch.commit = AsyncMock()
        self.mock_db.batch.return_value = mock_batch

        count = await self.store.delete_expired(86400)
        assert count == 3
        mock_batch.commit.assert_called()

    async def test_count_by_status(self):
        """count_by_status() aggregates correctly."""
        docs = [
            _make_mock_doc(_make_record('id-1', status='running')),
            _make_mock_doc(_make_record('id-2', status='done')),
            _make_mock_doc(_make_record('id-3', status='done')),
        ]

        async def mock_stream():
            for d in docs:
                yield d

        self.mock_col.stream.return_value = mock_stream()

        counts = await self.store.count_by_status()
        assert counts == {'running': 1, 'done': 2}

"""Tests for observability modules: metrics, tracing, storage, logging."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_emit_logs_metric(self):
        """emit() always logs the metric as structured JSON."""
        import src.metrics as m
        with patch.object(m.logger, 'info') as mock_log:
            m.emit('test_metric', 42.0, {'key': 'val'})
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args
        assert call_kwargs[1]['extra']['metric'] == 'test_metric'
        assert call_kwargs[1]['extra']['value'] == 42.0

    def test_emit_no_project_skips_cloud(self, monkeypatch):
        """emit() without GOOGLE_CLOUD_PROJECT does not submit to thread pool."""
        monkeypatch.delenv('GOOGLE_CLOUD_PROJECT', raising=False)
        import src.metrics as m
        with patch.object(m._pool, 'submit') as mock_submit:
            m.emit('test_metric', 1.0)
        mock_submit.assert_not_called()

    def test_singleton_client(self):
        """_get_monitoring_client returns the same cached instance."""
        import src.metrics as m
        # Reset cached client
        original = m._monitoring_client
        m._monitoring_client = None
        try:
            mock_client = MagicMock()
            with patch.dict('sys.modules', {'google.cloud.monitoring_v3': MagicMock(MetricServiceClient=MagicMock(return_value=mock_client))}):
                # Re-import to pick up mock
                import importlib
                importlib.reload(m)
                m._monitoring_client = None
                c1 = m._get_monitoring_client()
                c2 = m._get_monitoring_client()
                assert c1 is c2
                assert c1 is mock_client
        finally:
            m._monitoring_client = original
            import importlib
            importlib.reload(m)


# ---------------------------------------------------------------------------
# tracing.py
# ---------------------------------------------------------------------------


class TestTracing:
    def test_setup_without_project(self, monkeypatch):
        """setup_tracing without GOOGLE_CLOUD_PROJECT is a no-op."""
        monkeypatch.delenv('GOOGLE_CLOUD_PROJECT', raising=False)
        import src.tracing as t
        # Reset state
        original_configured = t._configured
        original_tracer = t._tracer
        t._configured = False
        t._tracer = None
        try:
            t.setup_tracing()
            assert t._tracer is None
        finally:
            t._configured = original_configured
            t._tracer = original_tracer

    def test_span_noop(self):
        """span() yields None when tracing is not configured."""
        import src.tracing as t
        original = t._tracer
        t._tracer = None
        try:
            with t.span('test-span') as s:
                assert s is None
        finally:
            t._tracer = original

    def test_shutdown_noop(self):
        """shutdown_tracing when provider is None does not raise."""
        import src.tracing as t
        original = t._provider
        t._provider = None
        try:
            t.shutdown_tracing()  # should not raise
        finally:
            t._provider = original


# ---------------------------------------------------------------------------
# storage.py
# ---------------------------------------------------------------------------


class TestStorage:
    def test_upload_no_bucket(self, monkeypatch):
        """upload_screenshot without GCS_BUCKET returns None."""
        monkeypatch.delenv('GCS_BUCKET', raising=False)
        from src.storage import upload_screenshot
        result = upload_screenshot(b'fake-png', '00000000-0000-0000-0000-000000000000', 1)
        assert result is None

    def test_upload_invalid_task_id(self, monkeypatch):
        """upload_screenshot with non-UUID task_id returns None (path traversal blocked)."""
        monkeypatch.setenv('GCS_BUCKET', 'test-bucket')
        from src.storage import upload_screenshot
        result = upload_screenshot(b'fake-png', '../../../etc/passwd', 1)
        assert result is None

    def test_upload_valid_task_id(self, monkeypatch):
        """upload_screenshot with valid UUID calls GCS."""
        monkeypatch.setenv('GCS_BUCKET', 'test-bucket')
        import src.storage as s
        original = s._gcs_client
        s._gcs_client = None
        try:
            mock_client = MagicMock()
            mock_blob = MagicMock()
            mock_blob.generate_signed_url.return_value = 'https://signed-url'
            mock_client.bucket.return_value.blob.return_value = mock_blob
            with patch('src.storage._get_gcs_client', return_value=mock_client):
                result = s.upload_screenshot(
                    b'fake-png', '00000000-0000-0000-0000-000000000000', 1
                )
            assert result == 'https://signed-url'
            mock_blob.upload_from_string.assert_called_once()
        finally:
            s._gcs_client = original


# ---------------------------------------------------------------------------
# logging_config.py
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    def test_valid_log_level(self, monkeypatch):
        """configure_logging with valid LOG_LEVEL sets the correct level."""
        monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
        from src.logging_config import configure_logging
        configure_logging()
        assert logging.getLogger().level == logging.DEBUG

    def test_invalid_log_level_falls_back(self, monkeypatch):
        """configure_logging with invalid LOG_LEVEL falls back to INFO."""
        monkeypatch.setenv('LOG_LEVEL', 'INVALID_LEVEL')
        from src.logging_config import configure_logging
        configure_logging()
        assert logging.getLogger().level == logging.INFO

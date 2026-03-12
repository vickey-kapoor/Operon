"""OpenTelemetry / Cloud Trace setup and span context manager."""

import contextlib
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_tracer = None
_provider = None
_configured = False


def setup_tracing() -> None:
    """
    Initialize OTel + Cloud Trace exporter if ``GOOGLE_CLOUD_PROJECT`` is set.

    Safe to call multiple times — initialises only once.
    """
    global _tracer, _configured
    if _configured:
        return
    _configured = True

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        logger.debug("Tracing disabled — GOOGLE_CLOUD_PROJECT not set")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        global _provider
        provider = TracerProvider()
        _provider = provider
        exporter = CloudTraceSpanExporter(project_id=project)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("ui-navigator")
        logger.info("Cloud Trace initialised for project %s", project)
    except Exception as exc:
        logger.warning("Failed to initialise tracing (non-fatal): %s", exc)


def shutdown_tracing() -> None:
    """Flush and shut down the TracerProvider. Call on app shutdown."""
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception as exc:
            logger.debug("Tracer shutdown error: %s", exc)


@contextlib.contextmanager
def span(name: str, attributes: Optional[Dict[str, str]] = None) -> "contextlib.AbstractContextManager":
    """
    Context manager that wraps a block of code in an OTel span.

    No-op (yields immediately) if tracing has not been initialised.
    Works in both sync and async contexts.
    """
    if _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, v)
        yield s

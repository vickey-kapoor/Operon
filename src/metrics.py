"""Cloud Monitoring metric emission — fire-and-forget."""

import concurrent.futures
import logging
import os
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
_METRIC_PREFIX = "custom.googleapis.com/ui_navigator"

# Bounded thread pool for metric emission — prevents unbounded thread creation.
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Cached MetricServiceClient — lazy-init on first use (thread-safe via GIL).
_monitoring_client = None


def emit(
    name: str,
    value: float = 1.0,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Emit a named metric data point.

    Always logs the metric as structured JSON.  If ``GOOGLE_CLOUD_PROJECT`` is
    configured, also writes to Cloud Monitoring via a thread pool (best-effort,
    never raises).
    """
    logger.info(
        "metric",
        extra={"metric": name, "value": value, "labels": labels or {}},
    )
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", _PROJECT)
    if project:
        _pool.submit(_emit_to_cloud_monitoring, name, value, labels or {}, time.time(), project)


def _get_monitoring_client():
    """Return a cached MetricServiceClient, creating it on first call."""
    global _monitoring_client
    if _monitoring_client is None:
        from google.cloud import monitoring_v3
        _monitoring_client = monitoring_v3.MetricServiceClient()
    return _monitoring_client


def _emit_to_cloud_monitoring(
    name: str,
    value: float,
    labels: Dict[str, str],
    ts: float,
    project: str,
) -> None:
    """Write a single GAUGE double time series to Cloud Monitoring (sync, thread-safe)."""
    try:
        from google.cloud import monitoring_v3
        from google.protobuf import timestamp_pb2

        client = _get_monitoring_client()
        project_name = f"projects/{project}"

        series = monitoring_v3.TimeSeries()
        series.metric.type = f"{_METRIC_PREFIX}/{name}"
        for k, v in labels.items():
            series.metric.labels[k] = v
        series.resource.type = "global"

        ts_proto = timestamp_pb2.Timestamp(
            seconds=int(ts),
            nanos=int((ts % 1) * 1e9),
        )
        interval = monitoring_v3.TimeInterval(end_time=ts_proto)

        point = monitoring_v3.Point(
            interval=interval,
            value=monitoring_v3.TypedValue(double_value=value),
        )
        series.points = [point]

        client.create_time_series(name=project_name, time_series=[series])
    except Exception as exc:
        logger.debug("Cloud Monitoring emit failed (non-fatal): %s", exc)

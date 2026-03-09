"""Cloud Monitoring metric emission — fire-and-forget."""

import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
_METRIC_PREFIX = "custom.googleapis.com/ui_navigator"


def emit(
    name: str,
    value: float = 1.0,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Emit a named metric data point.

    Always logs the metric as structured JSON.  If ``GOOGLE_CLOUD_PROJECT`` is
    configured, also writes to Cloud Monitoring in a daemon thread (best-effort,
    never raises).
    """
    logger.info(
        "metric",
        extra={"metric": name, "value": value, "labels": labels or {}},
    )
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", _PROJECT)
    if project:
        t = threading.Thread(
            target=_emit_to_cloud_monitoring,
            args=(name, value, labels or {}, time.time(), project),
            daemon=True,
        )
        t.start()


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

        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project}"

        series = monitoring_v3.TimeSeries()
        series.metric.type = f"{_METRIC_PREFIX}/{name}"
        for k, v in labels.items():
            series.metric.labels[k] = v
        series.resource.type = "global"

        ts_proto = timestamp_pb2.Timestamp()
        ts_proto.FromJsonString(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)))
        interval = monitoring_v3.TimeInterval(end_time=ts_proto)

        point = monitoring_v3.Point(
            interval=interval,
            value=monitoring_v3.TypedValue(double_value=value),
        )
        series.points = [point]

        client.create_time_series(name=project_name, time_series=[series])
    except Exception as exc:
        logger.debug("Cloud Monitoring emit failed (non-fatal): %s", exc)

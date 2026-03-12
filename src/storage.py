"""GCS screenshot upload -- returns a 7-day signed URL or None."""

import asyncio
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_SIGNED_URL_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

# Cached GCS client -- lazy-init on first use.
_gcs_client = None


def _get_gcs_client():
    """Return a cached GCS client, creating it on first call."""
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def _upload_sync(png_bytes: bytes, task_id: str, step: int, bucket_name: str) -> Optional[str]:
    """Blocking upload -- intended to run in a thread via asyncio.to_thread."""
    try:
        client = _get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob_name = f"screenshots/{task_id}/step_{step:04d}.png"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(png_bytes, content_type="image/png")
        url = blob.generate_signed_url(
            expiration=_SIGNED_URL_EXPIRY_SECONDS,
            method="GET",
            version="v4",
        )
        logger.debug("Screenshot uploaded to GCS: %s", blob_name)
        return url
    except Exception as exc:
        logger.warning("GCS screenshot upload failed (non-fatal): %s", exc)
        return None


def upload_screenshot(png_bytes: bytes, task_id: str, step: int) -> Optional[str]:
    """
    Upload a PNG screenshot to GCS and return a 7-day signed URL.

    Returns None if GCS_BUCKET env var is not configured or if the
    upload fails for any reason (failure is always non-fatal).
    """
    bucket_name = os.environ.get("GCS_BUCKET", "")
    if not bucket_name:
        return None

    # Validate task_id to prevent path traversal.
    if not _UUID_RE.match(task_id):
        logger.warning("Invalid task_id for GCS upload: %r", task_id)
        return None

    return _upload_sync(png_bytes, task_id, step, bucket_name)

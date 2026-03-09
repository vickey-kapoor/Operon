"""GCS screenshot upload — returns a 7-day signed URL or None."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_SIGNED_URL_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days


def upload_screenshot(png_bytes: bytes, task_id: str, step: int) -> Optional[str]:
    """
    Upload a PNG screenshot to GCS and return a 7-day signed URL.

    Returns ``None`` if ``GCS_BUCKET`` env var is not configured or if the
    upload fails for any reason (failure is always non-fatal).
    """
    bucket_name = os.environ.get("GCS_BUCKET", "")
    if not bucket_name:
        return None

    try:
        from google.cloud import storage

        client = storage.Client()
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

"""FilePorter: fetch a binary from a URL and upload it to a Google Drive folder.

Usage from the executor:
    result = await asyncio.to_thread(run_porter, url, folder_id)

Auth (checked in order):
    1. Path pointed to by GOOGLE_SERVICE_ACCOUNT_JSON env var (service-account key file).
    2. Application Default Credentials (gcloud auth application-default login, Workload Identity, etc.).

The optional 'drive' extra must be installed:
    pip install "operon[drive]"   # or: pip install google-api-python-client
"""

from __future__ import annotations

import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB resumable-upload chunks
_DEFAULT_MIME = "application/octet-stream"
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 120


@dataclass
class PorterResult:
    success: bool
    detail: str
    drive_file_id: str | None = None
    local_bytes: int = 0


def _derive_filename(url: str, content_type: str | None) -> str:
    """Return a best-effort filename from the URL path or Content-Type."""
    path = PurePosixPath(urlparse(url).path)
    name = path.name.strip()
    if name:
        return name
    ext = ""
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
    return f"download{ext}" if ext else "download.bin"


def _derive_mime(filename: str, content_type: str | None) -> str:
    if content_type:
        base = content_type.split(";")[0].strip()
        if base and base != "application/octet-stream":
            return base
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or _DEFAULT_MIME


def _build_drive_service():
    """Return an authenticated Google Drive v3 service resource."""
    try:
        from google.auth import (
            default as google_auth_default,  # type: ignore[import-untyped]
        )
        from google.oauth2 import service_account  # type: ignore[import-untyped]
        from googleapiclient.discovery import build  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "google-api-python-client is not installed. "
            'Run: pip install "operon[drive]"'
        ) from exc

    sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_path:
        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
    else:
        creds, _ = google_auth_default(
            scopes=["https://www.googleapis.com/auth/drive.file"]
        )

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def run_porter(url: str, folder_id: str) -> PorterResult:
    """Fetch *url* and upload the content to the Drive folder *folder_id*.

    This is a synchronous function; call it via asyncio.to_thread from async code.
    """
    # ── 1. Download ──────────────────────────────────────────────────────────
    try:
        resp = requests.get(
            url,
            stream=True,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            headers={"User-Agent": "Operon/FilePorter"},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return PorterResult(success=False, detail=f"Download failed: {exc}")

    content_type = resp.headers.get("Content-Type")
    filename = _derive_filename(url, content_type)
    mime_type = _derive_mime(filename, content_type)

    data = resp.content
    byte_count = len(data)
    logger.info("FilePorter: downloaded %d bytes (%s) from %s", byte_count, mime_type, url)

    # ── 2. Upload to Drive ───────────────────────────────────────────────────
    try:
        from googleapiclient.http import (
            MediaInMemoryUpload,  # type: ignore[import-untyped]
        )
    except ImportError:
        return PorterResult(
            success=False,
            detail=(
                "google-api-python-client is not installed. "
                'Run: pip install "operon[drive]"'
            ),
            local_bytes=byte_count,
        )

    try:
        service = _build_drive_service()
    except Exception as exc:
        return PorterResult(
            success=False,
            detail=f"Drive auth failed: {exc}",
            local_bytes=byte_count,
        )

    try:
        media = MediaInMemoryUpload(data, mimetype=mime_type, chunksize=_CHUNK_SIZE, resumable=True)
        file_meta = {"name": filename, "parents": [folder_id]}
        file_resource = (
            service.files()
            .create(body=file_meta, media_body=media, fields="id,name,size")
            .execute()
        )
        drive_file_id = file_resource.get("id", "")
        logger.info("FilePorter: uploaded '%s' → Drive file %s", filename, drive_file_id)
        return PorterResult(
            success=True,
            detail=f"Saved '{filename}' ({byte_count:,} bytes) to Drive folder {folder_id} (file id: {drive_file_id})",
            drive_file_id=drive_file_id,
            local_bytes=byte_count,
        )
    except Exception as exc:
        return PorterResult(
            success=False,
            detail=f"Drive upload failed: {exc}",
            local_bytes=byte_count,
        )

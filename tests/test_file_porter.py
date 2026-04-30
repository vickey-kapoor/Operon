from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from src.executor.desktop import DesktopExecutor
from src.models.common import FailureCategory
from src.models.policy import ActionType, AgentAction
from src.tools.file_porter import (
    PorterResult,
    _derive_filename,
    _derive_mime,
    run_porter,
)


class _Response:
    def __init__(self, *, content: bytes = b"hello", content_type: str = "text/plain") -> None:
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self) -> None:
        return None


def _install_google_http_module(monkeypatch: pytest.MonkeyPatch, media_cls) -> None:
    parent = ModuleType("googleapiclient")
    http_mod = ModuleType("googleapiclient.http")
    http_mod.MediaInMemoryUpload = media_cls
    parent.http = http_mod
    monkeypatch.setitem(sys.modules, "googleapiclient", parent)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http_mod)


def test_derive_filename_prefers_url_path() -> None:
    assert _derive_filename("https://example.com/files/report.pdf", "application/pdf") == "report.pdf"


def test_derive_filename_falls_back_to_content_type_extension() -> None:
    assert _derive_filename("https://example.com/", "image/png") == "download.png"


def test_derive_mime_prefers_specific_content_type() -> None:
    assert _derive_mime("report.bin", "application/pdf; charset=utf-8") == "application/pdf"


def test_derive_mime_falls_back_to_filename_guess() -> None:
    assert _derive_mime("report.pdf", None) == "application/pdf"


def test_run_porter_returns_download_failure() -> None:
    with patch("src.tools.file_porter.requests.get", side_effect=requests.RequestException("network down")):
        result = run_porter("https://example.com/file.txt", "folder-1")

    assert result.success is False
    assert "Download failed" in result.detail


def test_run_porter_returns_missing_google_dependency_after_download(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMediaUpload:
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError("missing google client")

    _install_google_http_module(monkeypatch, FakeMediaUpload)

    with patch("src.tools.file_porter.requests.get", return_value=_Response(content=b"abc")):
        result = run_porter("https://example.com/file.txt", "folder-1")

    assert result.success is False
    assert "google-api-python-client is not installed" in result.detail
    assert result.local_bytes == 3


def test_run_porter_returns_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMediaUpload:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    _install_google_http_module(monkeypatch, FakeMediaUpload)

    with (
        patch("src.tools.file_porter.requests.get", return_value=_Response(content=b"hello")),
        patch("src.tools.file_porter._build_drive_service", side_effect=RuntimeError("bad credentials")),
    ):
        result = run_porter("https://example.com/file.txt", "folder-1")

    assert result.success is False
    assert "Drive auth failed" in result.detail
    assert result.local_bytes == 5


def test_run_porter_returns_upload_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMediaUpload:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _CreateCall:
        def execute(self):
            raise RuntimeError("upload boom")

    class _FilesResource:
        def create(self, **_kwargs):
            return _CreateCall()

    service = SimpleNamespace(files=lambda: _FilesResource())
    _install_google_http_module(monkeypatch, FakeMediaUpload)

    with (
        patch("src.tools.file_porter.requests.get", return_value=_Response(content=b"payload")),
        patch("src.tools.file_porter._build_drive_service", return_value=service),
    ):
        result = run_porter("https://example.com/file.txt", "folder-1")

    assert result.success is False
    assert "Drive upload failed" in result.detail
    assert result.local_bytes == 7


def test_run_porter_success(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeMediaUpload:
        def __init__(self, data, mimetype, chunksize, resumable) -> None:
            created["data"] = data
            created["mimetype"] = mimetype
            created["chunksize"] = chunksize
            created["resumable"] = resumable

    class _CreateCall:
        def execute(self):
            return {"id": "drive-123", "name": "report.txt", "size": "5"}

    class _FilesResource:
        def create(self, **kwargs):
            created["create_kwargs"] = kwargs
            return _CreateCall()

    service = SimpleNamespace(files=lambda: _FilesResource())
    _install_google_http_module(monkeypatch, FakeMediaUpload)

    with (
        patch("src.tools.file_porter.requests.get", return_value=_Response(content=b"hello")),
        patch("src.tools.file_porter._build_drive_service", return_value=service),
    ):
        result = run_porter("https://example.com/report.txt", "folder-1")

    assert result.success is True
    assert result.drive_file_id == "drive-123"
    assert result.local_bytes == 5
    assert created["mimetype"] == "text/plain"
    assert created["resumable"] is True
    assert created["create_kwargs"]["body"] == {"name": "report.txt", "parents": ["folder-1"]}


@pytest.mark.asyncio
async def test_desktop_executor_file_porter_success() -> None:
    executor = DesktopExecutor.__new__(DesktopExecutor)
    action = AgentAction(
        action_type=ActionType.FILE_PORTER,
        url="https://example.com/report.txt",
        text="folder-1",
    )

    with patch(
        "src.tools.file_porter.run_porter",
        return_value=PorterResult(success=True, detail="saved to drive"),
    ):
        result = await DesktopExecutor._exec_file_porter(executor, action)

    assert result.success is True
    assert result.detail == "saved to drive"


@pytest.mark.asyncio
async def test_desktop_executor_file_porter_failure_maps_to_execution_error() -> None:
    executor = DesktopExecutor.__new__(DesktopExecutor)
    action = AgentAction(
        action_type=ActionType.FILE_PORTER,
        url="https://example.com/report.txt",
        text="folder-1",
    )

    with patch(
        "src.tools.file_porter.run_porter",
        return_value=PorterResult(success=False, detail="upload failed"),
    ):
        result = await DesktopExecutor._exec_file_porter(executor, action)

    assert result.success is False
    assert result.detail == "upload failed"
    assert result.failure_category is FailureCategory.EXECUTION_ERROR

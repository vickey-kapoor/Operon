import os
from uuid import uuid4

import pytest

from operon import cli
from operon.core.logging_utils import url_origin_for_log
from operon.store.background_writer import BackgroundWriter
from operon.store.cleanup import cleanup_browser_recordings


@pytest.mark.parametrize(("url", "expected"), [
    ("https://user:password@example.com:8443/private?token=value#fragment", "https://example.com:8443"),
    ("http://[::1]:8080/path", "http://[::1]:8080"),
    ("data:text/plain,private", "<unavailable URL origin>"),
    ("https://[malformed", "<unavailable URL origin>"),
    (None, None),
])
def test_log_url_contains_only_origin(url, expected):
    assert url_origin_for_log(url) == expected


def test_browser_video_retention(tmp_path):
    root = tmp_path / "browser"
    runs = tmp_path / "runs"

    def recording(*, old, recent_state=False):
        run_id = str(uuid4())
        video = root / run_id / "session_video" / "session.webm"
        video.parent.mkdir(parents=True)
        video.touch()
        if old:
            os.utime(video, (1, 1))
        if recent_state:
            state = runs / run_id / "state.json"
            state.parent.mkdir(parents=True)
            state.touch()
        return video

    orphan = recording(old=True)
    recent = recording(old=False)
    active = recording(old=True, recent_state=True)
    assert cleanup_browser_recordings(root, runs, -1) == 0
    assert orphan.exists()
    assert cleanup_browser_recordings(root, runs, 14) == 1
    assert not orphan.exists()
    assert recent.exists() and active.exists()


@pytest.mark.parametrize("method", ["append", "enqueue"])
def test_artifact_failure_warns(tmp_path, caplog, method):
    parent = tmp_path / "file"
    parent.touch()
    writer = BackgroundWriter(sync=True)
    getattr(writer, method)(parent / "artifact.json", "data")
    assert any(record.levelname == "WARNING" and "artifact.json" in record.message for record in caplog.records)


@pytest.mark.parametrize(("args", "level"), [([], "info"), (["--log-level", "debug"], "debug")])
def test_serve_sets_application_and_uvicorn_level(monkeypatch, caplog, args, level):
    import logging

    import uvicorn

    monkeypatch.setenv("API_KEYS", "test-key")
    caplog.set_level(logging.WARNING, logger="operon")
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    assert cli.main(["serve", *args]) == 0
    assert calls[0]["log_level"] == level
    assert logging.getLogger("operon").level == getattr(logging, level.upper())

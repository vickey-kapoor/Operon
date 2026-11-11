"""The live view check's bookkeeping (evals/live_view_check.py), offline."""

from __future__ import annotations

from evals import live_view_check


def test_only_binary_frames_count_and_takeover_frames_are_counted_apart() -> None:
    watch = live_view_check.Watch()
    watch.on_frame(b"\xff\xd8jpeg")
    watch.on_frame('{"type": "run_started"}')  # an event, not a frame
    watch.in_takeover = True
    watch.on_frame(bytearray(b"\xff\xd8jpeg"))

    assert watch.frames == 2
    assert watch.frames_in_takeover == 1


def test_main_needs_a_key(monkeypatch, capsys) -> None:
    import pytest

    monkeypatch.delenv("API_KEYS", raising=False)
    with pytest.raises(SystemExit):
        live_view_check.main([])
    assert "--key" in capsys.readouterr().err

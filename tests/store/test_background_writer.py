"""Tests for the background artifact writer."""

from pathlib import Path

import pytest

from operon.store.background_writer import BackgroundWriter


@pytest.mark.asyncio
async def test_enqueue_accepts_str_path_in_async_mode(tmp_path: Path) -> None:
    # Debug-artifact callers pass str paths; these used to be dropped silently.
    writer = BackgroundWriter()
    target = tmp_path / "step_1" / "verification_prompt.txt"

    writer.enqueue(str(target), "prompt")
    await writer.flush()

    assert target.read_text(encoding="utf-8") == "prompt"


def test_enqueue_accepts_str_path_in_sync_mode(tmp_path: Path) -> None:
    writer = BackgroundWriter(sync=True)
    target = tmp_path / "nested" / "out.json"

    writer.enqueue(str(target), "{}")

    assert target.read_text(encoding="utf-8") == "{}"


@pytest.mark.asyncio
async def test_append_accepts_str_path(tmp_path: Path) -> None:
    writer = BackgroundWriter()
    target = tmp_path / "run.jsonl"

    writer.append(str(target), "a\n")
    await writer.flush()
    writer.append(str(target), "b\n")
    await writer.flush()

    assert target.read_text(encoding="utf-8") == "a\nb\n"

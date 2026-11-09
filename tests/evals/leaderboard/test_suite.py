import logging

import pytest

from evals.leaderboard.suite import (
    SuiteMismatchError,
    check_hashes_match,
    hash_directory,
    hash_text,
)


def test_hash_changes_when_a_task_changes(tmp_path):
    (tmp_path / "a.json").write_text('{"id": "a"}', encoding="utf-8")
    before = hash_directory(tmp_path)
    (tmp_path / "a.json").write_text('{"id": "a", "x": 1}', encoding="utf-8")
    assert hash_directory(tmp_path) != before


def test_hash_ignores_line_endings_and_file_order(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_bytes(b"x\r\ny\r\n")
    (first / "b.txt").write_bytes(b"z")
    (second / "b.txt").write_bytes(b"z")
    (second / "a.txt").write_bytes(b"x\ny\n")
    assert hash_directory(first) == hash_directory(second)


def test_hash_includes_file_names(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("same", encoding="utf-8")
    (second / "b.txt").write_text("same", encoding="utf-8")
    assert hash_directory(first) != hash_directory(second)


def test_missing_directory_hashes_like_an_empty_one(tmp_path):
    (tmp_path / "empty").mkdir()
    assert hash_directory(tmp_path / "missing") == hash_directory(tmp_path / "empty")


def test_hash_text_is_stable():
    assert hash_text("task") == hash_text("task")
    assert hash_text("task") != hash_text("task2")


def test_matching_hashes_pass():
    assert check_hashes_match({"working": "a", "held_out": "b"}, {"working": "a", "held_out": "b"}) == []


def test_changed_hash_is_refused():
    with pytest.raises(SuiteMismatchError, match="working hash"):
        check_hashes_match({"working": "old", "held_out": "b"}, {"working": "new", "held_out": "b"})


def test_missing_name_counts_as_a_change():
    with pytest.raises(SuiteMismatchError, match="held_out"):
        check_hashes_match({"working": "a"}, {"working": "a", "held_out": "b"})


def test_changed_hash_allowed_on_purpose_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="evals.leaderboard.suite"):
        changed = check_hashes_match({"working": "x", "held_out": "y"}, {"working": "a", "held_out": "b"}, allow_change=True)
    assert changed == ["held_out", "working"]
    assert "held_out and working" in caplog.text


def test_nul_bytes_in_a_file_cannot_imitate_another_file(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "a").write_bytes(b"x\0b\0")
    (two / "a").write_bytes(b"x")
    (two / "b").write_bytes(b"")
    assert hash_directory(one) != hash_directory(two)

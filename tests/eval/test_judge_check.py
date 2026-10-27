"""Offline checks for the judge check: labels file, scoring and the judging loop."""

from __future__ import annotations

import json

from tests.eval.judge_check import (
    Graded,
    Label,
    check_row,
    judge_labels,
    read_labels,
    score,
    template,
)
from tests.eval.live_bench import Verdict


def test_template_lists_judged_tasks_with_their_kept_screenshots(tmp_path) -> None:
    report = tmp_path / "full-20261024T000000Z.json"
    screens = tmp_path / "full-20261024T000000Z"
    screens.mkdir()
    (screens / "click_link.png").write_bytes(b"png")
    report.write_text(json.dumps({"results": [
        {"task": "click_link", "outcome": "success", "final_screenshot": "/runner/a.png"},
        {"task": "go_back", "outcome": "false_success", "final_screenshot": "/runner/b.png"},
        {"task": "hn_new", "outcome": "timeout", "final_screenshot": "/runner/c.png"},
        {"task": "dropdown", "outcome": "failure", "final_screenshot": "-"},
    ]}))

    lines = template(report)

    assert [(line["task"], line["judge_then"]) for line in lines] == [("click_link", True), ("go_back", False)]
    assert lines[0]["screenshot"] == str(screens / "click_link.png")
    assert lines[1]["screenshot"] == "/runner/b.png"
    assert all(line["label"] is None for line in lines)


def test_read_labels_skips_unlabelled_lines(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"task": "a", "screenshot": "a.png", "label": true}\n\n'
        '{"task": "b", "screenshot": "b.png", "label": null}\n'
        '{"task": "c", "screenshot": "c.png", "label": false}\n'
    )
    labels, unlabelled = read_labels(path)
    assert labels == [Label("a", "a.png", True), Label("c", "c.png", False)]
    assert unlabelled == 1


def test_score_counts_agreement_and_self_agreement() -> None:
    graded = [
        Graded(Label("a", "-", True), (True, True, True)),    # right
        Graded(Label("b", "-", False), (True, True, False)),  # false pass, inconsistent
        Graded(Label("c", "-", False), (False, False, False)),
        Graded(Label("d", "-", True), (False, False, False)),  # missed pass
        Graded(Label("e", "-", True), (None, None, None)),     # no verdict
    ]
    summary = score(graded)
    assert (summary["true_positive"], summary["false_positive"]) == (1, 1)
    assert (summary["true_negative"], summary["false_negative"]) == (1, 1)
    assert summary["no_verdict"] == 1
    assert summary["accuracy"] == 0.5 and summary["precision"] == 0.5 and summary["recall"] == 0.5
    assert summary["self_agreement"] == 0.8


async def test_judge_labels_repeats_and_skips_missing_screens(tmp_path) -> None:
    shot = tmp_path / "s.png"
    shot.write_bytes(b"png")
    calls: list[str] = []

    async def judge(task, screenshot):
        calls.append(task.name)
        return Verdict(True, "ok")

    graded = await judge_labels(
        [Label("click_link", str(shot), True), Label("click_link", str(tmp_path / "gone.png"), True),
         Label("not_a_task", str(shot), False)],
        judge, repeat=2,
    )
    assert calls == ["click_link", "click_link"]
    assert [g.verdict for g in graded] == [True, None, None]


def test_check_row_keeps_disagreements_and_drops_local_paths() -> None:
    graded = [
        Graded(Label("click_link", "/home/me/screens/click_link.png", True), (True, True)),
        Graded(Label("hn_new", "/home/me/screens/hn_new.png", False), (True, True)),
        Graded(Label("go_back", "/home/me/screens/go_back.png", True), (None,)),
    ]

    row = check_row(graded, score(graded), "gemini-3.5-flash", 2)

    assert row["labels"] == 3
    assert row["repeats"] == 2
    assert row["false_passes"] == 1
    assert row["false_fails"] == 0
    assert row["precision"] == 0.5
    assert row["disagreements"] == [
        {"task": "hn_new", "screenshot": "hn_new.png", "label": False, "judge": True},
        {"task": "go_back", "screenshot": "go_back.png", "label": True, "judge": None},
    ]

"""Run videos (KAN-185): offline, from synthetic run folders."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import cv2
import pytest
from PIL import Image

from evals import run_video


def _run(root, name: str = "r1", steps: int = 3, *, jsonl: bool = True, intent: str = "Find the population of Paris"):
    run = root / name
    for i in range(1, steps + 1):
        step = run / f"step_{i}"
        step.mkdir(parents=True)
        Image.new("RGB", (1281, 801), (i * 20, 40, 90)).save(step / "before.png")
    Image.new("RGB", (1281, 801), (0, 200, 0)).save(run / f"step_{steps}" / "after.png")
    (run / "state.json").write_text(json.dumps({"intent": intent}), encoding="utf-8")
    if jsonl:
        lines = [
            {"step_index": 1, "executed_action": {"action": {"action_type": "click", "x": 10, "y": 20}}},
            {"step_index": 2, "executed_action": {"action": {"action_type": "type", "text": "Paris"}}},
            "not json",
            {"step_index": 3, "executed_action": {}},
        ]
        (run / "run.jsonl").write_text(
            "\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines), encoding="utf-8"
        )
    return run


def _info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
    finally:
        cap.release()


def test_make_video_writes_a_playable_mp4(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    run = _run(tmp_path)
    out = tmp_path / "out" / "r1.mp4"
    written = run_video.make_video(run, out, width=640, seconds_per_step=1.0, last_seconds=3.0, verdict="PASS")
    # 3 before frames x 2 + final after x 6 at 2 fps
    assert written == 3 * 2 + 6
    count, width, height = _info(out)
    assert abs(count - written) <= 2
    assert width == 640 and height % 2 == 0 and width % 2 == 0


def test_captions_use_actions_and_survive_a_missing_log(tmp_path) -> None:
    run = _run(tmp_path)
    assert run_video.actions_of(run) == {1: "click at (10, 20)", 2: 'type "Paris"'}
    bare = _run(tmp_path, "bare", jsonl=False)
    assert run_video.actions_of(bare) == {}


def test_make_video_without_run_log_still_works(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    run = _run(tmp_path, jsonl=False)
    assert run_video.make_video(run, tmp_path / "v.mp4", width=320) > 0


def test_no_screenshots_raises(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        run_video.make_video(empty, tmp_path / "x.mp4")


def test_save_videos_skips_a_broken_run_and_names_by_verdict(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    good = _run(tmp_path, "good")
    broken = tmp_path / "broken"
    broken.mkdir()
    items = [
        ("../weird name", broken, "t", "failure"),
        ("notepad hello", good, "Type hello", "success"),
    ]
    assert run_video.save_videos(items, tmp_path / "videos") == 1
    assert [p.name for p in (tmp_path / "videos").iterdir()] == ["notepad-hello-success.mp4"]
    assert "no video" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_ffmpeg_reencodes_to_h264(tmp_path) -> None:
    run = _run(tmp_path)
    out = tmp_path / "h264.mp4"
    run_video.make_video(run, out, width=320)
    assert b"avc1" in out.read_bytes()[:4096]
    assert _info(out)[0] > 0


def test_cli_run_and_all(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    root = tmp_path / "runs"
    root.mkdir()
    _run(root, "a")
    _run(root, "b")
    (root / "no-shots").mkdir()
    assert run_video.main(["--all", str(root), "--out-dir", str(tmp_path / "vids"), "--width", "320"]) == 0
    assert sorted(p.name for p in (tmp_path / "vids").iterdir()) == ["a.mp4", "b.mp4"]
    assert run_video.main(["--run", str(root / "a"), "--out", str(tmp_path / "one.mp4"), "--width", "320"]) == 0
    assert (tmp_path / "one.mp4").is_file()
    assert run_video.main(["--run", str(root / "no-shots"), "--out", str(tmp_path / "n.mp4")]) == 1
    assert "Wrote 2 of 2" in capsys.readouterr().out


def test_live_bench_never_makes_a_video_for_a_held_out_task(tmp_path, monkeypatch) -> None:
    from evals import live_bench

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    runs = {"r-open": _run(tmp_path, "open"), "r-held": _run(tmp_path, "held")}
    monkeypatch.setattr(live_bench, "_run_dir", lambda run_id: runs.get(run_id, tmp_path / "missing"))
    season = SimpleNamespace(held_out=[(0, SimpleNamespace(name="secret-task"))])
    results = [
        SimpleNamespace(task="open-task", attempt=1, run_id="r-open", outcome="success"),
        SimpleNamespace(task="secret-task", attempt=1, run_id="r-held", outcome="success"),
        SimpleNamespace(task="open-task", attempt=2, run_id="r-open", outcome="failure"),
        SimpleNamespace(task="never", attempt=1, run_id=None, outcome="error"),
    ]
    shown = live_bench.shown_results(results, season)
    assert [r.task for r in shown] == ["open-task", "open-task", "never"]
    assert live_bench.save_task_videos(shown, tmp_path / "videos") == 2
    names = sorted(p.name for p in (tmp_path / "videos").iterdir())
    assert names == ["open-task-success.mp4", "open-task-try2-failure.mp4"]
    assert live_bench.shown_results(results, None) == results
    assert live_bench.build_parser().parse_args(["--videos"]).videos is True


def test_write_videos_only_gets_the_shown_results(tmp_path, monkeypatch) -> None:
    from evals import live_bench

    captured = {}
    monkeypatch.setattr(live_bench, "save_task_videos", lambda results, target: captured.update(r=results, t=target) or 1)
    season = SimpleNamespace(held_out=[(0, SimpleNamespace(name="secret"))])
    results = [SimpleNamespace(task="open"), SimpleNamespace(task="secret")]
    assert live_bench.write_videos(results, season, tmp_path) == 1
    assert [r.task for r in captured["r"]] == ["open"]

    def boom(results, target):
        raise ImportError("no cv2")

    monkeypatch.setattr(live_bench, "save_task_videos", boom)
    assert live_bench.write_videos(results, season, tmp_path) == 0  # never raises


def test_mixed_frame_sizes_are_all_kept(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    run = _run(tmp_path, steps=3)
    Image.new("RGB", (1921, 1201), (9, 9, 9)).save(run / "step_2" / "before.png")
    written = run_video.make_video(run, tmp_path / "m.mp4", width=640)
    assert written == 12
    assert abs(_info(tmp_path / "m.mp4")[0] - written) <= 1


def test_cli_task_and_verdict(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    run = _run(tmp_path)
    rc = run_video.main(["--run", str(run), "--out", str(tmp_path / "c.mp4"), "--width", "320",
                         "--task", "chrome: abc", "--verdict", "pass"])
    assert rc == 0 and (tmp_path / "c.mp4").is_file()


def test_osworld_videos_run_under_the_operon_python(tmp_path) -> None:
    import os

    from evals import osworld

    out = tmp_path / "bench"
    (out / "osworld-runs" / "run-a").mkdir(parents=True)
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        ok = cmd[cmd.index("--out") + 1].endswith("abcdef12-pass.mp4")
        return SimpleNamespace(returncode=0 if ok else 1, stdout="", stderr="boom")

    results = [
        {"id": "abcdef12-0000", "domain": "chrome", "passed": True, "run_id": "run-a"},
        {"id": "99999999-0000", "domain": "gimp", "passed": False, "run_id": "run-a"},
        {"id": "no-run-id", "domain": "os", "passed": False, "run_id": None},
    ]
    assert osworld.save_task_videos(results, out, "/venv/python", run=fake_run) == 1
    assert len(calls) == 2
    cmd, kw = calls[0]
    assert cmd[:3] == ["/venv/python", "-m", "evals.run_video"]
    assert cmd[cmd.index("--verdict") + 1] == "pass"
    assert "evals" in os.listdir(kw["cwd"])

    def raising(cmd, **kw):
        raise OSError("no python")

    assert osworld.save_task_videos(results, out, "/nope", run=raising) == 0

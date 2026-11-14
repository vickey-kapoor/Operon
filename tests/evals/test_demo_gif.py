"""Demo GIF maker (KAN-141): offline, from fake run folders."""

from __future__ import annotations

import json
import os

import pytest
from PIL import Image

from evals import demo_gif


def _run(root, name: str, steps: int, *, after_last: bool = True, intent: str = "Find the population of Paris"):
    run = root / name
    for i in range(1, steps + 1):
        step = run / f"step_{i}"
        step.mkdir(parents=True)
        Image.new("RGB", (1280, 800), (i * 20, 40, 90)).save(step / "before.png")
        if after_last and i == steps:
            Image.new("RGB", (1280, 800), (0, 200, 0)).save(step / "after.png")
    (run / "state.json").write_text(json.dumps({"intent": intent}), encoding="utf-8")
    return run


def test_frames_are_in_step_order_with_the_final_after(tmp_path) -> None:
    run = _run(tmp_path, "r1", 11)
    shots = demo_gif.frames_of(run)
    assert [p.parent.name for p in shots[:3]] == ["step_1", "step_2", "step_3"]
    assert shots[9].parent.name == "step_10"  # numeric order, not step_1, step_10, step_11, step_2
    assert shots[-1].name == "after.png" and shots[-1].parent.name == "step_11"
    assert len(shots) == 12


def test_make_gif_scales_captions_and_holds_the_last_frame(tmp_path) -> None:
    run = _run(tmp_path, "r1", 3)
    out = tmp_path / "out" / "demo.gif"
    count = demo_gif.make_gif(run, out, width=640, frame_ms=500, last_ms=2000)
    assert count == 4
    with Image.open(out) as gif:
        assert gif.n_frames == 4
        assert gif.width == 640
        assert gif.height == 400 + demo_gif._CAPTION_H
        gif.seek(3)
        assert gif.info["duration"] == 2000


def test_no_caption_keeps_the_frame_size(tmp_path) -> None:
    run = _run(tmp_path, "r1", 2, after_last=False)
    out = tmp_path / "demo.gif"
    assert demo_gif.make_gif(run, out, width=640, caption="") == 2
    with Image.open(out) as gif:
        assert (gif.width, gif.height) == (640, 400)


def test_latest_run_and_cli(tmp_path, monkeypatch, capsys) -> None:
    old = _run(tmp_path, "old", 1)
    new = _run(tmp_path, "new", 2)
    os.utime(old / "state.json", (1, 1))
    assert demo_gif.latest_run(tmp_path) == new
    monkeypatch.setenv("OPERON_RUNS_ROOT", str(tmp_path))
    out = tmp_path / "x.gif"
    assert demo_gif.main(["--latest", "--out", str(out)]) == 0
    assert "3 frames" in capsys.readouterr().out


def test_an_empty_run_is_an_error(tmp_path, capsys) -> None:
    (tmp_path / "empty").mkdir()
    assert demo_gif.main(["--run", str(tmp_path / "empty"), "--out", str(tmp_path / "x.gif")]) == 1
    assert "no step screenshots" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError):
        demo_gif.latest_run(tmp_path / "missing")

"""Offline checks for the grounding eval: sampling, scoring and the run loop (no network, no model)."""

from __future__ import annotations

import io
import json
import zipfile

import httpx
from PIL import Image

from evals import grounding
from evals.grounding import (
    Sample,
    first_point,
    pro_samples,
    score,
    stratified,
    v2_samples,
)
from operon.models.policy import ActionType, AgentAction, PolicyDecision


def _png(width: int = 200, height: int = 100) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _sample(i: int, data_type: str = "text", group: str = "") -> Sample:
    return Sample(f"s{i}", "pro", f"do {i}", f"img{i}.png", (10, 10, 50, 50), data_type, "app", group)


def test_stratified_is_balanced_and_stable() -> None:
    pool = [_sample(i, "text" if i % 4 else "icon") for i in range(40)]  # 10 icons, 30 text

    picked = stratified(pool, 10, lambda s: s.data_type)

    assert [s.data_type for s in picked].count("icon") == 5
    assert picked == stratified(list(reversed(pool)), 10, lambda s: s.data_type)
    # A small stratum runs out and the others fill the rest.
    assert len(stratified(pool, 30, lambda s: s.data_type)) == 30
    assert len(stratified(pool, 100, lambda s: s.data_type)) == 40


def test_dataset_rows_become_samples_with_corner_boxes() -> None:
    (v2,) = v2_samples([{"img_filename": "a.png", "bbox": [10, 20, 30, 40], "instruction": "x", "data_type": "icon",
                         "data_source": "windows"}], "desktop")
    assert v2.bbox == (10, 20, 40, 60) and v2.set == "v2-desktop" and v2.id == "v2-desktop-0"
    (pro,) = pro_samples([{"id": "excel_0", "img_filename": "excel/a.png", "bbox": [1, 2, 3, 4], "instruction": "y",
                           "ui_type": "text", "application": "excel", "platform": "macos", "group": "Office"}])
    assert pro.bbox == (1, 2, 3, 4) and pro.image == "images/excel/a.png" and pro.group == "Office"


def test_score_and_first_point() -> None:
    assert score((30, 30), (10, 10, 50, 50)) == ("hit", 0.0)
    assert score((10, 50), (10, 10, 50, 50))[0] == "hit"  # edges count
    assert score((60, 30), (10, 10, 50, 50)) == ("miss", 30.0)
    assert score(None, (10, 10, 50, 50)) == ("no_click", None)

    click = AgentAction(action_type=ActionType.CLICK, x=5, y=6)
    assert first_point(click) == (5, 6)
    batch = AgentAction(action_type=ActionType.BATCH, actions=[AgentAction(action_type=ActionType.WAIT, wait_ms=100), click])
    assert first_point(batch) == (5, 6)
    assert first_point(AgentAction(action_type=ActionType.WAIT, wait_ms=100)) is None


def test_range_reader_lets_zipfile_read_one_member() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("screenspotv2_image/a.png", b"first")
        archive.writestr("screenspotv2_image/b.png", b"second" * 1000)
    blob = buffer.getvalue()

    def serve(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(blob))})
        start, end = request.headers["range"].removeprefix("bytes=").split("-")
        return httpx.Response(206, content=blob[int(start): int(end) + 1])

    http = httpx.Client(transport=httpx.MockTransport(serve))
    reader = grounding.RangeReader(http, "https://example.test/x.zip")
    archive = zipfile.ZipFile(io.BufferedReader(reader, buffer_size=64))
    assert archive.read("screenspotv2_image/b.png") == b"second" * 1000


class FakeDataset:
    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples

    def samples(self, set_name: str) -> list[Sample]:
        return [s for s in self._samples if s.set == set_name]

    def image(self, sample: Sample) -> bytes:
        return _png()


class FakeBackend:
    """Clicks where the sample's instruction says (``x,y``), or proposes no click."""

    def __init__(self, sample: Sample, usage) -> None:
        self.sample, self.usage = sample, usage

    async def perceive(self, screenshot, state):
        assert screenshot.width == 200 and screenshot.height == 100
        assert self.sample.instruction in state.intent
        return None

    async def choose_action(self, state, perception):
        if self.sample.instruction == "boom":
            raise RuntimeError("API down")
        if self.sample.instruction == "none":
            action = {"action_type": "wait", "wait_ms": 1000}
        else:
            x, y = map(int, self.sample.instruction.split(","))
            action = {"action_type": "click", "x": x, "y": y}
        return PolicyDecision.model_validate({"action": action, "rationale": "r", "confidence": 0.5, "active_subgoal": "g"})

    def latest_debug_artifacts(self):
        return type("Artifacts", (), {"usage": self.usage})()


async def test_run_scores_every_sample_and_saves_a_report(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    usage = type("Usage", (), {"estimated_cost_usd": 0.01, "input_tokens": 100, "output_tokens": 5,
                               "thoughts_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})()
    box = (10, 10, 50, 50)
    samples = [
        Sample("a", "v2-web", "30,30", "a.png", box, "text", "web"),
        Sample("b", "v2-web", "90,90", "b.png", box, "icon", "web"),
        Sample("c", "v2-desktop", "none", "c.png", box, "icon", "windows"),
        Sample("d", "v2-desktop", "boom", "d.png", box, "text", "windows"),
    ]
    args = grounding.build_parser().parse_args(["--per-set", "5", "--out-dir", str(tmp_path / "out"), "--set", "v2-web",
                                                "--set", "v2-desktop", "--backend", "claude_computer_use"])

    code = await grounding.run(args, make_backend=lambda s: FakeBackend(s, usage), dataset=FakeDataset(samples))

    assert code == 0
    (saved,) = (tmp_path / "out").glob("grounding-*.json")
    report = json.loads(saved.read_text())
    assert "results" not in report  # the dashboard importer must not take it for a live_bench report
    outcomes = {s["id"]: s["outcome"] for s in report["samples"]}
    assert outcomes == {"a": "hit", "b": "miss", "c": "no_click", "d": "error"}
    summary = report["summary"]
    assert summary["hit"] == 1 and summary["accuracy"] == round(1 / 3, 3)  # errors are not scored
    assert summary["by_set"]["v2-web"]["by_type"]["text"]["accuracy"] == 1.0
    assert summary["cost_usd"] == 0.03  # the errored sample made no call
    assert not (tmp_path / ".var" / "grounding").exists() or not any((tmp_path / ".var" / "grounding").iterdir())


async def test_run_stops_at_the_cost_cap(tmp_path) -> None:
    usage = type("Usage", (), {"estimated_cost_usd": 1.0})()
    samples = [Sample(f"s{i}", "pro", "30,30", "x.png", (10, 10, 50, 50), "text", "app") for i in range(5)]

    results, stopped = await grounding.run_samples(
        samples, FakeDataset(samples), lambda s: FakeBackend(s, usage), tmp_path, concurrency=1, max_cost_usd=2.0,
    )

    assert len(results) == 2
    assert stopped == {"limit_usd": 2.0, "spent_usd": 2.0, "tasks_planned": 5, "tasks_run": 2}

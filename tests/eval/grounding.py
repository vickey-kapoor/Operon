"""Grounding eval: does the agent click the right spot? (ScreenSpot-v2 and ScreenSpot-Pro)

Each sample is one screenshot, one instruction ("close this window") and the
target's box. The sample goes through the backend Operon actually runs, with the
same prompt, screenshot downscaling and coordinate scaling as a live step, and the
first click it proposes is scored as a hit when it lands inside the box. One model
call per sample: no browser, no API server, no desktop.

    .venv/bin/python -m tests.eval.grounding [--per-set 100] [--set v2-web] [--backend claude_computer_use]

Sets (a fixed, stratified sample of each, the same on every run):

- ``v2-desktop`` and ``v2-web``: ScreenSpot-v2 (OS-Copilot/ScreenSpot-v2), half text
  targets and half icons.
- ``pro``: ScreenSpot-Pro (likaixin/ScreenSpot-Pro), high-resolution screens of
  professional apps (CAD, IDEs, Office, creative tools), spread across its groups.

Claude runs web samples through the browser backend and desktop and Pro samples
through the desktop backend; Gemini (``--backend computer_use``) uses its browser
Computer Use backend for all of them, since Gemini has no desktop computer-use
backend. The datasets are pinned to a revision and their files cached under
``.var/grounding/cache``; ScreenSpot-v2's images come out of its 1.3 GB zip by
HTTP range requests, so only the sampled screenshots are downloaded.

Outcomes per sample: ``hit``, ``miss`` (clicked outside the box), ``no_click``
(the first turn proposed no click) and ``error``. The report (``.var/bench/grounding-<stamp>.json``)
gives accuracy per set and target type, and the cost. Spend is capped like the
live benchmark (``--max-cost-usd`` / ``MAX_RUN_COST_USD``, default $5).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx

from operon.models.capture import CaptureFrame
from operon.models.common import RunStatus
from operon.models.policy import AgentAction
from operon.models.state import AgentState

HF = "https://huggingface.co"
V2_REPO, V2_REVISION = "OS-Copilot/ScreenSpot-v2", "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c"
PRO_REPO, PRO_REVISION = "likaixin/ScreenSpot-Pro", "210e78d3844251110bff86c95835ebd37a6930fa"
V2_ZIP = "screenspotv2_image.zip"
SETS = ("v2-desktop", "v2-web", "pro")
OUTCOMES = ("hit", "miss", "no_click", "error")
BACKENDS = ("claude_computer_use", "computer_use")
DEFAULT_PER_SET = 100
DEFAULT_MAX_COST_USD = 5.0
BUDGET_EXIT_CODE = 3
SEED = "operon-grounding-v1"
CACHE_DIR = Path(".var/grounding/cache")
BENCH_DIR = Path(os.getenv("OPERON_BENCH_DIR", ".var/bench"))

INTENT = 'Click the element on the screen that matches this instruction: "{instruction}"'


@dataclass(frozen=True)
class Sample:
    id: str
    set: str
    instruction: str
    image: str  # path inside the dataset
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in screenshot pixels
    data_type: str  # text or icon
    source: str  # app or platform
    group: str = ""


@dataclass
class SampleResult:
    id: str
    set: str
    data_type: str
    source: str
    instruction: str
    outcome: str
    x: int | None
    y: int | None
    bbox: list[float]
    distance_px: float | None  # from the click to the box's centre
    image_width: int
    image_height: int
    seconds: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    detail: str


# ---------- dataset ----------


def hf_url(repo: str, revision: str, path: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/{revision}/{path}"


def _rank(key: str) -> str:
    return hashlib.sha1(f"{SEED}:{key}".encode()).hexdigest()


def stratified(samples: list[Sample], n: int, stratum) -> list[Sample]:
    """``n`` samples spread evenly over the strata, the same ones on every call.

    Within a stratum samples are ordered by a hash of their id, and the strata take
    turns giving one each until ``n`` are picked or all are used.
    """
    buckets: dict[str, list[Sample]] = {}
    for sample in samples:
        buckets.setdefault(stratum(sample), []).append(sample)
    queues = [sorted(bucket, key=lambda s: _rank(s.id)) for _, bucket in sorted(buckets.items())]
    picked: list[Sample] = []
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop(0))
    return picked


def v2_samples(rows: list[dict], platform: str) -> list[Sample]:
    """ScreenSpot-v2 rows (bbox is x, y, width, height)."""
    out = []
    for index, row in enumerate(rows):
        x, y, w, h = row["bbox"]
        out.append(Sample(
            id=f"v2-{platform}-{index}",
            set=f"v2-{platform}",
            instruction=row["instruction"],
            image=row["img_filename"],
            bbox=(x, y, x + w, y + h),
            data_type=row.get("data_type", "unknown"),
            source=row.get("data_source", ""),
        ))
    return out


def pro_samples(rows: list[dict]) -> list[Sample]:
    """ScreenSpot-Pro rows (bbox is x1, y1, x2, y2)."""
    return [
        Sample(
            id=str(row["id"]),
            set="pro",
            instruction=row["instruction"],
            image=f"images/{row['img_filename']}",
            bbox=tuple(row["bbox"]),
            data_type=row.get("ui_type", "unknown"),
            source=f"{row.get('application', '')}/{row.get('platform', '')}",
            group=row.get("group", ""),
        )
        for row in rows
    ]


class Dataset:
    """Fetches the pinned annotations and only the screenshots a run needs, with a disk cache."""

    def __init__(self, cache_dir: Path = CACHE_DIR, http: httpx.Client | None = None) -> None:
        self.cache_dir = cache_dir
        self.http = http or httpx.Client(follow_redirects=True, timeout=120.0)
        self._zip: zipfile.ZipFile | None = None

    def _cached(self, name: str, fetch) -> bytes:
        path = self.cache_dir / name
        if path.is_file():
            return path.read_bytes()
        data = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return data

    def _get(self, url: str) -> bytes:
        response = self.http.get(url)
        response.raise_for_status()
        return response.content

    def _json(self, repo: str, revision: str, path: str) -> Any:
        name = f"{repo.split('/')[1]}-{revision[:12]}/{path}"
        return json.loads(self._cached(name, lambda: self._get(hf_url(repo, revision, path))))

    def samples(self, set_name: str) -> list[Sample]:
        if set_name in ("v2-desktop", "v2-web"):
            platform = set_name.removeprefix("v2-")
            return v2_samples(self._json(V2_REPO, V2_REVISION, f"screenspot_{platform}_v2.json"), platform)
        if set_name == "pro":
            listing = self._cached(
                f"ScreenSpot-Pro-{PRO_REVISION[:12]}/annotations.index.json",
                lambda: self._get(f"{HF}/api/datasets/{PRO_REPO}/tree/{PRO_REVISION}/annotations"),
            )
            rows: list[dict] = []
            for entry in sorted(json.loads(listing), key=lambda e: e["path"]):
                if entry.get("type") == "file" and entry["path"].endswith(".json"):
                    rows.extend(self._json(PRO_REPO, PRO_REVISION, entry["path"]))
            return pro_samples(rows)
        raise ValueError(f"unknown set {set_name!r}")

    def image(self, sample: Sample) -> bytes:
        if sample.set == "pro":
            name = f"ScreenSpot-Pro-{PRO_REVISION[:12]}/{sample.image}"
            return self._cached(name, lambda: self._get(hf_url(PRO_REPO, PRO_REVISION, sample.image)))
        name = f"ScreenSpot-v2-{V2_REVISION[:12]}/images/{sample.image}"
        return self._cached(name, lambda: self._v2_zip().read(f"screenspotv2_image/{sample.image}"))

    def _v2_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            reader = RangeReader(self.http, hf_url(V2_REPO, V2_REVISION, V2_ZIP))
            self._zip = zipfile.ZipFile(io.BufferedReader(reader, buffer_size=1 << 16))
        return self._zip


class RangeReader(io.RawIOBase):
    """A seekable read-only file over HTTP range requests, so ``zipfile`` can pick out one member."""

    def __init__(self, http: httpx.Client, url: str) -> None:
        head = http.head(url)
        head.raise_for_status()
        self.http, self.url = http, str(head.url)
        self.size = int(head.headers["content-length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(buffer)) - 1
        response = self.http.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        response.raise_for_status()
        data = response.content
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def select_samples(dataset: Dataset, sets: list[str], per_set: int) -> list[Sample]:
    picked: list[Sample] = []
    for set_name in sets:
        pool = dataset.samples(set_name)
        stratum = (lambda s: f"{s.group}/{s.data_type}") if set_name == "pro" else (lambda s: s.data_type)
        picked.extend(stratified(pool, per_set, stratum))
    return picked


# ---------- scoring ----------


def first_point(action: AgentAction) -> tuple[int, int] | None:
    """Where the proposed action clicks first, if it clicks at all."""
    for candidate in [action, *(action.actions or [])]:
        if candidate.x is not None and candidate.y is not None and candidate.action_type.value != "batch":
            return candidate.x, candidate.y
    return None


def score(point: tuple[int, int] | None, bbox: tuple[float, float, float, float]) -> tuple[str, float | None]:
    if point is None:
        return "no_click", None
    x, y = point
    x1, y1, x2, y2 = bbox
    distance = ((x - (x1 + x2) / 2) ** 2 + (y - (y1 + y2) / 2) ** 2) ** 0.5
    return ("hit" if x1 <= x <= x2 and y1 <= y <= y2 else "miss"), round(distance, 1)


# ---------- running ----------


class Backend(Protocol):
    async def perceive(self, screenshot: CaptureFrame, state: AgentState) -> Any: ...
    async def choose_action(self, state: AgentState, perception: Any) -> Any: ...
    def latest_debug_artifacts(self) -> Any: ...


class BackendFactory:
    """One backend per sample (fresh conversation), sharing one HTTP client per kind."""

    def __init__(self, backend: str, model: str, effort: str | None) -> None:
        self.backend, self.model, self.effort = backend, model, effort
        self._clients: dict[str, Any] = {}

    def __call__(self, sample: Sample) -> Backend:
        from operon.agent.backends.browser_claude_computer_use import (
            BrowserClaudeComputerUseBackend,
        )
        from operon.agent.backends.browser_computer_use import BrowserComputerUseBackend
        from operon.agent.backends.desktop_claude_computer_use import (
            DesktopClaudeComputerUseBackend,
        )
        from operon.clients.anthropic_computer_use import (
            LAUNCH_APP_TOOL,
            NAVIGATE_TOOL,
            AnthropicComputerUseHttpClient,
        )
        from operon.clients.gemini_computer_use import GeminiComputerUseHttpClient
        from operon.core.paths import prompts_dir

        if self.backend == "computer_use":
            if "gemini" not in self._clients:
                self._clients["gemini"] = GeminiComputerUseHttpClient(model=self.model, thinking_level=self.effort)
            return BrowserComputerUseBackend(
                client=self._clients["gemini"], prompt_path=prompts_dir() / "browser_computer_use_prompt.txt"
            )
        kind = "browser" if sample.set == "v2-web" else "desktop"
        if kind not in self._clients:
            tools = (NAVIGATE_TOOL,) if kind == "browser" else (LAUNCH_APP_TOOL,)
            self._clients[kind] = AnthropicComputerUseHttpClient(model=self.model, effort=self.effort, custom_tools=tools)
        backend_cls = BrowserClaudeComputerUseBackend if kind == "browser" else DesktopClaudeComputerUseBackend
        return backend_cls(
            client=self._clients[kind], prompt_path=prompts_dir() / f"{kind}_claude_computer_use_prompt.txt"
        )


def _image_size(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.size


async def run_sample(sample: Sample, image: bytes, make_backend, work_dir: Path) -> SampleResult:
    started = time.monotonic()
    width, height = _image_size(image)
    sample_dir = work_dir / sample.id
    sample_dir.mkdir(parents=True, exist_ok=True)
    screen = sample_dir / "screen.png"
    screen.write_bytes(image)
    state = AgentState(
        run_id=f"grounding-{sample.id}",
        intent=INTENT.format(instruction=sample.instruction),
        status=RunStatus.RUNNING,
    )
    point, detail, usage = None, "", None
    try:
        backend = make_backend(sample)
        frame = CaptureFrame(artifact_path=str(screen), width=width, height=height)
        perception = await backend.perceive(frame, state)
        decision = await backend.choose_action(state, perception)
        artifacts = backend.latest_debug_artifacts()
        usage = getattr(artifacts, "usage", None)
        point = first_point(decision.action)
        detail = f"{decision.action.action_type.value}: {(decision.rationale or '')[:160]}"
        outcome, distance = score(point, sample.bbox)
    except Exception as exc:  # noqa: BLE001 - one broken sample must not end the run
        outcome, distance, detail = "error", None, f"{type(exc).__name__}: {exc}"[:300]
    return SampleResult(
        id=sample.id,
        set=sample.set,
        data_type=sample.data_type,
        source=sample.source,
        instruction=sample.instruction,
        outcome=outcome,
        x=point[0] if point else None,
        y=point[1] if point else None,
        bbox=list(sample.bbox),
        distance_px=distance,
        image_width=width,
        image_height=height,
        seconds=round(time.monotonic() - started, 2),
        cost_usd=round(getattr(usage, "estimated_cost_usd", None) or 0.0, 6),
        input_tokens=getattr(usage, "input_tokens", None) or 0,
        output_tokens=(getattr(usage, "output_tokens", None) or 0) + (getattr(usage, "thoughts_tokens", None) or 0),
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
        detail=detail,
    )


async def run_samples(
    samples: list[Sample],
    dataset: Dataset,
    make_backend,
    work_dir: Path,
    *,
    concurrency: int = 4,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    on_result=None,
) -> tuple[list[SampleResult], dict | None]:
    """Run every sample, ``concurrency`` at a time, until done or the cost cap is reached."""
    results: list[SampleResult] = []
    queue = list(samples)
    stopped: dict | None = None

    def spent() -> float:
        return round(sum(r.cost_usd for r in results), 6)

    async def worker() -> None:
        nonlocal stopped
        while queue:
            if max_cost_usd > 0 and spent() >= max_cost_usd:
                stopped = stopped or {"limit_usd": max_cost_usd, "spent_usd": spent(), "tasks_planned": len(samples)}
                return
            sample = queue.pop(0)
            try:
                image = await asyncio.to_thread(dataset.image, sample)
            except Exception as exc:  # noqa: BLE001
                print(f"  {sample.id}: could not fetch the screenshot: {exc}", file=sys.stderr)
                continue
            result = await run_sample(sample, image, make_backend, work_dir)
            results.append(result)
            if on_result:
                on_result(result, len(results), len(samples))

    await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    if stopped:
        stopped.update(spent_usd=spent(), tasks_run=len(results))
    order = {s.id: i for i, s in enumerate(samples)}
    results.sort(key=lambda r: order[r.id])
    return results, stopped


def _rate(results: list[SampleResult]) -> dict:
    counts = {outcome: sum(r.outcome == outcome for r in results) for outcome in OUTCOMES}
    scored = len(results) - counts["error"]
    return {"samples": len(results), **counts, "accuracy": round(counts["hit"] / scored, 3) if scored else 0.0}


def summarize(results: list[SampleResult]) -> dict:
    """Accuracy over all samples, per set, and per set and target type. Errors are left out of accuracy."""
    by_set: dict[str, dict] = {}
    for set_name in dict.fromkeys(r.set for r in results):
        in_set = [r for r in results if r.set == set_name]
        by_set[set_name] = {
            **_rate(in_set),
            "by_type": {t: _rate([r for r in in_set if r.data_type == t]) for t in sorted({r.data_type for r in in_set})},
        }
    cost = round(sum(r.cost_usd for r in results), 6)
    total = len(results)
    misses = [r.distance_px for r in results if r.outcome == "miss" and r.distance_px is not None]
    return {
        **_rate(results),
        "by_set": by_set,
        "median_miss_distance_px": sorted(misses)[len(misses) // 2] if misses else None,
        "mean_seconds": round(sum(r.seconds for r in results) / total, 2) if total else 0.0,
        "cost_usd": cost,
        "cost_per_sample_usd": round(cost / total, 6) if total else 0.0,
        **{key: sum(getattr(r, key) for r in results) for key in
           ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")},
    }


def format_report(summary: dict) -> str:
    lines = [f"{'set':12} {'type':6} {'n':>4} {'hit':>4} {'miss':>4} {'none':>4} {'err':>4} {'accuracy':>8}"]

    def row(name: str, kind: str, s: dict) -> str:
        return (f"{name:12} {kind:6} {s['samples']:>4} {s['hit']:>4} {s['miss']:>4} {s['no_click']:>4} "
                f"{s['error']:>4} {s['accuracy']:>8.1%}")

    for set_name, s in summary["by_set"].items():
        for kind, t in s["by_type"].items():
            lines.append(row(set_name, kind, t))
        lines.append(row(set_name, "all", s))
    lines.append(row("all", "", summary))
    lines.append("")
    lines.append(
        f"cost ${summary['cost_usd']:.4f} (${summary['cost_per_sample_usd']:.4f} per sample) · "
        f"{summary['mean_seconds']} s per sample · median miss {summary['median_miss_distance_px']} px from the target's centre"
    )
    return "\n".join(lines)


def _default_model(backend: str) -> str:
    if backend == "computer_use":
        return os.getenv("OPERON_BROWSER_MODEL", "gemini-3.5-flash")
    return os.getenv("OPERON_BROWSER_CLAUDE_MODEL", "claude-opus-5-5")


def _default_backend() -> str:
    explicit = os.getenv("OPERON_BROWSER_BACKEND")
    if explicit in BACKENDS:
        return explicit
    return "claude_computer_use" if os.getenv("ANTHROPIC_API_KEY") else "computer_use"


def _max_cost_default() -> float:
    raw = os.getenv("MAX_RUN_COST_USD", "").strip()
    try:
        return float(raw) if raw else DEFAULT_MAX_COST_USD
    except ValueError:
        sys.exit(f"MAX_RUN_COST_USD must be a number of US dollars, not {raw!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", action="append", choices=SETS, help="sets to run (default: all)")
    parser.add_argument("--per-set", type=int, default=DEFAULT_PER_SET, help="samples per set")
    parser.add_argument("--backend", choices=BACKENDS, default=_default_backend())
    parser.add_argument("--model", help="model (default: the backend's server default)")
    parser.add_argument("--effort", help="Claude effort, or Gemini thinking level (default: low)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--max-cost-usd", type=float, default=_max_cost_default(),
        help="stop once the run has spent this much (default $MAX_RUN_COST_USD or 5; 0: no cap)",
    )
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--out-dir", default=str(BENCH_DIR))
    parser.add_argument("--keep", action="store_true", help="keep each sample's prompt and model reply under .var/grounding/")
    parser.add_argument("--list", action="store_true", help="list the selected samples and exit (no model calls)")
    parser.add_argument("--json", action="store_true")
    return parser


async def run(args: argparse.Namespace, make_backend=None, dataset: Dataset | None = None) -> int:
    dataset = dataset or Dataset(Path(args.cache_dir))
    sets = args.set or list(SETS)
    samples = select_samples(dataset, sets, args.per_set)
    if args.list:
        for s in samples:
            print(f"{s.id:24} {s.set:10} {s.data_type:5} {s.source:28} {s.instruction}")
        return 0
    model = args.model or _default_model(args.backend)
    effort = args.effort or (os.getenv("OPERON_BROWSER_CLAUDE_EFFORT", "low") if args.backend != "computer_use"
                             else os.getenv("OPERON_BROWSER_THINKING", "low")) or None
    make_backend = make_backend or BackendFactory(args.backend, model, effort)
    started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    work_dir = Path(".var/grounding") / started_at

    def progress(result: SampleResult, done: int, total: int) -> None:
        if not args.json:
            print(f"[{done}/{total}] {result.id}: {result.outcome}", flush=True)

    results, stopped = await run_samples(
        samples, dataset, make_backend, work_dir,
        concurrency=args.concurrency, max_cost_usd=args.max_cost_usd, on_result=progress,
    )
    summary = summarize(results)
    # "samples", not "results": the dashboard importer takes any JSON with both
    # "summary" and "results" for a live benchmark report.
    report = {
        "kind": "grounding",
        "suite": "grounding",
        "started_at": started_at,
        "backend": args.backend,
        "model": model,
        "effort": effort,
        "zoom": args.backend != "computer_use" and _zoom_enabled(),
        "per_set": args.per_set,
        "datasets": {V2_REPO: V2_REVISION, PRO_REPO: PRO_REVISION},
        "budget_stop": stopped,
        "summary": summary,
        "samples": [asdict(r) for r in results],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"grounding-{started_at}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.keep:
        shutil.rmtree(work_dir, ignore_errors=True)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print()
        print(format_report(summary))
        if stopped:
            print(f"Stopped at the ${args.max_cost_usd:.2f} cost cap after {stopped['tasks_run']} of {len(samples)} samples.")
        print(f"\nsaved {path}")
    return BUDGET_EXIT_CODE if stopped else 0


def _zoom_enabled() -> bool:
    from operon.clients.anthropic_computer_use import zoom_enabled

    return zoom_enabled()


def main(argv: list[str]) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Live view check (KAN-89): one short real task, watched over the API.

Serves the local apps site, starts a short shop task through ``POST /app/api/runs`` of a
running API and watches ``/ws/stream`` with a plain WebSocket client: are frames
arriving, do they keep arriving during a takeover (``/takeover``, ``/input``,
``/handback``), and does the run end with a report (``/report``). A ``result.json``
lands in ``--out``.

Needs a running API with a real GEMINI_API_KEY; it spends a few cents of model calls::

    API_KEYS=check-key .venv/bin/python -m uvicorn operon.api.server:app --port 8080 &
    .venv/bin/python -m evals.live_view_check --key check-key
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

TASK = "Add one pack of AA batteries to the cart in the shop, then stop."


class Watch:
    """What the stream delivered."""

    def __init__(self) -> None:
        self.frames = 0
        self.frames_in_takeover = 0
        self.in_takeover = False
        self.sizes: set[int] = set()

    def on_frame(self, payload) -> None:
        if isinstance(payload, (bytes, bytearray)):
            self.frames += 1
            self.sizes.add(len(payload))
            if self.in_takeover:
                self.frames_in_takeover += 1


def ws_url(api: str) -> str:
    return api.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws/stream"


async def _listen(api: str, key: str, watch: Watch) -> None:
    """Count the frames on /ws/stream. A plain client sends no Origin, which the stream accepts."""
    import websockets

    async with websockets.connect(ws_url(api), subprotocols=["x-api-key", key]) as ws:
        async for message in ws:
            watch.on_frame(message)


async def check(api: str, key: str, out: Path, *, max_steps: int, timeout_s: float) -> dict:
    import httpx

    from evals.fixture_site import FixtureSite
    from evals.local_apps import build_apps
    from evals.safety_tasks import site_pages

    out.mkdir(parents=True, exist_ok=True)
    watch = Watch()
    started = time.time()
    took_over = False
    report: dict = {}
    status: dict = {}
    with FixtureSite(site_pages(), "127.0.0.1", 8765, apps=build_apps()) as site:
        async with httpx.AsyncClient(base_url=api, headers={"X-API-Key": key}, timeout=30.0) as http:
            response = await http.post("/app/api/runs", json={
                "intent": TASK, "environment": "browser", "start_url": site.url("/apps/shop/"),
                "max_steps": max_steps, "permission_mode": "auto",
            })
            response.raise_for_status()
            run_id = response.json()["run_id"]
            listener = asyncio.create_task(_listen(api, key, watch))
            try:
                while time.time() - started < timeout_s:
                    await asyncio.sleep(3)
                    if listener.done() and listener.exception() is not None:
                        raise listener.exception()  # type: ignore[misc]
                    status = (await http.get(f"/app/api/runs/{run_id}")).json()
                    if status["done"]:
                        break
                    # Take over as soon as frames show: a short task can end in two steps.
                    if not took_over and watch.frames >= 3:
                        took_over = True
                        (await http.post(f"/app/api/runs/{run_id}/takeover")).raise_for_status()
                        watch.in_takeover = True
                        for _ in range(4):  # Tab moves the focus ring: a change the person should see
                            await http.post(f"/app/api/runs/{run_id}/input", json={"input_type": "key", "key": "Tab"})
                            await asyncio.sleep(0.7)
                        await asyncio.sleep(1.5)
                        watch.in_takeover = False
                        (await http.post(
                            f"/app/api/runs/{run_id}/handback", json={"note": "I only moved the focus with Tab."},
                        )).raise_for_status()
                if status.get("done"):
                    report = (await http.get(f"/app/api/runs/{run_id}/report")).json()
            finally:
                listener.cancel()
        shop = site.state().get("shop", {})

    result = {
        "run_id": run_id,
        "ws_frames": watch.frames,
        "ws_frames_during_takeover": watch.frames_in_takeover,
        "distinct_frame_sizes": len(watch.sizes),
        "took_over": took_over,
        "final_status": status.get("status"),
        "steps": status.get("step_count"),
        "report": report,
        "cart": shop.get("cart"),
        "seconds": round(time.time() - started, 1),
    }
    result["checks"] = {
        "frames_arrived_on_the_stream": watch.frames >= 10,
        "frames_kept_coming_during_takeover": took_over and watch.frames_in_takeover >= 1,
        "run_ended_with_a_report": bool(report),
    }
    result["passed"] = all(result["checks"].values())
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default=os.getenv("OPERON_API_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--key", default=(os.getenv("API_KEYS", "").split(",")[0] or None))
    parser.add_argument("--out", type=Path, default=Path(".var/live-view-check"))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.key:
        parser.error("--key (or API_KEYS) is required")
    result = asyncio.run(check(args.api.rstrip("/"), args.key, args.out, max_steps=args.max_steps, timeout_s=args.timeout))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

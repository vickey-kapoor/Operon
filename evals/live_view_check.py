"""Live view check (KAN-89): one short real task on the task page, watched by a browser.

Serves the local apps site, opens ``/app`` of a running API in a separate headless
Chromium, starts a short shop task there and watches the page the way a person
would: are frames arriving over ``/ws/stream`` and showing in the live view, do they
keep arriving during a takeover (so the person sees their own clicks), and does the
run end with a report. Screenshots of the page and a ``result.json`` land in ``--out``.

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
# The result card the live run shows once the run has ended.
REPORT = "#done-card:not(.hidden)"


class Watch:
    """What the page received and showed."""

    def __init__(self) -> None:
        self.frames = 0
        self.frames_in_takeover = 0
        self.in_takeover = False
        self.shots: list[str] = []

    def on_frame(self, payload) -> None:
        if isinstance(payload, (bytes, bytearray)):
            self.frames += 1
            if self.in_takeover:
                self.frames_in_takeover += 1


async def _shown(page) -> dict:
    """The live view image as the page shows it."""
    return await page.evaluate(
        """() => { const img = document.getElementById('frame');
                   return { visible: !img.classList.contains('hidden'), width: img.naturalWidth,
                            height: img.naturalHeight, src: img.src,
                            status: document.getElementById('run-state-text').textContent,
                            steps: document.getElementById('run-meta').textContent }; }"""
    )


async def _snap(page, out: Path, watch: Watch, name: str) -> None:
    path = out / f"{len(watch.shots):02d}-{name}.png"
    await page.screenshot(path=str(path))
    watch.shots.append(path.name)


async def check(api: str, key: str, out: Path, *, max_steps: int, timeout_s: float) -> dict:
    from playwright.async_api import async_playwright

    from evals.fixture_site import FixtureSite
    from evals.local_apps import build_apps
    from evals.safety_tasks import site_pages

    out.mkdir(parents=True, exist_ok=True)
    watch = Watch()
    started = time.time()
    with FixtureSite(site_pages(), "127.0.0.1", 8765, apps=build_apps()) as site:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page(viewport={"width": 1400, "height": 900})
            page.on("websocket", lambda ws: ws.on("framereceived", watch.on_frame) if "/ws/stream" in ws.url else None)
            await page.goto(f"{api}/app#key={key}")
            await page.wait_for_selector("#view-home:not(.hidden)")
            await page.fill("#intent", TASK)
            # The run options sit in a collapsed "Where it runs and when it asks" section.
            await page.click("details.options > summary")
            await page.select_option("#environment", "browser")
            await page.fill("#start-url", site.url("/apps/shop/"))
            await page.fill("#max-steps", str(max_steps))
            await page.check("input[name=mode][value=auto]")
            await page.click("#btn-start")
            await page.wait_for_selector("#view-run:not(.hidden)", timeout=30_000)

            # Watch the run; take over once frames show, click, and hand back.
            took_over = False
            sizes: set[tuple[int, int]] = set()
            images: set[str] = set()
            while time.time() - started < timeout_s:
                await asyncio.sleep(3)
                shown = await _shown(page)
                if shown["visible"] and shown["width"]:
                    sizes.add((shown["width"], shown["height"]))
                    images.add(shown["src"])
                await _snap(page, out, watch, "watch")
                if await page.is_visible(REPORT):
                    break
                # Take over as soon as the live view shows the browser: a short task can end in two steps.
                if not took_over and watch.frames >= 3 and shown["visible"] and shown["width"]:
                    took_over = True
                    await page.click("#btn-takeover")
                    await page.wait_for_selector("#takeover-card:not(.hidden)", timeout=60_000)
                    watch.in_takeover = True
                    await _snap(page, out, watch, "takeover")
                    box = await page.locator("#frame").bounding_box()
                    if box:  # a click on an empty corner of the shop page: harmless
                        await page.mouse.click(box["x"] + box["width"] * 0.97, box["y"] + box["height"] * 0.97)
                    for _ in range(4):  # Tab moves the focus ring: a change the person should see
                        await page.click("#takeover-keys [data-key=Tab]")
                        await asyncio.sleep(0.7)
                    await asyncio.sleep(1.5)
                    await _snap(page, out, watch, "takeover-input")
                    watch.in_takeover = False
                    await page.fill("#handback-note", "I only moved the focus with Tab.")
                    await page.click("#btn-handback")
            await _snap(page, out, watch, "end")
            report_text = await page.inner_text(REPORT) if await page.is_visible(REPORT) else ""
            final = await _shown(page)
            await browser.close()
        shop = site.state().get("shop", {})

    result = {
        "ws_frames": watch.frames,
        "ws_frames_during_takeover": watch.frames_in_takeover,
        "distinct_images_shown": len(images),
        "image_sizes": sorted(sizes),
        "took_over": took_over,
        "final_status": final["status"],
        "steps": final["steps"],
        "report": report_text,
        "cart": shop.get("cart"),
        "seconds": round(time.time() - started, 1),
        "screenshots": watch.shots,
    }
    result["checks"] = {
        "frames_reached_the_page": watch.frames >= 10,
        "live_view_showed_them": bool(sizes) and len(images) >= 3,
        "frames_kept_coming_during_takeover": took_over and watch.frames_in_takeover >= 1,
        "run_ended_with_a_report": bool(report_text),
    }
    result["passed"] = all(result["checks"].values())
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default=os.getenv("OPERON_API_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--key", default=(os.getenv("API_KEYS", "").split(",")[0] or None))
    parser.add_argument("--out", type=Path, default=Path(".var/live-view-check"))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if not args.key:
        parser.error("--key (or API_KEYS) is required")
    result = asyncio.run(check(args.api.rstrip("/"), args.key, args.out, max_steps=args.max_steps, timeout_s=args.timeout))
    print(json.dumps({k: v for k, v in result.items() if k != "screenshots"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

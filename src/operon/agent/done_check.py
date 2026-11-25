"""One independent check of the model's "done" (KAN-208).

When the model ends a run with "done", a second model sees the task, the model's closing
message and the final screenshot, and answers one JSON object. Like the planner it is one
bounded call that can only advise:

* **A rejection is a hint.** Its reason reaches the acting model as a note on the next step
  and the run goes on. The action is never replaced and nothing is retried by the loop.
* **At most** ``MAX_DONE_CHECKS`` **checks per run.** After that the model's done stands.
* **Safe to fail.** An error, a timeout or an unusable answer leaves the done standing; the
  result says ``error`` and the claim is never counted as verified.

Off by default (``OPERON_DONE_CHECK``). The loop wires it in ``_check_done_claim``.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from operon.models.usage import ModelUsage

MAX_DONE_CHECKS = 2
# The run's last step waits on this, so the call gets a short overall deadline (seconds).
DONE_CHECK_DEADLINE_S = 20.0
DONE_CHECK_USAGE_FILE = "done_check_usage.json"
MAX_REASON_CHARS = 200

DONE_CHECK_PROMPT = """An agent that controls a computer says it has finished a task. The image is the
screen right after its last action. Decide whether the task is finished.

Task: {intent}

The agent's closing message: {closing_message}

Rules:
- Judge only from what the screen shows and what the task asks for.
- Answer done: true unless the screen clearly shows the task is not finished.
- If the task is not finished, say in one line what is still missing.

Return only JSON: {{"done": true or false, "reason": "one line"}}"""

Verdict = Literal["accepted", "rejected", "error"]


@dataclass
class DoneCheckResult:
    """What one check came to. ``error`` means no usable verdict: the done stands."""

    verdict: Verdict
    reason: str = ""
    error: str | None = None
    raw: str | None = None
    usage: ModelUsage | None = None
    prompt: str | None = None


def parse_verdict(raw: str) -> tuple[bool, str] | None:
    """``(done, reason)`` from the model's JSON answer; None if it is unusable."""
    text = raw.strip() if isinstance(raw, str) else ""
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("done"), bool):
        return None
    reason = data.get("reason")
    reason = " ".join(reason.split())[:MAX_REASON_CHARS] if isinstance(reason, str) else ""
    return data["done"], reason


def rejection_note(reason: str) -> str:
    """The note the acting model reads after a rejected done."""
    detail = reason.strip().rstrip(".") or "no reason given"
    return (
        f"An independent check of the final screen says the task is not done yet: {detail}. "
        "Look at the screen and keep working; say done again when it really is."
    )


async def check_done(client: Any, intent: str, closing_message: str, screenshot_path: str) -> DoneCheckResult:
    """Ask ``client`` (a Gemini client) whether the task is finished. Never raises."""
    prompt = DONE_CHECK_PROMPT.format(
        intent=intent.strip(), closing_message=" ".join((closing_message or "").split()) or "(none)"
    )
    try:
        raw = await asyncio.wait_for(client.generate_perception(prompt, screenshot_path), timeout=DONE_CHECK_DEADLINE_S)
    except Exception as exc:  # noqa: BLE001 - a checker failure must never fail a run (TimeoutError included)
        return DoneCheckResult("error", error=f"{type(exc).__name__}: {exc}"[:300], prompt=None)
    latest = getattr(client, "latest_usage", None)
    usage = latest() if callable(latest) else None
    usage = usage if isinstance(usage, ModelUsage) else None
    verdict = parse_verdict(raw)
    if verdict is None:
        return DoneCheckResult("error", error="unusable answer", raw=str(raw), usage=usage, prompt=prompt)
    done, reason = verdict
    return DoneCheckResult("accepted" if done else "rejected", reason=reason, raw=str(raw), usage=usage, prompt=prompt)

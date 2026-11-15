"""One-shot planner: a cheaper model writes a short plan once, at task start (KAN-112).

A small, cheap Gemini (Flash-Lite by default) reads only the task text and writes a few
short steps. The plan reaches the acting model as one advisory hint on its first step
("Notes from Operon for this step"), so it stays in the conversation for the whole run.

* **One call per run, no re-planning.** This is the narrow form of the cancelled KAN-83
  planner: no loop, no second opinion, and the actor still decides every action.
* **Text only, vision-only intact.** The planner never sees the screen and never names
  coordinates or selectors; it only says what to do, in order.
* **Safe to fail.** Any error, empty or unusable answer means no plan; the run goes on
  exactly as it would without the planner.

Off by default (``OPERON_PLANNER``). The plan is saved next to the run as ``plan.json``
with the call's cost, so an A/B (KAN-115) can count it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from operon.models.usage import ModelUsage

logger = logging.getLogger(__name__)

MAX_STEPS = 8
MAX_STEP_CHARS = 160

PLAN_PROMPT = """You plan tasks for a computer-use agent that sees the screen and acts with
the mouse and keyboard. Write a short plan for the task below.

Task: {intent}

Rules:
- 2 to {max_steps} steps, each one short sentence saying what to do, in order.
- Say what to do, never where to click: no coordinates, no selectors, no code.
- Do not invent facts about the screen. Do not add steps the task does not need.
- The last step says how to tell the task is finished.

Return only JSON: {{"steps": ["...", "..."]}}"""


def parse_plan(raw: str) -> list[str]:
    """The plan's steps from the model's JSON answer; empty if it is unusable."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        return []
    cleaned = [
        " ".join(step.split())[:MAX_STEP_CHARS]
        for step in steps
        if isinstance(step, str) and step.strip()
    ]
    return cleaned[:MAX_STEPS]


def plan_hint(steps: list[str]) -> str:
    """The plan as the one-line-per-step note the acting model reads."""
    numbered = "; ".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
    return (
        "A quick plan for this task (a guide, not an order: follow the screen if it "
        f"differs): {numbered}"
    )


async def make_plan(client: Any, intent: str) -> tuple[list[str], ModelUsage | None]:
    """Ask ``client`` (a Gemini text client) for the plan. Returns ([], None) on any failure."""
    if client is None or not intent.strip():
        return [], None
    prompt = PLAN_PROMPT.format(intent=intent.strip(), max_steps=MAX_STEPS)
    try:
        raw = await client.generate_policy(prompt)
    except Exception as exc:  # noqa: BLE001 - a planner failure must never fail a run
        logger.warning("planner: call failed, running without a plan: %s", exc)
        return [], None
    usage = client.latest_usage() if hasattr(client, "latest_usage") else None
    steps = parse_plan(raw)
    if len(steps) < 2:
        logger.info("planner: no usable plan, running without one")
        return [], usage if isinstance(usage, ModelUsage) else None
    return steps, usage if isinstance(usage, ModelUsage) else None

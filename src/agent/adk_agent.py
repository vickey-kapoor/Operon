"""Google ADK-based agent for the Chrome Extension session flow."""

import base64, io, json, logging, os
from typing import Optional

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from PIL import Image

from src.agent.planner import ActionPlan, ActionPlanner
from src.agent.vision import SYSTEM_PROMPT
from src import metrics

logger = logging.getLogger(__name__)

_APP_NAME = "ui_navigator"
_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# ADK agent definition
# ---------------------------------------------------------------------------

_agent = Agent(
    name="ui_navigator",
    model=_MODEL,
    instruction=SYSTEM_PROMPT,
)

_session_service = InMemorySessionService()

_runner = Runner(
    agent=_agent,
    app_name=_APP_NAME,
    session_service=_session_service,
)

# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


async def create_session(user_id: str = "extension") -> str:
    """Create a new ADK session and return its session_id."""
    session = await _session_service.create_session(
        app_name=_APP_NAME, user_id=user_id
    )
    metrics.emit("ext_session_created")
    logger.info("adk_session_created", extra={"session_id": session.id})
    return session.id


async def delete_session(session_id: str, user_id: str = "extension") -> bool:
    """Delete an ADK session. Returns True if it existed."""
    try:
        await _session_service.delete_session(
            app_name=_APP_NAME, user_id=user_id, session_id=session_id
        )
        logger.info("adk_session_deleted", extra={"session_id": session_id})
        return True
    except Exception:
        return False


async def session_exists(session_id: str, user_id: str = "extension") -> bool:
    s = await _session_service.get_session(
        app_name=_APP_NAME, user_id=user_id, session_id=session_id
    )
    return s is not None


# ---------------------------------------------------------------------------
# Step — send screenshot + task, get ActionPlan back
# ---------------------------------------------------------------------------


async def step(session_id: str, image_b64: str, task: str) -> ActionPlan:
    """Run one agent step via ADK Runner and return a parsed ActionPlan."""
    import time
    t0 = time.time()

    # Build multimodal user message.
    img_bytes = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    user_message = types.Content(role="user", parts=[
        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
        types.Part.from_text(
            text=f"User task: {task}\n\nAnalyze the screenshot and respond with the next action plan."
        ),
    ])

    # Run through ADK — collects multi-turn history automatically.
    response_text = ""
    async for event in _runner.run_async(
        user_id="extension",
        session_id=session_id,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    response_text += part.text

    step_ms = int((time.time() - t0) * 1000)

    # Parse response with existing ActionPlanner logic.
    planner = ActionPlanner(vision_client=None)
    plan = planner._parse_response(response_text) if response_text else planner._fallback_plan(response_text)

    action_types = [
        a.type.value if hasattr(a.type, "value") else str(a.type)
        for a in plan.actions
    ]

    metrics.emit("ext_step_latency_ms", step_ms)
    metrics.emit("ext_step_actions", len(plan.actions))
    if plan.done:
        metrics.emit("ext_session_completed")
    for at in action_types:
        metrics.emit("ext_action_type", labels={"type": at})

    logger.info("adk_step", extra={
        "session_id": session_id,
        "step_latency_ms": step_ms,
        "action_types": action_types,
        "action_count": len(plan.actions),
        "done": plan.done,
        "is_fallback": plan.observation.startswith("Unable to parse"),
    })
    return plan

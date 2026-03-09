import asyncio, base64, io, json, logging, os, time, uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from google.genai import types
from PIL import Image

from src.agent.vision import GeminiVisionClient
from src.agent.planner import ActionPlan, ActionPlanner
from src import metrics

logger = logging.getLogger(__name__)
_SESSION_IDLE_SECONDS = 3600


@dataclass
class Session:
    id: str
    history: List[types.Content] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class SessionManager:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._api_key = api_key
        self._vision: Optional[GeminiVisionClient] = None

    async def create_session(self) -> str:
        sid = str(uuid.uuid4())
        async with self._lock:
            self._sessions[sid] = Session(id=sid)
        metrics.emit("ext_session_created")
        logger.info("ext_session_created", extra={"session_id": sid})
        return sid

    async def get_session(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.get(session_id)
            existed = session is not None
            if existed:
                duration_s = round(time.time() - session.created_at, 1)
                turns = len(session.history)
            self._sessions.pop(session_id, None)
        if existed:
            metrics.emit("ext_session_duration_s", duration_s)
            metrics.emit("ext_session_turns", turns)
            logger.info("ext_session_deleted", extra={
                "session_id": session_id,
                "duration_s": duration_s,
                "history_turns": turns,
            })
        return existed

    async def cleanup_idle(self, max_age_seconds: float = _SESSION_IDLE_SECONDS) -> int:
        cutoff = time.time() - max_age_seconds
        async with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
            for sid in stale:
                del self._sessions[sid]
        if stale:
            logger.info("Cleaned up %d idle sessions", len(stale))
        return len(stale)

    async def step(self, session_id: str, image_b64: str, task: str) -> ActionPlan:
        session = await self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)

        t0 = time.time()
        planner = ActionPlanner(vision_client=self._get_vision_client())
        plan = await planner.plan(image=image_b64, task=task, history=list(session.history))
        step_ms = int((time.time() - t0) * 1000)

        img_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")

        user_content = types.Content(role="user", parts=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            types.Part.from_text(text=f"User task: {task}\n\nAnalyze the screenshot."),
        ])
        model_content = types.Content(role="model", parts=[
            types.Part.from_text(text=json.dumps(plan.model_dump())),
        ])

        action_types = [a.type.value if hasattr(a.type, "value") else str(a.type) for a in plan.actions]

        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].history += [user_content, model_content]
                self._sessions[session_id].last_active = time.time()
                history_turns = len(self._sessions[session_id].history)

        metrics.emit("ext_step_latency_ms", step_ms)
        metrics.emit("ext_step_actions", len(plan.actions))
        if plan.done:
            metrics.emit("ext_session_completed")
        for at in action_types:
            metrics.emit("ext_action_type", labels={"type": at})

        logger.info("ext_step", extra={
            "session_id": session_id,
            "step_latency_ms": step_ms,
            "action_types": action_types,
            "action_count": len(plan.actions),
            "done": plan.done,
            "history_turns": history_turns,
            "is_fallback": plan.observation.startswith("Unable to parse"),
        })
        return plan

    def _get_vision_client(self) -> GeminiVisionClient:
        if self._vision is None:
            self._vision = GeminiVisionClient(
                api_key=self._api_key or os.environ.get("GOOGLE_API_KEY")
            )
        return self._vision


_manager: Optional[SessionManager] = None


def get_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager

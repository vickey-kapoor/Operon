"""OSWorld task machine provider (KAN-169): drive an OSWorld VM from pixels.

OSWorld (xlang-ai) runs each task in an Ubuntu VM that exposes a small HTTP server
(``/screenshot``, ``/execute``). This provider points Operon at one such VM
(``OPERON_OSWORLD_URL``, for example ``http://127.0.0.1:5000``). The VM itself, task
setup and scoring belong to OSWorld and are driven by ``evals/osworld.py``; Operon's
own done-check is never the score.

Vision-only: the executor takes a screenshot and turns each action into plain
``pyautogui`` input on the VM. No accessibility tree, no DOM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from PIL import Image

from operon.core.paths import desktop_artifacts_dir
from operon.core.settings import get_settings
from operon.environments.base import EnvironmentProvider, ProviderCheck
from operon.models.capture import CaptureFrame
from operon.models.common import FailureCategory
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_KEY_ALIASES = {
    "control": "ctrl", "return": "enter", "esc": "esc", "escape": "esc", "page_down": "pagedown",
    "page_up": "pageup", "pgdn": "pagedown", "pgup": "pageup", "del": "delete", "super": "win",
    "cmd": "win", "meta": "win", "arrowup": "up", "arrowdown": "down", "arrowleft": "left",
    "arrowright": "right", "space": "space", "spacebar": "space",
}


def _key(name: str) -> str:
    low = name.strip().lower()
    return _KEY_ALIASES.get(low, low)


def pyautogui_snippet(action: AgentAction) -> str | None:
    """The pyautogui statements for one action, or None when it needs no input."""
    t = action.action_type
    x, y = action.x, action.y
    pos = f"{int(x)}, {int(y)}" if x is not None and y is not None else ""
    if t is ActionType.CLICK:
        return f"pyautogui.click({pos})"
    if t is ActionType.DOUBLE_CLICK:
        return f"pyautogui.doubleClick({pos})"
    if t is ActionType.TRIPLE_CLICK:
        return f"pyautogui.tripleClick({pos})"
    if t is ActionType.RIGHT_CLICK:
        return f"pyautogui.rightClick({pos})"
    if t is ActionType.HOVER:
        return f"pyautogui.moveTo({pos})"
    if t is ActionType.TYPE:
        lines = []
        if pos:
            lines.append(f"pyautogui.click({pos})")
        if action.clear_before_typing:
            lines.append("pyautogui.hotkey('ctrl', 'a'); pyautogui.press('delete')")
        lines.append(f"pyautogui.write({action.text or ''!r}, interval=0.02)")
        if action.press_enter:
            lines.append("pyautogui.press('enter')")
        return "\n".join(lines)
    if t is ActionType.PRESS_KEY:
        return f"pyautogui.press({_key(action.key or '')!r})"
    if t is ActionType.HOTKEY:
        keys = [_key(k) for k in (action.key or "").replace(" ", "").split("+") if k]
        return f"pyautogui.hotkey({', '.join(repr(k) for k in keys)})"
    if t is ActionType.SCROLL:
        fn = "hscroll" if action.horizontal else "scroll"
        at = f", x={int(x)}, y={int(y)}" if x is not None and y is not None else ""
        return f"pyautogui.{fn}({int(action.scroll_amount or 0)}{at})"
    if t is ActionType.DRAG:
        return (f"pyautogui.moveTo({pos}); pyautogui.dragTo({int(action.x_end or 0)}, "
                f"{int(action.y_end or 0)}, duration=0.5, button='left')")
    return None


class OSWorldProvider(EnvironmentProvider):
    name = "osworld"
    summary = "An OSWorld task VM reached at OPERON_OSWORLD_URL (benchmark runs, KAN-169)."
    runs_on_host = True

    def __init__(self, *, url: str | None = None) -> None:
        self.url = (url if url is not None else get_settings().osworld_url).strip().rstrip("/")

    def platform(self) -> str:
        return "linux"

    def check(self) -> ProviderCheck:
        if not self.url:
            return ProviderCheck(False, ("Set OPERON_OSWORLD_URL to the OSWorld VM's server, "
                                         "for example http://127.0.0.1:5000.",))
        try:
            ok = httpx.get(f"{self.url}/screenshot", timeout=15, trust_env=False).status_code == 200
        except httpx.HTTPError as exc:
            return ProviderCheck(False, (f"The OSWorld VM at {self.url} did not answer ({exc}).",))
        return ProviderCheck(ok, () if ok else (f"The OSWorld VM at {self.url} did not return a screenshot.",))

    def build_executor(self):
        return OSWorldExecutor(self.url)


class OSWorldExecutor:
    """The executor surface the loop uses, over an OSWorld VM's HTTP server."""

    def __init__(self, base_url: str, *, artifact_dir: str | Path | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._dir = Path(artifact_dir) if artifact_dir is not None else desktop_artifacts_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self.batch_child_check = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=_TIMEOUT,
                                             transport=self._transport, trust_env=False)
        return self._client

    async def _shell(self, argv: list[str]) -> dict[str, Any]:
        response = await self._http().post("/execute", json={"command": argv, "shell": False})
        response.raise_for_status()
        return response.json()

    async def capture(self) -> CaptureFrame:
        response = await self._http().get("/screenshot")
        response.raise_for_status()
        path = self._dir / f"osworld_{uuid4().hex[:8]}.png"
        path.write_bytes(response.content)
        with Image.open(path) as image:
            width, height = image.size
        return CaptureFrame(artifact_path=str(path), width=width, height=height)

    async def execute(self, action: AgentAction) -> ExecutedAction:
        if action.action_type is ActionType.BATCH:
            detail: list[str] = []
            for child in action.actions or []:
                result = await self.execute(child)
                detail.append(f"{child.action_type.value}:{'ok' if result.success else 'failed'}")
                if not result.success:
                    return ExecutedAction(action=action, success=False, detail=f"Batch failed: {result.detail}",
                                          failure_category=result.failure_category)
            return ExecutedAction(action=action, success=True, detail=f"Executed batch: {', '.join(detail)}")
        t = action.action_type
        try:
            if t in (ActionType.STOP, ActionType.WAIT_FOR_USER):
                return ExecutedAction(action=action, success=True, detail=f"{t.value}: nothing to do")
            if t is ActionType.WAIT:
                import asyncio

                await asyncio.sleep((action.wait_ms or 1000) / 1000)
                return ExecutedAction(action=action, success=True, detail=f"Waited {action.wait_ms or 1000} ms")
            if t is ActionType.LAUNCH_APP:
                from operon.agent.app_catalog import LINUX_APP_LAUNCH_TARGETS

                command = LINUX_APP_LAUNCH_TARGETS.get((action.text or "").strip().lower())
                if command is None:
                    return _failed(action, f"launch_app: '{action.text}' is not a catalog application")
                await self._shell(["bash", "-c", f"DISPLAY=:0 nohup {command} >/dev/null 2>&1 &"])
                return ExecutedAction(action=action, success=True, detail=f"Launched {action.text}")
            snippet = pyautogui_snippet(action)
            if snippet is None:
                return _failed(action, f"{t.value} is not supported on an OSWorld machine")
            script = "import pyautogui; pyautogui.FAILSAFE = False\n" + snippet
            reply = await self._shell(["python3", "-c", script])
            if reply.get("status") == "error" or reply.get("returncode", 0) not in (0, None):
                return _failed(action, f"VM refused {t.value}: {json.dumps(reply)[:300]}")
            return ExecutedAction(action=action, success=True, detail=f"Executed {t.value} on the OSWorld VM")
        except httpx.HTTPError as exc:
            return _failed(action, f"OSWorld VM unreachable: {exc}")

    # Surface the loop calls that has nothing to do on a task VM OSWorld resets itself.
    def set_current_run_id(self, run_id: str) -> None: ...
    async def reset_desktop(self) -> None: ...
    async def context_reset(self) -> None: ...
    async def session_reset(self, start_url: str | None = None) -> None: ...
    def close_apps_when_done(self, run_id: str) -> None: ...
    async def aclose_run(self, run_id: str) -> int:
        return 0

    async def live_frame_png(self, run_id: str) -> bytes | None:
        try:
            response = await self._http().get("/screenshot")
        except httpx.HTTPError:
            return None
        return response.content if response.status_code == 200 else None

    async def user_input(self, run_id: str, *, input_type: str, **kwargs: Any) -> bool:
        return False

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _failed(action: AgentAction, detail: str) -> ExecutedAction:
    return ExecutedAction(action=action, success=False, detail=detail,
                          failure_category=FailureCategory.EXECUTION_ERROR)

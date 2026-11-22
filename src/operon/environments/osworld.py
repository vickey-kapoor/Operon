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

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from PIL import Image

from operon.core.paths import desktop_artifacts_dir
from operon.core.settings import get_settings
from operon.core.vault import SecretError, redact, resolve_text, secret_endpoint_ok
from operon.environments.base import EnvironmentProvider, ProviderCheck
from operon.executor.scroll_units import wheel_clicks
from operon.models.capture import CaptureFrame
from operon.models.common import FailureCategory
from operon.models.execution import ExecutedAction
from operon.models.policy import ActionType, AgentAction

logger = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_SERVO_RADIUS = 50
_SERVO_THRESHOLD = 20.0  # DesktopExecutor._SERVO_THRESHOLD_DEFAULT: the uncalibrated visual-servo variance gate
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
        text = action.text or ""
        if text.isascii():
            lines.append(f"pyautogui.write({text!r}, interval=0.02)")
        else:
            # pyautogui.write drops characters X11 has no key for (cafe-with-accent -> caf):
            # paste through the VM's clipboard instead, typing only if pyperclip is missing.
            lines.append(
                f"try:\n    import pyperclip; pyperclip.copy({text!r}); pyautogui.hotkey('ctrl', 'v')\n"
                f"except Exception:\n    pyautogui.write({text!r}, interval=0.02)"
            )
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
        return f"pyautogui.{fn}({wheel_clicks(int(action.scroll_amount or 0))}{at})"
    if t is ActionType.DRAG:
        return (f"pyautogui.moveTo({pos}); pyautogui.dragTo({int(action.x_end or 0)}, "
                f"{int(action.y_end or 0)}, duration=0.5, button='left')")
    return None


class OSWorldProvider(EnvironmentProvider):
    name = "osworld"
    summary = "An OSWorld task VM reached at OPERON_OSWORLD_URL (benchmark runs, KAN-169)."
    # Not the host: Operon's file and shell tools would act on the machine running Operon, not
    # on the OSWorld VM, so the loop must never wrap this executor in the host ToolExecutor.
    runs_on_host = False

    def __init__(self, *, url: str | None = None) -> None:
        self.url = (url if url is not None else get_settings().osworld_url).strip().rstrip("/")

    def platform(self) -> str:
        return "linux"

    def check(self) -> ProviderCheck:
        tools = (get_settings().tools or "").strip().lower()
        if tools not in ("", "off"):
            return ProviderCheck(False, ("OPERON_TOOLS must be off for the osworld environment: "
                                         "the tools would run on this host, not on the OSWorld VM.",))
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
                 transport: httpx.AsyncBaseTransport | None = None, trusted_host: str | None = None,
                 post_action_wait_s: float | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._trusted_host = trusted_host  # the one non-loopback address a secret may go to
        self._dir = Path(artifact_dir) if artifact_dir is not None else desktop_artifacts_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self.batch_child_check = None
        self._post_action_wait_s = (get_settings().osworld_post_action_wait_s
                                    if post_action_wait_s is None else max(0.0, post_action_wait_s))

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
        result, did_input = await self._execute_one(action)
        if did_input and self._post_action_wait_s > 0:
            # The VM replies before the app has finished reacting (a dialog still closing, an app
            # still opening), so let it settle before the loop's next screenshot. Once per action,
            # once per batch; also after a batch that failed part-way, since its earlier input ran.
            await asyncio.sleep(self._post_action_wait_s)
        if did_input:
            # The loop moves this file to the step's after.png. A failed capture never fails the action.
            after = await self._capture_after()
            if after is not None:
                result = result.model_copy(update={"artifact_path": after})
        return result

    async def _capture_after(self) -> str | None:
        try:
            response = await self._http().get("/screenshot")
            response.raise_for_status()
            path = self._dir / f"osworld_after_{uuid4().hex[:8]}.png"
            path.write_bytes(response.content)
            return str(path)
        except Exception:  # noqa: BLE001 - the after shot is a record; losing it must not fail the action
            logger.debug("OSWorld after screenshot skipped", exc_info=True)
            return None

    async def _execute_one(self, action: AgentAction) -> tuple[ExecutedAction, bool]:
        """Run one action; the flag says whether it sent input to the VM (so a post-action wait applies).
        A failed single action sent no input; a part-failed batch reports its earlier children's input."""
        if action.action_type is ActionType.BATCH:
            detail: list[str] = []
            any_input = False
            for index, child in enumerate(action.actions or []):
                check = self.batch_child_check
                if index > 0 and check is not None:
                    blocked = await check(child)
                    if blocked is not None:
                        return ExecutedAction(
                            action=action, success=False,
                            detail=f"Batch stopped before {child.action_type.value}: {blocked}",
                            failure_category=FailureCategory.EXECUTION_ERROR, batch_failed_index=index), any_input
                result, child_input = await self._execute_one(child)
                any_input = any_input or child_input
                detail.append(f"{child.action_type.value}:{'ok' if result.success else 'failed'}")
                if not result.success:
                    return ExecutedAction(action=action, success=False, detail=f"Batch failed: {result.detail}",
                                          failure_category=result.failure_category), any_input
            return ExecutedAction(action=action, success=True, detail=f"Executed batch: {', '.join(detail)}"), any_input
        t = action.action_type
        try:
            if t in (ActionType.STOP, ActionType.WAIT_FOR_USER):
                return ExecutedAction(action=action, success=True, detail=f"{t.value}: nothing to do"), False
            if t is ActionType.WAIT:
                await asyncio.sleep((action.wait_ms or 1000) / 1000)
                return ExecutedAction(action=action, success=True, detail=f"Waited {action.wait_ms or 1000} ms"), False
            if t is ActionType.LAUNCH_APP:
                from operon.agent.app_catalog import launch_targets

                command = launch_targets().get((action.text or "").strip().lower())
                if command is None:
                    return _failed(action, f"launch_app: '{action.text}' is not a catalog application"), False
                await self._shell(["bash", "-c", f"DISPLAY=:0 nohup {command} >/dev/null 2>&1 &"])
                return ExecutedAction(action=action, success=True, detail=f"Launched {action.text}"), True
            if t in _SERVO_ACTIONS and action.x is not None and action.y is not None and not action.is_input_zone:
                blank = await self._blank_target(action)
                if blank is not None:
                    return blank, False
            # KAN-103: the value goes only into the snippet sent to the VM, never into `action`.
            secret_values: tuple[str, ...] = ()
            to_type = action
            if t is ActionType.TYPE and action.text:
                try:
                    resolved, secret_values = resolve_text(action.text)
                except SecretError as exc:
                    return _failed(action, f"type failed: {exc}"), False
                if secret_values and not secret_endpoint_ok(self.base_url, self._trusted_host):
                    return _failed(action, "type failed: the OSWorld VM is not at a loopback address"), False
                if secret_values and not resolved.isascii():
                    # Non-ASCII text is pasted through the VM's clipboard: a secret never is.
                    return _failed(action, "type failed: this secret has a character that cannot be typed without the clipboard"), False
                to_type = action.model_copy(update={"text": resolved})
            snippet = pyautogui_snippet(to_type)
            if snippet is None:
                return _failed(action, f"{t.value} is not supported on an OSWorld machine"), False
            script = "import pyautogui; pyautogui.FAILSAFE = False\n" + snippet
            reply = await self._shell(["python3", "-c", script])
            if reply.get("status") == "error" or reply.get("returncode", 0) not in (0, None):
                return _failed(action, redact(f"VM refused {t.value}: {json.dumps(reply)[:300]}", secret_values)), False
            return ExecutedAction(action=action, success=True, detail=f"Executed {t.value} on the OSWorld VM"), True
        except httpx.HTTPError as exc:
            return _failed(action, f"OSWorld VM unreachable: {exc}"), False

    async def _blank_target(self, action: AgentAction) -> ExecutedAction | None:
        """Visual servo (an invariant: every click goes through it): refuse a click whose
        target region of the VM's screen is uniform, so stale coordinates do not spend the action."""
        try:
            response = await self._http().get("/screenshot")
            response.raise_for_status()
            variance = region_variance(response.content, int(action.x or 0), int(action.y or 0))
        except Exception:  # noqa: BLE001 - a screenshot that cannot be read must not block the action
            logger.debug("OSWorld visual servo skipped", exc_info=True)
            return None
        if variance is None or variance > _SERVO_THRESHOLD:
            return None
        logger.warning("CoordDriftWarning: uniform region at (%s, %s)", action.x, action.y)
        return _failed(
            action, f"visual servo: blank region at ({action.x}, {action.y}); target may have shifted",
            FailureCategory.EXECUTION_TARGET_NOT_FOUND,
        ).model_copy(update={"visual_variance": variance})

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


def _failed(action: AgentAction, detail: str,
            category: FailureCategory = FailureCategory.EXECUTION_ERROR) -> ExecutedAction:
    return ExecutedAction(action=action, success=False, detail=detail, failure_category=category)


_SERVO_ACTIONS = {ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.TRIPLE_CLICK, ActionType.RIGHT_CLICK,
                  ActionType.TYPE}


def region_variance(png: bytes, x: int, y: int, radius: int = _SERVO_RADIUS) -> float | None:
    """Variance of the RGB bytes in the square around (x, y), as DesktopExecutor's servo measures it."""
    import io

    with Image.open(io.BytesIO(png)) as image:
        rgb = image.convert("RGB")
        box = (max(0, x - radius), max(0, y - radius), min(rgb.width, x + radius), min(rgb.height, y + radius))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        raw = rgb.crop(box).tobytes()
    mean = sum(raw) / len(raw)
    return sum((b - mean) ** 2 for b in raw) / len(raw)

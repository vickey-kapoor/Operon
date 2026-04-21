"""Human-in-the-Loop support: LLM message generation, notifications, escalation."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.clients.gemini import GeminiClient

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Page-hint keywords that require a human to unblock the agent.
# These are matched as substrings of the page_hint value.
# -------------------------------------------------------------------
HITL_PAGE_HINT_KEYWORDS: frozenset[str] = frozenset({
    "captcha", "recaptcha", "robot",
    "login", "sign_in", "signin",
    "cookie_consent", "cookie_banner", "gdpr", "consent",
    "age_verification", "age_gate",
    "two_factor", "2fa", "mfa", "otp", "verification_code",
    "terms_and_conditions", "terms_of_service",
    "payment", "checkout", "billing",
    "blocked", "access_denied", "forbidden", "bot_detection",
})

_CONTEXT_PROMPT = """\
You are an assistant explaining to a human why an automated agent needs their help.

The agent was trying to: {intent}
The current page state: {page_hint}
Current URL: {url}
What the agent sees on screen: {elements}

Write exactly 2 short sentences:
1. What happened (why the agent is stuck).
2. What specific action the human needs to take to unblock it.

Be direct and concrete. No filler phrases. Return only the 2 sentences, no labels or formatting."""


async def generate_hitl_message(
    *,
    intent: str,
    page_hint: str,
    url: str | None,
    visible_element_names: list[str],
    gemini_client: GeminiClient | None,
) -> str:
    """Call the LLM to produce a human-readable intervention message."""
    fallback = f"The agent reached a '{page_hint}' page and cannot proceed automatically. Please complete the required action in the browser window, then click Resume."
    if gemini_client is None:
        return fallback
    try:
        prompt = _CONTEXT_PROMPT.format(
            intent=intent,
            page_hint=page_hint,
            url=url or "unknown",
            elements=", ".join(visible_element_names[:10]) or "none visible",
        )
        raw = await gemini_client.generate_policy(prompt)
        msg = raw.strip()
        if msg:
            return msg
    except Exception as exc:
        logger.debug("hitl message generation failed: %s", exc)
    return fallback


def notify_desktop(title: str, message: str) -> None:
    """Fire a best-effort desktop notification without blocking."""
    try:
        if sys.platform == "win32":
            _notify_windows(title, message)
        elif sys.platform == "darwin":
            _notify_macos(title, message)
        else:
            _notify_linux(title, message)
    except Exception as exc:
        logger.debug("desktop notification failed: %s", exc)


def _notify_windows(title: str, message: str) -> None:
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        f'$n = New-Object System.Windows.Forms.NotifyIcon; '
        "$n.Icon = [System.Drawing.SystemIcons]::Information; "
        "$n.Visible = $true; "
        f'$n.ShowBalloonTip(8000, "{title}", "{message}", [System.Windows.Forms.ToolTipIcon]::Warning); '
        "Start-Sleep -Seconds 9; $n.Visible = $false"
    )
    subprocess.Popen(
        ["powershell", "-WindowStyle", "Hidden", "-Command", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _notify_macos(title: str, message: str) -> None:
    script = f'display notification "{message}" with title "{title}"'
    subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _notify_linux(title: str, message: str) -> None:
    subprocess.Popen(["notify-send", title, message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def start_escalation_timer(
    run_id: str,
    *,
    get_status_fn,
    notify_fn,
    intervals_seconds: tuple[int, ...] = (120, 600, 1800),
) -> None:
    """Re-notify at escalating intervals while the run stays WAITING_FOR_USER.

    ``get_status_fn`` should be a zero-arg async callable returning the current
    RunStatus string (or None if the run is gone).
    ``notify_fn`` should be a zero-arg callable that fires the notification.
    """
    labels = ["2 minutes", "10 minutes", "30 minutes"]
    for interval, label in zip(intervals_seconds, labels):
        await asyncio.sleep(interval)
        try:
            status = await get_status_fn()
        except Exception:
            return
        if status != "waiting_for_user":
            return
        logger.info("HITL escalation: run %s still waiting after %s", run_id, label)
        try:
            notify_fn()
        except Exception as exc:
            logger.debug("escalation notify failed: %s", exc)

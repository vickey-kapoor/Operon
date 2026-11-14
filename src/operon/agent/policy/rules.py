"""Explicit rule-first policy checks for deterministic agent cases.

Only rules that can fire under Gemini Computer Use live here. That backend reports
no element list and no page type, so a rule that needs either never runs (KAN-161
removed those).
"""

from __future__ import annotations

import logging
import re

from operon.agent import control as run_control
from operon.agent.app_catalog import launch_targets
from operon.models.memory import MemoryHint
from operon.models.perception import PageHint, ScreenPerception
from operon.models.policy import ActionType, AgentAction, PolicyDecision
from operon.models.state import AgentState

logger = logging.getLogger(__name__)


# Intent patterns that indicate the user wants to open/launch a desktop app.
_LAUNCH_INTENT_RE = re.compile(
    r"(?:open|launch|start|run|use)\s+([a-zA-Z0-9][a-zA-Z0-9 _\-]*?)(?:\s+and\b|\s+then\b|\s+to\b|\s+write\b|\s+in\b|$)",
    re.IGNORECASE,
)

_LEADING_WORDS_RE = re.compile(r"^(?:the|a|an|my|up)\s+")
_TRAILING_WORDS_RE = re.compile(r"\s+(?:app|application|program|window)$")


def _catalog_app_name(raw_name: str) -> str | None:
    """The catalog app the intent names, or None.

    The whole name must be a catalog app, give or take an article and a trailing
    "app": "open the settings page" or "open the account settings" names a page,
    not the Settings app, so a catalog name inside a longer phrase never counts."""
    name = raw_name.strip().lower()
    previous = None
    while name != previous:
        previous = name
        name = _TRAILING_WORDS_RE.sub("", _LEADING_WORDS_RE.sub("", name)).strip()
    return name if name in launch_targets() else None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PolicyRuleEngine:
    """Small explicit rule engine that runs before the LLM-backed policy.

    Engine primitives run unconditionally for every benchmark.
    """

    def __init__(self, *, launch_apps: bool = True) -> None:
        # Desktop only: in the browser "open X" means a page or a site, not an app.
        self.launch_apps = launch_apps
        self._last_fired_rule: str | None = None

    def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> PolicyDecision | None:
        self._last_fired_rule = None

        # Engine primitives — always active, checked in priority order
        _primitives: list[tuple[str, object]] = [
            ("_task_success_stop_rule", lambda: self._task_success_stop_rule(perception)),
            ("_prefer_launch_app_rule", lambda: self._prefer_launch_app_rule(state, perception)),
        ]
        _disabled = run_control.get_disabled_rules()

        for rule_name, call in _primitives:
            if rule_name in _disabled:
                continue
            decision = call()  # type: ignore[operator]
            if decision is not None:
                self._last_fired_rule = rule_name
                return decision
        return None

    def last_fired_rule_name(self) -> str | None:
        """Return the name of the rule that fired in the most recent choose_action call."""
        return self._last_fired_rule

    # ------------------------------------------------------------------
    # Engine primitives
    # ------------------------------------------------------------------

    def _prefer_launch_app_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision | None:
        """Force LAUNCH_APP when the intent is to open a known desktop application.

        Fires at higher priority than _search_query_rule so the agent never wastes
        steps typing an app name into the Windows Search bar.  Only triggers when:
        - The intent matches 'open/launch/start/run/use <app>'
        - The app name maps to a known launch_app command
        - A successful LAUNCH_APP for this app has not already been recorded
        """
        if not self.launch_apps:
            return None
        # Don't fire if we've already launched an app this run
        if any(
            h.action.action_type is ActionType.LAUNCH_APP and h.success
            for h in state.action_history
        ):
            return None

        intent = state.intent or ""
        m = _LAUNCH_INTENT_RE.search(intent)
        if m is None:
            return None

        raw_name = m.group(1).strip().lower()
        alias = _catalog_app_name(raw_name)
        if alias is None:
            return None

        logger.info(
            "_prefer_launch_app_rule: intent=%r → launch_app(%r)",
            intent[:80], alias,
        )
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.LAUNCH_APP, text=alias),
            rationale=(
                f"Intent requires '{raw_name}' — using launch_app directly "
                "instead of searching to avoid screen-state churn."
            ),
            confidence=0.98,
            active_subgoal=f"launch {raw_name}",
            expected_change="content",
        )

    def _task_success_stop_rule(self, perception: ScreenPerception) -> PolicyDecision | None:
        if perception.page_hint is not PageHint.FORM_SUCCESS and not _has_success_signal(perception):
            return None
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.STOP),
            rationale="Task success state is already visible.",
            confidence=1.0,
            active_subgoal="verify_success",
        )



def _has_success_signal(perception: ScreenPerception) -> bool:
    success_tokens = (
        "thank you",
        "submitted successfully",
        "submission successful",
        "submission complete",
        "task completed",
        "message submitted",
        "form submitted",
        "successfully submitted",
        "sent successfully",
        "message sent",
    )
    for element in perception.visible_elements:
        if any(token in element.primary_name.lower() for token in success_tokens):
            return True
    return any(token in perception.summary.lower() for token in success_tokens)

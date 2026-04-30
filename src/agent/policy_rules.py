"""Explicit rule-first policy checks for deterministic agent cases."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from urllib.parse import quote_plus, urljoin

from src.agent.hitl import HITL_PAGE_HINT_KEYWORDS
from src.agent.selector import DeterministicTargetSelector
from src.models.common import FailureCategory
from src.models.memory import MemoryHint
from src.models.perception import PageHint, ScreenPerception, UIElement, UIElementType
from src.models.policy import ActionType, AgentAction, PolicyDecision
from src.models.selector import (
    SelectorConfidenceBand,
    SelectorTrace,
    TargetEvidence,
    TargetIntent,
    TargetIntentAction,
)
from src.models.state import AgentState
from src.models.verification import VerificationStatus

logger = logging.getLogger(__name__)

# Distance threshold (px) for the semantic anchor check.
# A targeted coordinate more than this many pixels from every visible element's
# bounding box is treated as empty-space hallucination and intercepted.
_ANCHOR_DISTANCE_THRESHOLD: float = 15.0

# Action types whose (x, y) coordinates represent intended element targets.
# Excluded: SCROLL (viewport centre), HOVER (may target empty space), DRAG (start/end
# coords are often off-element), SCREENSHOT_REGION (bounding box endpoints).
_COORD_ANCHOR_ACTION_TYPES: frozenset[ActionType] = frozenset({
    ActionType.CLICK,
    ActionType.DOUBLE_CLICK,
    ActionType.RIGHT_CLICK,
    ActionType.TYPE,
})

# Intent patterns that indicate the user wants to open/launch a desktop app.
_LAUNCH_INTENT_RE = re.compile(
    r"(?:open|launch|start|run|use)\s+([a-zA-Z0-9][a-zA-Z0-9 _\-]*?)(?:\s+and\b|\s+then\b|\s+to\b|\s+write\b|\s+in\b|$)",
    re.IGNORECASE,
)

# Canonical app name → argument passed to launch_app executor.
# Mirrors _APP_ALIASES in desktop.py but lives here so policy_rules has no
# circular dependency on the executor layer.
_LAUNCH_APP_NAMES: dict[str, str] = {
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "paint": "mspaint",
    "mspaint": "mspaint",
    "explorer": "explorer",
    "file explorer": "explorer",
    "vs code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "terminal": "wt",
    "windows terminal": "wt",
    "powershell": "powershell",
    "cmd": "cmd",
    "command prompt": "cmd",
    "task manager": "taskmgr",
    "settings": "ms-settings:",
}

# Labels that suggest a button dismisses / rejects rather than accepts.
_DISMISS_TOKENS: frozenset[str] = frozenset({
    "don't", "dont", "dismiss", "close", "cancel", "no thanks", "skip",
    "not now", "maybe later", "decline", "reject", "later", "no", "×", "✕",
})

# A benchmark rule plugin is any callable that matches this signature.
# Return a PolicyDecision to short-circuit the engine, or None to pass through.
BenchmarkRulePlugin = Callable[
    [AgentState, ScreenPerception, list[MemoryHint]],
    PolicyDecision | None,
]


# ---------------------------------------------------------------------------
# Form benchmark-specific rules
# ---------------------------------------------------------------------------

def form_submit_when_ready_rule(
    state: AgentState,
    perception: ScreenPerception,
    memory_hints: list[MemoryHint],
) -> PolicyDecision | None:
    if perception.page_hint is not PageHint.FORM_PAGE:
        return None
    if not _form_fields_completed(state, perception):
        return None
    submit_button = _submit_button(perception)
    if submit_button is None:
        return None
    return PolicyDecision(
        action=AgentAction(
            action_type=ActionType.CLICK,
            target_element_id=submit_button.element_id,
            x=submit_button.x + max(1, submit_button.width // 2),
            y=submit_button.y + max(1, submit_button.height // 2),
        ),
        rationale="Required form fields are already filled; submit the form.",
        confidence=0.97,
        active_subgoal="submit_form",
    )


# ---------------------------------------------------------------------------
# Default plugin registry — benchmark name → plugins
# ---------------------------------------------------------------------------

BENCHMARK_PLUGINS: dict[str, list[BenchmarkRulePlugin]] = {}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PolicyRuleEngine:
    """Small explicit rule engine that runs before the LLM-backed policy.

    Engine primitives run unconditionally for every benchmark.
    Benchmark-specific plugins are registered per benchmark name and only
    run when that benchmark is active.
    """

    def __init__(
        self,
        selector: DeterministicTargetSelector | None = None,
        plugins: dict[str, list[BenchmarkRulePlugin]] | None = None,
    ) -> None:
        self.selector = selector or DeterministicTargetSelector()
        self._plugins = plugins if plugins is not None else dict(BENCHMARK_PLUGINS)
        self._latest_selector_traces: list[SelectorTrace] = []
        self._cached_intermediates: tuple[list[UIElement], list[list[UIElement]]] | None = None
        # Tracks consecutive steps where a HITL keyword matched, per run_id.
        # HITL only fires after HITL_DEBOUNCE_THRESHOLD consecutive matches to
        # prevent false positives from a single bad perception.
        self._hitl_consecutive: dict[str, int] = {}
        # Tracks form option labels (lowercased) that this rule has already clicked,
        # keyed by run_id. Survives page scrolls — coord-based tracking does not.
        self._form_options_clicked: dict[str, set[str]] = {}
        # Tracks which text field TYPES (name, email, password, message) have been
        # filled by the rule this run. Keyed by run_id to prevent cross-run bleed.
        self._form_fields_filled: dict[str, set[str]] = {}

    _HITL_DEBOUNCE_THRESHOLD: int = 2

    def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
        benchmark_name: str | None = None,
    ) -> PolicyDecision | None:
        self._latest_selector_traces = []
        self._last_fired_rule: str | None = None
        self._cached_intermediates = (
            self.selector._label_like_text_candidates(perception),
            self.selector._visual_groups(perception),
        )

        # Benchmark-specific plugins run first (higher priority / more specific)
        if benchmark_name is not None:
            from src.benchmarks.registry import BENCHMARK_REGISTRY
            for plugin in BENCHMARK_REGISTRY.get_rules(benchmark_name) or self._plugins.get(benchmark_name, []):
                decision = plugin(state, perception, memory_hints)
                if decision is not None:
                    self._last_fired_rule = getattr(plugin, "__name__", type(plugin).__name__)
                    return decision

        # Engine primitives — always active, checked in priority order
        _primitives: list[tuple[str, object]] = [
            ("_human_intervention_rule", lambda: self._human_intervention_rule(state, perception)),
            ("_task_success_stop_rule", lambda: self._task_success_stop_rule(perception)),
            ("_prefer_launch_app_rule", lambda: self._prefer_launch_app_rule(state, perception)),
            ("_form_visible_field_fill_rule", lambda: self._form_visible_field_fill_rule(state, perception)),
            ("_dropdown_menu_select_rule", lambda: self._dropdown_menu_select_rule(state, perception)),
            ("_avoid_identical_type_retry", lambda: self._avoid_identical_type_retry(state, perception, memory_hints)),
            ("_no_progress_recovery_rule", lambda: self._no_progress_recovery_rule(state, perception)),
            ("_dismiss_blocking_overlay_rule", lambda: self._dismiss_blocking_overlay_rule(state, perception)),
            ("_search_query_rule", lambda: self._search_query_rule(state, perception)),
        ]
        for rule_name, call in _primitives:
            decision = call()  # type: ignore[operator]
            if decision is not None:
                self._last_fired_rule = rule_name
                return decision
        return None

    def last_fired_rule_name(self) -> str | None:
        """Return the name of the rule that fired in the most recent choose_action call."""
        return getattr(self, "_last_fired_rule", None)

    def register_plugins(self, benchmark_name: str, plugins: list[BenchmarkRulePlugin]) -> None:
        """Register additional plugins for a benchmark at runtime."""
        self._plugins.setdefault(benchmark_name, []).extend(plugins)

    def latest_selector_traces(self) -> list[SelectorTrace]:
        return list(self._latest_selector_traces)

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
        app_cmd = _LAUNCH_APP_NAMES.get(raw_name)
        if app_cmd is None:
            # Partial-match fallback: "open vs code" → matches "vs code" key
            for alias, cmd in _LAUNCH_APP_NAMES.items():
                if alias in raw_name or raw_name.startswith(alias):
                    app_cmd = cmd
                    raw_name = alias
                    break
        if app_cmd is None:
            return None

        logger.info(
            "_prefer_launch_app_rule: intent=%r → launch_app(%r)",
            intent[:80], app_cmd,
        )
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.LAUNCH_APP, text=app_cmd),
            rationale=(
                f"Intent requires '{raw_name}' — using launch_app directly "
                "instead of searching to avoid screen-state churn."
            ),
            confidence=0.98,
            active_subgoal=f"launch {raw_name}",
            expected_change="content",
        )

    def _avoid_identical_type_retry(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
    ) -> PolicyDecision | None:
        if not _has_hint(memory_hints, "avoid_identical_type_retry"):
            return None

        last_action = state.action_history[-1].action if state.action_history else None
        if last_action is None or last_action.action_type is not ActionType.TYPE:
            return None

        last_failure = None
        if state.verification_history:
            last_failure = state.verification_history[-1].failure_category
        if last_failure is None and state.action_history:
            last_failure = state.action_history[-1].failure_category
        if last_failure not in {
            FailureCategory.EXECUTION_TARGET_NOT_FOUND,
            FailureCategory.EXECUTION_TARGET_NOT_EDITABLE,
        }:
            return None

        if last_action.target_element_id is None:
            return None
        if not _has_recent_target_failure_signal(state, last_action.target_element_id, last_failure):
            return None

        target = self._resolve_target_element(
            perception,
            target_element_id=last_action.target_element_id,
            subgoal=state.current_subgoal,
        )
        if target is None:
            target = self._fallback_input_target(perception, state.current_subgoal, last_action.target_element_id)

        if target is not None and perception.focused_element_id != target.element_id:
            return _click_decision(target, f"Re-establish focus on {target.primary_name} before retrying type.")

        return PolicyDecision(
            action=AgentAction(action_type=ActionType.WAIT, wait_ms=500),
            rationale="Previous type action failed on the same target; avoid repeating the identical type action blindly.",
            confidence=0.93,
            active_subgoal=f"re-assess {last_action.target_element_id or 'input'}",
        )

    # Substrings in element_id that signal a secondary menu / dropdown child item.
    _DROPDOWN_ID_SIGNALS: frozenset[str] = frozenset({
        "dropdown", "submenu", "menu_item", "menuitem", "popover", "flyout", "child",
    })

    def _dropdown_menu_select_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision | None:
        """Fire when a dropdown/submenu is open and the agent should pick a child item.

        Fires when ALL of the following hold:
        - The last action was a successful CLICK on a non-dropdown element (the parent trigger).
        - The current perception contains at least one element whose element_id includes a
          known dropdown-child signal substring (dropdown, submenu, menu_item, etc.).
        - The last-clicked element is still visible (meaning no navigation occurred — same page).

        Chooses the best-matching child item by scoring against intent + subgoal keywords.
        This prevents the LLM from either re-clicking the parent or picking an unrelated element.
        """
        # Form pages have always-present select elements whose IDs contain "dropdown" —
        # these are not navigation overlay menus and must not trigger this rule.
        if perception.page_hint is PageHint.FORM_PAGE:
            return None

        if not state.action_history:
            return None

        last_entry = state.action_history[-1]
        last = last_entry.action
        if last.action_type is not ActionType.CLICK:
            return None
        # Only fire when the preceding click actually succeeded — a failed click
        # leaves the screen unchanged, so there is no open dropdown to select from.
        if not last_entry.success:
            return None
        last_id = last.target_element_id
        if last_id is None:
            return None

        # Only fire when the last-clicked element is itself NOT a dropdown child
        # (prevents rule from firing repeatedly after we've already selected a child).
        if any(sig in last_id.lower() for sig in self._DROPDOWN_ID_SIGNALS):
            return None

        # Collect visible child items — elements whose IDs signal secondary menu membership
        # AND whose element_type is consistent with a real menu item (not a text area,
        # window chrome, or input field that Gemini mis-labelled with "dropdown" in its id).
        _MENU_TYPES: frozenset[UIElementType] = frozenset({
            UIElementType.BUTTON, UIElementType.LINK, UIElementType.ICON,
        })
        child_items = [
            e for e in perception.visible_elements
            if any(sig in e.element_id.lower() for sig in self._DROPDOWN_ID_SIGNALS)
            and e.usable_for_targeting
            and e.element_type in _MENU_TYPES
        ]
        if not child_items:
            return None

        # Score each child by how many intent/subgoal keywords appear in its label or id.
        all_tokens = _match_tokens(state.intent) | _match_tokens(state.current_subgoal)

        def _score(el: UIElement) -> int:
            text = (el.primary_name + " " + el.element_id).lower()
            return sum(1 for t in all_tokens if t in text)

        best = max(child_items, key=_score)
        best_score = _score(best)
        # Don't fire if no child item matches any intent/subgoal token — this prevents
        # the rule from accidentally selecting unrelated site menus (e.g. Wikipedia's
        # "Tools" dropdown) whose element_ids happen to contain "dropdown".
        if best_score == 0:
            return None
        logger.info(
            "Dropdown-menu rule: last click was '%s', %d child items visible. "
            "Selecting best match '%s' (score=%d).",
            last_id, len(child_items), best.element_id, best_score,
        )
        return PolicyDecision(
            action=AgentAction(
                action_type=ActionType.CLICK,
                target_element_id=best.element_id,
                x=best.x + max(1, best.width // 2),
                y=best.y + max(1, best.height // 2),
            ),
            rationale=(
                f"Dropdown menu is open after clicking '{last_id}'. "
                f"Selecting child item '{best.primary_name}' that best matches the current intent."
            ),
            confidence=0.92,
            active_subgoal=f"select_menu_item:{best.element_id}",
        )

    def _human_intervention_rule(
        self, state: AgentState, perception: ScreenPerception
    ) -> PolicyDecision | None:
        """Fire WAIT_FOR_USER when the page requires human action (CAPTCHA, login, etc.).

        Requires HITL_DEBOUNCE_THRESHOLD consecutive steps with a matching page
        hint before pausing — prevents false positives from a single bad perception.
        """
        hint = perception.page_hint.value.lower()
        matched = next((kw for kw in HITL_PAGE_HINT_KEYWORDS if kw in hint), None)
        run_id = state.run_id

        if matched is None:
            # Page no longer looks like a HITL page — reset the counter.
            self._hitl_consecutive.pop(run_id, None)
            return None

        count = self._hitl_consecutive.get(run_id, 0) + 1
        self._hitl_consecutive[run_id] = count

        if count < self._HITL_DEBOUNCE_THRESHOLD:
            logger.debug(
                "HITL keyword %r matched on run %s (hit %d/%d) — holding off for next step",
                matched, run_id, count, self._HITL_DEBOUNCE_THRESHOLD,
            )
            return None

        # Threshold reached — fire and reset so resuming works cleanly.
        self._hitl_consecutive.pop(run_id, None)
        return PolicyDecision(
            action=AgentAction(
                action_type=ActionType.WAIT_FOR_USER,
                text=f"hitl:{hint}",
            ),
            rationale=f"Page requires human action ({hint}). Agent pausing for human to complete this step.",
            confidence=0.99,
            active_subgoal=f"human_intervention_required:{hint}",
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

    _NO_PROGRESS_REPEAT_THRESHOLD: int = 3

    def _no_progress_recovery_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision | None:
        """Fire when the same action+target has been repeated N times with no progress.

        Issues a visual perturbation (Escape key) to clear stale OS/DOM render
        state and sets force_fresh_perception so the next capture waits for the
        UI to settle. Safer than a geometry-based adjacent click, which risks
        hitting Cancel buttons or diverging form state.
        """
        n = self._NO_PROGRESS_REPEAT_THRESHOLD
        if len(state.action_history) < n:
            return None

        recent = [h.action for h in state.action_history[-n:]]
        if len({a.action_type for a in recent}) != 1:
            return None
        if len({a.target_element_id for a in recent}) != 1:
            return None
        stuck_target_id = recent[0].target_element_id
        if stuck_target_id is None:
            return None

        if not any(e.element_id == stuck_target_id for e in perception.visible_elements):
            return None

        stuck_type = recent[0].action_type.value
        logger.info(
            "No-progress rule: %r repeated %d× on '%s'. Issuing Escape to clear stale render state.",
            stuck_type, n, stuck_target_id,
        )
        state.force_fresh_perception = True
        return PolicyDecision(
            action=AgentAction(
                action_type=ActionType.PRESS_KEY,
                key="escape",
            ),
            rationale=(
                f"'{stuck_type}' on '{stuck_target_id}' repeated {n}× with no progress. "
                "Pressing Escape to clear stale OS/render state before retrying."
            ),
            confidence=0.85,
            active_subgoal=f"visual_perturbation_before_{stuck_target_id}",
        )

    def _dismiss_blocking_overlay_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision | None:
        """Dismiss any blocking dialog/overlay when the agent is stuck on a target.

        Fires when ALL of the following hold:
        - A dialog, modal, banner, or non-interactable overlay element is visible.
        - The last 2 consecutive actions were clicks on the same target (stuck signal).
        - A dismiss-like button exists within or near the blocking element's bounds.
        - The dismiss button is not itself the stuck target (avoid infinite loops).
        """
        dialogs = _blocking_overlay_elements(perception)
        if not dialogs:
            return None

        # Stuck signal: 2+ consecutive clicks on the same target_element_id
        if len(state.action_history) < 2:
            return None
        recent = [h.action for h in state.action_history[-2:]]
        if not all(a.action_type is ActionType.CLICK for a in recent):
            return None
        stuck_ids = {a.target_element_id for a in recent}
        if len(stuck_ids) != 1 or None in stuck_ids:
            return None
        stuck_target = next(iter(stuck_ids))

        for overlay in dialogs:
            btn = _best_dismiss_button(overlay, perception, exclude_id=stuck_target)
            if btn is None:
                continue
            logger.info(
                "Dismiss-overlay rule: blocking element %r detected, stuck on %r. "
                "Dismissing via %r.",
                overlay.element_id, stuck_target, btn.element_id,
            )
            return PolicyDecision(
                action=AgentAction(
                    action_type=ActionType.CLICK,
                    target_element_id=btn.element_id,
                    x=btn.x + max(1, btn.width // 2),
                    y=btn.y + max(1, btn.height // 2),
                ),
                rationale=(
                    f"Blocking overlay '{overlay.element_id}' is present while agent is stuck "
                    f"clicking '{stuck_target}'. Dismissing via '{btn.primary_name}' to unblock."
                ),
                confidence=0.93,
                active_subgoal="dismiss_blocking_overlay",
            )
        return None

    def _search_query_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision | None:
        query = _extract_search_query(state)
        if query is None:
            return None

        _POST_SEARCH_HINTS = {"article_page", "search_results", "form_success", "search_page"}
        _hint_str = str(perception.page_hint)

        # Compute _already_searched BEFORE the page-hint bail so we can distinguish
        # "homepage mis-identified as article_page" (not searched yet → proceed)
        # from "genuinely on post-search page after searching" (done → bail).
        _last_trace = state.last_rule_trace or ""
        _already_searched = any(
            h.action.action_type is ActionType.TYPE
            and h.action.text == query
            and h.action.press_enter
            and h.success
            for h in state.action_history
        )

        # Don't re-issue a search if already on a known post-search page AND the
        # search was already submitted. When the page_hint is article_page but we
        # haven't searched yet (e.g. Wikipedia homepage mis-labelled), still fire.
        if _hint_str in _POST_SEARCH_HINTS and _already_searched:
            return None

        if _already_searched:
            # Search was already submitted but we're still not on a results/article page —
            # keyboard Enter didn't navigate. This applies whether the rule or the LLM typed.
            _needs_navigate = (
                (_hint_str not in _POST_SEARCH_HINTS)
                and (
                    "_search_query_rule" in _last_trace and "action=type" in _last_trace
                    or "_hint_str" not in _POST_SEARCH_HINTS  # LLM-typed: check page didn't change
                )
            )
            if _hint_str not in _POST_SEARCH_HINTS:
                # Build a site-aware search URL.
                _base = state.start_url or ""
                if "wikipedia.org" in _base:
                    from urllib.parse import urlparse
                    _origin = f"{urlparse(_base).scheme}://{urlparse(_base).netloc}"
                    _search_url = f"{_origin}/w/index.php?search={quote_plus(query)}"
                elif "github.com" in _base:
                    # Translate common natural-language patterns to GitHub search syntax.
                    _lang_m = re.search(r"\b(python|javascript|typescript|rust|go|java|c\+\+|ruby|swift|kotlin)\b", query, re.IGNORECASE)
                    _star_m = re.search(r"(\d[\d,]*)\s*(?:k|,?000)?\s*\+?\s*stars?", query, re.IGNORECASE)
                    _star_thresh_m = re.search(r"more than\s+(\d[\d,]*)", query, re.IGNORECASE)
                    if _lang_m and (_star_m or _star_thresh_m):
                        _lang = _lang_m.group(1).lower().replace("+", "%2B")
                        _raw = (_star_thresh_m or _star_m).group(1).replace(",", "")
                        _stars = int(_raw) if len(_raw) <= 6 else int(_raw)
                        _search_url = f"https://github.com/search?q=language%3A{_lang}+stars%3A%3E{_stars}&type=repositories&s=stars&o=desc"
                    elif _lang_m:
                        _search_url = f"https://github.com/trending/{_lang_m.group(1).lower()}"
                    else:
                        _search_url = f"https://github.com/search?q={quote_plus(query)}&type=repositories"
                else:
                    _search_url = urljoin(_base, f"search?q={quote_plus(query)}")
                return PolicyDecision(
                    action=AgentAction(action_type=ActionType.NAVIGATE, url=_search_url),
                    rationale=f"Search submitted but page did not navigate — using direct search URL for '{query}'.",
                    confidence=0.92,
                    active_subgoal=f"navigate to search results for {query}",
                )
            return None

        search_input = self._search_input_target(perception)
        if search_input is not None:
            mid_x = search_input.x + max(1, search_input.width // 2)
            mid_y = search_input.y + max(1, search_input.height // 2)
            if perception.focused_element_id != search_input.element_id:
                return _click_decision(
                    search_input,
                    f"Focus {search_input.primary_name} so the search query can be entered.",
                )
            # Include coordinates so the executor clicks the input to guarantee
            # focus before typing — prevents keyboard events going to document body.
            return PolicyDecision(
                action=AgentAction(
                    action_type=ActionType.TYPE,
                    target_element_id=search_input.element_id,
                    x=mid_x,
                    y=mid_y,
                    text=query,
                    press_enter=True,
                ),
                rationale=f"Type the search query '{query}' into {search_input.primary_name} and submit it.",
                confidence=0.97,
                active_subgoal=f"search for {query}",
            )

        search_trigger = _search_trigger_target(perception)
        if search_trigger is None:
            return None

        # Multi-step search trigger flow: click → wait (overlay animation) → type.
        # If that still fails, fall back to navigating directly to a search URL.
        # Checking last_rule_trace tells us which step of the flow we're in.
        _last_trace = state.last_rule_trace or ""
        if "_search_query_rule" in _last_trace:
            if "action=type" in _last_trace:
                # TYPE didn't land in any focused input — navigate directly to search URL.
                _search_url = urljoin(state.start_url or "", f"search?q={quote_plus(query)}")
                return PolicyDecision(
                    action=AgentAction(action_type=ActionType.NAVIGATE, url=_search_url),
                    rationale=f"Search overlay TYPE failed — navigating directly to search URL for '{query}'.",
                    confidence=0.92,
                    active_subgoal=f"navigate to search results for {query}",
                )
            if "action=wait" in _last_trace:
                # Waited for overlay — now type into the focused input.
                # Derive coordinates from perception so the executor can click-to-focus
                # before dispatching keystrokes; without coords the TYPE lands nowhere.
                _overlay_input = self._search_input_target(perception)
                if _overlay_input is not None:
                    _mid_x = _overlay_input.x + max(1, _overlay_input.width // 2)
                    _mid_y = _overlay_input.y + max(1, _overlay_input.height // 2)
                    return PolicyDecision(
                        action=AgentAction(
                            action_type=ActionType.TYPE,
                            target_element_id=_overlay_input.element_id,
                            x=_mid_x,
                            y=_mid_y,
                            text=query,
                            press_enter=True,
                        ),
                        rationale=f"Search overlay open — typing '{query}' at ({_mid_x},{_mid_y}).",
                        confidence=0.94,
                        active_subgoal=f"search for {query}",
                    )
                return PolicyDecision(
                    action=AgentAction(
                        action_type=ActionType.TYPE,
                        text=query,
                        press_enter=True,
                    ),
                    rationale=f"Search overlay open after wait — typing '{query}' into focused input.",
                    confidence=0.90,
                    active_subgoal=f"search for {query}",
                )
            if "action=click" in _last_trace:
                # Just clicked the trigger — wait 600ms for overlay to open and focus.
                return PolicyDecision(
                    action=AgentAction(action_type=ActionType.WAIT, wait_ms=600),
                    rationale="Search trigger clicked — waiting for overlay to open and focus input.",
                    confidence=0.94,
                    active_subgoal="wait for search overlay",
                )

        return PolicyDecision(
            action=AgentAction(
                action_type=ActionType.CLICK,
                target_element_id=search_trigger.element_id,
                x=search_trigger.x + max(1, search_trigger.width // 2),
                y=search_trigger.y + max(1, search_trigger.height // 2),
            ),
            rationale=f"Activate {search_trigger.primary_name} so the search input becomes available.",
            confidence=0.94,
            active_subgoal=f"focus {search_trigger.element_id}",
        )

    def _form_visible_field_fill_rule(
        self,
        state: AgentState,
        perception: ScreenPerception,
    ) -> PolicyDecision | None:
        """Deterministically fill visible form fields and click matching checkboxes/radios.

        Fires on form_page when a visible input matches a field value or option
        extracted from intent that has not yet been acted on this run. This
        bypasses LLM confusion about stale subgoals after verification failures.
        """
        if perception.page_hint is not PageHint.FORM_PAGE:
            return None

        intent = state.intent or ""

        # Extract structured values from intent
        email_match = re.search(r"[\w.+\-]+@[\w.\-]+\.\w+", intent)
        email = email_match.group() if email_match else None

        name_match = re.search(
            r"[Nn]ame(?:\s+as)?\s+['\"]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)['\"]?",
            intent,
        )
        name = name_match.group(1) if name_match else None

        # Per-run field-type fill tracking: prevents re-filling the same FIELD TYPE
        # even when the same value is used in multiple fields (e.g. email+password).
        run_id = state.run_id
        _fields_filled: set[str] = self._form_fields_filled.setdefault(run_id, set())

        # Collect text values already typed this run (reserved for future duplicate-type detection)
        _recently_typed: set[str] = {
            a.action.text
            for a in state.action_history
            if a.action.action_type is ActionType.TYPE and a.action.text
        }
        # Coord-based type tracking was removed — form scrolling displaces elements
        # by 100-200px between steps, making positional deduplication unreliable.
        # Value-based deduplication (recently_typed) is the single source of truth.

        intent_options: list[str] = re.findall(r"['\"]([^'\"]+)['\"]", intent)

        # Also capture unquoted color/option mentions e.g. "set the color to blue"
        color_match = re.search(r"(?:color|colour)(?:\s+to)?\s+(\w+)", intent, re.IGNORECASE)
        if color_match:
            color_val = color_match.group(1)
            if color_val.lower() not in {o.lower() for o in intent_options}:
                intent_options.append(color_val)

        # Track which option labels have been clicked this run using the rule engine's
        # per-run label set. This survives page scrolls (coord-based tracking does not).
        run_id = state.run_id
        _clicked_labels: set[str] = self._form_options_clicked.setdefault(run_id, set())

        # Coord fallback for elements not yet in the label set (e.g. first-pass clicks).
        click_coords: list[tuple[int, int]] = [
            (a.action.x, a.action.y)
            for a in state.action_history
            if a.action.action_type is ActionType.CLICK
            and a.action.x is not None
            and a.action.y is not None
        ]

        def _coord_already_clicked(cx: int, cy: int) -> bool:
            return any(abs(px - cx) < 60 and abs(py - cy) < 60 for px, py in click_coords)

        def _option_already_handled(opt_lower: str, cx: int, cy: int) -> bool:
            return opt_lower in _clicked_labels or _coord_already_clicked(cx, cy)

        for element in perception.visible_elements:
            if not element.is_interactable:
                continue
            label = element.primary_name.lower()
            mid_x = element.x + max(1, element.width // 2)
            mid_y = element.y + max(1, element.height // 2)

            if element.element_type is UIElementType.INPUT:
                # Text inputs: name first, then email (matches typical form top-to-bottom order)
                if "name" in label and name and "name" not in _fields_filled:
                    _fields_filled.add("name")
                    return PolicyDecision(
                        action=AgentAction(
                            action_type=ActionType.TYPE,
                            target_element_id=element.element_id,
                            x=mid_x,
                            y=mid_y,
                            text=name,
                            press_enter=False,
                            clear_before_typing=True,
                        ),
                        rationale=f"Form name field visible — typing '{name}' per intent.",
                        confidence=0.97,
                        active_subgoal="fill name field",
                    )

                # Fill message/textarea field ONLY when the intent explicitly requests it.
                # Filling a message field that's not required by intent can disrupt forms
                # that validate all filled fields, causing unexpected form behavior.
                _intent_wants_message = bool(re.search(r"\bmessage\b", intent, re.IGNORECASE))
                _text_fields_partial_done = (name is None or "name" in _fields_filled) and (email is None or "email" in _fields_filled or "password" in _fields_filled)
                _msg_match = re.search(r"message(?:\s+as)?\s+['\"]([^'\"]+)['\"]", intent, re.IGNORECASE)
                _msg_text = _msg_match.group(1) if _msg_match else "Hello, I am filling out this form."
                if "message" in label and _intent_wants_message and _text_fields_partial_done and (name or email) and "message" not in _fields_filled:
                    _fields_filled.add("message")
                    return PolicyDecision(
                        action=AgentAction(
                            action_type=ActionType.TYPE,
                            target_element_id=element.element_id,
                            x=mid_x,
                            y=mid_y,
                            text=_msg_text,
                            press_enter=False,
                            clear_before_typing=True,
                        ),
                        rationale="Form message field visible and intent requires it — typing message.",
                        confidence=0.92,
                        active_subgoal="fill message field",
                    )

                # Match email field by "email" label. Only fill "password" labeled fields
                # with the email value when the intent explicitly mentions a password
                # context (e.g., "login with password X"). For forms that just have a
                # Password field that isn't required by the task, skip it — attempting
                # to fill it with the email value can break form validation.
                _intent_wants_password = bool(re.search(r"\bpassword\b", intent, re.IGNORECASE))
                _field_type_for_email = None
                if "email" in label:
                    _field_type_for_email = "email"
                elif "password" in label and _intent_wants_password and "name" not in label and "message" not in label:
                    # Only fill a "password" field with the email value when the intent
                    # explicitly references a password context.
                    _field_type_for_email = "password"
                _email_label_match = _field_type_for_email is not None
                _email_needs_fill = email and _field_type_for_email not in _fields_filled
                if _email_label_match and _email_needs_fill:
                    _fields_filled.add(_field_type_for_email)
                    return PolicyDecision(
                        action=AgentAction(
                            action_type=ActionType.TYPE,
                            target_element_id=element.element_id,
                            x=mid_x,
                            y=mid_y,
                            text=email,
                            press_enter=False,
                            clear_before_typing=True,
                        ),
                        rationale=f"Form {_field_type_for_email} field visible — typing email value per intent.",
                        confidence=0.97,
                        active_subgoal=f"fill {_field_type_for_email} field",
                    )

                # Checkboxes and radio buttons: click if label matches an intent option.
                # Exclude structured values (email addresses, "First Last" names) — they
                # appear in intent_options from the quoted-string regex but are not clickable.
                _clickable_options = [
                    opt for opt in intent_options
                    if "@" not in opt and opt != name and opt != email
                ]
                for option in _clickable_options:
                    if option.lower() in label and not _option_already_handled(option.lower(), mid_x, mid_y):
                        _clicked_labels.add(option.lower())
                        return PolicyDecision(
                            action=AgentAction(
                                action_type=ActionType.CLICK,
                                target_element_id=element.element_id,
                                x=mid_x,
                                y=mid_y,
                            ),
                            rationale=f"Form option '{option}' visible and unclicked — clicking now.",
                            confidence=0.96,
                            active_subgoal=f"select {option}",
                        )

        text_fields_done = (
            (name is None or "name" in _fields_filled)
            and (email is None or "email" in _fields_filled or "password" in _fields_filled)
        )

        if text_fields_done and intent_options and (name or email):
            # Check if all intent options have been coord-clicked already.
            all_options_handled = all(
                any(
                    opt.lower() in el.primary_name.lower() and _coord_already_clicked(
                        el.x + max(1, el.width // 2), el.y + max(1, el.height // 2)
                    )
                    for el in perception.visible_elements
                    if el.is_interactable
                )
                for opt in intent_options
            )
            # Only count options that actually appear as labels on visible clickable
            # elements. Quoted strings in intent like "Jane Doe" / "jane@example.com"
            # are text values, not selectable options — they must not block submit.
            interactable_elements = [el for el in perception.visible_elements if el.is_interactable]
            clickable_options = [
                opt for opt in intent_options
                if "@" not in opt and opt != name and opt != email
            ]
            # An option is handled if its label was recorded in _clicked_labels (survives
            # page scrolls) OR if a coord-proximate click exists in history.
            all_options_handled = bool(clickable_options) and all(
                opt.lower() in _clicked_labels or any(
                    opt.lower() in el.primary_name.lower() and _coord_already_clicked(
                        el.x + max(1, el.width // 2), el.y + max(1, el.height // 2)
                    )
                    for el in interactable_elements
                )
                for opt in clickable_options
            )
            if all_options_handled:
                # All fields filled and all options clicked — fire submit.
                submit_button = _submit_button(perception)
                _viewport_h = 1080  # standard viewport height assumption
                if submit_button is not None:
                    _mid_y = submit_button.y + max(1, submit_button.height // 2)
                    if _mid_y <= _viewport_h:
                        return PolicyDecision(
                            action=AgentAction(
                                action_type=ActionType.CLICK,
                                target_element_id=submit_button.element_id,
                                x=submit_button.x + max(1, submit_button.width // 2),
                                y=_mid_y,
                            ),
                            rationale="All form fields and options handled — submitting the form.",
                            confidence=0.97,
                            active_subgoal="submit_form",
                        )
                # Submit not visible or below viewport — scroll DOWN to reveal it.
                # Do NOT use Press_Key End: when a form element has focus, End moves
                # the cursor within the element instead of scrolling the page.
                # Negative scroll_amount → wheel deltaY positive → scrolls DOWN.
                return PolicyDecision(
                    action=AgentAction(
                        action_type=ActionType.SCROLL,
                        x=960,
                        y=540,
                        scroll_amount=-3000,
                    ),
                    rationale="All fields and options done — scrolling to page bottom to reveal submit button.",
                    confidence=0.95,
                    active_subgoal="scroll to submit",
                )

            # Options not all handled but not visible — scroll up to find them.
            # Use _clickable_options (filtered) so email/name values in the form don't
            # falsely appear as "visible options" and block the scroll trigger.
            any_option_visible = bool(clickable_options) and any(
                any(opt.lower() in el.primary_name.lower() for opt in clickable_options)
                for el in perception.visible_elements
                if el.is_interactable
            )
            if not any_option_visible:
                return PolicyDecision(
                    action=AgentAction(
                        action_type=ActionType.SCROLL,
                        x=960,
                        y=540,
                        scroll_amount=800,
                    ),
                    rationale=f"Name/email filled; no option elements visible — scrolling up to find {clickable_options}.",
                    confidence=0.95,
                    active_subgoal="scroll to form options",
                )

        return None

    # ------------------------------------------------------------------
    # Selector helpers
    # ------------------------------------------------------------------

    def _resolve_target_element(
        self,
        perception: ScreenPerception,
        *,
        target_element_id: str | None,
        subgoal: str | None,
    ) -> UIElement | None:
        if target_element_id is not None:
            target = next((e for e in perception.visible_elements if e.element_id == target_element_id), None)
            if target is not None and target.element_type is UIElementType.INPUT and target.usable_for_targeting:
                self._latest_selector_traces.append(
                    SelectorTrace(
                        intent=_input_target_intent(subgoal, target_element_id),
                        candidate_count=1,
                        top_candidates=[
                            TargetEvidence(
                                element_id=target.element_id,
                                element_type=target.element_type,
                                primary_name=target.primary_name,
                                total_score=999.0,
                                matched_signals=["exact_target_element_id"],
                                rejected_by=[],
                                action_compatible=True,
                                exact_semantic_match=True,
                                uses_unlabeled_fallback=target.is_unlabeled,
                                nearest_matched_text_candidate_id=None,
                                spatial_grounding_contributed=False,
                                confidence_band=SelectorConfidenceBand.HIGH,
                            )
                        ],
                        selected_element_id=target.element_id,
                        decision_reason="Accepted exact target_element_id match before semantic scoring.",
                        rejection_reason=None,
                        score_margin=None,
                    )
                )
                return target

        intent = _input_target_intent(subgoal, target_element_id)
        return self._select_target(perception, intent)

    def _fallback_input_target(
        self,
        perception: ScreenPerception,
        subgoal: str | None,
        target_element_id: str | None,
    ) -> UIElement | None:
        intent = _input_target_intent(subgoal, target_element_id)
        return self._select_target(perception, intent)

    def _search_input_target(self, perception: ScreenPerception) -> UIElement | None:
        candidates = [e for e in _input_candidates(perception) if "search" in e.primary_name.lower()]
        return candidates[0] if candidates else None

    def _semantic_anchor_check(
        self,
        state: AgentState,
        perception: ScreenPerception,
        decision: PolicyDecision,
    ) -> PolicyDecision | None:
        """Intercept hallucinated coordinates that target empty space.

        Fires when ALL of the following hold:
        - The action type is CLICK, DOUBLE_CLICK, RIGHT_CLICK, or TYPE.
        - The action has explicit (x, y) coordinates.
        - There are visible elements in the current perception to compare against.
        - The targeted coordinate is more than _ANCHOR_DISTANCE_THRESHOLD pixels
          from the bounding box of every visible element.

        Returns a WAIT decision with force_fresh_perception=True and an anchor
        hint so the next LLM prompt knows what went wrong and where to look.
        Returns None when the coordinates look plausible (i.e., do not fire).
        """
        action = decision.action
        if action.action_type not in _COORD_ANCHOR_ACTION_TYPES:
            return None
        if action.x is None or action.y is None:
            return None
        if not perception.visible_elements:
            return None

        nearest_result = _nearest_element_by_box(action.x, action.y, perception.visible_elements)
        if nearest_result is None:
            return None
        nearest_elem, min_dist = nearest_result

        if min_dist <= _ANCHOR_DISTANCE_THRESHOLD:
            return None

        anchor_cx = nearest_elem.x + nearest_elem.width // 2
        anchor_cy = nearest_elem.y + nearest_elem.height // 2
        anchor_hint = (
            f"You targeted empty space at ({action.x}, {action.y}) — "
            f"the nearest element is '{nearest_elem.primary_name}' "
            f"({nearest_elem.element_id}) at approximately ({anchor_cx}, {anchor_cy}), "
            f"{min_dist:.0f} px away. "
            "Please refine your coordinates to target this element or another visible element directly."
        )

        logger.warning(
            "semantic_anchor_check: coord (%d, %d) is %.1f px from nearest element "
            "'%s' (%s) — intercepting and requesting re-perceive",
            action.x, action.y, min_dist,
            nearest_elem.primary_name, nearest_elem.element_id,
        )

        state.force_fresh_perception = True
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.WAIT, wait_ms=300),
            rationale=anchor_hint,
            confidence=0.99,
            active_subgoal=f"anchor_recheck:({action.x},{action.y})→{nearest_elem.element_id}",
            rule_name="_semantic_anchor_check",
        )

    def _select_target(self, perception: ScreenPerception, intent: TargetIntent) -> UIElement | None:
        result = self.selector.select(perception, intent, _cached_intermediates=self._cached_intermediates)
        self._latest_selector_traces.append(result.trace)
        return result.selected


# ---------------------------------------------------------------------------
# Module-level helpers shared by engine primitives and benchmark plugins
# ---------------------------------------------------------------------------

def _has_hint(memory_hints: list[MemoryHint], key: str) -> bool:
    return any(hint.key == key for hint in memory_hints)


def _has_recent_target_failure_signal(
    state: AgentState,
    target_element_id: str,
    failure_category: FailureCategory,
) -> bool:
    key = f"type:{target_element_id}:{failure_category.value}"
    return state.target_failure_counts.get(key, 0) > 0


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


def _input_candidates(perception: ScreenPerception) -> list[UIElement]:
    return [e for e in perception.visible_elements if e.element_type is UIElementType.INPUT and e.usable_for_targeting]


def _click_decision(target: UIElement, rationale: str) -> PolicyDecision:
    return PolicyDecision(
        action=AgentAction(
            action_type=ActionType.CLICK,
            target_element_id=target.element_id,
            x=target.x + max(1, target.width // 2),
            y=target.y + max(1, target.height // 2),
        ),
        rationale=rationale,
        confidence=0.95,
        active_subgoal=f"focus {target.element_id}",
    )


def _match_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = "".join(c.lower() if c.isalnum() else " " for c in value)
    return {t for t in normalized.split() if len(t) >= 2}


def _input_target_intent(subgoal: str | None, target_element_id: str | None) -> TargetIntent:
    lowered_subgoal = (subgoal or "").lower()
    expected_section: str | None = None
    target_text = None

    subgoal_map = (
        ("name", {"name"}),
        ("email", {"email", "recipient", "to"}),
        ("message", {"message", "body", "draft"}),
        ("subject", {"subject"}),
    )
    for canonical, tokens in subgoal_map:
        if any(token in lowered_subgoal for token in tokens):
            target_text = canonical
            break

    if target_text is None:
        id_tokens = _match_tokens(target_element_id)
        for canonical, tokens in subgoal_map:
            if id_tokens & tokens:
                target_text = canonical
                break

    if "form" in lowered_subgoal or "submit" in lowered_subgoal:
        expected_section = "form"

    return TargetIntent(
        action=TargetIntentAction.CLICK,
        target_text=target_text,
        expected_element_types=[UIElementType.INPUT],
        expected_section=expected_section,
    )


def _submit_button(perception: ScreenPerception) -> UIElement | None:
    from src.agent.selector import DeterministicTargetSelector
    result = DeterministicTargetSelector().select(
        perception,
        TargetIntent(
            action=TargetIntentAction.CLICK,
            target_text="submit",
            expected_element_types=[UIElementType.BUTTON, UIElementType.LINK],
            expected_section="form",
        ),
    )
    return result.selected


def _extract_search_query(state: AgentState) -> str | None:
    for source in (state.current_subgoal, state.intent):
        if not source:
            continue
        match = re.search(r"search for\s+(.+?)(?:,|\sand\s|\sthen\s|$)", source, flags=re.IGNORECASE)
        if match is None:
            continue
        query = match.group(1).strip(" .")
        # Drop trailing " on <site>" qualifiers first (e.g. "'GPT-4' on Wikipedia")
        query = re.sub(r"\s+on\s+\w[\w.]*$", "", query, flags=re.IGNORECASE).strip()
        # Strip surrounding quotes (single or double) after site-qualifier removal
        query = re.sub(r"^['\"](.+)['\"]$", r"\1", query)
        if query:
            return query
    return None


def _search_trigger_target(perception: ScreenPerception) -> UIElement | None:
    for element in perception.visible_elements:
        if not element.usable_for_targeting:
            continue
        if element.element_type not in {UIElementType.BUTTON, UIElementType.LINK, UIElementType.ICON}:
            continue
        if "search" in element.primary_name.lower():
            return element
    return None


def _blocking_overlay_elements(perception: ScreenPerception) -> list[UIElement]:
    """Return elements that look like blocking overlays: dialogs or non-interactable
    banners/windows that might intercept clicks intended for other elements."""
    overlay_types = {UIElementType.DIALOG, UIElementType.WINDOW}
    results = []
    for e in perception.visible_elements:
        if e.element_type in overlay_types:
            results.append(e)
            continue
        # Non-interactable elements with overlay-like labels also qualify
        if not e.is_interactable and e.element_type not in {UIElementType.TEXT, UIElementType.ICON}:
            name = e.primary_name.lower()
            if any(tok in name for tok in ("banner", "promo", "modal", "overlay", "popup", "notification", "toast", "cookie", "consent")):
                results.append(e)
    return results


def _best_dismiss_button(
    overlay: UIElement,
    perception: ScreenPerception,
    *,
    exclude_id: str | None,
) -> UIElement | None:
    """Find the most dismiss-like interactable button within or near the overlay bounds.

    Search radius expands 80px beyond the overlay boundary to catch buttons
    that are visually attached but not strictly inside the bounding box.
    """
    margin = 80
    x1, y1 = overlay.x - margin, overlay.y - margin
    x2, y2 = overlay.x + overlay.width + margin, overlay.y + overlay.height + margin

    candidates: list[tuple[int, UIElement]] = []
    for e in perception.visible_elements:
        if not e.is_interactable:
            continue
        if e.element_type not in {UIElementType.BUTTON, UIElementType.LINK, UIElementType.ICON}:
            continue
        if e.element_id == exclude_id:
            continue
        cx = e.x + e.width // 2
        cy = e.y + e.height // 2
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        score = sum(1 for tok in _DISMISS_TOKENS if tok in e.primary_name.lower())
        candidates.append((score, e))

    if not candidates:
        return None
    # Prefer highest dismiss-token score; break ties by shorter label (X > "No thanks")
    candidates.sort(key=lambda item: (-item[0], len(item[1].primary_name)))
    return candidates[0][1]


def _nearest_element_by_box(
    px: int,
    py: int,
    elements: list[UIElement],
) -> tuple[UIElement, float] | None:
    """Return the (element, distance) pair where distance is the minimum pixel gap
    from point (px, py) to the nearest edge of each element's bounding box.

    A point inside a bounding box has distance 0.  The box includes the full
    width×height extent reported by perception — no padding applied.
    """
    import math
    best: tuple[UIElement, float] | None = None
    for elem in elements:
        # Clamp point to the box, then measure straight-line distance to that clamped point.
        cx = max(elem.x, min(px, elem.x + elem.width))
        cy = max(elem.y, min(py, elem.y + elem.height))
        dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        if best is None or dist < best[1]:
            best = (elem, dist)
    return best


def _form_fields_completed(state: AgentState, perception: ScreenPerception) -> bool:
    # Build a map of visible text input fields we can match to intent values.
    # Accept "name", "email", "message", OR "password" (Gemini sometimes
    # mis-labels email-type inputs as "password").
    _FIELD_LABELS = {"name", "email", "message", "password"}
    required_targets: dict[str, str] = {}
    for element in _input_candidates(perception):
        label = element.primary_name.lower()
        for field in _FIELD_LABELS:
            if field in label and field not in required_targets:
                required_targets[field] = element.element_id
                break

    # Need at least 1 identified field to evaluate (avoids firing on blank pages).
    if not required_targets:
        return False

    completed_targets = {
        executed.action.target_element_id
        for executed, verification in zip(state.action_history, state.verification_history)
        if executed.action.action_type is ActionType.TYPE
        and verification.status is VerificationStatus.SUCCESS
        and executed.action.target_element_id is not None
    }
    # Also accept typed-by-text: if the value was typed (regardless of element_id
    # stability), count the field as completed.
    typed_values: set[str] = {
        executed.action.text
        for executed in state.action_history
        if executed.action.action_type is ActionType.TYPE and executed.action.text
    }
    intent = state.intent or ""
    email_match = re.search(r"[\w.+\-]+@[\w.\-]+\.\w+", intent)
    name_match = re.search(
        r"[Nn]ame(?:\s+as)?\s+['\"]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)['\"]?", intent
    )
    intent_email = email_match.group() if email_match else None
    intent_name = name_match.group(1) if name_match else None

    for field, target_id in required_targets.items():
        if target_id in completed_targets:
            continue
        # Fallback: check if the expected value for this field was typed
        if field in ("email", "password") and intent_email and intent_email in typed_values:
            continue
        if field == "name" and intent_name and intent_name in typed_values:
            continue
        return False
    return True

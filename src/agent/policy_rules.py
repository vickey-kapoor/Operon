"""Explicit rule-first policy checks for deterministic agent cases."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

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
# Gmail-specific rules
# ---------------------------------------------------------------------------

def gmail_login_page_guardrail(
    state: AgentState,
    perception: ScreenPerception,
    memory_hints: list[MemoryHint],
) -> PolicyDecision | None:
    if perception.page_hint != "google_sign_in":
        return None
    if _has_hint(memory_hints, "authenticated_start_required"):
        return PolicyDecision(
            action=AgentAction(action_type=ActionType.STOP),
            rationale="Benchmark requires an authenticated Gmail start state; login/auth screens are out of scope.",
            confidence=1.0,
            active_subgoal="stop for benchmark setup",
        )
    return PolicyDecision(
        action=AgentAction(
            action_type=ActionType.WAIT_FOR_USER,
            text="Login page detected. Please sign in using the browser window, then click Resume.",
        ),
        rationale="Login/auth page detected — requires user credentials that Operon cannot provide autonomously.",
        confidence=1.0,
        active_subgoal="wait for user authentication",
    )


def gmail_compose_already_visible_rule(
    state: AgentState,
    perception: ScreenPerception,
    memory_hints: list[MemoryHint],
) -> PolicyDecision | None:
    current_subgoal = (state.current_subgoal or "").lower()
    if "open compose" not in current_subgoal and "compose" not in current_subgoal:
        return None
    if perception.page_hint != "gmail_compose" and not _compose_form_visible(perception):
        return None
    target = _preferred_compose_input(perception)
    if target is None:
        return None
    return _click_decision(target, "Compose form is already visible; move to the form instead of reopening compose.")


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

    _HITL_DEBOUNCE_THRESHOLD: int = 2

    def choose_action(
        self,
        state: AgentState,
        perception: ScreenPerception,
        memory_hints: list[MemoryHint],
        benchmark_name: str | None = None,
    ) -> PolicyDecision | None:
        self._latest_selector_traces = []
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
                    return decision

        # Engine primitives — always active
        return (
            self._human_intervention_rule(state, perception)
            or self._task_success_stop_rule(perception)
            or self._dropdown_menu_select_rule(state, perception)
            or self._avoid_identical_type_retry(state, perception, memory_hints)
            or self._no_progress_recovery_rule(state, perception)
            or self._dismiss_blocking_overlay_rule(state, perception)
            or self._search_query_rule(state, perception)
        )

    def register_plugins(self, benchmark_name: str, plugins: list[BenchmarkRulePlugin]) -> None:
        """Register additional plugins for a benchmark at runtime."""
        self._plugins.setdefault(benchmark_name, []).extend(plugins)

    def latest_selector_traces(self) -> list[SelectorTrace]:
        return list(self._latest_selector_traces)

    # ------------------------------------------------------------------
    # Engine primitives
    # ------------------------------------------------------------------

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
        if not state.action_history:
            return None

        last = state.action_history[-1].action
        if last.action_type is not ActionType.CLICK:
            return None
        last_id = last.target_element_id
        if last_id is None:
            return None

        # Only fire when the last-clicked element is itself NOT a dropdown child
        # (prevents rule from firing repeatedly after we've already selected a child).
        if any(sig in last_id.lower() for sig in self._DROPDOWN_ID_SIGNALS):
            return None

        # Collect visible child items — elements whose IDs signal secondary menu membership.
        child_items = [
            e for e in perception.visible_elements
            if any(sig in e.element_id.lower() for sig in self._DROPDOWN_ID_SIGNALS)
            and e.usable_for_targeting
        ]
        if not child_items:
            return None

        # Score each child by how many intent/subgoal keywords appear in its label or id.
        all_tokens = _match_tokens(state.intent) | _match_tokens(state.current_subgoal)

        def _score(el: UIElement) -> int:
            text = (el.primary_name + " " + el.element_id).lower()
            return sum(1 for t in all_tokens if t in text)

        best = max(child_items, key=_score)
        logger.info(
            "Dropdown-menu rule: last click was '%s', %d child items visible. "
            "Selecting best match '%s' (score=%d).",
            last_id, len(child_items), best.element_id, _score(best),
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

        search_input = self._search_input_target(perception)
        if search_input is not None:
            if perception.focused_element_id != search_input.element_id:
                return _click_decision(
                    search_input,
                    f"Focus {search_input.primary_name} so the search query can be entered.",
                )
            return PolicyDecision(
                action=AgentAction(
                    action_type=ActionType.TYPE,
                    target_element_id=search_input.element_id,
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

    if "compose" in lowered_subgoal or "gmail" in lowered_subgoal:
        expected_section = "compose"
    elif "form" in lowered_subgoal or "submit" in lowered_subgoal:
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


def _preferred_compose_input(perception: ScreenPerception) -> UIElement | None:
    from src.agent.selector import DeterministicTargetSelector
    selector = DeterministicTargetSelector()
    intents = (
        TargetIntent(action=TargetIntentAction.CLICK, target_text="recipient", expected_element_types=[UIElementType.INPUT], expected_section="compose"),
        TargetIntent(action=TargetIntentAction.CLICK, target_text="to", expected_element_types=[UIElementType.INPUT], expected_section="compose"),
        TargetIntent(action=TargetIntentAction.CLICK, target_text="subject", expected_element_types=[UIElementType.INPUT], expected_section="compose"),
        TargetIntent(action=TargetIntentAction.CLICK, target_text="message", expected_element_types=[UIElementType.INPUT], expected_section="compose"),
    )
    for intent in intents:
        result = selector.select(perception, intent)
        if result.selected is not None:
            return result.selected
    return None


def _compose_form_visible(perception: ScreenPerception) -> bool:
    return _preferred_compose_input(perception) is not None


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


def _form_fields_completed(state: AgentState, perception: ScreenPerception) -> bool:
    required_targets: dict[str, str] = {}
    for element in _input_candidates(perception):
        label = element.primary_name.lower()
        if "name" in label:
            required_targets["name"] = element.element_id
        elif "email" in label:
            required_targets["email"] = element.element_id
        elif "message" in label:
            required_targets["message"] = element.element_id

    if len(required_targets) < 3:
        return False

    completed_targets = {
        executed.action.target_element_id
        for executed, verification in zip(state.action_history, state.verification_history)
        if executed.action.action_type is ActionType.TYPE
        and verification.status is VerificationStatus.SUCCESS
        and executed.action.target_element_id is not None
    }
    return all(target_id in completed_targets for target_id in required_targets.values())

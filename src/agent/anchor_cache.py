"""Per-run cache of post-click coordinate anchors.

A successful CLICK records (x, y) for that element_id under the run. A subsequent
TYPE on the same element snaps back to the anchor when fresh-perception drift is
small (< _ANCHOR_SNAP_THRESHOLD_PX). Defeats perception jitter that otherwise
causes the visual servo to abort harmless TYPE actions.

Also exposes `tag_input_zone()` — classifies a CLICK/TYPE on a blank input
surface so the visual servo's variance gate is bypassed (a Notepad text area is
intentionally white, but otherwise looks like a "moved" target).
"""

from __future__ import annotations

import logging

from src.models.execution import AnchorSnapInfo
from src.models.perception import UIElementType
from src.models.policy import ActionType, AgentAction

logger = logging.getLogger(__name__)


# Element-id substrings that indicate a valid but blank interactive surface
# (text editors, input fields, search boxes). element_type is also checked.
_INPUT_ZONE_ID_TOKENS: frozenset[str] = frozenset({
    "text_area", "text_field", "text_input", "notepad_text_area",
    "input", "textarea", "editor", "search_input",
})


class AnchorCache:
    """Per-run anchor store. Keys are run_id then element_id."""

    ANCHOR_SNAP_THRESHOLD_PX: int = 50

    def __init__(self) -> None:
        self._anchors: dict[str, dict[str, tuple[int, int]]] = {}

    def discard_run(self, run_id: str) -> None:
        """Drop the anchor table for a completed run. Idempotent."""
        self._anchors.pop(run_id, None)

    def update(self, run_id: str, action: AgentAction) -> None:
        """Record the click coordinates for an element as the known-good anchor.

        Called after every successful CLICK so subsequent TYPE actions on the same
        element can snap back to this position if perception jitter shifts the
        reported coordinates by a small amount.
        """
        target_id = action.target_element_id
        if target_id is None or action.x is None or action.y is None:
            return
        self._anchors.setdefault(run_id, {})[target_id] = (action.x, action.y)
        logger.debug("coord_anchor: stored %r → (%d, %d) for run %s", target_id, action.x, action.y, run_id[:8])

    def apply(self, run_id: str, decision):
        """Snap TYPE action coordinates to the post-click anchor when drift is small.

        Returns (possibly-modified decision, AnchorSnapInfo | None).  The snap only
        fires when all of the following hold:
        - Action is TYPE with a target_element_id and x/y coordinates.
        - An anchor exists for that element_id in this run.
        - Euclidean distance between fresh perception coords and the anchor is
          strictly less than ANCHOR_SNAP_THRESHOLD_PX (small jitter, not a
          genuine element move).
        """
        action = decision.action
        if action.action_type is not ActionType.TYPE:
            return decision, None
        target_id = action.target_element_id
        if target_id is None or action.x is None or action.y is None:
            return decision, None
        anchors = self._anchors.get(run_id, {})
        anchor = anchors.get(target_id)
        if anchor is None:
            return decision, None
        ax, ay = anchor
        drift = ((action.x - ax) ** 2 + (action.y - ay) ** 2) ** 0.5
        if drift >= self.ANCHOR_SNAP_THRESHOLD_PX:
            logger.debug(
                "coord_anchor: %r drift=%.1fpx ≥ threshold — trusting fresh perception",
                target_id, drift,
            )
            return decision, None
        logger.info(
            "snap_to_anchor: %r drift=%.1fpx original=(%d,%d) anchor=(%d,%d) run=%s",
            target_id, drift, action.x, action.y, ax, ay, run_id[:8],
        )
        snapped_action = action.model_copy(update={"x": ax, "y": ay})
        snap_info = AnchorSnapInfo(
            element_id=target_id,
            original_x=action.x,
            original_y=action.y,
            anchored_x=ax,
            anchored_y=ay,
            drift_px=round(drift, 1),
        )
        return decision.model_copy(update={"action": snapped_action}), snap_info


def tag_input_zone(decision, perception):
    """Mark the action as is_input_zone=True if the target is a blank input surface.

    The visual servo rejects clicks on uniform (low-variance) regions — but a
    blank Notepad text area or empty input field is intentionally white/blank and
    must still accept clicks.  This method detects that case and sets the bypass
    flag so _region_has_content skips the variance gate.
    """
    action = decision.action
    if action.action_type not in (ActionType.CLICK, ActionType.TYPE):
        return decision
    if action.is_input_zone:
        return decision

    target_id = (action.target_element_id or "").lower()
    if any(tok in target_id for tok in _INPUT_ZONE_ID_TOKENS):
        return decision.model_copy(update={"action": action.model_copy(update={"is_input_zone": True})})

    if action.target_element_id and perception is not None:
        for el in perception.visible_elements:
            if el.element_id == action.target_element_id:
                if el.element_type is UIElementType.INPUT:
                    return decision.model_copy(update={"action": action.model_copy(update={"is_input_zone": True})})
                break

    return decision

"""Perception, capture, geometry, and screen-diff components."""

from operon.agent.perception.service import *  # noqa: F401,F403
from operon.agent.perception.service import (  # noqa: F401
    _apply_weak_canonicalization,
    _low_quality_reason,
    _normalize_visible_elements,
    _quality_metrics,
    _strip_json_fence,
)

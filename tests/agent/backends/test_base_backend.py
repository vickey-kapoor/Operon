"""Default hooks every backend inherits (backends.base.AgentBackend)."""

from __future__ import annotations

import pytest

from operon.agent.backends.base import AgentBackend


class _Minimal(AgentBackend):
    async def perceive(self, screenshot, state):
        raise NotImplementedError

    async def choose_action(self, state, perception):
        raise NotImplementedError

    def latest_debug_artifacts(self):
        return None


def test_abstract_methods_are_required() -> None:
    class _Incomplete(AgentBackend):
        def latest_debug_artifacts(self):
            return None

    with pytest.raises(TypeError):
        _Incomplete()


def test_optional_hooks_are_harmless_no_ops() -> None:
    backend = _Minimal()
    assert backend.add_advisory_hints(["h"], source="s", run_id="r") is None
    assert backend.clear_advisory_hints("r") is None
    assert backend._reset_advisory_hints_for_test(["h"]) is None
    assert backend.note_not_executed("r", "blocked") is None
    assert backend.discard_held_actions("r", "correction") is None
    assert backend.has_held_actions("r") is False
    assert backend.supports_resampling("r") is True


def test_release_run_drops_only_that_run_from_every_known_cache() -> None:
    backend = _Minimal()
    for name in AgentBackend._PER_RUN_CACHES:
        setattr(backend, name, {"r1": object(), "r2": object()})

    backend.release_run("r1")

    for name in AgentBackend._PER_RUN_CACHES:
        assert set(getattr(backend, name)) == {"r2"}, name


def test_release_run_tolerates_missing_and_non_dict_caches() -> None:
    backend = _Minimal()
    backend._conversations = ["not", "a", "dict"]
    backend._runs = {"r1": 1}

    backend.release_run("r1")
    backend.release_run("unknown")

    assert backend._conversations == ["not", "a", "dict"]
    assert backend._runs == {}

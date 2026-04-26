"""Benchmark plugin registry — the single registration point for all benchmark-specific logic.

Engine code reads from BENCHMARK_REGISTRY using the explicit benchmark name carried on
AgentState. Nothing in the engine hardcodes benchmark identifiers or task-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.policy_rules import BenchmarkRulePlugin
    from src.models.memory import MemoryRecord
    from src.models.perception import PageHint


@dataclass
class BenchmarkPlugin:
    """Everything a benchmark contributes to the engine — rules, seeds, signals, metadata."""

    name: str
    rules: list[BenchmarkRulePlugin] = field(default_factory=list)
    memory_seeds: list[MemoryRecord] = field(default_factory=list)
    success_tokens: tuple[str, ...] = field(default_factory=tuple)
    # Maps a PageHint to the element-section label used by the selector.
    section_map: dict[PageHint, str] = field(default_factory=dict)
    default_url: str | None = None
    task_type: str = "generic"
    expected_completion_signal: str = "task completed"


class BenchmarkRegistry:
    """Registry of benchmark plugins. Benchmarks register themselves at import time."""

    def __init__(self) -> None:
        self._plugins: dict[str, BenchmarkPlugin] = {}

    def register(self, plugin: BenchmarkPlugin) -> None:
        self._plugins[plugin.name] = plugin

    def get(self, name: str | None) -> BenchmarkPlugin | None:
        if name is None:
            return None
        return self._plugins.get(name)

    def get_rules(self, name: str | None) -> list[BenchmarkRulePlugin]:
        plugin = self.get(name)
        return plugin.rules if plugin else []

    def get_seeds(self, name: str | None) -> list[MemoryRecord]:
        plugin = self.get(name)
        return plugin.memory_seeds if plugin else []

    def get_success_tokens(self, name: str | None) -> tuple[str, ...]:
        plugin = self.get(name)
        return plugin.success_tokens if plugin else ()

    def get_section(self, name: str | None, page_hint: PageHint) -> str | None:
        plugin = self.get(name)
        if plugin is None:
            return None
        return plugin.section_map.get(page_hint)

    def all_seeds(self) -> list[MemoryRecord]:
        """All seeds across every registered benchmark — used for startup seeding."""
        seeds: list[MemoryRecord] = []
        for plugin in self._plugins.values():
            seeds.extend(plugin.memory_seeds)
        return seeds

    def registered_names(self) -> list[str]:
        return list(self._plugins.keys())


BENCHMARK_REGISTRY = BenchmarkRegistry()

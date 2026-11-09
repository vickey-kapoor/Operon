"""Paired bootstrap comparisons with Improved / Regressed / No clear change labels.

Tasks are the unit of sampling, so repeats are averaged within a task first. Tasks
are grouped into categories (for Operon: suites such as ``long`` or ``grounding``);
the overall score weights every category equally. Corrected intervals use
Bonferroni over the comparisons in a family and the metrics reported.

Adapted from model-release-benchmark's ``benchmark/stats.py``; the categories are
now passed in rather than fixed to that repo's text categories.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence

ITERATIONS = 10_000
ALPHA = 0.05
OVERALL = "overall"

IMPROVED = "Improved"
REGRESSED = "Regressed"
NO_CLEAR_CHANGE = "No clear change"
POSSIBLE_DRIFT = "Possible drift"


def _bootstrap_means(values: Sequence[float], iterations: int, rng: random.Random) -> list[float]:
    size = len(values)
    return [sum(rng.choices(values, k=size)) / size for _ in range(iterations)]


def _percentile_interval(sorted_values: Sequence[float], alpha: float) -> tuple[float, float]:
    last = len(sorted_values) - 1
    return sorted_values[round(alpha / 2 * last)], sorted_values[round((1 - alpha / 2) * last)]


def group_by_category(task_values: Mapping[str, float], task_categories: Mapping[str, str]) -> dict[str, list[float]]:
    """Group per-task values by category; tasks without a category are skipped.

    ``"overall"`` is reserved for the combined score, so a category with that name is refused.
    """
    if OVERALL in task_categories.values():
        raise ValueError(f"{OVERALL!r} is reserved for the combined score; rename that category")
    grouped: dict[str, list[float]] = {}
    for task_id, value in sorted(task_values.items()):
        if task_id in task_categories:
            grouped.setdefault(task_categories[task_id], []).append(value)
    return grouped


def _estimates(grouped: Mapping[str, list[float]]) -> dict[str, float]:
    estimates = {category: statistics.fmean(values) for category, values in grouped.items()}
    if len(estimates) > 1:
        estimates[OVERALL] = statistics.fmean(estimates.values())
    return estimates


def _distributions(grouped: Mapping[str, list[float]], iterations: int, rng: random.Random) -> dict[str, list[float]]:
    samples = {category: _bootstrap_means(values, iterations, rng) for category, values in sorted(grouped.items())}
    if len(samples) > 1:
        arrays = list(samples.values())
        samples[OVERALL] = [sum(values) / len(arrays) for values in zip(*arrays)]
    return samples


def summarize(
    task_means: Mapping[str, float],
    task_categories: Mapping[str, str],
    iterations: int = 2_000,
    seed: str = "summary",
) -> dict[str, dict]:
    """Score per category (and overall, if more than one) with a 95% interval, 0-1 scale."""
    grouped = group_by_category(task_means, task_categories)
    if not grouped:
        return {}
    estimates = _estimates(grouped)
    samples = _distributions(grouped, iterations, random.Random(seed))
    result = {}
    for metric, estimate in estimates.items():
        low, high = _percentile_interval(sorted(samples[metric]), ALPHA)
        tasks = len(grouped[metric]) if metric in grouped else sum(len(v) for v in grouped.values())
        result[metric] = {"score": estimate, "low": low, "high": high, "tasks": tasks}
    return result


def label_for(corrected_low: float, corrected_high: float) -> str:
    if corrected_low > 0:
        return IMPROVED
    if corrected_high < 0:
        return REGRESSED
    return NO_CLEAR_CHANGE


def compare(
    means_new: Mapping[str, float],
    means_old: Mapping[str, float],
    task_categories: Mapping[str, str],
    comparisons_in_family: int = 1,
    iterations: int = ITERATIONS,
    seed: str = "compare",
) -> dict[str, dict]:
    """Compare two sets of per-task scores on the tasks both have. Positive favors ``means_new``.

    Use it for two Operon versions on the same model, or two models on the same Operon.
    """
    shared = set(means_new) & set(means_old)
    differences = {task: means_new[task] - means_old[task] for task in shared}
    grouped = group_by_category(differences, task_categories)
    if not grouped:
        return {}
    estimates = _estimates(grouped)
    samples = _distributions(grouped, iterations, random.Random(seed))
    corrected_alpha = ALPHA / (comparisons_in_family * len(estimates))
    result = {}
    for metric, estimate in estimates.items():
        ordered = sorted(samples[metric])
        low, high = _percentile_interval(ordered, ALPHA)
        corrected_low, corrected_high = _percentile_interval(ordered, corrected_alpha)
        result[metric] = {
            "difference": estimate,
            "low": low,
            "high": high,
            "corrected_low": corrected_low,
            "corrected_high": corrected_high,
            "label": label_for(corrected_low, corrected_high),
            "tasks": len(grouped[metric]) if metric in grouped else sum(len(v) for v in grouped.values()),
        }
    return result


def drift_check(
    baseline: Mapping[str, Mapping[str, float]],
    current: Mapping[str, Mapping[str, float]],
    iterations: int = ITERATIONS,
) -> dict[str, dict]:
    """Compare two runs of the same tasks per model ({model: {task: score}}).

    Flags "Possible drift" when a model's score moved beyond the corrected interval
    with nothing in Operon changed, which points at a silent model change.
    """
    models = sorted(set(baseline) & set(current))
    output = {}
    for model in models:
        pooled = {task: "all" for task in set(current[model]) & set(baseline[model])}
        if not pooled:
            continue
        result = compare(current[model], baseline[model], pooled, len(models), iterations, seed=f"drift|{model}")["all"]
        output[model] = {**result, "label": POSSIBLE_DRIFT if result["label"] != NO_CLEAR_CHANGE else NO_CLEAR_CHANGE}
    return output

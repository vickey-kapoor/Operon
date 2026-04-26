"""Stateless evaluation logic for WebArena task grading."""

from __future__ import annotations

from src.benchmarks.webarena_models import WebArenaTask


def string_match(must_include: list[str], text: str) -> bool:
    """All strings in must_include must appear in text (case-insensitive)."""
    text_lower = text.lower()
    return all(s.lower() in text_lower for s in must_include)


def url_match(url_contains: str, url: str) -> bool:
    """url_contains must appear as a substring of url (case-insensitive)."""
    return url_contains.lower() in url.lower()


def evaluate_task(
    task: WebArenaTask,
    final_url: str,
    final_text: str,
) -> dict[str, bool]:
    """Grade a completed task run against the WebArena reference answers.

    Returns a dict mapping each eval_type to its pass/fail result.
    The overall task passes only when all eval_types pass.
    """
    results: dict[str, bool] = {}
    answers = task.eval.reference_answers

    for eval_type in task.eval.eval_types:
        if eval_type == "string_match":
            raw = answers.get("must_include", [])
            must_include = raw if isinstance(raw, list) else [raw]
            results["string_match"] = string_match(must_include, final_text)
        elif eval_type == "url_match":
            url_contains = str(answers.get("url_contains", ""))
            results["url_match"] = url_match(url_contains, final_url)
        else:
            # Unknown eval type — conservatively fail
            results[eval_type] = False

    return results

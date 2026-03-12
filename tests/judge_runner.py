"""
UI Navigator Judge Runner
==========================
Reads /tmp/wp_test_report.json, calls the Claude API to evaluate the test
results, and writes the judge output to /tmp/wp_judge_output.json.

Usage:
    python tests/judge_runner.py [--report /tmp/wp_test_report.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_JUDGE_PROMPT_PATH = _REPO_ROOT / "tests" / "judge_prompt.md"
_DEFAULT_REPORT = Path(tempfile.gettempdir()) / "wp_test_report.json"
_JUDGE_OUTPUT = Path(tempfile.gettempdir()) / "wp_judge_output.json"

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_output(data: dict) -> None:
    _JUDGE_OUTPUT.write_text(json.dumps(data, indent=2))


def _strip_fences(text: str) -> str:
    """Remove leading/trailing ```json … ``` or ``` … ``` fences if present."""
    text = text.strip()
    # Match optional language tag after opening fence
    match = re.match(r"^```[a-z]*\n?([\s\S]*?)\n?```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(report_path: Path) -> None:
    # 1. Check for API key before importing anthropic (gives a clear error fast)
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _write_output({"error": "ANTHROPIC_API_KEY not set"})
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # 2. Read test report
    if not report_path.exists():
        _write_output({"error": f"Report file not found: {report_path}"})
        print(f"Error: report file not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    try:
        report_text = report_path.read_text()
        json.loads(report_text)  # validate it's real JSON before sending
    except Exception as exc:
        _write_output({"error": f"Failed to read report: {exc}"})
        print(f"Error reading report: {exc}", file=sys.stderr)
        sys.exit(1)

    # 3. Read judge system prompt
    if not _JUDGE_PROMPT_PATH.exists():
        _write_output({"error": f"Judge prompt not found: {_JUDGE_PROMPT_PATH}"})
        print(f"Error: judge prompt not found: {_JUDGE_PROMPT_PATH}", file=sys.stderr)
        sys.exit(1)

    system_prompt = _JUDGE_PROMPT_PATH.read_text()

    # 4. Call the Anthropic API
    try:
        import anthropic
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic", "-q"])
        import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": report_text,
                }
            ],
        )
    except anthropic.AuthenticationError:
        _write_output({"error": "ANTHROPIC_API_KEY is invalid or revoked"})
        print("Error: invalid API key", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIStatusError as exc:
        _write_output({"error": f"API error {exc.status_code}: {exc.message}"})
        print(f"API error: {exc}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIConnectionError as exc:
        _write_output({"error": f"Connection error: {exc}"})
        print(f"Connection error: {exc}", file=sys.stderr)
        sys.exit(1)

    # 5. Extract text from response
    raw = next(
        (block.text for block in response.content if block.type == "text"),
        "",
    )

    # 6. Parse JSON (strip fences if present)
    cleaned = _strip_fences(raw)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "error": "judge_parse_failed",
            "raw": raw[:500],
        }
        _write_output(result)
        print(f"Warning: judge returned non-JSON response. Raw: {raw[:200]}", file=sys.stderr)
        sys.exit(1)

    # 7. Write output
    _write_output(result)
    print("Judge complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM judge on a test report")
    parser.add_argument(
        "--report",
        type=Path,
        default=_DEFAULT_REPORT,
        help=f"Path to the JSON test report (default: {_DEFAULT_REPORT})",
    )
    args = parser.parse_args()
    run(args.report)


if __name__ == "__main__":
    main()

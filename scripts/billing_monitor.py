"""
Real-time billing monitor for Operon Gemini / Vertex AI usage.

Tails run.jsonl log files and parses `gemini_usage` INFO lines emitted by
GeminiHttpClient to track spend, image count, and remaining Vertex AI credit.

Usage:
    python scripts/billing_monitor.py                     # watches all runs/
    python scripts/billing_monitor.py runs/<run_id>/run.jsonl
    python scripts/billing_monitor.py --credit 1000.00    # override starting credit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

CREDIT_START_DATE = date(2025, 4, 26)   # date the $1,000 Vertex credit was activated
CREDIT_TOTAL_USD = 1_000.0
SINGLE_CALL_WARN_USD = 0.05             # warn if one inference call exceeds this
POLL_INTERVAL_S = 1.0

# Matches the log line format emitted by GeminiHttpClient._post_json_payload:
# gemini_usage kind=image model=gemini-2.5-flash in=1234 out=456 total=1690 cost_usd=0.001234 vertex=True
_USAGE_RE = re.compile(
    r"gemini_usage\s+"
    r"kind=(?P<kind>\S+)\s+"
    r"model=(?P<model>\S+)\s+"
    r"in=(?P<input>\S+)\s+"
    r"out=(?P<output>\S+)\s+"
    r"total=(?P<total>\S+)\s+"
    r"cost_usd=(?P<cost>\S+)\s+"
    r"vertex=(?P<vertex>\S+)"
)

# Also parse the structured perception_usage lines from perception.py:
# perception_usage step=N attempt=N in=X out=Y total=Z cost_usd=0.000123
_PERCEPTION_RE = re.compile(
    r"perception_usage\s+"
    r"step=(?P<step>\d+)\s+"
    r"attempt=(?P<attempt>\d+)\s+"
    r"in=(?P<input>\S+)\s+"
    r"out=(?P<output>\S+)\s+"
    r"total=(?P<total>\S+)\s+"
    r"cost_usd=(?P<cost>\S+)"
)


# ── State ─────────────────────────────────────────────────────────────────────

class SessionTally:
    def __init__(self, starting_credit: float) -> None:
        self.starting_credit = starting_credit
        self.total_cost_usd: float = 0.0
        self.images_processed: int = 0
        self.calls: int = 0
        self.warnings: list[str] = []

    def record(self, kind: str, cost: float, is_image: bool) -> None:
        self.total_cost_usd += cost
        self.calls += 1
        if is_image:
            self.images_processed += 1
        if cost > SINGLE_CALL_WARN_USD:
            self.warnings.append(
                f"  ⚠  CONTEXT BLOAT — single {kind} call: ${cost:.4f} "
                f"(threshold: ${SINGLE_CALL_WARN_USD:.2f})"
            )

    def remaining_credit(self) -> float:
        days_elapsed = (date.today() - CREDIT_START_DATE).days
        # Google Vertex AI $1,000 trial credits do not expire on a daily burn;
        # remaining = starting_credit minus what we've spent this session.
        return max(0.0, self.starting_credit - self.total_cost_usd)

    def render(self) -> str:
        remaining = self.remaining_credit()
        pct_used = (self.total_cost_usd / self.starting_credit * 100) if self.starting_credit else 0.0
        days_elapsed = (date.today() - CREDIT_START_DATE).days
        lines = [
            "",
            "┌─────────────────────────────────────────┐",
            "│        Operon Billing Monitor           │",
            "├─────────────────────────────────────────┤",
            f"│  Session spend :  ${self.total_cost_usd:>10.6f}           │",
            f"│  Calls         :  {self.calls:>10}           │",
            f"│  Images        :  {self.images_processed:>10}           │",
            "├─────────────────────────────────────────┤",
            f"│  Credit start  :  {CREDIT_START_DATE} ({days_elapsed}d ago)  │",
            f"│  Credit total  :  ${self.starting_credit:>10.2f}           │",
            f"│  Remaining     :  ${remaining:>10.4f}  ({100-pct_used:.2f}% left)  │",
            "└─────────────────────────────────────────┘",
        ]
        if self.warnings:
            lines.append("")
            lines.extend(self.warnings[-5:])  # show last 5 warnings only
        return "\n".join(lines)


# ── Log parsing ───────────────────────────────────────────────────────────────

def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def parse_line(line: str, tally: SessionTally) -> bool:
    """Return True if the line contained a usage record."""
    m = _USAGE_RE.search(line)
    if m:
        cost = _parse_float(m.group("cost"))
        kind = m.group("kind")
        tally.record(kind=kind, cost=cost, is_image=(kind == "image"))
        return True

    m = _PERCEPTION_RE.search(line)
    if m:
        cost = _parse_float(m.group("cost"))
        tally.record(kind="perception", cost=cost, is_image=True)
        return True

    # Also try parsing as a JSON StepLog line (run.jsonl entries)
    try:
        obj = json.loads(line)
        usage = obj.get("usage") or obj.get("model_usage")
        if isinstance(usage, dict):
            cost = _parse_float(usage.get("estimated_cost_usd") or 0)
            kind = usage.get("request_kind", "unknown")
            if cost > 0:
                tally.record(kind=kind, cost=cost, is_image=(kind == "image"))
                return True
    except (json.JSONDecodeError, AttributeError):
        pass

    return False


# ── File tailing ──────────────────────────────────────────────────────────────

def _collect_log_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("run.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def tail_logs(paths: list[Path], tally: SessionTally) -> None:
    """Open all log files, seek to end, then poll for new lines."""
    handles: dict[Path, object] = {}
    for p in paths:
        try:
            fh = p.open("r", encoding="utf-8", errors="replace")
            fh.seek(0, 2)  # seek to end — only watch new entries
            handles[p] = fh
        except OSError:
            pass

    print(f"Monitoring {len(handles)} log file(s). Press Ctrl+C to stop.\n")

    try:
        while True:
            updated = False
            for path, fh in list(handles.items()):
                line = fh.readline()
                while line:
                    if parse_line(line.rstrip(), tally):
                        updated = True
                    line = fh.readline()

            # Pick up any new run.jsonl files that appeared since we started
            for new_path in _collect_log_files(paths[0].parent if len(paths) == 1 else Path("runs")):
                if new_path not in handles:
                    try:
                        fh = new_path.open("r", encoding="utf-8", errors="replace")
                        fh.seek(0, 2)
                        handles[new_path] = fh
                    except OSError:
                        pass

            if updated:
                sys.stdout.write("\033[2J\033[H")  # clear terminal
                print(tally.render())
                sys.stdout.flush()

            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\n\nFinal tally:")
        print(tally.render())
    finally:
        for fh in handles.values():
            fh.close()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time Operon billing monitor")
    parser.add_argument(
        "log_path",
        nargs="?",
        default="runs",
        help="Path to run.jsonl or a runs/ directory (default: runs/)",
    )
    parser.add_argument(
        "--credit",
        type=float,
        default=CREDIT_TOTAL_USD,
        help=f"Starting Vertex AI credit in USD (default: {CREDIT_TOTAL_USD})",
    )
    args = parser.parse_args()

    root = Path(args.log_path)
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        sys.exit(1)

    tally = SessionTally(starting_credit=args.credit)
    log_files = _collect_log_files(root)
    if not log_files:
        print(f"No run.jsonl files found under {root}", file=sys.stderr)
        sys.exit(1)

    tail_logs(log_files, tally)


if __name__ == "__main__":
    main()

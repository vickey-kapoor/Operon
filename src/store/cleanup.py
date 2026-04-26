"""Delete run artifact directories that are not from today.

Usage:
    python -m src.store.cleanup           # dry-run: show what would be deleted
    python -m src.store.cleanup --delete  # actually delete
    python -m src.store.cleanup --days 3  # keep runs from the last N days (default: today only)
"""

from __future__ import annotations

import argparse
import shutil
from datetime import date, timedelta
from pathlib import Path


def cleanup_old_runs(
    root_dir: str | Path = "runs",
    keep_days: int = 0,
    delete: bool = False,
) -> tuple[list[Path], int]:
    """Return (dirs_to_delete, bytes_freed) and optionally remove them.

    Args:
        root_dir: Path to the runs directory.
        keep_days: Keep runs modified within this many days (0 = today only).
        delete: If False, dry-run only.
    """
    root = Path(root_dir)
    if not root.exists():
        return [], 0

    cutoff = date.today() - timedelta(days=keep_days)
    to_delete: list[Path] = []

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        # Use state.json mtime when available for accuracy; fall back to dir mtime.
        state_json = run_dir / "state.json"
        ref = state_json if state_json.exists() else run_dir
        run_date = date.fromtimestamp(ref.stat().st_mtime)
        if run_date < cutoff:
            to_delete.append(run_dir)

    total_bytes = 0
    for path in to_delete:
        total_bytes += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if delete:
            shutil.rmtree(path)

    return to_delete, total_bytes


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up old run artifact directories.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete (default is dry-run).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        metavar="N",
        help="Keep runs from the last N days inclusive (default: 0 = today only).",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        metavar="PATH",
        help="Root runs directory (default: runs).",
    )
    args = parser.parse_args()

    dirs, freed = cleanup_old_runs(
        root_dir=args.runs_dir,
        keep_days=args.days,
        delete=args.delete,
    )

    if not dirs:
        print("Nothing to clean up.")
        return

    action = "Deleted" if args.delete else "Would delete"
    print(f"{action} {len(dirs)} run director{'y' if len(dirs) == 1 else 'ies'} ({_fmt_bytes(freed)})")
    if not args.delete:
        print("Run with --delete to remove them.")


if __name__ == "__main__":
    main()

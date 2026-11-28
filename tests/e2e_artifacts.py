"""Opt-in evidence copy for the free e2e tests (KAN-213). Stdlib only.

Nothing is copied unless ``OPERON_E2E_ARTIFACTS_DIR`` is set (CI sets it). Only screenshots,
``run.jsonl`` and ``state.json`` are copied: never settings.json, host.json, memory or model I/O.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import warnings
from pathlib import Path

ENV_VAR = "OPERON_E2E_ARTIFACTS_DIR"
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TEST_BYTES = 30 * 1024 * 1024
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_EXACT_NAMES = {"run.jsonl", "state.json"}


def safe_name(nodeid: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", nodeid).strip("_.")
    if len(name) > 120:
        digest = hashlib.sha1(nodeid.encode("utf-8")).hexdigest()[:8]  # noqa: S324 - not security
        name = name[:111].rstrip("_.") + "_" + digest
    return name


def _wanted(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES or path.name in _EXACT_NAMES


def save_run_artifacts(src: Path, nodeid: str, dest_root: str | None = None) -> Path | None:
    """Copy the allowlisted evidence under ``src`` to ``<dest_root>/<safe_name>/``; never raises."""
    root = dest_root or os.environ.get(ENV_VAR)
    if not root or not root.strip():
        return None
    dest = Path(root) / safe_name(nodeid)
    try:
        src = Path(src)
        dest.mkdir(parents=True, exist_ok=True)
        skipped: list[str] = []
        total = 0
        for path in sorted(src.rglob("*")):
            if path.is_symlink() or not path.is_file() or not _wanted(path):
                continue
            rel = path.relative_to(src)
            if "memory" in rel.parts[:-1]:
                continue
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                skipped.append(f"{rel.as_posix()} (too large: {size} bytes)")
                continue
            if total + size > MAX_TEST_BYTES:
                skipped.append(f"{rel.as_posix()} (per-test cap reached)")
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            total += size
        lines = [nodeid, ""]
        if skipped:
            lines += ["skipped:", *skipped]
        (dest / "nodeid.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        warnings.warn(f"could not save e2e artifacts for {nodeid}: {exc}", stacklevel=2)
        return None
    return dest

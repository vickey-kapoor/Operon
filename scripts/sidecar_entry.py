"""
Operon API server entry point for PyInstaller / Tauri sidecar packaging.

In release builds Tauri launches this as an external sidecar process.
In dev, run directly: .venv/Scripts/python scripts/sidecar_entry.py
"""

import os
import sys
from pathlib import Path

# ── PyInstaller bundle setup ────────────────────────────────────────────────
# Prompt files use Path(__file__).resolve().parents[N] so they self-locate
# correctly inside the bundle — no patching needed for reads.
# We only redirect mutable runtime dirs to a persistent app data location
# outside the ephemeral sys._MEIPASS temp dir.

if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(sys._MEIPASS)  # type: ignore[attr-defined]

    if sys.platform == "win32":
        _base = os.environ.get("APPDATA") or str(Path.home())
        APP_DATA = Path(_base) / "Operon"
    elif sys.platform == "darwin":
        APP_DATA = Path.home() / "Library" / "Application Support" / "Operon"
    else:
        _xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        APP_DATA = Path(_xdg) / "Operon"

    APP_DATA.mkdir(parents=True, exist_ok=True)
    (APP_DATA / "runs").mkdir(exist_ok=True)
    (APP_DATA / "memory").mkdir(exist_ok=True)

    # Redirect mutable dirs — server.py and cleanup.py both read these
    os.environ.setdefault("OPERON_RUNS_ROOT", str(APP_DATA / "runs"))
    os.environ.setdefault("OPERON_MEMORY_ROOT", str(APP_DATA / "memory"))

    # Load user-supplied .env from app data (API keys, model overrides, etc.)
    _env = APP_DATA / ".env"
    if _env.exists():
        from dotenv import load_dotenv
        load_dotenv(_env, override=False)

    # Ensure bundle root is on sys.path so `src.*` imports resolve
    sys.path.insert(0, str(BUNDLE_DIR))

import uvicorn


def main() -> None:
    host = os.environ.get("OPERON_HOST", "127.0.0.1")
    port = int(os.environ.get("OPERON_PORT", "8080"))

    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        log_level=os.environ.get("OPERON_LOG_LEVEL", "info"),
        reload=False,
    )


if __name__ == "__main__":
    main()

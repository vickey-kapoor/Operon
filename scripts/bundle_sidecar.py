"""
Build and stage the operon-api sidecar binary for Tauri desktop packaging.

Usage:
    .venv\\Scripts\\python scripts\\bundle_sidecar.py           # build + stage
    .venv\\Scripts\\python scripts\\bundle_sidecar.py --check   # verify deps only
    .venv\\Scripts\\python scripts\\bundle_sidecar.py --triple aarch64-apple-darwin

Output:
    src-tauri/binaries/operon-api-<triple>[.exe]

The Rust host triple is derived from `rustc --print host` so it always matches
what Tauri expects. Falls back to platform-module detection if rustc is absent.

Cross-compilation is NOT supported by PyInstaller. Run this script on each
target OS to produce that platform's binary:
  - Windows  → operon-api-x86_64-pc-windows-msvc.exe
  - macOS    → operon-api-x86_64-apple-darwin  or  aarch64-apple-darwin
  - Linux    → operon-api-x86_64-unknown-linux-gnu

Requires PyInstaller in the active venv:
    pip install pyinstaller
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARIES_DIR = ROOT / "src-tauri" / "binaries"
ENTRY = ROOT / "scripts" / "sidecar_entry.py"
DIST_DIR = ROOT / "dist"

# Data separator: semicolon on Windows, colon on Unix (PyInstaller convention)
SEP = ";" if platform.system() == "Windows" else ":"


def get_triple(override: str | None = None) -> str:
    if override:
        return override
    try:
        result = subprocess.run(
            ["rustc", "--print", "host"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        machine = platform.machine().lower()
        arch = "x86_64" if machine in ("amd64", "x86_64") else "aarch64"
        sys_name = platform.system().lower()
        if sys_name == "windows":
            return f"{arch}-pc-windows-msvc"
        elif sys_name == "darwin":
            return f"{arch}-apple-darwin"
        else:
            return f"{arch}-unknown-linux-gnu"


def check_deps() -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            capture_output=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: PyInstaller not found.")
        print("       Install with: pip install pyinstaller")
        sys.exit(1)


def build(verbose: bool = False) -> Path:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "operon-api",
        "--noconfirm",
        "--clean",

        # ── Data files ──────────────────────────────────────────────────────
        # Prompt templates (loaded via Path(__file__).parents[N] — must mirror
        # the src/ tree structure inside the bundle so relative resolution works)
        "--add-data", f"prompts{SEP}prompts",
        # Static HTML served by FastAPI routes
        "--add-data", f"src/api/static{SEP}src/api/static",

        # ── Hidden imports ───────────────────────────────────────────────────
        # uvicorn auto-selects event loop / protocol implementations at runtime
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan.on",
        # FastAPI / Starlette internals loaded dynamically
        "--hidden-import", "fastapi",
        "--hidden-import", "starlette.routing",
        "--hidden-import", "starlette.middleware.cors",
        # Pydantic v2 core
        "--hidden-import", "pydantic",
        "--hidden-import", "pydantic_core",
        # dotenv used at import time in server.py
        "--hidden-import", "dotenv",
        # httpx http2 support
        "--hidden-import", "httpx",
        "--hidden-import", "h2",
        # Benchmark plugin imported dynamically in server.py
        "--hidden-import", "src.benchmarks.form_plugin",
        # All src.* submodules (captures dynamic imports inside the agent loop)
        "--collect-submodules", "src",

        # ── Exclusions ───────────────────────────────────────────────────────
        # Playwright ships its own Chromium (~170 MB). Exclude it — users must
        # run `playwright install chromium` separately. The sidecar only needs
        # Playwright's Python API, not the bundled browser.
        "--exclude-module", "playwright._impl._chromium_launcher",
        # Dev / test tooling — never needed at runtime
        "--exclude-module", "pytest",
        "--exclude-module", "ruff",
        "--exclude-module", "IPython",
        "--exclude-module", "notebook",

        str(ENTRY),
    ]

    if verbose:
        print("PyInstaller command:")
        print("  " + " \\\n  ".join(cmd))

    subprocess.run(cmd, cwd=ROOT, check=True)

    exe_name = "operon-api.exe" if platform.system() == "Windows" else "operon-api"
    exe = DIST_DIR / exe_name
    if not exe.exists():
        print(f"ERROR: expected output not found at {exe}")
        sys.exit(1)
    return exe


def stage(src: Path, triple: str) -> Path:
    BINARIES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if platform.system() == "Windows" else ""
    dest = BINARIES_DIR / f"operon-api-{triple}{suffix}"
    shutil.copy2(src, dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle operon-api sidecar binary for Tauri",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify deps only, skip build")
    parser.add_argument("--triple", metavar="TRIPLE", help="Override platform triple (e.g. aarch64-apple-darwin)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print PyInstaller command")
    args = parser.parse_args()

    check_deps()

    triple = get_triple(args.triple)
    print(f"Target triple : {triple}")

    if args.check:
        print("Deps OK — run without --check to build.")
        return

    print("Building with PyInstaller (this takes ~60–120s)…")
    exe = build(verbose=args.verbose)
    size_mb = exe.stat().st_size / (1024 * 1024)
    print(f"Built         : {exe.relative_to(ROOT)}  ({size_mb:.1f} MB)")

    dest = stage(exe, triple)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Staged        : {dest.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    print()
    print("Tauri will find the sidecar at:")
    print(f"  src-tauri/binaries/operon-api-{triple}{'/.exe' if platform.system() == 'Windows' else ''}")
    print()
    print("Next: cargo tauri build")


if __name__ == "__main__":
    main()

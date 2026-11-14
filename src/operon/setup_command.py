"""``operon setup``: get a new install ready to run a task in one command (KAN-140).

    operon setup              ask for the Gemini key, install the browser, run the self-check
    operon setup --check      only run the self-check
    operon setup --new-key    replace the saved key

Three steps:

1. **Key.** Uses ``GEMINI_API_KEY`` if one is already set (environment, a ``.env`` in this
   folder, or a key saved earlier); otherwise asks for it with the input hidden, checks it
   with Gemini, and saves it to ``~/.operon/operon.env`` (``OPERON_CONFIG_DIR``), readable
   only by you. Every ``operon`` command reads that file, so it works from any folder.
2. **Browser.** ``python -m playwright install chromium``, the browser Operon drives.
3. **Self-check.** The key is set and Gemini accepts it (listing models, which costs
   nothing), the browser starts and renders a page, and the settings are valid.

Nothing here runs a task or spends model credits.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

KEY_NAME = "GEMINI_API_KEY"
KEY_URL = "https://aistudio.google.com/apikey"
_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_TRY = 'operon run "Find the population of Paris on Wikipedia"'


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


# ── the key ──────────────────────────────────────────────────────────────────


def check_key(key: str, *, transport=None, timeout_s: float = 15.0) -> Check:
    """Ask Gemini to list models with ``key``: free, and fails on a wrong or disabled key."""
    import httpx

    url = os.getenv("GEMINI_API_BASE_URL", "").strip() or _MODELS_URL
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as http:
            response = http.get(url, params={"pageSize": 1}, headers={"x-goog-api-key": key})
    except httpx.HTTPError as exc:
        return Check("Gemini accepts the key", False, f"could not reach Gemini ({type(exc).__name__}); check your connection")
    if response.status_code == 200:
        return Check("Gemini accepts the key", True)
    if response.status_code in {400, 401, 403}:
        return Check("Gemini accepts the key", False, f"Gemini refused it (HTTP {response.status_code}); get a new key at {KEY_URL} and run `operon setup --new-key`")
    return Check("Gemini accepts the key", False, f"unexpected answer from Gemini (HTTP {response.status_code}); try again later")


def save_key(key: str, path: Path) -> None:
    """Write ``GEMINI_API_KEY=key`` into ``path``, keeping its other lines, readable only by you."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    kept = [line for line in lines if not line.strip().startswith(f"{KEY_NAME}=")]
    kept.append(f"{KEY_NAME}={key}")
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(kept) + "\n")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - Windows ignores POSIX modes
        pass


def _read_key(ask_secret: Callable[[str], str]) -> str:
    prompt = f"Paste your Gemini API key (free at {KEY_URL}; input is hidden): "
    return ask_secret(prompt).strip()


# ── the browser ──────────────────────────────────────────────────────────────


def install_browser(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Check:
    """Install the Chromium build Playwright drives (a no-op when it is already there)."""
    result = run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)
    if result.returncode == 0:
        return Check("Browser installed", True)
    hint = " On Linux, also run: python -m playwright install-deps chromium" if sys.platform.startswith("linux") else ""
    return Check("Browser installed", False, f"playwright install exited with {result.returncode}.{hint}")


async def _render_page() -> int:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 640, "height": 400})
            await page.set_content("<h1 style='font:48px sans-serif'>Operon self-check</h1>")
            return len(await page.screenshot())
        finally:
            await browser.close()


def check_browser(render: Callable[[], int] | None = None) -> Check:
    """Start the browser headless, render a page and take a screenshot, as a run would."""
    try:
        size = render() if render is not None else asyncio.run(_render_page())
    except Exception as exc:  # noqa: BLE001 - any failure here means "not ready"
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return Check("Browser starts and takes a screenshot", False, f"{first[:200]}; run `operon setup` again")
    if size <= 0:
        return Check("Browser starts and takes a screenshot", False, "the screenshot came back empty")
    return Check("Browser starts and takes a screenshot", True)


def check_settings() -> Check:
    from operon.api.runtime_config import validate_modes

    errors = validate_modes()
    if errors:
        return Check("Settings are valid", False, "; ".join(errors))
    return Check("Settings are valid", True)


# ── the command ──────────────────────────────────────────────────────────────


@dataclass
class Steps:
    """What ``setup`` calls for each step, so tests can run it with no network or browser."""

    ask_secret: Callable[[str], str] = getpass.getpass
    check_key: Callable[[str], Check] = check_key
    install_browser: Callable[[], Check] = install_browser
    check_browser: Callable[[], Check] = check_browser
    check_settings: Callable[[], Check] = check_settings


def _print_check(check: Check, out: Callable[[str], None]) -> None:
    mark = "ok  " if check.ok else "FAIL"
    out(f"  [{mark}] {check.name}" + (f": {check.detail}" if check.detail else ""))


def self_check(*, offline: bool, skip_browser: bool, steps: Steps, out: Callable[[str], None]) -> list[Check]:
    key = os.getenv(KEY_NAME, "").strip()
    checks = [Check(f"{KEY_NAME} is set", bool(key), "" if key else "run `operon setup` to add it")]
    if key and not offline:
        checks.append(steps.check_key(key))
    if not skip_browser:
        checks.append(steps.check_browser())
    checks.append(steps.check_settings())
    for check in checks:
        _print_check(check, out)
    return checks


def setup(args: argparse.Namespace, *, steps: Steps | None = None, out: Callable[[str], None] = print) -> int:
    """Run the setup steps; 0 when the self-check passes, 1 otherwise."""
    from operon.core.paths import user_env_file

    steps = steps or Steps()
    if not args.check:
        out("Operon setup\n")
        existing = os.getenv(KEY_NAME, "").strip()
        if existing and not args.new_key:
            out("1. Gemini key: already set, keeping it (use --new-key to replace it).")
        else:
            out("1. Gemini key")
            key = _read_key(steps.ask_secret)
            if not key:
                out(f"   No key given. Get one at {KEY_URL}, then run `operon setup` again.")
                return 1
            if not args.offline:
                verdict = steps.check_key(key)
                if not verdict.ok:
                    out(f"   Not saved: {verdict.detail}")
                    return 1
            path = user_env_file()
            save_key(key, path)
            os.environ[KEY_NAME] = key
            out(f"   Saved to {path} (only you can read it).")
        if args.skip_browser:
            out("2. Browser: skipped (--skip-browser).")
        else:
            out("2. Browser: installing Chromium for Operon (first time about 150 MB)...")
            installed = steps.install_browser()
            if not installed.ok:
                out(f"   {installed.detail}")
                return 1
        out("3. Self-check")
    checks = self_check(offline=args.offline, skip_browser=args.skip_browser, steps=steps, out=out)
    if all(check.ok for check in checks):
        out(f"\nReady. Try: {_TRY}")
        return 0
    out("\nNot ready yet: fix the FAIL lines above, then run `operon setup` again.")
    return 1


def add_parser(sub) -> None:
    parser = sub.add_parser("setup", help="get ready to run: Gemini key, browser, self-check")
    parser.add_argument("--check", action="store_true", help="only run the self-check")
    parser.add_argument("--new-key", action="store_true", help="ask for a new Gemini key even if one is set")
    parser.add_argument("--skip-browser", action="store_true", help="don't install or check the browser")
    parser.add_argument("--offline", action="store_true", help="don't ask Gemini to confirm the key")
    parser.set_defaults(func=setup)

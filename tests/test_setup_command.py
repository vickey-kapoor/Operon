"""`operon setup` (KAN-140): key, browser, self-check; offline, no real browser or network."""

from __future__ import annotations

import os
import stat
import subprocess
import sys

import httpx
import pytest

from operon import cli, setup_command
from operon.core import paths
from operon.setup_command import Check, Steps


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("OPERON_CONFIG_DIR", str(tmp_path / "cfg"))
    # setenv first so monkeypatch restores the original value even after setup writes os.environ
    for name in ("GEMINI_API_KEY", "GEMINI_API_BASE_URL"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return tmp_path / "cfg"


class Fake:
    """Records what setup asked for; every step passes unless told otherwise."""

    def __init__(self, *, key: str = "good-key", key_ok: bool = True, install_ok: bool = True, browser_ok: bool = True):
        self.calls: list[str] = []
        self.key, self.key_ok, self.install_ok, self.browser_ok = key, key_ok, install_ok, browser_ok

    def steps(self) -> Steps:
        def ask(prompt: str) -> str:
            self.calls.append("ask")
            return self.key

        def key(value: str) -> Check:
            self.calls.append(f"check_key:{value}")
            return Check("Gemini accepts the key", self.key_ok, "" if self.key_ok else "refused")

        def install() -> Check:
            self.calls.append("install")
            return Check("Browser installed", self.install_ok, "" if self.install_ok else "exit 1")

        def browser() -> Check:
            self.calls.append("browser")
            return Check("Browser starts and takes a screenshot", self.browser_ok)

        return Steps(ask_secret=ask, check_key=key, install_browser=install, check_browser=browser,
                     check_settings=lambda: Check("Settings are valid", True))


def _args(*argv: str):
    return cli.build_parser().parse_args(["setup", *argv])


def _run(fake: Fake, *argv: str) -> tuple[int, str]:
    lines: list[str] = []
    code = setup_command.setup(_args(*argv), steps=fake.steps(), out=lines.append)
    return code, "\n".join(lines)


def test_first_setup_saves_the_key_installs_and_checks(home) -> None:
    fake = Fake()
    code, text = _run(fake)
    assert code == 0, text
    assert fake.calls == ["ask", "check_key:good-key", "install", "check_key:good-key", "browser"]
    saved = (home / "operon.env").read_text(encoding="utf-8")
    assert saved == "GEMINI_API_KEY=good-key\n"
    assert os.environ["GEMINI_API_KEY"] == "good-key"
    assert "Ready. Try: operon run" in text
    assert "good-key" not in text  # the key is never printed


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_saved_key_file_is_private(home) -> None:
    _run(Fake())
    assert stat.S_IMODE((home / "operon.env").stat().st_mode) == 0o600


def test_a_refused_key_is_not_saved(home) -> None:
    fake = Fake(key_ok=False)
    code, text = _run(fake)
    assert code == 1
    assert "Not saved: refused" in text
    assert not (home / "operon.env").exists()
    assert "install" not in fake.calls


def test_an_empty_key_stops_setup(home) -> None:
    code, text = _run(Fake(key="  "))
    assert code == 1
    assert "No key given" in text


def test_an_existing_key_is_kept_unless_new_key(home, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "already-here")
    fake = Fake()
    code, text = _run(fake)
    assert code == 0
    assert "ask" not in fake.calls
    assert "already set" in text
    assert not (home / "operon.env").exists()

    fake = Fake(key="replacement")
    code, _ = _run(fake, "--new-key")
    assert code == 0
    assert fake.calls[0] == "ask"
    assert (home / "operon.env").read_text(encoding="utf-8") == "GEMINI_API_KEY=replacement\n"


def test_offline_and_skip_browser_skip_their_steps(home) -> None:
    fake = Fake()
    code, text = _run(fake, "--offline", "--skip-browser")
    assert code == 0, text
    assert fake.calls == ["ask"]
    assert "skipped (--skip-browser)" in text


def test_a_failed_browser_install_stops_before_the_check(home) -> None:
    fake = Fake(install_ok=False)
    code, text = _run(fake)
    assert code == 1
    assert "browser" not in fake.calls
    assert "exit 1" in text


def test_check_only_reports_a_missing_key_and_fails(home) -> None:
    fake = Fake(browser_ok=False)
    code, text = _run(fake, "--check")
    assert code == 1
    assert "ask" not in fake.calls and "install" not in fake.calls
    assert "[FAIL] GEMINI_API_KEY is set" in text
    assert "[FAIL] Browser starts" in text
    assert "Not ready yet" in text


def test_save_key_keeps_other_lines_and_replaces_the_old_key(tmp_path) -> None:
    path = tmp_path / "operon.env"
    path.write_text("OPERON_PERMISSION_MODE=ask\nGEMINI_API_KEY=old\n", encoding="utf-8")
    setup_command.save_key("new", path)
    assert path.read_text(encoding="utf-8") == "OPERON_PERMISSION_MODE=ask\nGEMINI_API_KEY=new\n"


@pytest.mark.parametrize(
    ("status", "ok", "phrase"),
    [(200, True, ""), (400, False, "refused"), (403, False, "refused"), (500, False, "unexpected")],
)
def test_check_key_reads_geminis_answer(home, status, ok, phrase) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json={})

    result = setup_command.check_key("k-123", transport=httpx.MockTransport(handler))
    assert result.ok is ok
    assert phrase in result.detail
    assert seen[0].headers["x-goog-api-key"] == "k-123"
    assert "k-123" not in str(seen[0].url)  # the key goes in a header, not the URL


def test_check_key_reports_no_connection(home) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    result = setup_command.check_key("k", transport=httpx.MockTransport(handler))
    assert not result.ok
    assert "could not reach Gemini" in result.detail


def test_install_browser_runs_playwright_with_this_python() -> None:
    seen: list[list[str]] = []

    def run(cmd, check):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    assert setup_command.install_browser(run).ok
    assert seen == [[sys.executable, "-m", "playwright", "install", "chromium"]]


def test_check_browser_turns_an_error_into_a_fail() -> None:
    def boom() -> int:
        raise RuntimeError("Executable doesn't exist at /x/chrome\nmore")

    result = setup_command.check_browser(boom)
    assert not result.ok
    assert "Executable doesn't exist" in result.detail and "more" not in result.detail
    assert setup_command.check_browser(lambda: 1234).ok


def test_user_env_file_is_loaded_without_overriding(home, monkeypatch) -> None:
    home.mkdir(parents=True)
    (home / "operon.env").write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    assert paths.load_user_env() is True
    assert os.environ["GEMINI_API_KEY"] == "from-file"

    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    paths.load_user_env()
    assert os.environ["GEMINI_API_KEY"] == "from-env"


def test_no_user_env_file_is_fine(home) -> None:
    assert paths.load_user_env() is False

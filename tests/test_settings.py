"""One typed settings object (KAN-157): parsing, startup checks, caching, per-run record."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from operon.core import settings as settings_mod
from operon.core.settings import (
    DIRECT_READS,
    SECRET_VARIABLES,
    Settings,
    SettingsError,
    check_startup_settings,
    get_settings,
)

ROOT = Path(__file__).resolve().parents[1]
_LISTED = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)=", re.MULTILINE)
# A read: os.getenv("X"), os.environ.get("X") or os.environ["X"], but not os.environ["X"] = ...
_DIRECT = re.compile(r"""(?:getenv\(|environ\.get\(|environ\[)\s*["']([A-Z][A-Z0-9_]+)["'](?!\]\s*=[^=])""")


def _listed() -> set[str]:
    return set(_LISTED.findall((ROOT / ".env.example").read_text(encoding="utf-8")))


def test_every_setting_is_listed_in_env_example() -> None:
    names = set(Settings.env_names().values()) | set(DIRECT_READS)
    assert names == _listed()


def test_only_settings_and_the_named_tools_read_operon_variables_directly() -> None:
    allowed = set(DIRECT_READS) | {"GEMINI_API_BASE_URL"}  # operon setup's key check
    for path in (ROOT / "src" / "operon").rglob("*.py"):
        if path.name == "settings.py" and path.parent.name == "core":
            continue
        read = set(_DIRECT.findall(path.read_text(encoding="utf-8")))
        stray = sorted(read - allowed)
        assert not stray, f"{path.relative_to(ROOT)} reads {stray} directly; add them to Settings"


def test_defaults_match_the_old_call_sites() -> None:
    s = Settings.from_env({})
    assert s.desktop_backend == s.browser_backend == "computer_use"
    assert s.browser_model == "gemini-3.5-flash"
    assert s.browser_thinking == "low" and s.browser_fallback_backend == "json"
    assert s.trust_gate is True and s.trust_confirm == ()
    assert s.permission_mode == "" and s.tools is None
    assert (s.zoom, s.zoom_min_side, s.zoom_factor) == ("dense", 2000, 2.0)
    assert (s.browser_width, s.browser_height, s.browser_headless) == (1920, 1080, False)
    assert s.runs_retain_days == 14 and s.stream_token_ttl_s == 600
    assert s.environment == "live" and s.desktop_platform == "windows"
    assert (s.sandbox_memory_mb, s.sandbox_start_timeout) == (4096, 240)
    assert s.errors == () and s.warnings == ()


def test_values_parse_as_before() -> None:
    s = Settings.from_env(
        {
            "API_KEYS": " a , ,b",
            "OPERON_ALLOWED_HOSTS": "Box.Local, ",
            "OPERON_BROWSER_THINKING": "",
            "OPERON_BROWSER_MODEL": "",
            "OPERON_DESKTOP_BACKEND": "  ",
            "OPERON_TRUST_GATE": "No",
            "OPERON_TRUST_DENY": "Rm -rf, Format",
            "OPERON_ZOOM": "yes",
            "OPERON_ZOOM_FACTOR": "9",
            "OPERON_STREAM_TOKEN_TTL_S": "1",
            "OPERON_BESTOFN_N": "0",
            "OPERON_ENVIRONMENT": " Sandbox ",
            "OPERON_DESKTOP_PLATFORM": "LINUX",
            "OPERON_TEST_SAFE_MODE": "TRUE",
            "OPERON_STRICT_ERRORS": "on",
        }
    )
    assert s.api_keys == ("a", "b")
    assert s.allowed_hosts == ("box.local",)
    assert s.browser_thinking is None  # blank turns thinking off; unset means "low"
    assert s.browser_model == "gemini-3.5-flash"
    assert s.desktop_backend == "computer_use"
    assert s.trust_gate is False
    assert s.trust_deny == ("rm -rf", "format")
    assert s.zoom == "dense" and s.zoom_factor == 6.0
    assert s.stream_token_ttl_s == 10
    assert s.bestofn_n == 1
    assert s.environment == "sandbox" and s.desktop_platform == "linux"
    assert s.test_safe_mode is True and s.strict_errors is True


def test_unusable_values_fall_back_with_a_warning() -> None:
    s = Settings.from_env({"OPERON_ZOOM": "maybe", "OPERON_ZOOM_MIN_SIDE": "big", "OPERON_PERMISSION_MODE": "yolo"})
    assert s.zoom == "dense" and s.zoom_min_side == 2000 and s.permission_mode == ""
    assert s.errors == ()
    assert len(s.warnings) == 3
    assert any("OPERON_ZOOM='maybe'" in w for w in s.warnings)


def test_values_that_used_to_crash_later_are_errors() -> None:
    s = Settings.from_env({"BROWSER_WIDTH": "wide", "OPERON_RUNS_RETAIN_DAYS": "two weeks"})
    assert s.browser_width == 1920
    assert len(s.errors) == 2
    assert any("BROWSER_WIDTH='wide' is not a number" in e for e in s.errors)


def test_startup_check_raises_with_every_problem(monkeypatch, caplog) -> None:
    monkeypatch.setenv("BROWSER_HEIGHT", "tall")
    monkeypatch.setenv("OPERON_BROWSER_BACKEND", "conputer_use")
    monkeypatch.setenv("OPERON_ZOOM", "maybe")
    with pytest.raises(SettingsError) as raised:
        check_startup_settings()
    message = str(raised.value)
    assert "BROWSER_HEIGHT='tall'" in message
    assert "OPERON_BROWSER_BACKEND='conputer_use'" in message
    assert "OPERON_ZOOM='maybe'" in caplog.text


def test_startup_check_passes_on_defaults(monkeypatch) -> None:
    for name in ("BROWSER_WIDTH", "BROWSER_HEIGHT", "OPERON_RUNS_RETAIN_DAYS"):
        monkeypatch.delenv(name, raising=False)
    assert isinstance(check_startup_settings(Settings.from_env({})), Settings)


def test_get_settings_follows_the_environment_and_caches(monkeypatch) -> None:
    monkeypatch.setenv("OPERON_GROUNDER", "snap")
    first = get_settings()
    assert first.grounder == "snap"
    assert get_settings() is first  # nothing changed: the same object
    monkeypatch.setenv("OPERON_GROUNDER", "Deterministic")
    assert get_settings().grounder == "deterministic"


def test_effective_settings_hide_secrets() -> None:
    s = Settings.from_env({"API_KEYS": "k1", "GEMINI_API_KEY": "sekrit", "OPERON_RUNS_ROOT": "/tmp/r"})
    effective = s.effective()
    assert effective["API_KEYS"] == "<set>" and effective["GEMINI_API_KEY"] == "<set>"
    assert effective["GOOGLE_API_KEY"] is None
    assert effective["OPERON_RUNS_ROOT"] == str(Path("/tmp/r"))
    assert "sekrit" not in json.dumps(effective)
    assert set(effective) == set(Settings.env_names().values())
    assert SECRET_VARIABLES <= set(effective)


def test_each_run_records_its_settings(tmp_path, monkeypatch) -> None:
    from operon.store.run_store import FileBackedRunStore

    monkeypatch.setenv("GEMINI_API_KEY", "sekrit")
    monkeypatch.setenv("OPERON_HARNESS_PROFILE", "thin")
    record = FileBackedRunStore(root_dir=tmp_path).create_run("do a thing")

    saved = json.loads((tmp_path / record.run_id / "settings.json").read_text(encoding="utf-8"))
    assert saved["OPERON_HARNESS_PROFILE"] == "thin"
    assert saved["GEMINI_API_KEY"] == "<set>"


def test_settings_module_has_no_import_time_reads() -> None:
    # The cache is filled lazily, so importing operon never freezes the environment.
    assert settings_mod._ENV_NAMES and "OPERON_ZOOM" in settings_mod._ENV_NAMES


def test_expect_check_flag_defaults_off_and_parses_on():
    from operon.core.settings import Settings

    assert Settings.from_env({}).expect_check is False
    assert Settings.from_env({"OPERON_EXPECT_CHECK": "on"}).expect_check is True


# --- Experimental flags default to off (CLAUDE.md invariant) ------------------------------
# Settings.from_env({}) reads only the dict it is given, so this is a clean environment.
# Every Settings field must be in exactly one list below, so adding a setting forces a
# conscious choice: experimental (off), plain switch (off), graduated (on), or plain config.
_EXPERIMENTAL_OFF = [
    ("claude_zoom", False),
    ("progress_revisit", False),
    ("settle_detect", False),
    ("change_note", False),
    ("focus_guard", False),
    ("trust_categories", False),
    ("secrets", False),
    ("dpi_aware", False),
    ("planner", False),
    ("task_note", False),
    ("checkpoints", False),
    ("click_text", False),
    ("expect_check", False),
    ("harness_profile", ""),  # "" is the full harness; "thin" is opt-in
    ("critic", ""),
    ("site_adapters", ""),
    ("model_profile", ""),
    ("grounder", "deterministic"),
    ("bestofn_n", 1),
    ("bestofn_confidence", 1.0),
    ("parallel_runs", 1),
]
# Boolean switches that are off by default but not experimental features.
_PLAIN_SWITCHES_OFF = [
    "auto_confirm",
    "builtin_browser",
    "use_vertex",
    "allow_unauthenticated",
    "browser_headless",
    "test_safe_mode",
    "strict_errors",
    "trace",
]
# Graduated: on by default on purpose. (`tools` is unset by default and resolves to on.)
_GRADUATED_ON = [("trust_gate", True), ("zoom", "dense")]
_GRADUATED_RESOLVED = ["tools"]
# Plain configuration: models, paths, hosts, keys, timeouts and sizes; no feature to turn on.
_PLAIN_CONFIG = {
    "api_keys", "cors_origins", "allowed_hosts", "api_url", "runs_retain_days", "stream_token_ttl_s",
    "hitl_webhook_url", "config_dir", "runtime_root", "runs_root", "browser_artifacts_root",
    "desktop_artifacts_root", "test_artifacts_root", "gemini_api_key", "google_api_key", "gemini_model",
    "gemini_api_base_url", "vertex_project", "vertex_location", "model_registry", "anthropic_api_key",
    "anthropic_api_base_url", "openai_api_key", "openai_api_base_url", "desktop_backend", "desktop_model",
    "desktop_thinking", "desktop_fallback_model", "desktop_verifier_model", "desktop_verifier_thinking",
    "browser_backend", "browser_model", "browser_thinking", "browser_fallback_backend",
    "browser_fallback_model", "browser_verifier_model", "browser_verifier_thinking", "settle_max_s",
    "planner_model", "task_note_max_items", "trust_deny", "trust_confirm", "trust_allow_domains", "trust_category_list",
    "permission_mode",
    "allowed_file_roots", "zoom_min_side", "zoom_factor", "environment", "desktop_platform",
    "sandbox_memory_mb", "sandbox_start_timeout", "container_image", "container_start_timeout",
    "container_ca_bundle", "osworld_url", "browser_width", "browser_height", "browser_profile_dir",
}


@pytest.mark.parametrize(("field", "default"), _EXPERIMENTAL_OFF)
def test_experimental_flags_default_off(field: str, default: object) -> None:
    assert getattr(Settings.from_env({}), field) == default


@pytest.mark.parametrize("field", _PLAIN_SWITCHES_OFF)
def test_plain_switches_default_off(field: str) -> None:
    assert getattr(Settings.from_env({}), field) is False


@pytest.mark.parametrize(("field", "default"), _GRADUATED_ON)
def test_graduated_flags_default_on(field: str, default: object) -> None:
    assert getattr(Settings.from_env({}), field) == default


def test_tools_are_on_by_default() -> None:
    from operon.agent.tools import DEFAULT_TOOLS, enabled_tools

    assert DEFAULT_TOOLS
    assert enabled_tools(Settings.from_env({}).tools) == DEFAULT_TOOLS


def test_every_setting_is_classified() -> None:
    fields = set(Settings.model_fields) - {"errors", "warnings"}
    groups = [
        {f for f, _ in _EXPERIMENTAL_OFF},
        set(_PLAIN_SWITCHES_OFF),
        {f for f, _ in _GRADUATED_ON},
        set(_GRADUATED_RESOLVED),
        _PLAIN_CONFIG,
    ]
    classified = set().union(*groups)
    assert sum(len(g) for g in groups) == len(classified), "a setting is in more than one list"
    assert fields == classified, (
        "classify new settings as experimental (off), plain switch (off), graduated (on) or plain config: "
        f"{sorted(fields ^ classified)}"
    )

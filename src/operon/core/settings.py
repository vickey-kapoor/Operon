"""Operon's settings as one typed object (KAN-157).

Every ``OPERON_*`` (and the few other) environment variables the agent reads are
parsed here, into :class:`Settings`, instead of by ``os.getenv`` calls spread over
the code. ``get_settings()`` returns the current settings; code reads fields off it.

* **Same values as before.** Each field is parsed exactly as its old call site did:
  the same defaults, the same accepted spellings, and the same quiet fallback for a
  value that is not understood.
* **Validated once at startup.** ``operon serve`` / the API and ``operon run`` call
  :func:`check_startup_settings`: a value that would crash later (``BROWSER_WIDTH=wide``)
  or that names something unknown (``OPERON_BROWSER_BACKEND=conputer_use``) stops
  startup with every problem listed; a value that quietly falls back to its default
  (``OPERON_ZOOM=maybe``) is logged as a warning.
* **Recorded with each run.** :meth:`Settings.effective` is the set of variables a run
  saw, with secrets replaced by ``"<set>"``; the run store writes it to the run's
  ``settings.json``.
* **Follows the environment.** The result is cached against the current values of the
  variables it reads, so a ``.env`` loaded at startup, ``operon setup`` saving a key, or
  a test's ``monkeypatch.setenv`` is picked up on the next call without a reload step.

``.env.example`` stays the contract: every variable listed there is a field here (or
one of :data:`DIRECT_READS`), and ``tests/test_settings.py`` checks both ways.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}

# Variables some module still reads itself, and why (the contract test allows these).
DIRECT_READS: dict[str, str] = {
    # `operon setup` reads and writes the key before any settings exist, and tells a
    # key set in the shell from one in a file; it also checks against GEMINI_API_BASE_URL.
    "GEMINI_API_KEY": "operon setup (src/operon/setup_command.py)",
    # The Windows Sandbox guest runs inside the sandbox, as its own process.
    "OPERON_GUEST_TOKEN": "sandbox guest (src/operon/environments/guest.py)",
    # The Azure helper is a developer tool with injectable settings of its own.
    "OPERON_CLOUD_RESOURCE_GROUP": "operon cloud (src/operon/cloud.py)",
    "OPERON_CLOUD_VM": "operon cloud (src/operon/cloud.py)",
    "OPERON_CLOUD_LOCATION": "operon cloud (src/operon/cloud.py)",
    "OPERON_CLOUD_SIZE": "operon cloud (src/operon/cloud.py)",
    "OPERON_CLOUD_ADMIN": "operon cloud (src/operon/cloud.py)",
    "OPERON_CLOUD_LOCAL_PORT": "operon cloud (src/operon/cloud.py)",
}

# Values never written to a run's settings.json or a log.
SECRET_VARIABLES = frozenset({"API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY", "HITL_WEBHOOK_URL"})


def _env_field(env: str, default: Any = None, **kwargs: Any) -> Any:
    return Field(default=default, json_schema_extra={"env": env}, **kwargs)


class Settings(BaseModel):
    """Operon's effective settings, parsed from the environment."""

    model_config = ConfigDict(frozen=True)

    # --- API and CLI --------------------------------------------------------------
    api_keys: tuple[str, ...] = _env_field("API_KEYS", ())
    allow_unauthenticated: bool = _env_field("OPERON_ALLOW_UNAUTHENTICATED", False)
    cors_origins: tuple[str, ...] = _env_field("CORS_ORIGINS", ())
    allowed_hosts: tuple[str, ...] = _env_field("OPERON_ALLOWED_HOSTS", ())
    api_url: str | None = _env_field("OPERON_API_URL")
    runs_retain_days: int = _env_field("OPERON_RUNS_RETAIN_DAYS", 14)
    stream_token_ttl_s: int = _env_field("OPERON_STREAM_TOKEN_TTL_S", 600)
    hitl_webhook_url: str = _env_field("HITL_WEBHOOK_URL", "")

    # --- Folders (None: the default under runtime_dir()) -----------------------------
    config_dir: Path | None = _env_field("OPERON_CONFIG_DIR")
    runtime_root: Path | None = _env_field("OPERON_RUNTIME_ROOT")
    runs_root: Path | None = _env_field("OPERON_RUNS_ROOT")
    browser_artifacts_root: Path | None = _env_field("OPERON_BROWSER_ARTIFACTS_ROOT")
    desktop_artifacts_root: Path | None = _env_field("OPERON_DESKTOP_ARTIFACTS_ROOT")
    test_artifacts_root: Path | None = _env_field("OPERON_TEST_ARTIFACTS_ROOT")

    # --- Gemini ------------------------------------------------------------------
    gemini_api_key: str | None = _env_field("GEMINI_API_KEY")
    google_api_key: str | None = _env_field("GOOGLE_API_KEY")
    gemini_model: str = _env_field("GEMINI_MODEL", "gemini-3.5-flash")
    gemini_api_base_url: str = _env_field(
        "GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models"
    )
    use_vertex: bool = _env_field("OPERON_USE_VERTEX", False)
    vertex_project: str | None = _env_field("VERTEX_PROJECT")
    vertex_location: str = _env_field("VERTEX_LOCATION", "us-central1")

    # --- Per-environment models (None for thinking: the model's own default) --------
    desktop_backend: str = _env_field("OPERON_DESKTOP_BACKEND", "computer_use")
    desktop_model: str = _env_field("OPERON_DESKTOP_MODEL", "gemini-3.5-flash")
    desktop_thinking: str | None = _env_field("OPERON_DESKTOP_THINKING", "low")
    desktop_fallback_model: str = _env_field("OPERON_DESKTOP_FALLBACK_MODEL", "gemini-3.5-flash")
    desktop_verifier_model: str = _env_field("OPERON_DESKTOP_VERIFIER_MODEL", "gemini-3.5-flash")
    desktop_verifier_thinking: str | None = _env_field("OPERON_DESKTOP_VERIFIER_THINKING", "low")
    browser_backend: str = _env_field("OPERON_BROWSER_BACKEND", "computer_use")
    browser_model: str = _env_field("OPERON_BROWSER_MODEL", "gemini-3.5-flash")
    browser_thinking: str | None = _env_field("OPERON_BROWSER_THINKING", "low")
    browser_fallback_backend: str = _env_field("OPERON_BROWSER_FALLBACK_BACKEND", "json")
    browser_fallback_model: str = _env_field("OPERON_BROWSER_FALLBACK_MODEL", "gemini-3.5-flash")
    browser_verifier_model: str = _env_field("OPERON_BROWSER_VERIFIER_MODEL", "gemini-3.5-flash")
    browser_verifier_thinking: str | None = _env_field("OPERON_BROWSER_VERIFIER_THINKING", "low")

    # --- Harness and policy ---------------------------------------------------------
    harness_profile: str = _env_field("OPERON_HARNESS_PROFILE", "")
    critic: str = _env_field("OPERON_CRITIC", "")
    grounder: str = _env_field("OPERON_GROUNDER", "deterministic")
    site_adapters: str = _env_field("OPERON_SITE_ADAPTERS", "")
    progress_revisit: bool = _env_field("OPERON_PROGRESS_REVISIT", False)
    bestofn_n: int = _env_field("OPERON_BESTOFN_N", 1)
    bestofn_confidence: float = _env_field("OPERON_BESTOFN_CONFIDENCE", 1.0)
    trust_gate: bool = _env_field("OPERON_TRUST_GATE", True)
    trust_deny: tuple[str, ...] = _env_field("OPERON_TRUST_DENY", ())
    # Empty: the gate's built-in confirm list.
    trust_confirm: tuple[str, ...] = _env_field("OPERON_TRUST_CONFIRM", ())
    trust_allow_domains: tuple[str, ...] = _env_field("OPERON_TRUST_ALLOW_DOMAINS", ())
    # "" when unset or not a known mode: the server default (risky).
    permission_mode: str = _env_field("OPERON_PERMISSION_MODE", "")
    allowed_file_roots: tuple[str, ...] = _env_field("OPERON_ALLOWED_FILE_ROOTS", ())
    # None: unset (the tool list's own default, "on").
    tools: str | None = _env_field("OPERON_TOOLS")
    zoom: str = _env_field("OPERON_ZOOM", "off")
    zoom_min_side: int = _env_field("OPERON_ZOOM_MIN_SIDE", 2000)
    zoom_factor: float = _env_field("OPERON_ZOOM_FACTOR", 2.0)

    # --- Machines ------------------------------------------------------------------
    environment: str = _env_field("OPERON_ENVIRONMENT", "live")
    desktop_platform: str = _env_field("OPERON_DESKTOP_PLATFORM", "windows")
    sandbox_memory_mb: int = _env_field("OPERON_SANDBOX_MEMORY_MB", 4096)
    sandbox_start_timeout: int = _env_field("OPERON_SANDBOX_START_TIMEOUT", 240)
    browser_width: int = _env_field("BROWSER_WIDTH", 1920)
    browser_height: int = _env_field("BROWSER_HEIGHT", 1080)
    browser_headless: bool = _env_field("BROWSER_HEADLESS", False)

    # --- Diagnostics -----------------------------------------------------------------
    test_safe_mode: bool = _env_field("OPERON_TEST_SAFE_MODE", False)
    strict_errors: bool = _env_field("OPERON_STRICT_ERRORS", False)
    trace: bool = _env_field("OPERON_TRACE", False)

    # Problems found while parsing (not settings themselves).
    errors: tuple[str, ...] = Field(default=(), exclude=True)
    warnings: tuple[str, ...] = Field(default=(), exclude=True)

    @classmethod
    def env_names(cls) -> dict[str, str]:
        """Field name -> environment variable, for every setting."""
        names: dict[str, str] = {}
        for name, info in cls.model_fields.items():
            extra = info.json_schema_extra
            if isinstance(extra, dict) and "env" in extra:
                names[name] = str(extra["env"])
        return names

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Parse settings from ``env`` (default ``os.environ``)."""
        return _Parser(os.environ if env is None else env).build()

    def effective(self) -> dict[str, Any]:
        """The settings a run used, by variable name; secrets show only as ``"<set>"``."""
        out: dict[str, Any] = {}
        for field_name, env_name in self.env_names().items():
            value = getattr(self, field_name)
            if env_name in SECRET_VARIABLES:
                value = "<set>" if value else None
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            out[env_name] = value
        return out


def _split(raw: str, *, lower: bool = False) -> tuple[str, ...]:
    items = (item.strip() for item in raw.split(","))
    return tuple(item.lower() if lower else item for item in items if item)


class _Parser:
    """Reads each variable as its old call site did, noting values it cannot use."""

    def __init__(self, env: Mapping[str, str]) -> None:
        self.env = env
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def raw(self, name: str) -> str | None:
        return self.env.get(name)

    def text(self, name: str, default: str = "") -> str:
        return self.env.get(name, default)

    def or_default(self, name: str, default: str) -> str:
        """``os.getenv(name) or default``: unset and blank both mean the default."""
        return self.env.get(name) or default

    def stripped(self, name: str, default: str = "") -> str:
        return (self.env.get(name) or "").strip() or default

    def flag(self, name: str, default: str = "", *, on: set[str] | None = None) -> bool:
        return self.env.get(name, default).strip().lower() in (on or _TRUE)

    def path(self, name: str) -> Path | None:
        value = self.env.get(name, "")
        return Path(value) if value.strip() else None

    def number(self, name: str, default: str, convert: Callable[[str], Any], *, fatal: bool) -> Any:
        raw = self.env.get(name, default)
        try:
            return convert(raw)
        except ValueError:
            message = f"{name}={raw!r} is not a number"
            if fatal:
                self.errors.append(message)
            else:
                self.warnings.append(f"{message}; using {default}")
            return convert(default)

    def choice(self, name: str, value: str, allowed: set[str], fallback: str) -> str:
        if value in allowed:
            return value
        if value:
            self.warnings.append(f"{name}={value!r} is not one of {sorted(allowed)}; using {fallback!r}")
        return fallback

    def build(self) -> Settings:
        from operon.agent.actions import zoom as zoom_mod

        zoom_mode = (self.raw("OPERON_ZOOM") or zoom_mod.OFF).strip().lower()
        if zoom_mode in _TRUE:
            zoom_mode = zoom_mod.DENSE
        zoom_mode = self.choice("OPERON_ZOOM", zoom_mode, set(zoom_mod.MODES), zoom_mod.OFF)

        permission = self.text("OPERON_PERMISSION_MODE").strip().lower()
        permission = self.choice("OPERON_PERMISSION_MODE", permission, {"ask", "risky", "auto"}, "")

        platform = (self.raw("OPERON_DESKTOP_PLATFORM") or "").strip().lower()
        platform = "linux" if platform == "linux" else "windows"

        def thinking(name: str) -> str | None:
            return self.text(name, "low") or None

        def sandbox_int(name: str, default: int) -> int:
            raw = self.raw(name) or str(default)
            try:
                return int(raw)
            except ValueError:
                self.warnings.append(f"{name}={raw!r} is not a number; using {default}")
                return default

        values: dict[str, Any] = dict(
            api_keys=_split(self.text("API_KEYS")),
            allow_unauthenticated=self.flag("OPERON_ALLOW_UNAUTHENTICATED", "false"),
            cors_origins=_split(self.text("CORS_ORIGINS")),
            allowed_hosts=_split(self.text("OPERON_ALLOWED_HOSTS"), lower=True),
            api_url=self.raw("OPERON_API_URL"),
            runs_retain_days=self.number("OPERON_RUNS_RETAIN_DAYS", "14", int, fatal=True),
            stream_token_ttl_s=max(10, min(self.number("OPERON_STREAM_TOKEN_TTL_S", "600", int, fatal=False), 86400)),
            hitl_webhook_url=self.text("HITL_WEBHOOK_URL").strip(),
            config_dir=self.path("OPERON_CONFIG_DIR"),
            runtime_root=self.path("OPERON_RUNTIME_ROOT"),
            runs_root=self.path("OPERON_RUNS_ROOT"),
            browser_artifacts_root=self.path("OPERON_BROWSER_ARTIFACTS_ROOT"),
            desktop_artifacts_root=self.path("OPERON_DESKTOP_ARTIFACTS_ROOT"),
            test_artifacts_root=self.path("OPERON_TEST_ARTIFACTS_ROOT"),
            gemini_api_key=self.raw("GEMINI_API_KEY"),
            google_api_key=self.raw("GOOGLE_API_KEY"),
            gemini_model=self.text("GEMINI_MODEL", "gemini-3.5-flash"),
            gemini_api_base_url=self.text("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models"),
            use_vertex=self.text("OPERON_USE_VERTEX").lower() in {"1", "true", "yes"},
            vertex_project=self.raw("VERTEX_PROJECT"),
            vertex_location=self.text("VERTEX_LOCATION", "us-central1"),
            desktop_backend=self.stripped("OPERON_DESKTOP_BACKEND", "computer_use"),
            desktop_model=self.or_default("OPERON_DESKTOP_MODEL", "gemini-3.5-flash"),
            desktop_thinking=thinking("OPERON_DESKTOP_THINKING"),
            desktop_fallback_model=self.or_default("OPERON_DESKTOP_FALLBACK_MODEL", "gemini-3.5-flash"),
            desktop_verifier_model=self.or_default("OPERON_DESKTOP_VERIFIER_MODEL", "gemini-3.5-flash"),
            desktop_verifier_thinking=thinking("OPERON_DESKTOP_VERIFIER_THINKING"),
            browser_backend=self.stripped("OPERON_BROWSER_BACKEND", "computer_use"),
            browser_model=self.or_default("OPERON_BROWSER_MODEL", "gemini-3.5-flash"),
            browser_thinking=thinking("OPERON_BROWSER_THINKING"),
            browser_fallback_backend=self.text("OPERON_BROWSER_FALLBACK_BACKEND", "json"),
            browser_fallback_model=self.or_default("OPERON_BROWSER_FALLBACK_MODEL", "gemini-3.5-flash"),
            browser_verifier_model=self.or_default("OPERON_BROWSER_VERIFIER_MODEL", "gemini-3.5-flash"),
            browser_verifier_thinking=thinking("OPERON_BROWSER_VERIFIER_THINKING"),
            harness_profile=(self.raw("OPERON_HARNESS_PROFILE") or "").strip().lower(),
            critic=(self.raw("OPERON_CRITIC") or "").strip().lower(),
            grounder=self.text("OPERON_GROUNDER", "deterministic").strip().lower(),
            site_adapters=self.text("OPERON_SITE_ADAPTERS").strip(),
            progress_revisit=self.flag("OPERON_PROGRESS_REVISIT", "off"),
            bestofn_n=max(1, self.number("OPERON_BESTOFN_N", "1", int, fatal=False)),
            bestofn_confidence=self.number("OPERON_BESTOFN_CONFIDENCE", "1.0", float, fatal=False),
            trust_gate=self.text("OPERON_TRUST_GATE", "on").strip().lower() not in {"off", "false", "0", "no"},
            trust_deny=_split(self.text("OPERON_TRUST_DENY"), lower=True),
            trust_confirm=_split(self.text("OPERON_TRUST_CONFIRM"), lower=True),
            trust_allow_domains=_split(self.text("OPERON_TRUST_ALLOW_DOMAINS"), lower=True),
            permission_mode=permission,
            allowed_file_roots=_split(self.text("OPERON_ALLOWED_FILE_ROOTS")),
            tools=self.raw("OPERON_TOOLS"),
            zoom=zoom_mode,
            zoom_min_side=max(1, self.number("OPERON_ZOOM_MIN_SIDE", str(zoom_mod.DEFAULT_MIN_SIDE), int, fatal=False)),
            zoom_factor=min(6.0, max(1.5, self.number("OPERON_ZOOM_FACTOR", str(zoom_mod.DEFAULT_FACTOR), float, fatal=False))),
            environment=(self.raw("OPERON_ENVIRONMENT") or "").strip().lower() or "live",
            desktop_platform=platform,
            sandbox_memory_mb=sandbox_int("OPERON_SANDBOX_MEMORY_MB", 4096),
            sandbox_start_timeout=sandbox_int("OPERON_SANDBOX_START_TIMEOUT", 240),
            browser_width=self.number("BROWSER_WIDTH", "1920", int, fatal=True),
            browser_height=self.number("BROWSER_HEIGHT", "1080", int, fatal=True),
            browser_headless=self.text("BROWSER_HEADLESS", "false").lower() == "true",
            test_safe_mode=self.text("OPERON_TEST_SAFE_MODE", "false").lower() == "true",
            strict_errors=self.flag("OPERON_STRICT_ERRORS"),
            trace=self.text("OPERON_TRACE").lower() in {"1", "true", "yes"},
        )
        return Settings(**values, errors=tuple(self.errors), warnings=tuple(self.warnings))


_ENV_NAMES: tuple[str, ...] = tuple(Settings.env_names().values())
_lock = threading.Lock()
_cached: tuple[tuple[str | None, ...], Settings] | None = None


def get_settings() -> Settings:
    """The current settings, re-parsed only when one of their variables changed."""
    global _cached
    key = tuple(os.environ.get(name) for name in _ENV_NAMES)
    cached = _cached
    if cached is not None and cached[0] == key:
        return cached[1]
    with _lock:
        settings = Settings.from_env(dict(os.environ))
        _cached = (key, settings)
        return settings


class SettingsError(RuntimeError):
    """Raised at startup when a setting cannot be used."""


def startup_problems(settings: Settings | None = None) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` for the current settings, including model/backend checks."""
    from operon.api.runtime_config import validate_modes

    current = settings or get_settings()
    return [*current.errors, *validate_modes(current)], list(current.warnings)


def check_startup_settings(settings: Settings | None = None) -> Settings:
    """Validate the settings once at startup: log warnings, raise on errors."""
    current = settings or get_settings()
    errors, warnings = startup_problems(current)
    for warning in warnings:
        logger.warning("setting ignored: %s", warning)
    if errors:
        joined = "\n  - ".join(errors)
        raise SettingsError(f"Invalid Operon runtime configuration:\n  - {joined}")
    return current

"""Model registry: ``evals/models.yaml`` (or ``$OPERON_MODEL_REGISTRY``).

Each top-level key is a profile name; its value says how Operon talks to that model.
A typo in a key fails loudly at load time rather than silently using a default.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from operon.core.paths import project_root

# Env vars an ``openai_compat`` profile reads unless it names others.
DEFAULT_BASE_URL_ENV = "OPENWEIGHT_BASE_URL"
DEFAULT_API_KEY_ENV = "OPENWEIGHT_API_KEY"

Adapter = Literal["openai_compat", "claude_native", "gemini_native", "openai_native"]
Parser = Literal["neutral_tools", "ui_tars", "opencua", "native"]
CoordSpace = Literal["pixels", "relative_1000", "normalized"]
Environment = Literal["browser", "desktop"]

# Adapters whose actions come from Operon's existing native backends; their profile
# only picks the model, so format, coordinates and images are the backend's own.
NATIVE_ADAPTERS = frozenset({"claude_native", "gemini_native", "openai_native"})


class ModelRegistryError(ValueError):
    """The registry file or a profile in it is invalid, or a profile is unknown."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageRule(_Strict):
    """How a screenshot is resized before it is sent.

    ``long_edge``: scale down so the longer side is at most ``max_long_edge`` px.
    ``smart``: Qwen-VL style ``smart_resize``, both sides rounded to multiples of
    ``factor`` (28 for Qwen2.5-VL and UI-TARS, 32 for Qwen3-VL) with the pixel count
    kept between ``min_pixels`` and ``max_pixels``."""

    resize: Literal["long_edge", "smart"] = "long_edge"
    max_long_edge: int = Field(default=1280, ge=64)
    factor: int = Field(default=28, ge=1)
    min_pixels: int = Field(default=3136, ge=784)
    max_pixels: int = Field(default=1_003_520, ge=784)
    format: Literal["jpeg", "png"] = "jpeg"


class Pricing(_Strict):
    """List prices for cost reports: per million tokens, or per GPU hour when self-hosted."""

    input_per_mtok: float | None = Field(default=None, ge=0)
    output_per_mtok: float | None = Field(default=None, ge=0)
    # Cached input tokens (prompt-cache hits); left out, they bill at input_per_mtok.
    cached_input_per_mtok: float | None = Field(default=None, ge=0)
    gpu_hour_usd: float | None = Field(default=None, ge=0)


class ModelProfile(_Strict):
    name: str = Field(min_length=1)
    adapter: Adapter
    parser: Parser = "neutral_tools"
    model: str = Field(min_length=1)
    # Where an openai_compat request goes: ``base_url`` itself, else the env var it names.
    base_url: str | None = None
    base_url_env: str = DEFAULT_BASE_URL_ENV
    api_key_env: str = DEFAULT_API_KEY_ENV
    coords: CoordSpace = "pixels"
    image: ImageRule = ImageRule()
    # Screenshots kept in the request: the current one plus up to N - 1 earlier ones.
    # Older turns stay in as text, so the model still sees what it did.
    history_screens: int = Field(default=3, ge=1, le=20)
    temperature: float | None = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=16, le=32768)
    # Native adapters: Claude's effort or Gemini's thinking level ("low", "high", ...).
    effort: str | None = None
    pricing: Pricing = Pricing()
    environments: tuple[Environment, ...] = ("browser", "desktop")
    notes: str = ""

    @model_validator(mode="after")
    def _check_pairing(self) -> "ModelProfile":
        if self.adapter in NATIVE_ADAPTERS and self.parser not in {"native", "neutral_tools"}:
            raise ValueError(f"adapter {self.adapter} has its own action format; drop parser={self.parser}")
        if self.adapter == "openai_compat" and self.parser == "native":
            raise ValueError("adapter openai_compat needs parser neutral_tools, ui_tars or opencua")
        return self

    @property
    def is_native(self) -> bool:
        return self.adapter in NATIVE_ADAPTERS

    def resolved_base_url(self) -> str:
        url = (self.base_url or os.getenv(self.base_url_env) or "").strip()
        if not url:
            raise ModelRegistryError(f"profile {self.name}: set {self.base_url_env} to the model server's base URL")
        return url.rstrip("/")

    def api_key(self) -> str | None:
        return (os.getenv(self.api_key_env) or "").strip() or None


def registry_path() -> Path:
    value = (os.getenv("OPERON_MODEL_REGISTRY") or "").strip()
    return Path(value) if value else project_root() / "evals" / "models.yaml"


def load_registry(path: Path | None = None) -> dict[str, ModelProfile]:
    """Every profile in the registry file, by name."""
    path = path or registry_path()
    return dict(_load(str(path.resolve())))


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[tuple[str, ModelProfile], ...]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ModelRegistryError(f"cannot read model registry {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ModelRegistryError(f"model registry {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelRegistryError(f"model registry {path} must map profile names to settings")
    profiles = []
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ModelRegistryError(f"profile {name!r} in {path} must be a mapping")
        try:
            profiles.append((str(name), ModelProfile(name=str(name), **body)))
        except ValidationError as exc:
            raise ModelRegistryError(f"profile {name!r} in {path} is invalid: {exc}") from exc
    return tuple(profiles)


def resolve_profile(name: str | None = None, *, path: Path | None = None) -> ModelProfile:
    """The profile ``name``, or ``$OPERON_MODEL_PROFILE`` when no name is given."""
    name = (name or os.getenv("OPERON_MODEL_PROFILE") or "").strip()
    if not name:
        raise ModelRegistryError("backend=model needs OPERON_MODEL_PROFILE set to a profile in the model registry")
    profiles = load_registry(path)
    if name not in profiles:
        raise ModelRegistryError(f"unknown model profile {name!r}; known: {', '.join(sorted(profiles))}")
    return profiles[name]

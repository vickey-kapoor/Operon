"""Shared Pydantic models for the UI Navigator API."""

import ipaddress
import time
import urllib.parse
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agent.core import AgentResult

_MAX_TASK_CHARS = 2_000

# Schemes that must never reach the browser executor from user input.
_BLOCKED_SCHEMES = frozenset({"javascript", "file", "data", "vbscript", "ftp"})

# RFC-1918, loopback, link-local (metadata services), and other reserved ranges.
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # AWS/GCP metadata endpoint
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _validate_start_url(url: str) -> str:
    """
    Reject URLs that could be used for SSRF attacks.

    Blocks:
    - Dangerous schemes (file://, javascript://, data://, etc.)
    - Direct navigation to localhost variants
    - Direct navigation to private / reserved IP ranges

    Note: DNS-rebinding attacks are out of scope for this validation layer.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        raise ValueError(f"Malformed URL: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme in _BLOCKED_SCHEMES:
        raise ValueError(f"URL scheme {scheme!r} is not permitted")
    if scheme and scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs are permitted (got {scheme!r})")

    hostname = (parsed.hostname or "").lower()
    if hostname in ("localhost", ""):
        raise ValueError("Navigation to localhost is not permitted")

    # Reject raw private/reserved IP addresses.
    try:
        addr = ipaddress.ip_address(hostname)
        for net in _PRIVATE_NETWORKS:
            if addr in net:
                raise ValueError(
                    f"Navigation to private/reserved address {hostname!r} is not permitted"
                )
    except ValueError as exc:
        if "is not permitted" in str(exc):
            raise
        # hostname is a domain name, not a raw IP — allowed.

    return url


class TaskStatus(str, Enum):
    """Valid lifecycle states for a navigation task."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class TaskRecord(BaseModel):
    """Internal record tracking an async navigation task."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    task: str
    start_url: Optional[str] = None
    max_steps: int = 20
    result: Optional[AgentResult] = None
    events: List[dict] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)

    model_config = ConfigDict(arbitrary_types_allowed=True)


_ALLOWED_MODELS = frozenset({"gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"})


class NavigateRequest(BaseModel):
    task: str = Field(
        ...,
        description="High-level user intent",
        max_length=_MAX_TASK_CHARS,
    )
    start_url: Optional[str] = Field(None, description="Optional URL to open first")
    max_steps: int = Field(20, ge=1, le=50, description="Max agent steps")
    model: Optional[str] = Field(None, description="Gemini model override")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt override")

    @field_validator("start_url", mode="before")
    @classmethod
    def validate_start_url(cls, v):
        if v is None:
            return v
        return _validate_start_url(str(v))

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _ALLOWED_MODELS:
            raise ValueError(f"Model '{v}' is not allowed. Permitted models: {sorted(_ALLOWED_MODELS)}")
        return v


class NavigateResponse(BaseModel):
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    task: str
    start_url: Optional[str]
    max_steps: int
    result: Optional[AgentResult]


class TaskListResponse(BaseModel):
    tasks: List[TaskStatusResponse]
    total: int
    limit: int
    offset: int

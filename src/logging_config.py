"""Shared structured JSON logging configuration.

Call ``configure_logging()`` once at application startup.  All subsequent
``logging.getLogger(__name__)`` calls automatically inherit the JSON format.

``request_id_var`` is a ContextVar that, when set, is automatically injected
into every log record as ``request_id``.  Set it in a request middleware so all
logs within that request context are correlated.
"""

import contextvars
import json
import logging
import os
import sys

# Per-request correlation ID.  Set this in request middleware; it is
# automatically included in every log record emitted from the same
# asyncio task context.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


class _ContextFilter(logging.Filter):
    """Inject the current ``request_id`` into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("")
        return True


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    # Standard LogRecord attributes that are captured via dedicated keys
    # and should not be duplicated in the extra-fields pass.
    _SKIP = frozenset({
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs",
        "msg", "name", "pathname", "process", "processName", "relativeCreated",
        "stack_info", "taskName", "thread", "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        # Merge any extra= fields passed to the logger call.
        for key, value in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure the root logger to emit structured JSON to stdout."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    handler.addFilter(_ContextFilter())

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)

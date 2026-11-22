"""The environment a program the agent launches (an app or a shell command) starts with."""

from __future__ import annotations

import os
import re

# Environment variables such a program must not inherit: a shell the model (or a
# prompt injection on screen) drives could otherwise print or send the server's API
# keys and tokens.
_SECRET_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|WEBHOOK|DATABASE_URL", re.IGNORECASE)


def launch_env() -> dict[str, str]:
    """This process's environment minus anything that looks like a secret."""
    return {name: value for name, value in os.environ.items() if not _SECRET_ENV_RE.search(name)}

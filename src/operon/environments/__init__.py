"""Environment providers (KAN-80): where a desktop run happens.

A provider owns one machine Operon drives: it gets the machine ready, builds the
desktop executor for it, and cleans it up. The loop only ever sees the executor,
so a new place to run (Windows Sandbox, a Linux container, a cloud VM) is a new
provider, not a change to the loop.

Pick one with ``OPERON_ENVIRONMENT`` (default ``live``, today's machine).
"""

from operon.environments.base import EnvironmentProvider, ProviderCheck
from operon.environments.registry import (
    DEFAULT_ENVIRONMENT,
    environment_name,
    get_provider,
    provider_names,
)

__all__ = [
    "DEFAULT_ENVIRONMENT",
    "EnvironmentProvider",
    "ProviderCheck",
    "environment_name",
    "get_provider",
    "provider_names",
]

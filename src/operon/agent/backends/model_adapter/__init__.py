"""Cross-model adapter layer (``OPERON_BROWSER_BACKEND=model``, ``OPERON_MODEL_PROFILE=<name>``).

Any model plugs into the same loop, executors and graders through one
``AgentBackend``, split into three swappable parts:

- an action *format*: the prompt, the tools (if any) and the parser that turns a
  reply into neutral actions (``formats/``);
- a *transport*: how a request reaches the model (``transports/``);
- the *profile*: one entry of the model registry (``evals/models.yaml``) naming the
  format, transport, coordinate space, image rule and history length.

Experimental and off by default: nothing here runs unless the backend is ``model``.
"""

from operon.agent.backends.model_adapter.registry import (
    ModelProfile,
    ModelRegistryError,
    load_registry,
    resolve_profile,
)

__all__ = ["ModelProfile", "ModelRegistryError", "load_registry", "resolve_profile"]

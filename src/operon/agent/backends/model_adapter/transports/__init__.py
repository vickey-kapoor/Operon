"""How a request reaches a model."""

from operon.agent.backends.model_adapter.transports.openai_compat import (
    ModelTransportError,
    OpenAICompatClient,
)
from operon.agent.backends.model_adapter.transports.openai_responses import (
    OpenAIResponsesClient,
)

__all__ = ["ModelTransportError", "OpenAICompatClient", "OpenAIResponsesClient"]
